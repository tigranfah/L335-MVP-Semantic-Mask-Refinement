# requirements:
# pip install pillow numpy pycocotools torch torchvision

import os
import json
from pathlib import Path
from collections import defaultdict
from typing import Optional, Callable, Tuple, List, Dict

import numpy as np
from PIL import Image, ImageDraw

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms.functional as TF

# pycocotools for RLE / polygon -> mask
from pycocotools import mask as mask_utils


class COCOSegmentationDataset(Dataset):
    """
    COCO instances -> semantic segmentation dataset.

    Returns: (image_tensor: FloatTensor[C,H,W], mask_tensor: LongTensor[H,W])
    mask values are contiguous class indices starting from 0 (0 = background).
    """

    def __init__(
        self,
        coco_json_path: str,
        images_root: str,
        category: dict,
        transforms: Optional[Callable[[Image.Image, Image.Image], Tuple[Image.Image, Image.Image]]] = None,
        keep_crowd: bool = False,
        skip_images_without_annotations: bool = True,
    ):
        """
        coco_json_path: path to instances_train*.json
        images_root: directory with the image files (file_name from JSON is relative to this)
        transforms: callable that accepts (PIL.Image, PIL.Image(mask)) and returns (image, mask)
          - transforms should be "joint" so image and mask align
        keep_crowd: if False, skip annotations with iscrowd==1
        skip_images_without_annotations: if True, dataset excludes images with no anns
        """
        self.coco_json_path = coco_json_path
        self.images_root = Path(images_root)
        self.transforms = transforms
        self.keep_crowd = keep_crowd

        with open(coco_json_path, "r") as f:
            coco = json.load(f)

        # Build image index and annotations index
        self.images = {img["id"]: img for img in coco["images"]}
        anns_by_image = defaultdict(list)
        for ann in coco["annotations"]:
            if (not keep_crowd) and ann.get("iscrowd", 0) == 1:
                # skipping crowd annotations by default
                continue
            if ann["category_id"] != category["id"]:
                continue
            anns_by_image[ann["image_id"]].append(ann)
        self.anns_by_image = dict(anns_by_image)

        # Category id -> name (from COCO)
        self.catid_to_name = {category["id"]: category["name"]}
        # print(self.catid_to_name)
        # Build mapping to contiguous labels: 0 = background, 1..N = classes
        sorted_cat_ids = sorted(self.catid_to_name.keys())
        self.catid_to_contiguous = {cid: i + 1 for i, cid in enumerate(sorted_cat_ids)}
        # print(self.catid_to_contiguous)
        # Map background (id 0) implicitly
        self.contiguous_to_catid = {v: k for k, v in self.catid_to_contiguous.items()}
        self.num_classes = len(sorted_cat_ids) + 1  # include background

        # Build list of image_ids we will serve
        if skip_images_without_annotations:
            self.image_ids = [img_id for img_id, img in self.images.items() if img_id in self.anns_by_image]
        else:
            self.image_ids = list(self.images.keys())

    def __len__(self):
        return len(self.image_ids)

    def _annotation_mask_to_binary(self, ann, h: int, w: int) -> np.ndarray:
        """
        Convert a single annotation's segmentation to a binary mask (h,w) uint8.
        Handles polygon and RLE. Uses pycocotools for RLE and frPyObjects.
        """
        seg = ann.get("segmentation", None)
        if seg is None:
            return np.zeros((h, w), dtype=np.uint8)

        # If segmentation is a dict -> RLE
        if isinstance(seg, dict):
            # segmentation: {"size": [h,w], "counts": ...}
            rle = seg
            mask = mask_utils.decode(rle)
            if mask.ndim == 3:
                # pycocotools may return (H,W,1) or multiple masks; collapse
                mask = mask.sum(axis=2)
            return (mask > 0).astype(np.uint8)

        # If segmentation is list:
        # either polygon(s): [[x1,y1, x2,y2, ...], ...]
        # or RLE encoded as list of ints (uncommon here)
        if isinstance(seg, list):
            # polygons = list of polygons OR RLE list
            # determine which: if list of floats and first element is a number of coords? assume polygons.
            if len(seg) == 0:
                return np.zeros((h, w), dtype=np.uint8)

            # If first element is a list -> polygon list
            if isinstance(seg[0], list):
                # polygons (many)
                mask = np.zeros((h, w), dtype=np.uint8)
                for poly in seg:
                    # pycocotools expects an RLE-like structure for frPyObjects
                    rles = mask_utils.frPyObjects([poly], h, w)
                    m = mask_utils.decode(rles)
                    if m.ndim == 3:
                        # m may be (H,W,1)
                        m = m.sum(axis=2)
                    mask = np.logical_or(mask, m > 0)
                return mask.astype(np.uint8)
            else:
                # If seg is a list of ints -> it might be RLE (uncompressed)
                # we can try to use mask_utils.frPyObjects directly
                try:
                    rle = mask_utils.frPyObjects(seg, h, w)
                    mask = mask_utils.decode(rle)
                    if mask.ndim == 3:
                        mask = mask.sum(axis=2)
                    return (mask > 0).astype(np.uint8)
                except Exception:
                    # fallback: return zeros
                    return np.zeros((h, w), dtype=np.uint8)

        # default fallback
        return np.zeros((h, w), dtype=np.uint8)

    def _build_semantic_mask(self, image_id: int) -> np.ndarray:
        """
        Build a H x W int mask with contiguous labels.
        0 == background, 1..N == classes
        """
        img_info = self.images[image_id]
        w, h = img_info["width"], img_info["height"]
        mask = np.zeros((h, w), dtype=np.uint16)  # use uint16 to allow lots of classes

        anns = self.anns_by_image[image_id]
        # Option: sort by area so smaller objects overwrite bigger ones or vice versa
        # anns = sorted(anns, key=lambda x: x.get("area", 0))

        for ann in anns:
            cat_id = ann["category_id"]
            if cat_id not in self.catid_to_contiguous:
                # unseen category (rare) -> skip
                continue
            label = self.catid_to_contiguous[cat_id]
            binary = self._annotation_mask_to_binary(ann, h, w)  # uint8 (0/1)
            if binary.sum() == 0:
                continue
            # assign label where binary==1. Later anns overwrite earlier ones.
            mask[binary.astype(bool)] = label

        return mask  # dtype uint16

    def __getitem__(self, idx: int):
        image_id = self.image_ids[idx]
        img_info = self.images[image_id]
        file_name = img_info["file_name"]
        img_path = self.images_root / file_name
        image = Image.open(img_path).convert("RGB")

        # build mask
        mask_np = self._build_semantic_mask(image_id)  # H x W (uint16)
        mask = Image.fromarray(mask_np.astype(np.uint16), mode="I;16")  # PIL mode for 16-bit

        # convert to tensors
        image_tensor = TF.to_tensor(image)  # float32 CxHxW, values [0,1]
        # normalize if you want with TF.normalize(image_tensor, mean, std)

        # mask: convert to LongTensor HxW
        mask_arr = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_arr).long()

        return image_tensor, mask_tensor, mask_tensor


if __name__ == "__main__":
    train_dataset = COCOSegmentationDataset(
        coco_json_path="../coco/annotations/instances_train2017.json",
        images_root="../coco/train2017",
        category={"id": 3, "name": "car"},
        transforms=None
    )
    val_dataset = COCOSegmentationDataset(
        coco_json_path="../coco/annotations/instances_val2017.json",
        images_root="../coco/val2017",
        category={"id": 3, "name": "car"},
        transforms=None
    )
    print(len(train_dataset), len(val_dataset))