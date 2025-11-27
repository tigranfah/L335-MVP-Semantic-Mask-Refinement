import torch
from torch.utils.data import DataLoader, Subset, Dataset
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
import random
import numpy as np
import string
import os
import glob
import albumentations as A

from scripts.coco_dataset import COCOSegmentationDataset


def random_blob_image(
    height=256,
    width=256,
    num_blobs=10,
    blob_radius=10,
    device="cpu"
):
    # Base empty image
    img = torch.zeros((1, 1, height, width), device=device)

    for _ in range(num_blobs):
        # Random center
        cx = torch.randint(0, width, (1,))
        cy = torch.randint(0, height, (1,))
        
        # Random amplitude for blob
        amp = torch.rand(1).item() * 1.0
        
        # Create a 2D Gaussian kernel
        y = torch.arange(height, device=device).view(-1, 1)
        x = torch.arange(width, device=device).view(1, -1)

        blob = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * blob_radius ** 2))
        blob = blob * amp
        
        img += blob

    # Normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return (img.squeeze() >= 0.5)  # shape: (H, W)

def block_image(img, num_blobs, blob_radius, device):
    blob = random_blob_image(img.shape[1], img.shape[2], num_blobs=num_blobs, blob_radius=blob_radius).to(device)
    blob_img = blob.logical_not().unsqueeze(0) * img.to(device)
    return blob_img

# def get_loaders(image_size=256, batch_size=32, data_split=(0.8, 0.1, 0.1), num_workers=2, root="./data",):
#     """
#     Creates and returns DataLoaders for the Oxford-IIIT Pet dataset.
#     Splits dataset into Train / Refiner / Dev / Test.

#     Args:
#         image_size: image resolution
#         batch_size: batch size for DataLoaders
#         data_split: tuple of 4 floats (train, refiner, dev, test) summing to 1
#         num_workers: DataLoader worker count
#         root: dataset directory

#     Returns:
#         dict of DataLoaders:
#             {
#                 'train': ...,
#                 'dev': ...,
#                 'test': ...,
#             }
#     """

#     assert abs(sum(data_split) - 1.0) == 0, "Splits must sum to 1.0"

#     # Transforms
#     image_transform = transforms.Compose([
#         transforms.Resize((image_size, image_size)),
#         transforms.ToTensor(),  # [0, 1]
#         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
#     ])

#     mask_transform = transforms.Compose([
#         transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
#         transforms.PILToTensor(),
#         lambda x: ((x == 1) | (x == 3)).float(),  # binary pet mask
#         transforms.Normalize((0.5,), (0.5,))  # [0,1] → [-1,1]
#     ])

#     # Dataset
#     dataset = OxfordIIITPet(
#         root=root,
#         split="trainval",
#         download=True,
#         target_types="segmentation",
#         transform=image_transform,
#         target_transform=mask_transform,
#     )

#     # Indices and splits
#     indices = list(range(len(dataset)))
#     random.seed(42)
#     random.shuffle(indices)
#     total_len = len(indices)

#     train_end = int(data_split[0] * total_len)
#     dev_end = int((data_split[0] + data_split[1]) * total_len)

#     train_idx = indices[:train_end]
#     dev_idx = indices[train_end:dev_end]
#     test_idx = indices[dev_end:]

#     print(f"Total samples: {total_len}")
#     print(f"Train: {len(train_idx)} | Dev: {len(dev_idx)} | Test: {len(test_idx)}")

#     # Subsets
#     train_dataset = Subset(dataset, train_idx)
#     dev_dataset = Subset(dataset, dev_idx)
#     test_dataset = Subset(dataset, test_idx)

#     # DataLoaders
#     loaders = {
#         "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
#         "dev": DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
#         "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers),
#     }

#     return loaders

def dynamic_augment_collate_fn(batch_data, image_transforms):
    augmented_images = []
    augmented_gt_masks = []
    augmented_coarse_masks = []
    for batch in batch_data:
        image, coarse_mask, gt_mask = batch

        # image: [C, H, W] float -> [H, W, C] float
        image = image.permute(1, 2, 0).cpu().numpy()
        # masks: [H, W] long -> [H, W] long
        gt_mask = gt_mask.cpu().numpy()
        coarse_mask = coarse_mask.cpu().numpy()
        
        transformed = image_transforms(image=image, gt_mask=gt_mask, coarse_mask=coarse_mask)
        image = transformed["image"]
        gt_mask = transformed["gt_mask"]
        coarse_mask = transformed["coarse_mask"]

        # image: [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        # masks: [H, W] -> [H, W]
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0).to(image.dtype)
        coarse_mask = torch.from_numpy(coarse_mask).unsqueeze(0).to(image.dtype)
        # accumulate augmented images, coarse masks, and gt masks
        augmented_images.append(image)
        augmented_gt_masks.append(gt_mask)
        augmented_coarse_masks.append(coarse_mask)

    return torch.stack(augmented_images), torch.stack(augmented_coarse_masks), torch.stack(augmented_gt_masks)

def seed_worker(worker_id):
    base_seed = torch.initial_seed()
    np.random.seed(base_seed % (2**32 - 1))
    random.seed(base_seed % (2**32 - 1))

def generate_random_id(length=6):
    return "".join(random.sample(string.ascii_lowercase + string.digits, k=length))

class CoarseOxfordIIITPet(Dataset):
    """
    A PyTorch Dataset that loads pre-generated tensor files from disk.
    
    It assumes a directory structure created by 'pregenerate_dataset.py':
    root_dir/
    ├── images/
    │   ├── 000000.pt
    │   ├── 000001.pt
    │   └── ...
    ├── coarse_masks/
    │   ├── 000000.pt
    │   └── ...
    └── gt_masks/
        ├── 000000.pt
        └── ...
    """
    def __init__(
        self, root_dir
    ):
        self.root_dir = root_dir
        
        self.image_dir = os.path.join(root_dir, "images")
        self.coarse_mask_dir = os.path.join(root_dir, "coarse_masks")
        self.gt_mask_dir = os.path.join(root_dir, "gt_masks")
        
        # Get the list of file names (e.g., "000000.pt")
        # We assume all directories are in sync
        self.file_names = sorted(
            [os.path.basename(f) for f in glob.glob(os.path.join(self.image_dir, "*.pt"))]
        )
        
        if not self.file_names:
            raise FileNotFoundError(f"No '.pt' files found in {self.image_dir}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # Load the pre-saved tensors
        image = torch.load(os.path.join(self.image_dir, file_name))
        coarse_mask = torch.load(os.path.join(self.coarse_mask_dir, file_name))
        gt_mask = torch.load(os.path.join(self.gt_mask_dir, file_name))
        
        return image, coarse_mask, gt_mask
    

def augment_batch(batch_data, image_transforms):
    augmented_images = []
    augmented_coarse_masks = []
    augmented_gt_masks = []

    images, coarse_masks, gt_masks = batch_data
    for image, coarse_mask, gt_mask in zip(images, coarse_masks, gt_masks, strict=True):
        # image: [C, H, W] float -> [H, W, C] float
        image = image.permute(1, 2, 0).cpu().numpy()
        # masks: [H, W] long -> [H, W] long
        coarse_mask = coarse_mask.cpu().numpy().squeeze(0)
        gt_mask = gt_mask.cpu().numpy().squeeze(0)
        
        transformed = image_transforms(image=image, coarse_mask=coarse_mask, gt_mask=gt_mask)
        image = transformed["image"]
        coarse_mask = transformed["coarse_mask"]
        gt_mask = transformed["gt_mask"]

        # image: [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        # masks: [H, W] -> [H, W]
        coarse_mask = torch.from_numpy(coarse_mask).unsqueeze(0)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)

        # accumulate augmented images, coarse masks, and gt masks
        augmented_images.append(image)
        augmented_coarse_masks.append(coarse_mask)
        augmented_gt_masks.append(gt_mask)

    return torch.stack(augmented_images), torch.stack(augmented_coarse_masks), torch.stack(augmented_gt_masks)

def oxford_get_train_val_dataloaders(batch_size, root_dir):
    train_dataset = CoarseOxfordIIITPet(
        os.path.join(root_dir, "train")
    )
    val_dataset = CoarseOxfordIIITPet(
        os.path.join(root_dir, "val")
    )
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    test_size = 16

    test_dataset = Subset(val_dataset, list(range(len(val_dataset)))[:test_size])
    
    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )
    
    return train_dataloader, val_dataloader, test_dataloader

def generate_random_id(length=6):
    return "".join(random.sample(string.ascii_lowercase + string.digits, k=length))

def coco_get_train_val_dataloaders(image_size, batch_size, coco_path):
    # Transforms for the RGB image
    img_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        # A.Rotate(limit=15, p=0.5, fill=0, fill_mask=-1),
        # A.Affine(translate_percent=(0.1, 0.1), p=0.5, fill=0, fill_mask=-1),
        # A.HorizontalFlip(p=0.5),
    ], additional_targets={"gt_mask": "mask", "coarse_mask": "mask"})

    train_dataset = COCOSegmentationDataset(
        coco_json_path=os.path.join(coco_path, "annotations/instances_train2017.json"),
        images_root=os.path.join(coco_path, "train2017"),
        category={"id": 3, "name": "car"},
    )
    dev_dataset = COCOSegmentationDataset(
        coco_json_path=os.path.join(coco_path, "annotations/instances_val2017.json"),
        images_root=os.path.join(coco_path, "val2017"),
        category={"id": 3, "name": "car"},
    )
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(dev_dataset)}")

    test_size = 16

    test_dataset = Subset(dev_dataset, list(range(len(dev_dataset)))[:test_size])
    
    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        collate_fn=lambda x: dynamic_augment_collate_fn(x, img_transforms),
        worker_init_fn=seed_worker,
    )
    val_dataloader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        collate_fn=lambda x: dynamic_augment_collate_fn(x, img_transforms),
        worker_init_fn=seed_worker,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=test_size,
        shuffle=False,
        num_workers=2,
        drop_last=False,
        collate_fn=lambda x: dynamic_augment_collate_fn(x, img_transforms),
        worker_init_fn=seed_worker,
    )
    
    return train_dataloader, val_dataloader, test_dataloader