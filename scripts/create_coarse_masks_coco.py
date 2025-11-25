import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from baseline_unet import UNetSmall
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from coco_dataset import COCOSegmentationDataset
from data_utils import dynamic_augment_collate_fn, seed_worker
import argparse
import albumentations as A


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_train_val_dataloaders(image_size, batch_size):
    """Create dataloaders for train, validation, and test sets."""
    # Transforms for the RGB image
    img_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        # A.Rotate(limit=15, p=0.5, fill=0, fill_mask=-1),
        # A.Affine(translate_percent=(0.1, 0.1), p=0.5, fill=0, fill_mask=-1),
        # A.HorizontalFlip(p=0.5),
    ], additional_targets={"gt_mask": "mask", "coarse_mask": "mask"})

    train_dataset = COCOSegmentationDataset(
        coco_json_path="../coco/annotations/instances_train2017.json",
        images_root="../coco/train2017",
        category={"id": 3, "name": "car"},
    )
    dev_dataset = COCOSegmentationDataset(
        coco_json_path="../coco/annotations/instances_val2017.json",
        images_root="../coco/val2017",
        category={"id": 3, "name": "car"},
    )
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(dev_dataset)}")

    test_size = 16

    test_dataset = Subset(dev_dataset, list(range(len(dev_dataset)))[:test_size])
    
    # 5. Create DataLoaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle for consistent saving
        num_workers=4,
        drop_last=False,  # Don't drop last for complete dataset
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


def generate_and_save_masks(model, dataloader, dataset_dir, split_name, device):
    """Generate and save coarse masks for a given split (train, val, or test)."""
    # Create directories
    os.makedirs(os.path.join(dataset_dir, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split_name, "coarse_masks"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split_name, "gt_masks"), exist_ok=True)
    
    model.eval()  # Set model to evaluation mode
    sample_index = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc=f"Generating {split_name} masks"):
            # COCOSegmentationDataset returns (image, mask, mask)
            # After dynamic_augment_collate_fn, we get (images, coarse_masks_placeholder, gt_masks)
            clean_images, coarse_masks_placeholder, gt_masks = batch
            clean_images = clean_images.to(device)  # (B, 3, H, W)
            
            # Generate coarse masks
            coarse_masks_logits = model(clean_images)  # (B, 1, H, W) raw logits
            
            # Apply sigmoid and threshold to get binary masks
            coarse_masks_prob = torch.sigmoid(coarse_masks_logits)  # (B, 1, H, W) in [0, 1]
            coarse_masks_binary = (coarse_masks_prob > 0.5).float()  # (B, 1, H, W) binary {0, 1}

            # Move to CPU for saving
            clean_images = clean_images.cpu()
            coarse_masks_binary = coarse_masks_binary.cpu()
            gt_masks = gt_masks.cpu()

            # Iterate over each item in the batch
            for i in range(clean_images.shape[0]):
                # Get the individual tensor for this sample
                img_tensor = clean_images[i]  # (3, H, W)
                coarse_mask_tensor = coarse_masks_binary[i]  # (1, H, W)
                gt_mask_tensor = gt_masks[i]  # (1, H, W)
                
                # Define the output file path using a zero-padded index
                file_name = f"{sample_index:06d}.pt"
                
                # Save each tensor individually
                torch.save(img_tensor, os.path.join(dataset_dir, split_name, "images", file_name))
                torch.save(coarse_mask_tensor, os.path.join(dataset_dir, split_name, "coarse_masks", file_name))
                torch.save(gt_mask_tensor, os.path.join(dataset_dir, split_name, "gt_masks", file_name))
                
                sample_index += 1
    
    return sample_index


def main(args):
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Check if model checkpoint exists
    if not os.path.exists(args.model_path):
        raise FileNotFoundError(
            f"Model checkpoint not found at {args.model_path}. "
            f"Please train the model first using train_baseline_coco.py"
        )
    
    # Load model
    print(f"Loading model from {args.model_path}")
    BASELINE_OUTPUT_CHANNELS = 1
    model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    print("Model loaded successfully!")
    
    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader = get_train_val_dataloaders(
        args.image_size,
        args.batch_size,
    )
    
    # Generate masks for training set
    print("\nGenerating masks for training set...")
    train_count = generate_and_save_masks(
        model, train_dataloader, args.output_dir, "train", device
    )
    print(f"Saved {train_count} training samples")
    
    # Generate masks for validation set
    print("\nGenerating masks for validation set...")
    val_count = generate_and_save_masks(
        model, val_dataloader, args.output_dir, "val", device
    )
    print(f"Saved {val_count} validation samples")
    
    # Generate masks for test set
    print("\nGenerating masks for test set...")
    test_count = generate_and_save_masks(
        model, test_dataloader, args.output_dir, "test", device
    )
    print(f"Saved {test_count} test samples")
    
    print(f"\n✓ Coarse masks generated and saved to {args.output_dir}")
    print(f"Total samples saved: {train_count} train + {val_count} val + {test_count} test = {train_count + val_count + test_count}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate coarse masks from trained segmentation model")
    parser.add_argument(
        "--model_path", 
        type=str, 
        default="checkpoints/baseline_unet_coco.pth",
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="data/coco_coarse",
        help="Directory to save generated coarse masks"
    )
    parser.add_argument(
        "--image_size", 
        type=int, 
        default=128,
        help="Image size for resizing"
    )
    parser.add_argument(
        "--batch_size", 
        type=int, 
        default=32,
        help="Batch size for processing"
    )
    
    args = parser.parse_args()
    main(args)