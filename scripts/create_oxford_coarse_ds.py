from argparse import ArgumentParser
import torch
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset
from tqdm.auto import tqdm
import os
import random

#ugly import of model, could be made a package
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from model.baseline_unet import UNetSmall


def generate_and_save_masks(baseline_model, dataloader, dataset_dir, split_name, device):
    """Generate and save coarse masks for a given split (train or val)"""
    os.makedirs(os.path.join(dataset_dir, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split_name, "gt_masks"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split_name, "coarse_masks"), exist_ok=True)
    
    sample_index = 0
    with torch.no_grad():
        for images, gt_masks in tqdm(dataloader, desc=f"Generating {split_name} masks"):
            images = images.to(device)  # (B, 3, H, W) in [-1, 1]
            
            # Generate coarse masks
            if baseline_model:
                coarse_masks_logits = baseline_model(images)  # (B, 1, H, W) raw logits
                coarse_masks = (torch.sigmoid(coarse_masks_logits) - 0.5) * 2 # [-1, 1]
            else:
                coarse_masks = torch.zeros_like(gt_masks)

            # Iterate over each item in the batch
            for i in range(images.shape[0]):
                # Get the individual tensor for this sample
                img_tensor = images[i].cpu()  # (3, H, W)
                coarse_mask_tensor = coarse_masks[i].cpu()  # (1, H, W)
                gt_mask_tensor = gt_masks[i].cpu()  # (1, H, W)
                
                # Define the output file path using a zero-padded index
                file_name = f"{sample_index:06d}.pt"
                
                # Save each tensor individually
                torch.save(img_tensor, os.path.join(dataset_dir, split_name, "images", file_name))
                torch.save(gt_mask_tensor, os.path.join(dataset_dir, split_name, "gt_masks", file_name))
                torch.save(coarse_mask_tensor, os.path.join(dataset_dir, split_name, "coarse_masks", file_name))
                
                sample_index += 1
    
    return sample_index


def create_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Transforms for the RGB image
    image_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
    ])
    
    # Transforms for the segmentation mask
    mask_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),
        # The mask values are 1 (pet), 2 (background), 3 (border).
        # We only care about the pet. Let's make a binary mask.
        lambda x: ((x == 1) | (x == 3)).float(), # 1 if pet, 0 if background/border
        transforms.Normalize((0.5,), (0.5,)) # Normalize [0, 1] to [-1, 1]
    ])

    dataset = OxfordIIITPet(
        root="./data",
        split="trainval", 
        download=True, 
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform
    )
    
    # ============================================================
    # Split dataset into Train / Refiner / Dev / Test (40 / 40 / 10 / 10)
    # ============================================================
    indices = list(range(len(dataset)))
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    
    train_end = int(0.8 * len(indices))
    dev_end = int(0.9 * len(indices))

    train_indices = indices[:train_end]
    dev_indices = indices[train_end:dev_end]
    test_indices = indices[dev_end:]

    print(f"Total samples: {len(dataset)}")
    print(f"Train: {len(train_indices)} | "
        f"Dev: {len(dev_indices)} | Test: {len(test_indices)}")

    # Create subsets
    train_dataset = Subset(dataset, train_indices)
    dev_dataset = Subset(dataset, dev_indices)
    test_dataset = Subset(dataset, test_indices)

    # ------------------------------------------------------------
    # Data loaders
    # ------------------------------------------------------------
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, drop_last=False)

    # Load model if checkpoint exists, otherwise create new model
    if args.checkpoint_path:
        baseline_model = UNetSmall(in_ch=3, out_ch=1).to(device)
        baseline_model.load_state_dict(torch.load(args.checkpoint_path))
        baseline_model.to(device)
        baseline_model.eval()  # Set model to evaluation mode
    
        total_params = sum(p.numel() for p in baseline_model.parameters())
        print(f"Baseline model parameters: {total_params:,}")
    else:
        baseline_model = None
    
    # Generate masks for training set (model was NOT trained on this)
    print("\nGenerating masks for training set...")
    train_count = generate_and_save_masks(baseline_model, train_loader, args.dataset_dir, "train", device)
    print(f"Saved {train_count} training samples")
    
    # Generate masks for DEV set (model was NOT trained on this)
    print("\nGenerating masks for dev set...")
    dev_count = generate_and_save_masks(baseline_model, dev_loader, args.dataset_dir, "val", device)
    print(f"Saved {dev_count} validation samples")

    # Generate masks for TEST set (model was NOT trained on this)
    print("\nGenerating masks for test set...")
    test_count = generate_and_save_masks(baseline_model, test_loader, args.dataset_dir, "test", device)
    print(f"Saved {test_count} test samples")
    
    print(f"\nCoarse masks generated and saved to {args.dataset_dir}")
    print(f"Total samples saved: {train_count} train + {dev_count} val = {train_count + dev_count}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=False, default=None)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    create_dataset(args)