import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, DataLoader, Subset
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from tqdm.auto import tqdm
from PIL import Image
import os
from matplotlib import pyplot as plt
import random

#ugly import of model, could be made a package
import sys
from pathlib import Path
from model.baseline_unet import UNetSmall

if __name__ == "__main__":

    image_size = 256
    BASELINE_OUTPUT_CHANNELS = 1 #with one chanel we can do background + (pet&boundry)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64

    # Transforms for the RGB image
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),  # [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # → [-1, 1]
    ])
    
    # Transforms for the segmentation mask
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
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
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False)
    dev_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)

    # Model checkpoint path
    model_checkpoint_path = "checkpoints/baseline_unet.pth"
    os.makedirs("checkpoints", exist_ok=True)
    
    # Load model if checkpoint exists, otherwise create new model
    model_loaded = os.path.exists(model_checkpoint_path)
    if model_loaded:
        print(f"Loading model from {model_checkpoint_path}")
        baseline_model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)
        baseline_model.load_state_dict(torch.load(model_checkpoint_path, map_location=device))
        print("Model loaded successfully!")
    else:
        print("Creating new model...")
        baseline_model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)
    
    total_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline model parameters: {total_params:,}")

    # Only train if model was not loaded
    if not model_loaded:
        num_epochs=75
        # For binary segmentation with 1 output channel, use BCEWithLogitsLoss
        # Masks: 1=Pet, 2=Background, 3=Boundary -> binary: Pet+Boundary (1) vs Background (0)
        criterion = nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(baseline_model.parameters(), lr=1e-4)
        baseline_model.train()

        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                images = images.to(device)           # (B, 3, H, W) in [-1, 1]
                masks = masks.to(device)             # (B, 1, H, W) binary mask normalized to [-1, 1]

                optimizer.zero_grad()
                outputs = baseline_model(images)     # (B, 1, H, W) raw logits

                # BCEWithLogitsLoss expects (B, 1, H, W) logits and (B, 1, H, W) targets in [0, 1]
                # Masks are normalized to [-1, 1] by transform, so denormalize back to [0, 1]
                # Denormalize: (x + 1) / 2 converts [-1, 1] -> [0, 1]
                binary_targets = (masks + 1) / 2  # (B, 1, H, W) with values in [0, 1]
                loss = criterion(outputs, binary_targets)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

        # Save model after training
        print(f"\nSaving model to {model_checkpoint_path}")
        torch.save(baseline_model.state_dict(), model_checkpoint_path)
        print("Model saved successfully!")
    else:
        print("\nSkipping training (model already loaded)")

    # Generate and save coarse masks separately for train and validation sets
    print("\nGenerating coarse masks...")
    baseline_model.eval()  # Set model to evaluation mode
    
    def generate_and_save_masks(dataloader, dataset_dir, split_name):
        """Generate and save coarse masks for a given split (train or val)"""
        os.makedirs(os.path.join(dataset_dir, split_name, "images"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split_name, "coarse_masks"), exist_ok=True)
        os.makedirs(os.path.join(dataset_dir, split_name, "gt_masks"), exist_ok=True)
        
        sample_index = 0
        with torch.no_grad():
            for images, gt_masks in tqdm(dataloader, desc=f"Generating {split_name} masks"):
                images = images.to(device)  # (B, 3, H, W) in [-1, 1]
                
                # Generate coarse masks
                coarse_masks_logits = baseline_model(images)  # (B, 1, H, W) raw logits
                thresholded_masks = (torch.sigmoid(coarse_masks_logits) > 0.5).float()
                coarse_masks = (thresholded_masks - 0.5) * 2 # [-1, 1]

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
                    torch.save(coarse_mask_tensor, os.path.join(dataset_dir, split_name, "coarse_masks", file_name))
                    torch.save(gt_mask_tensor, os.path.join(dataset_dir, split_name, "gt_masks", file_name))
                    
                    sample_index += 1
        
        return sample_index
    
    dataset_dir = "data/oxcoarse"
    
    # Generate masks for training set (model was NOT trained on this)
    print("\nGenerating masks for training set...")
    train_count = generate_and_save_masks(train_loader, dataset_dir, "train")
    print(f"Saved {train_count} training samples")
    
    # Generate masks for DEV set (model was NOT trained on this)
    print("\nGenerating masks for dev set...")
    dev_count = generate_and_save_masks(dev_loader, dataset_dir, "val")
    print(f"Saved {dev_count} validation samples")
    
    print(f"\nCoarse masks generated and saved to {dataset_dir}")
    print(f"Total samples saved: {train_count} train + {dev_count} val = {train_count + dev_count}")





    # sample_index = 0

    # dataset_dir = "data/oxcoarse"
    # os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    # os.makedirs(os.path.join(dataset_dir, "coarse_masks"), exist_ok=True)
    # os.makedirs(os.path.join(dataset_dir, "gt_masks"), exist_ok=True)

    # for batch in tqdm(dataloader, desc="Generating and saving samples"):
    #     images, gt_masks = batch

    #     # Images are already tensors from the dataset, now normalize for model input
    #     normalized_images = model_input_transform(images.float())
    #     with torch.no_grad():
    #         coarse_masks = loaded_model(normalized_images)
    #         coarse_masks = coarse_masks.sigmoid()

    #     # Iterate over each item in the batch
    #     for i in range(images.shape[0]):
    #         # Get the individual tensor for this sample
    #         img_tensor = images[i]
    #         coarse_mask_tensor = coarse_masks[i]
    #         gt_mask_tensor = gt_masks[i]
            
    #         # Define the output file path using a zero-padded index
    #         file_name = f"{sample_index:06d}.pt"
            
    #         # Save each tensor individually
    #         # Use args.output_dir
    #         torch.save(img_tensor, os.path.join(dataset_dir, "images", file_name))
    #         torch.save(coarse_mask_tensor, os.path.join(dataset_dir, "coarse_masks", file_name))
    #         torch.save(gt_mask_tensor, os.path.join(dataset_dir, "gt_masks", file_name))
            
    #         sample_index += 1