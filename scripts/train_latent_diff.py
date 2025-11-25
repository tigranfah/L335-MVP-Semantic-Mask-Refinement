import torch
import numpy as np

from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers import AutoencoderKL

from torch.utils.data import DataLoader, Dataset, Subset

from model.mask_autoencoder import MaskAutoencoder
from scripts.data_utils import get_loaders

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt

from typing import NoReturn
import torch
import torch.nn.functional as F
from torchvision.datasets import OxfordIIITPet
from torchvision import transforms
import albumentations as A
import numpy as np
import torchvision.tv_tensors as tv_tensors
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio
import random
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os
import torchvision
# import wandb
import random
import string
import argparse
import glob

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
        self, root_dir,
        img_transforms
    ):
        self.root_dir = root_dir
        self.img_transforms = img_transforms
        
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

        # image: [C, H, W] float -> [H, W, C] float
        image = image.permute(1, 2, 0).cpu().numpy()
        # masks: [H, W] long -> [H, W] long
        coarse_mask = coarse_mask.cpu().numpy().squeeze(0)
        gt_mask = gt_mask.cpu().numpy().squeeze(0)
        
        transformed = self.img_transforms(image=image, coarse_mask=coarse_mask, gt_mask=gt_mask)
        image = transformed["image"]
        coarse_mask = transformed["coarse_mask"]
        gt_mask = transformed["gt_mask"]

        # image: [H, W, C] -> [C, H, W]
        image = torch.from_numpy(image).permute(2, 0, 1)
        # masks: [H, W] -> [H, W]
        coarse_mask = torch.from_numpy(coarse_mask).unsqueeze(0)
        gt_mask = torch.from_numpy(gt_mask).unsqueeze(0)
        
        return image, coarse_mask, gt_mask

def get_train_val_dataloaders(image_size, batch_size, root_dir, val_split=0.2):    
    # Transforms for the RGB image
    img_transforms = A.Compose([
        A.Resize(height=image_size, width=image_size),
        A.Rotate(limit=15, p=0.5, fill=0, fill_mask=-1),
        A.Affine(translate_percent=(0.1, 0.1), p=0.5, fill=0, fill_mask=-1),
        A.HorizontalFlip(p=0.5),
    ], additional_targets={"coarse_mask": "mask", "gt_mask": "mask"})

    train_dataset = CoarseOxfordIIITPet(
        os.path.join(root_dir, "train"),
        img_transforms=img_transforms
    )
    dev_dataset = CoarseOxfordIIITPet(
        os.path.join(root_dir, "dev"),
        img_transforms=img_transforms
    )
    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(dev_dataset)}")

    test_size = 16

    test_dataset = Subset(dev_dataset, list(range(len(dev_dataset)))[:test_size])
    
    # 5. Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, num_workers=2, drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader

def sample_and_save_images(
    model,
    noise_scheduler,
    test_dataloader,
    device,
    epoch,
    output_dir,
    metrics_dict
):
    """Samples and saves a grid of images for visual inspection."""
    model.eval()
    
    batch = next(iter(test_dataloader))
    clean_images, coarse_mask, gt_mask = batch
    clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
    
    batch_size = clean_images.shape[0]
    
    # encode  to latent space
    with torch.no_grad():
        # clean_images = vae.encode(clean_images).latent_dist.sample()
        # coarse_mask = vae_mask.encode(coarse_mask).latent_dist.sample()
        # gt_mask = vae_mask.encode(gt_mask).latent_dist.sample()
        image_latents_dist = image_autoencoder.encode(clean_images).latent_dist
        image_latents = image_latents_dist.sample()
        image_latents = image_latents * 0.18215
        
        masks_conv = coarse_mask.repeat(1, 3, 1, 1)
        masks_latents_dist = image_autoencoder.encode(masks_conv).latent_dist
        masks_latents = masks_latents_dist.sample()
        masks_latents = masks_latents * 0.18215

    # Start with pure noise for the mask
    noisy_masks = torch.randn_like(masks_latents)
    
    with torch.no_grad():
        # Loop backwards through the diffusion timesteps
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            
            # 1. Concatenate the noisy mask and the condition (RGB image)
            # Input shape: (batch_size, 4, H, W)
            model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)
            
            # 2. Predict the noise
            noise_pred = model(model_input, t).sample

            # 3. Use the scheduler to "denoise" one step
            scheduler_output = noise_scheduler.step(noise_pred, t, noisy_masks)
            noisy_masks = scheduler_output.prev_sample

        # Update the coarse mask
        # noisy_masks = coarse_mask - noisy_masks

    # diff_mask = coarse_mask - gt_mask
    # diff_mask = (diff_mask * 0.5 + 0.5).clamp(0, 1)
    # diff_mask = diff_mask.repeat(1, 3, 1, 1)

    # --- Un-normalize all images for saving ---
    # Un-normalize from [-1, 1] to [0, 1]

    # decode the produced mask from latent space
    with torch.no_grad():
        predicted_masks = image_autoencoder.decode(noisy_masks / 0.18215).sample
        predicted_masks = torch.mean(predicted_masks, dim=1).unsqueeze(1)

    clean_images = (clean_images * 0.5 + 0.5).clamp(0, 1)
    coarse_mask = (coarse_mask * 0.5 + 0.5).clamp(0, 1)
    predicted_masks = (predicted_masks * 0.5 + 0.5).clamp(0, 1)
    gt_mask = (gt_mask * 0.5 + 0.5).clamp(0, 1)
    
    # Make masks 3-channel for grid
    coarse_mask = coarse_mask.repeat(1, 3, 1, 1)
    predicted_masks = predicted_masks.repeat(1, 3, 1, 1)
    gt_mask = gt_mask.repeat(1, 3, 1, 1)

    iou_score = metrics_dict["iou"](
        predicted_masks, 
        gt_mask
    )
    psnr = metrics_dict["psnr"](
        predicted_masks, 
        gt_mask
    )
    baseline_iou_score = metrics_dict["iou"](
        coarse_mask,
        gt_mask
    )
    baseline_psnr = metrics_dict["psnr"](
        coarse_mask, 
        gt_mask
    )
    # wandb.log({
    #     "val/baseline_iou_score": baseline_iou_score,
    #     "val/baseline_psnr": baseline_psnr,
    #     "val/iou_score": iou_score,
    #     "val/psnr": psnr,
    # })

    # Create a grid: [Image | Predicted Mask | Ground Truth]
    comparison_grid = torch.cat([clean_images, coarse_mask, predicted_masks, gt_mask], dim=0)
    
    # Save the grid
    save_path = os.path.join(output_dir, f"sample_epoch_{epoch}.png")
    torchvision.utils.save_image(comparison_grid, save_path, nrow=clean_images.shape[0])
    print(f"Saved sample grid to {save_path}")

    model.train()

def validate(model, noise_scheduler, val_dataloader, device):
    """Validates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            clean_images, coarse_mask, gt_mask = batch
            clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
            
            batch_size = clean_images.shape[0]
            
            # encode  to latent space
            with torch.no_grad():
                # clean_images = vae.encode(clean_images).latent_dist.sample()
                # coarse_mask = vae_mask.encode(coarse_mask).latent_dist.sample()
                # gt_mask = vae_mask.encode(gt_mask).latent_dist.sample()
                image_latents_dist = image_autoencoder.encode(clean_images).latent_dist
                image_latents = image_latents_dist.sample()
                image_latents = image_latents * 0.18215
                
                masks_conv = coarse_mask.repeat(1, 3, 1, 1)
                masks_latents_dist = image_autoencoder.encode(masks_conv).latent_dist
                masks_latents = masks_latents_dist.sample()
                masks_latents = masks_latents * 0.18215

                gt_conv = gt_mask.repeat(1, 3, 1, 1)
                gt_latents_dist = image_autoencoder.encode(gt_conv).latent_dist
                gt_latents = gt_latents_dist.sample()
                gt_latents = gt_latents * 0.18215

            # 1. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()

            # 2. Sample a random noise map *for the mask*
            noise = torch.randn_like(masks_latents)

            # 3. Add noise to the *clean masks*
            noisy_masks = noise_scheduler.add_noise(gt_latents, noise, timesteps)
            
            # 4. Concatenate noisy mask and clean image
            #    Input shape: (batch_size, 4, H, W)
            model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)
            
            # 5. Get the model's noise prediction
            noise_pred = model(model_input, timesteps).sample

            # 6. Calculate the loss (MSE on the noise)
            loss = F.mse_loss(noise_pred, noise)

            # clean_images, coarse_mask, gt_mask = batch
            # clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
            
            # batch_size = clean_images.shape[0]

            # # encode the images to latent space
            # with torch.no_grad():
            #     clean_images = vae.encode(clean_images).latent_dist.sample()
            #     coarse_mask = vae_mask.encode(coarse_mask).latent_dist.sample()
            #     gt_mask = vae_mask.encode(gt_mask).latent_dist.sample()

            # # 1. Sample random timesteps
            # timesteps = torch.randint(
            #     0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            # ).long()

            # # 2. Sample a random noise map *for the mask*
            # noise = torch.randn_like(coarse_mask)

            # # 3. Add noise to the *clean masks*
            # noisy_masks = noise_scheduler.add_noise(gt_mask, noise, timesteps)
            
            # # 4. Concatenate noisy mask and clean image
            # model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
            
            # # 5. Get the model's noise prediction
            # noise_pred = model(model_input, timesteps).sample
            
            # # 6. Calculate the loss (MSE on the noise)
            # loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"device: {device}")
image_size = 256
batch_size = 32
val_split = 0.2

data_root_dir = "./data"

jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device)
psnr_metric = PeakSignalNoiseRatio().to(device)

# dataloaders
print("Loading dataset...")
# loaders = get_loaders(image_size=image_size, batch_size=batch_size)
# train_loader = loaders["train"]
# dev_loader = loaders["dev"]
# test_loader = loaders["test"]

train_loader, val_loader, test_loader_for_sampling = get_train_val_dataloaders(
        image_size,
        batch_size,
        os.path.join(data_root_dir, "oxcoarse"),
        val_split=val_split
    )

# Image Autoencoder
url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
image_autoencoder = AutoencoderKL.from_single_file(url).to(device)
image_autoencoder.eval()

# Mask Autoencoder
# LATENT_DIM = 128
# mask_autoencoder = MaskAutoencoder(
#     base_channel_size=32,
#     latent_dim=LATENT_DIM,
#     num_output_channels=1  # Single channel output for binary masks
# ).to(device)
# weights = torch.load("checkpoints/autoencoder.pth", map_location="cpu").get("model_state_dict")
# mask_autoencoder.load_state_dict(weights)
# mask_autoencoder.to(device)
# mask_autoencoder.eval()

# num_samples = 4

# images, masks = next(iter(train_loader))
# images = images[:num_samples].to(device)
# masks = masks[:num_samples].to(device)
# # images = images[0].to(device)
# # masks = masks[0].to(device)

# with torch.no_grad():
#     print(images.shape)
#     image_latents_dist = image_autoencoder.encode(images).latent_dist
#     image_latents = image_latents_dist.sample()
#     image_latents = image_latents * 0.18215

#     # mask_latents = mask_autoencoder.encoder(masks)
#     masks_conv = masks.repeat(1, 3, 1, 1)
#     masks_latents_dist = image_autoencoder.encode(masks_conv).latent_dist
#     masks_latents = masks_latents_dist.sample()
#     masks_latents = masks_latents * 0.18215

# print(image_latents.shape, masks_latents.shape)

# image_size=32
num_epochs = 0
epoch = 0
batch_size = 32
num_train_timesteps = 1000
learning_rate = 1e-4
output_dir = "../test_segmentations"

# Latent Diffusion model
model = UNet2DModel(
        sample_size=32,
        in_channels=12,
        out_channels=1,
        layers_per_block=1,
        block_out_channels=(32, 64, 128), # Halve the channels
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"), # No attention
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),     # No attention
    )

noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

lr_scheduler = get_scheduler(
    "cosine",
    optimizer=optimizer,
    num_warmup_steps=40,
    num_training_steps=(len(train_loader) * 100),
)

model.to(device)

print("Starting training...")
for epoch in range(num_epochs):
    progress_bar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}")
    
    for step, batch in enumerate(train_loader):
    # for clean_images, coarse_mask, gt_mask in enumerate(train_loader):
        clean_images, coarse_mask, gt_mask = batch
        clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
        
        batch_size = clean_images.shape[0]
        
        # encode  to latent space
        with torch.no_grad():
            # clean_images = vae.encode(clean_images).latent_dist.sample()
            # coarse_mask = vae_mask.encode(coarse_mask).latent_dist.sample()
            # gt_mask = vae_mask.encode(gt_mask).latent_dist.sample()
            image_latents_dist = image_autoencoder.encode(clean_images).latent_dist
            image_latents = image_latents_dist.sample()
            image_latents = image_latents * 0.18215
            
            masks_conv = coarse_mask.repeat(1, 3, 1, 1)
            masks_latents_dist = image_autoencoder.encode(masks_conv).latent_dist
            masks_latents = masks_latents_dist.sample()
            masks_latents = masks_latents * 0.18215

            gt_conv = gt_mask.repeat(1, 3, 1, 1)
            gt_latents_dist = image_autoencoder.encode(gt_conv).latent_dist
            gt_latents = gt_latents_dist.sample()
            gt_latents = gt_latents * 0.18215

        # 1. Sample random timesteps
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
        ).long()

        # 2. Sample a random noise map *for the mask*
        noise = torch.randn_like(masks_latents)

        # 3. Add noise to the *clean masks*
        noisy_masks = noise_scheduler.add_noise(gt_latents, noise, timesteps)
        
        # 4. Concatenate noisy mask and clean image
        #    Input shape: (batch_size, 4, H, W)
        model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)
        
        # 5. Get the model's noise prediction
        noise_pred = model(model_input, timesteps).sample

        # 6. Calculate the loss (MSE on the noise)
        loss = F.mse_loss(noise_pred, noise)

        # 7. Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        # Log metrics to wandb
        # current_lr = lr_scheduler.get_last_lr()[0]
        # global_step = epoch * len(train_dataloader) + step
        # wandb.log({
        #     "train/loss": loss.item(),
        #     "train/learning_rate": current_lr,
        #     "train/epoch": epoch,
        #     "train/global_step": global_step
        # })

        progress_bar.update(1)
        progress_bar.set_postfix(loss=loss.item())

    progress_bar.close()

# --- 5. Validation at end of epoch ---
print(f"Running validation for epoch {epoch+1}...")
val_loss = validate(model, noise_scheduler, val_loader, device)
print(f"Validation loss: {val_loss:.4f}")

# Log validation metrics to wandb
# wandb.log({
#     "val/loss": val_loss,
#     "val/epoch": epoch
# })

# --- 6. Sample and Save Images at end of epoch ---
os.makedirs(output_dir, exist_ok=True)
sample_and_save_images(
    model, 
    noise_scheduler, 
    test_loader_for_sampling,
    device, 
    epoch, 
    output_dir,
    metrics_dict={
        "iou": jaccard_index,
        "psnr": psnr_metric,
    }
)

print("Training finished.")

# Save the final model's UNet component
model.save_pretrained(output_dir)
print(f"Model saved to {output_dir}")