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
from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os
import torchvision
import wandb
import random
import string
import argparse
import glob
from model.mask_autoencoder import MaskAutoencoder
# import time

from scripts.data_utils import CoarseOxfordIIITPet, augment_batch, oxford_get_train_val_dataloaders, generate_random_id, coco_get_train_val_dataloaders

def sample_and_save_images(
    model,
    noise_scheduler,
    test_dataloader,
    image_transforms,
    device,
    epoch,
    output_dir,
    metrics_dict,
    autoencoder
):
    """Samples and saves a grid of images for visual inspection."""
    model.eval()
    
    # Get a batch of test data
    try:
        batch = next(iter(test_dataloader))
        clean_images, coarse_mask, gt_mask = augment_batch(batch, image_transforms)
        clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
    except Exception as e:
        print(f"Error getting test batch: {e}")
        return
    
    if args.latent:
        image_latents_dist = autoencoder.encode(clean_images).latent_dist
        image_latents = image_latents_dist.sample()
        image_latents = image_latents * 0.18215
        
        masks_rep = coarse_mask.repeat(1, 3, 1, 1)
        masks_latents_dist = autoencoder.encode(masks_rep).latent_dist
        masks_latents = masks_latents_dist.sample()
        masks_latents = masks_latents * 0.18215

        gt_conv = gt_mask.repeat(1, 3, 1, 1)
        gt_latents_dist = autoencoder.encode(gt_conv).latent_dist
        gt_latents = gt_latents_dist.sample()
        gt_latents = gt_latents * 0.18215

        noisy_masks = torch.randn_like(gt_latents)
    else:
        noisy_masks = torch.randn_like(gt_mask)
    
    with torch.no_grad():

        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            if args.latent:
                model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)
            else:

                model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
            noise_pred = model(model_input, t).sample
            scheduler_output = noise_scheduler.step(noise_pred, t, noisy_masks)
            noisy_masks = scheduler_output.prev_sample

        # Update the coarse mask
        # noisy_masks = coarse_mask - noisy_masks

    # diff_mask = coarse_mask - gt_mask
    # diff_mask = (diff_mask * 0.5 + 0.5).clamp(0, 1)
    # diff_mask = diff_mask.repeat(1, 3, 1, 1)

    if args.latent:
        with torch.no_grad():
            noisy_masks = autoencoder.decode(noisy_masks / 0.18215).sample
            noisy_masks = torch.mean(noisy_masks, dim=1).unsqueeze(1)

    # --- Un-normalize all images for saving ---
    # Un-normalize from [-1, 1] to [0, 1]
    clean_images = (clean_images * 0.5 + 0.5).clamp(0, 1)
    coarse_mask = (coarse_mask * 0.5 + 0.5).clamp(0, 1)
    predicted_masks = (noisy_masks * 0.5 + 0.5).clamp(0, 1)
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
    wandb.log({
        "val/baseline_iou_score": baseline_iou_score,
        "val/baseline_psnr": baseline_psnr,
        "val/iou_score": iou_score,
        "val/psnr": psnr,
    })

    # Create a grid: [Image | Predicted Mask | Ground Truth]
    comparison_grid = torch.cat([clean_images, coarse_mask, predicted_masks, gt_mask], dim=0)
    
    # Save the grid
    save_path = os.path.join(output_dir, f"sample_epoch_{epoch}.png")
    torchvision.utils.save_image(comparison_grid, save_path, nrow=clean_images.shape[0])
    print(f"Saved sample grid to {save_path}")

    model.train()


def validate(model, noise_scheduler, val_dataloader, image_transforms, device, autoencoder):
    """Validates the model on the validation set."""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            clean_images, coarse_mask, gt_mask = augment_batch(batch, image_transforms)
            clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
            
            batch_size = clean_images.shape[0]

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()

            if args.latent:
                image_latents_dist = autoencoder.encode(clean_images).latent_dist
                image_latents = image_latents_dist.sample()
                image_latents = image_latents * 0.18215
                
                masks_rep = coarse_mask.repeat(1, 3, 1, 1)
                masks_latents_dist = autoencoder.encode(masks_rep).latent_dist
                masks_latents = masks_latents_dist.sample()
                masks_latents = masks_latents * 0.18215

                gt_conv = gt_mask.repeat(1, 3, 1, 1)
                gt_latents_dist = autoencoder.encode(gt_conv).latent_dist
                gt_latents = gt_latents_dist.sample()
                gt_latents = gt_latents * 0.18215

                noise = torch.randn_like(gt_latents)
                noisy_masks = noise_scheduler.add_noise(gt_latents, noise, timesteps)
                model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)
                noise_pred = model(model_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)

            else:
                noise = torch.randn_like(coarse_mask)
                noisy_masks = noise_scheduler.add_noise(gt_mask, noise, timesteps)
                model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
                noise_pred = model(model_input, timesteps).sample
                loss = F.mse_loss(noise_pred, noise)
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss


def train_model(args):
    """Main end-to-end training function."""
    
    exp_id = generate_random_id()
    args.output_dir = os.path.join(args.output_dir, exp_id)
    # --- 1. Initialize wandb ---
    wandb.init(
        project="combined",
        config=vars(args),
        name=exp_id,
        entity="tf426-cam"
    )
    
    # --- 2. Load and Preprocess Dataset ---
    print("Loading dataset...")
    if args.dataset == "oxford":
        train_dataloader, val_dataloader, test_dataloader_for_sampling = oxford_get_train_val_dataloaders(
            args.batch_size,
            os.path.join(args.data_root_dir, "oxcoarse")
        )

        # Transforms for the RGB image
        image_transforms = A.Compose([
            A.Resize(height=args.image_size, width=args.image_size),
            A.Rotate(limit=15, p=0.5, fill=-1, fill_mask=-1),
            A.Affine(translate_percent=(0.1, 0.1), p=0.5, fill=-1, fill_mask=-1),
            A.HorizontalFlip(p=0.5),
        ], additional_targets={"coarse_mask": "mask", "gt_mask": "mask"})

    elif args.dataset == "coco":
        train_dataloader, val_dataloader, test_dataloader_for_sampling = coco_get_train_val_dataloaders(
            args.image_size,
            args.batch_size,
            os.path.join(args.data_root_dir, "coco")
        )

        image_transforms = A.Compose([])

    else:
        print("unknown dataset")
        exit(1)

    

    # --- 3. Define the Model, Scheduler, and Optimizer ---
    print("Initializing model...")
    # KEY CHANGE: in_channels=4 (1 for mask + 3 for RGB)
    #             out_channels=1 (to predict noise for the mask)
    # model = UNet2DModel(
    #     sample_size=args.image_size,
    #     in_channels=5,  # <-- The CRITICAL change
    #     out_channels=1, # <-- The CRITICAL change
    #     layers_per_block=1,
    #     block_out_channels=(64, 128, 256), # A slightly larger UNet
    #     down_block_types=("DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
    #     up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D"),
    # )
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=12 if args.latent else 5,
        out_channels=4 if args.latent else 1,
        layers_per_block=1,
        block_out_channels=(32, 64, 128), # Halve the channels
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"), # No attention
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),     # No attention
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Training on device: {device}")
    
    # Log model parameters
    num_params = sum(p.numel() for p in model.parameters())
    wandb.config.update({"model_parameters": num_params})
    print(f"Model has {num_params:,} parameters")
    
    os.makedirs(args.output_dir, exist_ok=True)

    jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    if args.latent:
        url = "https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors"
        image_autoencoder = AutoencoderKL.from_single_file(url).to(device)
        image_autoencoder.eval()

    # --- Training Loop ---
    print("Starting training...")
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images, coarse_mask, gt_mask = augment_batch(batch, image_transforms)
            if args.perturb_coarse_mask:
                flip_mask = torch.rand_like(coarse_mask) <= 0.05
                coarse_mask[flip_mask] = 1 - coarse_mask[flip_mask]
            clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
            
            batch_size = clean_images.shape[0]
            
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            
            if args.latent:
                with torch.no_grad():
                    image_latents_dist = image_autoencoder.encode(clean_images).latent_dist
                    image_latents = image_latents_dist.sample()
                    image_latents = image_latents * 0.18215
                    
                    masks_rep = coarse_mask.repeat(1, 3, 1, 1)
                    masks_latents_dist = image_autoencoder.encode(masks_rep).latent_dist
                    masks_latents = masks_latents_dist.sample()
                    masks_latents = masks_latents * 0.18215

                    gt_conv = gt_mask.repeat(1, 3, 1, 1)
                    gt_latents_dist = image_autoencoder.encode(gt_conv).latent_dist
                    gt_latents = gt_latents_dist.sample()
                    gt_latents = gt_latents * 0.18215

                noise = torch.randn_like(gt_latents)
                noisy_masks = noise_scheduler.add_noise(gt_latents, noise, timesteps)
                model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)

            else:
                noise = torch.randn_like(coarse_mask)
                noisy_masks = noise_scheduler.add_noise(gt_mask, noise, timesteps)
                model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)

            noise_pred = model(model_input, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            # Log metrics to wandb
            current_lr = lr_scheduler.get_last_lr()[0]
            global_step = epoch * len(train_dataloader) + step
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch,
                "train/global_step": global_step
            })

            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())

        progress_bar.close()
        
        # --- 5. Validation at end of epoch ---
        print(f"Running validation for epoch {epoch+1}...")
        val_loss = validate(model, noise_scheduler, val_dataloader, image_transforms, device, image_autoencoder if args.latent else None)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Log validation metrics to wandb
        wandb.log({
            "val/loss": val_loss,
            "val/epoch": epoch
        })
        
        if epoch % 10 == 0:
            # --- 6. Sample and Save Images at end of epoch ---
            sample_and_save_images(
                model, 
                noise_scheduler, 
                test_dataloader_for_sampling, 
                image_transforms,
                device, 
                epoch, 
                args.output_dir,
                {
                    "iou": jaccard_index,
                    "psnr": psnr_metric,
                },
                image_autoencoder if args.latent else None
            )

    print("Training finished.")
    
    # Save the final model's UNet component
    model.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    
    # Finish wandb run
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--perturb_coarse_mask", action="store_true")
    parser.add_argument("--output_dir", type=str, default="../test_segmentations")
    parser.add_argument("--data_root_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="oxford")
    parser.add_argument("--latent", action="store_true")
    
    args = parser.parse_args()
    train_model(args)