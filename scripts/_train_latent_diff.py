#!/usr/bin/env python3
"""
Fixed latent-diffusion training / validation / sampling script.
Notes:
 - Assumes an AutoencoderKL (VAE) compatible with Stable Diffusion (latent scale 0.18215).
 - The dataset expects pregenerated .pt files under:
     <root>/train/images/*.pt, <root>/train/coarse_masks/*.pt, <root>/train/gt_masks/*.pt
   and similarly for <root>/dev/.
 - The DataLoader returns images normalized to [-1, 1] and masks as single-channel floats in [-1,1].
 - UNet input channels = 12 because we're concatenating three latent tensors (noisy_mask_latent, coarse_mask_latent, image_latent)
   each with 4 channels (Stable Diffusion VAE latents typically have 4 channels).
"""

import os
import glob
import argparse
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

import numpy as np
from tqdm.auto import tqdm

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torchvision
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio

from diffusers import DDPMScheduler, UNet2DModel, AutoencoderKL
from diffusers.optimization import get_scheduler


class CoarseOxfordIIITPet(Dataset):
    """
    Dataset loading pregenerated tensors (.pt) saved on disk.
    Each file contains a tensor for the image or mask with shape:
      image: [C, H, W] (float32, 0..1 or -1..1 — we'll convert)
      mask:  [1, H, W] (0/1 or 0..1)
    Directory layout:
      root_dir/
        images/*.pt
        coarse_masks/*.pt
        gt_masks/*.pt
    """
    def __init__(self, root_dir, img_transforms=None):
        self.root_dir = root_dir
        self.img_transforms = img_transforms

        self.image_dir = os.path.join(root_dir, "images")
        self.coarse_mask_dir = os.path.join(root_dir, "coarse_masks")
        self.gt_mask_dir = os.path.join(root_dir, "gt_masks")

        self.file_names = sorted([os.path.basename(f) for f in glob.glob(os.path.join(self.image_dir, "*.pt"))])

        if not self.file_names:
            raise FileNotFoundError(f"No '.pt' files found in {self.image_dir}")

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        fname = self.file_names[idx]

        image = torch.load(os.path.join(self.image_dir, fname))  # [C, H, W]
        coarse_mask = torch.load(os.path.join(self.coarse_mask_dir, fname))  # [1, H, W]
        gt_mask = torch.load(os.path.join(self.gt_mask_dir, fname))  # [1, H, W]

        # to numpy HWC for albumentations
        image = image.detach().cpu().numpy().astype(np.float32).transpose(1, 2, 0)  # H,W,C
        # Ensure image is 0..255 if values look small (albumentations works with floats but Normalize expects 0..255 by default)
        # We'll treat inputs as 0..1 and convert to 0..255 for better albumentations behavior, then Normalize to [-1,1].
        if image.max() <= 1.0:
            image = (image * 255.0).astype(np.float32)

        coarse_mask = coarse_mask.detach().cpu().numpy().squeeze(0).astype(np.float32)  # H,W
        gt_mask = gt_mask.detach().cpu().numpy().squeeze(0).astype(np.float32)  # H,W

        # Albumentations expects masks to be 0..255 or 0/1; keep them 0/1 floats
        if coarse_mask.max() > 1.0:
            coarse_mask = (coarse_mask / 255.0).astype(np.float32)
        if gt_mask.max() > 1.0:
            gt_mask = (gt_mask / 255.0).astype(np.float32)

        transformed = self.img_transforms(image=image, coarse_mask=coarse_mask, gt_mask=gt_mask)
        img = transformed["image"]  # tensor CHW because of ToTensorV2
        c_mask = transformed["coarse_mask"][None, ...].astype(np.float32)  # H,W -> 1,H,W
        g_mask = transformed["gt_mask"][None, ...].astype(np.float32)

        # Convert to torch tensors (ToTensorV2 already returns torch.Tensor for image)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.asarray(img)).permute(2, 0, 1).float()
        c_mask = torch.from_numpy(c_mask).float()
        g_mask = torch.from_numpy(g_mask).float()

        # Output: image tensor in [-1,1] and masks single-channel in [-1,1]
        return img, c_mask, g_mask


def get_train_val_dataloaders(image_size, batch_size, root_dir, val_split=0.2, num_workers=4):
    img_transforms = A.Compose(
        [
            A.Resize(height=image_size, width=image_size),
            A.Rotate(limit=15, p=0.5, border_mode=0),
            A.Affine(translate_percent=(0.1, 0.1), p=0.5, border_mode=0),
            A.HorizontalFlip(p=0.5),
            A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), max_pixel_value=255.0),
            ToTensorV2(),
        ],
        additional_targets={"coarse_mask": "mask", "gt_mask": "mask"},
    )

    train_dataset = CoarseOxfordIIITPet(os.path.join(root_dir, "train"), img_transforms=img_transforms)
    dev_dataset = CoarseOxfordIIITPet(os.path.join(root_dir, "dev"), img_transforms=img_transforms)

    print(f"Train samples: {len(train_dataset)}, Validation samples: {len(dev_dataset)}")

    # small test subset for sampling
    test_size = min(16, len(dev_dataset))
    test_dataset = Subset(dev_dataset, list(range(test_size)))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_dataset, batch_size=test_size, shuffle=False, num_workers=num_workers, drop_last=False)

    return train_loader, val_loader, test_loader


def validate(model, noise_scheduler, val_dataloader, device, image_autoencoder):
    model.eval()
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            clean_images, coarse_mask, gt_mask = batch
            clean_images = clean_images.to(device)
            coarse_mask = coarse_mask.to(device)
            gt_mask = gt_mask.to(device)

            batch_size = clean_images.shape[0]

            # Encode to VAE latents
            img_latent = image_autoencoder.encode(clean_images).latent_dist.sample() * 0.18215
            masks_conv = coarse_mask.repeat(1, 3, 1, 1)  # convert single channel to 3-channel image for VAE
            masks_latent = image_autoencoder.encode(masks_conv).latent_dist.sample() * 0.18215
            gt_conv = gt_mask.repeat(1, 3, 1, 1)
            gt_latent = image_autoencoder.encode(gt_conv).latent_dist.sample() * 0.18215

            # timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            noise = torch.randn_like(gt_latent)
            noisy_masks = noise_scheduler.add_noise(gt_latent, noise, timesteps)

            model_input = torch.cat([noisy_masks, masks_latent, img_latent], dim=1)
            noise_pred = model(model_input, timesteps).sample
            loss = F.mse_loss(noise_pred, noise)

            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / n_batches if n_batches else 0.0
    model.train()
    return avg_loss


def sample_and_save_images(model, noise_scheduler, test_dataloader, device, epoch, output_dir, image_autoencoder, metrics_dict):
    model.eval()

    try:
        batch = next(iter(test_dataloader))
    except StopIteration:
        print("No test samples available for sampling.")
        model.train()
        return

    clean_images, coarse_mask, gt_mask = batch
    clean_images = clean_images.to(device)
    coarse_mask = coarse_mask.to(device)
    gt_mask = gt_mask.to(device)
    batch_size = clean_images.shape[0]

    with torch.no_grad():
        img_latent = image_autoencoder.encode(clean_images).latent_dist.sample() * 0.18215
        masks_conv = coarse_mask.repeat(1, 3, 1, 1)
        masks_latent = image_autoencoder.encode(masks_conv).latent_dist.sample() * 0.18215

    # Start noise for mask latents (we sample from standard normal)
    noisy_masks = torch.randn_like(masks_latent).to(device)

    # iterate timesteps in reverse
    for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
        # t is a scalar value - scheduler.step accepts a scalar time or a tensor of timesteps depending on implementation
        # Prepare model input
        model_input = torch.cat([noisy_masks, masks_latent, img_latent], dim=1)
        # model returns a ModelOutput with .sample (diffusers API), but UNet2DModel returns a tensor with shape [B,C,H,W]
        out = model(model_input, t)
        # if it returns ModelOutput with .sample attribute:
        if hasattr(out, "sample"):
            noise_pred = out.sample
        else:
            noise_pred = out  # assume tensor

        step_output = noise_scheduler.step(noise_pred, t, noisy_masks)
        noisy_masks = step_output.prev_sample

    # noisy_masks are latents (scaled by 0.18215) -- decode to image space
    with torch.no_grad():
        decoded = image_autoencoder.decode(noisy_masks / 0.18215).sample  # decoded: [B, 3, H, W]

    # Convert decoded mask to single-channel mask approximation (average RGB -> scalar mask)
    predicted_masks = decoded.mean(dim=1, keepdim=True)  # [B,1,H,W]

    # Our tensors are in [-1,1] (because of Normalize); convert to [0,1] for metrics and saving
    def to_0_1(x):
        return (x * 0.5 + 0.5).clamp(0.0, 1.0)

    clean_images_disp = to_0_1(clean_images)
    coarse_mask_disp = to_0_1(coarse_mask)  # [B,1,H,W]
    predicted_masks_disp = to_0_1(predicted_masks)
    gt_mask_disp = to_0_1(gt_mask)

    # Make masks 3-channel for saving a single grid visually
    coarse_mask_3c = coarse_mask_disp.repeat(1, 3, 1, 1)
    predicted_mask_3c = predicted_masks_disp.repeat(1, 3, 1, 1)
    gt_mask_3c = gt_mask_disp.repeat(1, 3, 1, 1)

    # Metrics: convert to same device and shape
    iou_score = metrics_dict["iou"](
        (predicted_masks_disp > 0.5).float(), (gt_mask_disp > 0.5).float()
    )
    psnr_score = metrics_dict["psnr"](predicted_masks_disp, gt_mask_disp)
    baseline_iou = metrics_dict["iou"]((coarse_mask_disp > 0.5).float(), (gt_mask_disp > 0.5).float())
    baseline_psnr = metrics_dict["psnr"](coarse_mask_disp, gt_mask_disp)

    print(f"Sample metrics: IoU {iou_score:.4f}, PSNR {psnr_score:.4f} | Baseline IoU {baseline_iou:.4f}, Baseline PSNR {baseline_psnr:.4f}")

    # Create grid: for each sample show [Image, CoarseMask, PredMask, GT]
    comparison = torch.cat([clean_images_disp, coarse_mask_3c, predicted_mask_3c, gt_mask_3c], dim=0)
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f"sample_epoch_{epoch}.png")
    torchvision.utils.save_image(comparison, save_path, nrow=batch_size, pad_value=1.0)
    print(f"Saved sample grid to {save_path}")

    model.train()


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print(f"Using device: {device}")

    # Dataloaders
    train_loader, val_loader, test_loader = get_train_val_dataloaders(
        args.image_size, args.batch_size, args.data_root_dir, val_split=args.val_split, num_workers=args.num_workers
    )

    # Load VAE (AutoencoderKL) - you can replace with a local path or huggingface repo id
    print("Loading image autoencoder (VAE)...")
    if args.vae_file and os.path.exists(args.vae_file):
        image_autoencoder = AutoencoderKL.from_single_file(args.vae_file).to(device)
    else:
        # try to load from model repo id
        image_autoencoder = AutoencoderKL.from_single_file(args.vae_pretrained_id).to(device)
    image_autoencoder.eval()

    # Metrics
    jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device)
    psnr_metric = PeakSignalNoiseRatio().to(device)

    # Model: UNet expecting concatenated latents -> 12 channels (3 latents * 4 channels each)
    sample_size = args.image_size // 8  # stable-diffusion VAE typical downscale factor 8 -> latent spatial size
    model = UNet2DModel(
        sample_size=sample_size,
        in_channels=12,  # noisy_mask_latent (4) + coarse_mask_latent (4) + image_latent (4)
        out_channels=4,  # predict noise in latent channel space (4 channels)
        layers_per_block=1,
        block_out_channels=(32, 64, 128),
        down_block_types=("DownBlock2D", "DownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "UpBlock2D", "UpBlock2D"),
    )
    model.to(device)

    # Scheduler and optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    total_training_steps = max(1, args.num_epochs * len(train_loader))
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=total_training_steps,
    )

    # Training loop
    os.makedirs(args.output_dir, exist_ok=True)

    print("Starting training...")
    global_step = 0
    for epoch in range(args.num_epochs):
        model.train()
        progress = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}/{args.num_epochs}")
        for step, batch in enumerate(train_loader):
            clean_images, coarse_mask, gt_mask = batch
            clean_images = clean_images.to(device)
            coarse_mask = coarse_mask.to(device)
            gt_mask = gt_mask.to(device)

            bsz = clean_images.shape[0]

            # Encode to latents (no grad)
            with torch.no_grad():
                image_latents = image_autoencoder.encode(clean_images).latent_dist.sample() * 0.18215
                masks_conv = coarse_mask.repeat(1, 3, 1, 1)
                masks_latents = image_autoencoder.encode(masks_conv).latent_dist.sample() * 0.18215
                gt_conv = gt_mask.repeat(1, 3, 1, 1)
                gt_latents = image_autoencoder.encode(gt_conv).latent_dist.sample() * 0.18215

            # sample timesteps
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device).long()
            noise = torch.randn_like(gt_latents)
            noisy_masks = noise_scheduler.add_noise(gt_latents, noise, timesteps)

            model_input = torch.cat([noisy_masks, masks_latents, image_latents], dim=1)  # [B,12,H,W]
            out = model(model_input, timesteps)
            noise_pred = out.sample if hasattr(out, "sample") else out
            loss = F.mse_loss(noise_pred, noise)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()

            progress.update(1)
            progress.set_postfix({"loss": f"{loss.item():.4f}"})
            global_step += 1

        progress.close()

        # Validation
        val_loss = validate(model, noise_scheduler, val_loader, device, image_autoencoder)
        print(f"Epoch {epoch+1} validation loss: {val_loss:.4f}")

        # Sample and save images
        sample_and_save_images(
            model=model,
            noise_scheduler=noise_scheduler,
            test_dataloader=test_loader,
            device=device,
            epoch=epoch + 1,
            output_dir=args.output_dir,
            image_autoencoder=image_autoencoder,
            metrics_dict={"iou": jaccard_index, "psnr": psnr_metric},
        )

        # Save checkpoint every epoch (UNet weights)
        ckpt_path = os.path.join(args.output_dir, f"unet_epoch_{epoch+1}.pt")
        torch.save(model.state_dict(), ckpt_path)
        print(f"Saved UNet checkpoint to {ckpt_path}")

    # Final save via diffusers API
    try:
        model.save_pretrained(args.output_dir)
        print(f"Saved model (diffusers) to {args.output_dir}")
    except Exception as e:
        print(f"Could not save via save_pretrained(): {e}")

    print("Training finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="./output_segmentations")
    parser.add_argument("--data_root_dir", type=str, default="./data/oxcoarse")
    parser.add_argument("--val_split", type=float, default=0.1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--warmup_steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vae_file", type=str, default="")
    parser.add_argument("--vae_pretrained_id", type=str, default="https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors")
    args = parser.parse_args()

    main(args)