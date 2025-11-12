from typing import NoReturn
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Dataset, Subset
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio
import random
from diffusers import DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from tqdm.auto import tqdm
import os
import torchvision
import wandb
import random
import string
import argparse
import glob


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
    def __init__(self, root_dir, image_transform=None, mask_transform=None, coarse_mask_transform=None):
        self.root_dir = root_dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform
        self.coarse_mask_transform = coarse_mask_transform
        
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
        
        # Apply any *additional* transforms (e.g., data augmentation)
        # Note: Resizing/Normalization is already done!
        if self.image_transform:
            image = self.image_transform(image)
        if self.mask_transform:
            gt_mask = self.mask_transform(gt_mask)
        if self.coarse_mask_transform:
            coarse_mask = self.coarse_mask_transform(coarse_mask)
            
        return image, coarse_mask, gt_mask


def get_train_val_dataloaders(image_size, batch_size, root_dir, val_split=0.2):    
    # Transforms for the RGB image
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        lambda x: x.float(),
        # Normalize to [-1, 1]
        transforms.Normalize((255 / 2, 255 / 2, 255 / 2), (255 / 2, 255 / 2, 255 / 2)),
    ])
    
    # Transforms for the segmentation mask
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        # The mask values are 1 (pet), 2 (background), 3 (border).
        # We only care about the pet. Let's make a binary mask.
        lambda x: ((x == 1) | (x == 3)).float(), # 1 if pet, 0 if background/border
        transforms.Normalize((0.5,), (0.5,)) # Normalize [0, 1] to [-1, 1]
    ])
    coarse_mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.Normalize((0.5,), (0.5,)) # Normalize [0, 1] to [-1, 1]
    ])

    dataset = CoarseOxfordIIITPet(
        root_dir,
        image_transform=image_transform,
        mask_transform=mask_transform,
        coarse_mask_transform=coarse_mask_transform
    )

    indices = list(range(len(dataset)))
    print(f"Found {len(indices)} samples for dataset.")
    
    # 3. Split indices into train and validation
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    test_size = 64
    
    # 4. Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, val_indices[:test_size])
    
    # 5. Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
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
    
    # Get a batch of test data
    try:
        batch = next(iter(test_dataloader))
        clean_images, coarse_mask, gt_mask = batch
        clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
    except Exception as e:
        print(f"Error getting test batch: {e}")
        return

    # Start with pure noise for the mask
    noisy_masks = torch.randn_like(gt_mask)
    
    with torch.no_grad():
        # Loop backwards through the diffusion timesteps
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            
            # 1. Concatenate the noisy mask and the condition (RGB image)
            # Input shape: (batch_size, 4, H, W)
            model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
            
            # 2. Predict the noise
            noise_pred = model(model_input, t).sample
            
            # 3. Use the scheduler to "denoise" one step
            scheduler_output = noise_scheduler.step(noise_pred, t, noisy_masks)
            noisy_masks = scheduler_output.prev_sample

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
            
            # 1. Sample a random noise map for the mask
            noise = torch.randn_like(coarse_mask)
            
            # 2. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            
            # 3. Add noise to the clean masks
            noisy_masks = noise_scheduler.add_noise(gt_mask, noise, timesteps)
            
            # 4. Concatenate noisy mask and clean image
            model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
            
            # 5. Get the model's noise prediction
            noise_pred = model(model_input, timesteps).sample
            
            # 6. Calculate the loss (MSE on the noise)
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
        project="diffusion-segmentation",
        config=vars(args),
        name=exp_id,
        entity="tf426-cam"
    )
    
    # --- 2. Load and Preprocess Dataset ---
    print("Loading dataset...")
    train_dataloader, val_dataloader, test_dataloader_for_sampling = get_train_val_dataloaders(
        args.image_size,
        args.batch_size,
        os.path.join(args.data_root_dir, "oxcoarse"),
        val_split=args.val_split
    )

    # --- 3. Define the Model, Scheduler, and Optimizer ---
    print("Initializing model...")
    # KEY CHANGE: in_channels=4 (1 for mask + 3 for RGB)
    #             out_channels=1 (to predict noise for the mask)
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=5,  # <-- The CRITICAL change
        out_channels=1, # <-- The CRITICAL change
        layers_per_block=2,
        block_out_channels=(64, 128, 128, 256), # A slightly larger UNet
        down_block_types=("DownBlock2D", "DownBlock2D", "AttnDownBlock2D", "DownBlock2D"),
        up_block_types=("UpBlock2D", "AttnUpBlock2D", "UpBlock2D", "UpBlock2D"),
    )

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=40,
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

    # --- 4. The Training Loop ---
    print("Starting training...")
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images, coarse_mask, gt_mask = batch
            clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)
            
            batch_size = clean_images.shape[0]

            # 1. Sample a random noise map *for the mask*
            noise = torch.randn_like(coarse_mask)
            
            # 2. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()

            # 3. Add noise to the *clean masks*
            noisy_masks = noise_scheduler.add_noise(gt_mask, noise, timesteps)
            
            # 4. Concatenate noisy mask and clean image
            #    Input shape: (batch_size, 4, H, W)
            model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
            
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
        val_loss = validate(model, noise_scheduler, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")
        
        # Log validation metrics to wandb
        wandb.log({
            "val/loss": val_loss,
            "val/epoch": epoch
        })
        
        # --- 6. Sample and Save Images at end of epoch ---
        sample_and_save_images(
            model, 
            noise_scheduler, 
            test_dataloader_for_sampling, 
            device, 
            epoch, 
            args.output_dir,
            metrics_dict={
                "iou": jaccard_index,
                "psnr": psnr_metric,
            }
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
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="../test_segmentations")
    parser.add_argument("--data_root_dir", type=str, default="./data")
    parser.add_argument("--val_split", type=float, default=0.1)
    args = parser.parse_args()
    train_model(args)