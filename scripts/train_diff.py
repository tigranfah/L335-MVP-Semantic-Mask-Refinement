from typing import NoReturn
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import DataLoader, Subset
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


def generate_random_id(length=6):
    return "".join(random.sample(string.ascii_lowercase + string.digits, k=length))


def get_dataloader(image_size, batch_size, root_dir):
    """Loads the Oxford-IIIT Pet dataset."""
    
    # Transforms for the RGB image
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Normalize to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Transforms for the segmentation mask
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        # The mask is 1, 2, 3. We want 0, 1, 2.
        # Then, we'll just normalize [0, 2] to [-1, 1] for diffusion
        lambda x: (x.float() - 1.0) / 1.0, # Maps [1, 3] -> [0, 2] -> [0, 2]... wait, let's rethink.
        # Let's just scale [1, 3] to [-1, 1]
        lambda x: (x.float() - 2.0) / 1.0 # Maps [1, 2, 3] -> [-1, 0, 1]
        
        # A simpler way: The mask is 1, 2, 3. Let's just normalize to [0, 1]
        # and then to [-1, 1].
        # lambda x: (x.float() - 1.0) / 2.0, # Maps [1, 2, 3] -> [0, 1, 2] -> [0, 0.5, 1]
        # transforms.Normalize((0.5,), (0.5,)) # Maps [0, 1] -> [-1, 1]
    ])

    # --- Let's try the simplest mask transform ---
    # ToTensor maps [1,2,3] (as PIL) to a [0,1,2] float tensor...
    # No, it maps pixel values.
    # Let's just normalize the 1-channel mask to [-1, 1]
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(), # This will scale 0-255 to [0, 1]. The mask is 1, 2, 3.
        # The mask values are 1 (pet), 2 (background), 3 (border).
        # We only care about the pet. Let's make a binary mask.
        lambda x: (x == 1).float(), # 1 if pet, 0 if background/border
        transforms.Normalize((0.5,), (0.5,)) # Normalize [0, 1] to [-1, 1]
    ])

    dataset = OxfordIIITPet(
        root=root_dir, 
        split="trainval", 
        download=True, 
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform
    )

    pet_class_name = "Abyssinian"
    print(f"Filtering dataset for pet class: '{pet_class_name}'")
    
    # 1. Find the numerical index of the desired class
    try:
        # Note: Class names in the dataset are like 'Abyssinian', 'basset_hound'
        class_idx = dataset.classes.index(pet_class_name)
    except ValueError:
        print(f"Error: Pet class '{pet_class_name}' not found.")
        print(f"Available classes are: {dataset.classes}")
        return None # Return None to indicate failure
        
    # 2. Get the indices of all samples matching this class index
    #    The dataset object stores the class label for each image in ._labels
    indices = [i for i, label in enumerate(dataset._labels) if label == class_idx]
    
    if not indices:
        print(f"No samples found for class: {pet_class_name}")
        return None
        
    print(f"Found {len(indices)} samples for class '{pet_class_name}'.")
    
    # 3. Create a Subset dataset using these indices
    dataset = Subset(dataset, indices)
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    print(dataloader)
    return dataloader


def get_train_val_dataloaders(image_size, batch_size, root_dir, val_split=0.2):
    """Loads the Oxford-IIIT Pet dataset and splits it into train and validation sets."""
    
    # Transforms for the RGB image
    image_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        # Normalize to [-1, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    
    # Transforms for the segmentation mask
    mask_transform = transforms.Compose([
        transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(), # This will scale 0-255 to [0, 1]. The mask is 1, 2, 3.
        # The mask values are 1 (pet), 2 (background), 3 (border).
        # We only care about the pet. Let's make a binary mask.
        lambda x: (x == 1).float(), # 1 if pet, 0 if background/border
        transforms.Normalize((0.5,), (0.5,)) # Normalize [0, 1] to [-1, 1]
    ])

    dataset = OxfordIIITPet(
        root=root_dir, 
        split="trainval", 
        download=True, 
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform
    )

    # pet_class_name = "asd"
    # print(f"Filtering dataset for pet class: '{pet_class_name}'")
    
    # # 1. Find the numerical index of the desired class
    # try:
    #     class_idx = dataset.classes.index(pet_class_name)
    # except ValueError:
    #     print(f"Error: Pet class '{pet_class_name}' not found.")
    #     print(f"Available classes are: {dataset.classes}")
    #     return None, None
        
    # # 2. Get the indices of all samples matching this class index
    # indices = [i for i, label in enumerate(dataset._labels) if label == class_idx]
    
    # if not indices:
    #     print(f"No samples found for class: {pet_class_name}")
    #     return None, None
    
    indices = list(range(len(dataset)))
    print(f"Found {len(indices)} samples for dataset.")
    
    # 3. Split indices into train and validation
    random.seed(42)  # For reproducibility
    random.shuffle(indices)
    split_idx = int(len(indices) * (1 - val_split))
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    print(f"Train samples: {len(train_indices)}, Validation samples: {len(val_indices)}")
    
    # 4. Create Subset datasets
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, val_indices[:6])
    
    # 5. Create DataLoaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False)
    
    return train_dataloader, val_dataloader, test_dataloader


def sample_and_save_images(model, noise_scheduler, test_dataloader, device, epoch, output_dir):
    """Samples and saves a grid of images for visual inspection."""
    model.eval()
    
    # Get a batch of test data
    try:
        batch = next(iter(test_dataloader))
        clean_images, ground_truth_masks = batch
        clean_images, ground_truth_masks = clean_images.to(device), ground_truth_masks.to(device)
    except Exception as e:
        print(f"Error getting test batch: {e}")
        return

    # Start with pure noise for the mask
    noisy_masks = torch.randn_like(ground_truth_masks)
    
    with torch.no_grad():
        # Loop backwards through the diffusion timesteps
        for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
            
            # 1. Concatenate the noisy mask and the condition (RGB image)
            # Input shape: (batch_size, 4, H, W)
            model_input = torch.cat([noisy_masks, clean_images], dim=1)
            
            # 2. Predict the noise
            noise_pred = model(model_input, t).sample
            
            # 3. Use the scheduler to "denoise" one step
            scheduler_output = noise_scheduler.step(noise_pred, t, noisy_masks)
            noisy_masks = scheduler_output.prev_sample

    # --- Un-normalize all images for saving ---
    # Un-normalize from [-1, 1] to [0, 1]
    clean_images = (clean_images * 0.5 + 0.5).clamp(0, 1)
    predicted_masks = (noisy_masks * 0.5 + 0.5).clamp(0, 1)
    ground_truth_masks = (ground_truth_masks * 0.5 + 0.5).clamp(0, 1)
    
    # Make masks 3-channel for grid
    predicted_masks = predicted_masks.repeat(1, 3, 1, 1)
    ground_truth_masks = ground_truth_masks.repeat(1, 3, 1, 1)

    # Create a grid: [Image | Predicted Mask | Ground Truth]
    comparison_grid = torch.cat([clean_images, predicted_masks, ground_truth_masks], dim=0)
    
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
            clean_images, clean_masks = batch
            clean_images, clean_masks = clean_images.to(device), clean_masks.to(device)
            
            batch_size = clean_images.shape[0]
            
            # 1. Sample a random noise map for the mask
            noise = torch.randn_like(clean_masks)
            
            # 2. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()
            
            # 3. Add noise to the clean masks
            noisy_masks = noise_scheduler.add_noise(clean_masks, noise, timesteps)
            
            # 4. Concatenate noisy mask and clean image
            model_input = torch.cat([noisy_masks, clean_images], dim=1)
            
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
    args.output_dir = f"{exp_id}_" + args.output_dir
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
        args.data_root_dir,
        val_split=args.val_split
    )
    
    if train_dataloader is None or val_dataloader is None:
        print("Failed to load dataset. Exiting.")
        return
    
    # --- 3. Define the Model, Scheduler, and Optimizer ---
    print("Initializing model...")
    # KEY CHANGE: in_channels=4 (1 for mask + 3 for RGB)
    #             out_channels=1 (to predict noise for the mask)
    model = UNet2DModel(
        sample_size=args.image_size,
        in_channels=4,  # <-- The CRITICAL change
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

    # --- 4. The Training Loop ---
    print("Starting training...")
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), desc=f"Epoch {epoch+1}")
        
        for step, batch in enumerate(train_dataloader):
            clean_images, clean_masks = batch
            clean_images, clean_masks = clean_images.to(device), clean_masks.to(device)
            
            batch_size = clean_images.shape[0]

            # 1. Sample a random noise map *for the mask*
            noise = torch.randn_like(clean_masks)
            
            # 2. Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device
            ).long()

            # 3. Add noise to the *clean masks*
            noisy_masks = noise_scheduler.add_noise(clean_masks, noise, timesteps)
            
            # 4. Concatenate noisy mask and clean image
            #    Input shape: (batch_size, 4, H, W)
            model_input = torch.cat([noisy_masks, clean_images], dim=1)
            
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
            args.output_dir
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
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--output_dir", type=str, default="test_segmentation")
    parser.add_argument("--data_root_dir", type=str, default="./data")
    parser.add_argument("--val_split", type=float, default=0.2)
    args = parser.parse_args()
    train_model(args)