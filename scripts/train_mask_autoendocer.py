import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm.auto import tqdm
from torchmetrics import JaccardIndex
import matplotlib.pyplot as plt
from model.mask_autoencoder import MaskAutoencoder
from scripts.data_utils import get_loaders


def compute_loss(model, images, masks, device):
    """Compute cross-entropy loss for mask reconstruction"""
    masks = masks.to(device)
    
    # Forward pass
    logits = model(masks)  # (B, 1, H, W) logits
    
    # Convert masks from [-1, 1] to [0, 1] for BCE loss
    targets = (masks + 1) / 2  # [-1, 1] → [0, 1]
    
    # Use BCE with logits loss
    loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    
    return loss


def compute_iou(model, images, masks, device):
    """Compute IoU metric for mask reconstruction"""
    masks = masks.to(device)
    
    model.eval()
    with torch.no_grad():
        logits = model(masks)  # (B, 1, H, W)
        pred_masks = torch.sigmoid(logits)  # Convert to probabilities
        pred_binary = (pred_masks > 0.5).float()  # Threshold at 0.5
        
        # Convert true masks from [-1, 1] to [0, 1]
        true_binary = ((masks + 1) / 2 > 0.5).float()
        
        # Compute IoU using torchmetrics
        jaccard = JaccardIndex(task="binary", threshold=0.5).to(device)
        iou = jaccard(pred_binary, true_binary)
    
    model.train()
    return iou.item()


def evaluate(model, data_loader, device):
    """Evaluate model on a dataset"""
    model.eval()
    total_loss = 0.0
    total_iou = 0.0
    num_batches = 0
    
    jaccard = JaccardIndex(task="binary", threshold=0.5).to(device)
    
    with torch.no_grad():
        for images, masks in data_loader:
            images, masks = images.to(device), masks.to(device)
            
            # Compute loss
            logits = model(masks)
            targets = (masks + 1) / 2
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            
            # Compute IoU
            pred_masks = torch.sigmoid(logits)
            pred_binary = (pred_masks > 0.5).float()
            true_binary = (targets > 0.5).float()
            iou = jaccard(pred_binary, true_binary)
            
            total_loss += loss.item()
            total_iou += iou.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches
    avg_iou = total_iou / num_batches
    
    model.train()
    return avg_loss, avg_iou


def visualize_reconstructions(model, data_loader, device, num_samples=4, save_path=None):
    """Visualize mask reconstructions"""
    model.eval()
    
    images, masks = next(iter(data_loader))
    images = images[:num_samples].to(device)
    masks = masks[:num_samples].to(device)
    
    with torch.no_grad():
        logits = model(masks)
        pred_masks = torch.sigmoid(logits)
        pred_binary = (pred_masks > 0.5).float()
    
    # Convert to CPU for visualization
    images = images.cpu()
    masks = masks.cpu()
    pred_binary = pred_binary.cpu()
    
    fig, axes = plt.subplots(3, num_samples, figsize=(12, 9))
    
    for i in range(num_samples):
        # Original image (denormalize from [-1, 1] to [0, 1])
        img = images[i].permute(1, 2, 0).numpy()
        img = (img + 1) / 2  # [-1, 1] → [0, 1]
        img = img.clip(0, 1)
        axes[0, i].imshow(img)
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_ylabel('Image', fontsize=12)
        
        # True mask (denormalize from [-1, 1] to [0, 1])
        true_mask = ((masks[i, 0] + 1) / 2).numpy()
        axes[1, i].imshow(true_mask, cmap='gray', vmin=0, vmax=1)
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_ylabel('True Mask', fontsize=12)
        
        # Reconstructed mask
        recon_mask = pred_binary[i, 0].numpy()
        axes[2, i].imshow(recon_mask, cmap='gray', vmin=0, vmax=1)
        axes[2, i].axis('off')
        if i == 0:
            axes[2, i].set_ylabel('Reconstructed', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")
    
    plt.show()
    plt.close()
    
    model.train()


def train_autoencoder(
    model,
    train_loader,
    dev_loader,
    device,
    model_path="checkpoints/autoencoder.pth",
    num_epochs=50,
    lr=1e-4
):
    """Train the mask autoencoder"""
    
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully! (from epoch {checkpoint.get('epoch', 'unknown')})")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully!")
        return model

    print("Training new autoencoder model...")
    model.train()
    
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=10, min_lr=5e-5
    )

    # Training history
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_iou': [],
        'lr': []
    }
    
    best_val_loss = float('inf')
    checkpoint_dir = os.path.dirname(model_path)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        
        # Training loop
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for images, masks in pbar:
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            
            # Compute loss (autoencoder reconstructs masks)
            logits = model(masks)
            targets = (masks + 1) / 2  # [-1, 1] → [0, 1]
            loss = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = epoch_loss / len(train_loader)
        
        # Validation
        val_loss, val_iou = evaluate(model, dev_loader, device)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(val_loss)
        history['val_iou'].append(val_iou)
        history['lr'].append(current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch+1}/{num_epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val IoU: {val_iou:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_iou': val_iou,
                'history': history
            }, model_path)
            print(f"  → Saved best model (val_loss: {val_loss:.4f})")
        
        # Visualize every 5 epochs
        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(checkpoint_dir, f'reconstruction_epoch{epoch+1}.png')
            visualize_reconstructions(model, dev_loader, device, save_path=save_path)
    
    # Load best model
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Plot training history
    plot_training_history(history, checkpoint_dir)
    
    return model


def plot_training_history(history, save_dir):
    """Plot training curves"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Loss
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # IoU
    axes[1].plot(history['val_iou'], label='Val IoU', color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('IoU')
    axes[1].set_title('Validation IoU')
    axes[1].legend()
    axes[1].grid(True)
    
    # Learning Rate
    axes[2].plot(history['lr'], label='Learning Rate', color='red')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Learning Rate')
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_yscale('log')
    axes[2].legend()
    axes[2].grid(True)
    
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'training_history.png')
    plt.savefig(save_path, dpi=150)
    print(f"Saved training history to {save_path}")
    plt.close()


if __name__ == "__main__":    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 256
    batch_size = 32
    model_path = "checkpoints/autoencoder.pth"
    LATENT_DIM = 128

    # Get dataloaders
    loaders = get_loaders(image_size=image_size, batch_size=batch_size)
    train_loader = loaders["train"]
    dev_loader = loaders["dev"]

    # Build model
    model = MaskAutoencoder(
        base_channel_size=32,
        latent_dim=LATENT_DIM,
        num_output_channels=1  # Single channel output for binary masks
    ).to(device)

    # Train or load model
    train_autoencoder(model, train_loader, dev_loader, device, model_path, num_epochs=50)