import argparse
import os
import torch
import torch.nn as nn
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio
from tqdm.auto import tqdm
import wandb
from data_utils import generate_random_id
from train_baseline_coco import set_seed
from train_diff import get_train_val_dataloaders
from baseline_unet import UNetSmall
from diffusers.optimization import get_scheduler



def validate(model, val_dataloader, device):
    """Validates the model on the validation set."""
    jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device) #pass as arg
    psnr_metric = PeakSignalNoiseRatio().to(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    total_iou = 0.0
    total_psnr = 0.0

    criterion = nn.BCEWithLogitsLoss()
    class_labels = {0: "background", 1: "pet"}

    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validating")):
        # for batch in tqdm(val_dataloader, desc="Validating"):
            clean_images, _, gt_mask = batch
            clean_images, gt_mask = clean_images.to(device), gt_mask.to(device)
            
            outputs_logits = model(clean_images)    
            targets = (gt_mask + 1) / 2  # [-1,1] → [0,1]
      
            loss = criterion(outputs_logits, targets)

            outputs_probs = torch.sigmoid(outputs_logits)

            iou_score = jaccard_index(
                outputs_probs, 
                targets
            )
            psnr = psnr_metric(
                outputs_probs, 
                targets
            )
            
            total_loss += loss.item()
            total_iou += iou_score
            total_psnr += psnr

            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss, avg_iou, avg_psnr

def train_baseline(
    model,
    args
):
    train_loader, val_dataloader, test_dataloader_for_sampling = get_train_val_dataloaders(
        args.batch_size,
        args.data_root_dir
    )


    
    print("Training new baseline model...")
    set_seed(42)
    exp_id = generate_random_id()

    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    wandb.init(
        project="baseline-segmentation-oxford",
        config=vars(args),
        name=exp_id,
        entity="tf426-cam"
    )
    wandb_id = wandb.run.id

    

    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_loader) * args.num_epochs),
    )

    for epoch in range(args.num_epochs):
        model.train()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader)):
            images, _, masks = batch
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            targets = (masks + 1) / 2  # [-1,1] → [0,1]
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

            global_step = epoch * len(train_loader) + step
            current_lr = lr_scheduler.get_last_lr()[0]  # Get the actual current LR

            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch,
                "train/global_step": global_step
            })

        val_loss, val_iou, val_psnr = validate(
            model,
            val_dataloader,
            device,
        )

        should_log_images = (epoch % args.viz_interval == 0)
        print(f"Validation loss: {val_loss:.4f}")

        wandb.log({
            "val/loss": val_loss,
            "val/iou": val_iou,
            "val/psnr": val_psnr,
            "val/epoch": epoch
        })

        if should_log_images:
            filename = f"baseline_oxpets_{epoch}_{wandb_id}.pth"
            save_path = os.path.join(args.output_dir, filename)
        
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")

    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--viz_interval", type=int, default=10, help="Visualize every N epochs")
    parser.add_argument("--data_root_dir", type=str, default="data/oxford", help="data path")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    BASELINE_OUTPUT_CHANNELS = 1    
    
    # Build model
    model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)

    # Train or load model
    train_baseline(
    model,
    args)