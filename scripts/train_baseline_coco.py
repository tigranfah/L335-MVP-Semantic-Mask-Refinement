import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from baseline_unet import UNetSmall
import wandb
import random
import numpy as np
from torch.utils.data import DataLoader, Subset
from coco_dataset import COCOSegmentationDataset
from data_utils import dynamic_augment_collate_fn, seed_worker
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio
from data_utils import generate_random_id
import argparse
import albumentations as A
from diffusers.optimization import get_scheduler


# from coco_viz import visualize_batch, visualize_predictions, print_batch_statistics

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def get_train_val_dataloaders(image_size, batch_size):
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
        shuffle=True,
        num_workers=4,
        drop_last=True,
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

def validate(model, val_dataloader, device, current_epoch, log_images=False):
    """Validates the model on the validation set."""
    jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device) #pass as arg
    psnr_metric = PeakSignalNoiseRatio().to(device)

    model.eval()
    total_loss = 0.0
    num_batches = 0

    total_iou = 0.0
    total_psnr = 0.0

    criterion = nn.BCEWithLogitsLoss()
    class_labels = {0: "background", 1: "car"}

    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(val_dataloader, desc="Validating")):
        # for batch in tqdm(val_dataloader, desc="Validating"):
            clean_images, _, gt_mask = batch
            clean_images, gt_mask = clean_images.to(device), gt_mask.to(device)
            
            outputs_logits = model(clean_images)          
            loss = criterion(outputs_logits, gt_mask)

            outputs_probs = torch.sigmoid(outputs_logits)

            iou_score = jaccard_index(
                outputs_probs, 
                gt_mask
            )
            psnr = psnr_metric(
                outputs_probs, 
                gt_mask
            )
            
            total_loss += loss.item()
            total_iou += iou_score
            total_psnr += psnr

            num_batches += 1

            # 4. Visualization Logic (Only for the first batch, if flag is set)
            if log_images and i == 0:
                viz_images = []
                # Limit to first 8 images
                num_samples = min(clean_images.shape[0], 8)
                
                # --- COLOR MAPPING STRATEGY ---
                # We define 3 classes so WandB assigns distinct colors:
                # 0: Background (Transparent)
                # 1: Ground Truth (Usually Red in WandB default)
                # 2: Prediction (Usually distinct, e.g., Blue/Purple/Cyan)
                viz_labels = {
                    0: "background", 
                    1: "ground_truth", 
                    2: "prediction"
                }
                
                for idx in range(num_samples):
                    # Prepare Image: (C,H,W) -> (H,W,C)
                    img_np = clean_images[idx].cpu().permute(1, 2, 0).numpy()
                    
                    # Prepare Ground Truth: Class 1
                    # Ensure it is integer type (uint8)
                    gt_np = gt_mask[idx].cpu().squeeze().numpy().astype(np.uint8)
                    
                    # Prepare Prediction: Class 2
                    # 1. Threshold to get binary (0 or 1)
                    # 2. Multiply by 2 so the valid pixels become Class ID 2
                    pred_raw = (outputs_probs[idx] > 0.5).float().cpu().squeeze().numpy()
                    pred_np = (pred_raw * 2).astype(np.uint8) 

                    # Create WandB Image
                    # We use the SAME dictionary 'viz_labels' for both, 
                    # but the data arrays contain different integers (1 vs 2).
                    viz_images.append(wandb.Image(
                        img_np,
                        masks={
                            "ground_truth": {
                                "mask_data": gt_np,
                                "class_labels": viz_labels
                            },
                            "predictions": {
                                "mask_data": pred_np,
                                "class_labels": viz_labels
                            }
                        },
                        caption=f"Epoch {current_epoch} | Sample {idx}"
                    ))
                
                # Log the list
                wandb.log({"val_samples": viz_images, "val/epoch": current_epoch})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_iou = total_iou / num_batches if num_batches > 0 else 0.0
    avg_psnr = total_psnr / num_batches if num_batches > 0 else 0.0
    model.train()
    return avg_loss, avg_iou, avg_psnr

def train_baseline(
    model,
    args
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_epochs=args.num_epochs
    set_seed(42)
    os.makedirs(os.path.dirname(args.output_dir), exist_ok=True)

    # Get dataloaders
    train_dataloader, val_dataloader, test_dataloader_for_sampling = get_train_val_dataloaders(
        args.image_size,
        args.batch_size,
    )


    exp_id = generate_random_id()
    wandb.init(
        project="baseline-segmentation-coco",
        config=vars(args),
        name=exp_id,
        entity="tf426-cam"
    )
    wandb_id = wandb.run.id

    base_path, ext = os.path.splitext(args.output_dir)
    output_dir_with_id = f"{base_path}_{wandb.run.id}{ext}"
    best_model_path = f"{base_path}_{wandb.run.id}_best{ext}"

    if os.path.exists(output_dir_with_id):
        print(f"Loading pretrained model from {output_dir_with_id}")
        model.load_state_dict(torch.load(output_dir_with_id, map_location=device))
        print("Model loaded successfully!")
        return model

    print("Training new baseline model...")
    model.train()
    criterion = criterion = nn.BCEWithLogitsLoss() #with CE -0.0
    lr=args.learning_rate
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)#, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params:,}")

    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=(len(train_dataloader) * num_epochs),
    )
    print(f"Total training steps: {len(train_dataloader) * num_epochs}")
    print(f"Warmup steps: {args.warmup_steps}")

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0

        for step, batch in enumerate(train_dataloader):
            clean_images, _, gt_mask = batch
            clean_images, gt_mask = clean_images.to(device), gt_mask.to(device)
            

            optimizer.zero_grad()
            outputs = model(clean_images)           

            loss = criterion(outputs, gt_mask)
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            epoch_loss += loss.item()

            global_step = epoch * len(train_dataloader) + step
            # print(f"Epoch [{epoch+1}/{num_epochs}], Step [{step+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}")
            current_lr = lr_scheduler.get_last_lr()[0]  # Get the actual current LR
            wandb.log({
                "train/loss": loss.item(),
                "train/learning_rate": current_lr,
                "train/epoch": epoch,
                "train/global_step": global_step
            })
        
        should_log_images = (epoch % args.viz_interval == 0)

        val_loss, val_iou, val_psnr = validate(
            model, 
            val_dataloader, 
            device, 
            current_epoch=epoch, 
            log_images=should_log_images
        )

        # val_loss, val_iou, val_psnr = validate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")

        wandb.log({
            "val/loss": val_loss,
            "val/iou": val_iou,
            "val/psnr": val_psnr,
            "val/epoch": epoch
        })

        if should_log_images:
            filename = f"baseline_coco_{epoch}_{wandb_id}.pth"
            save_path = os.path.join(args.output_dir, filename)
        
            torch.save(model.state_dict(), save_path)
            print(f"Checkpoint saved to {save_path}")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # image_size = 128
    # batch_size =64
    # model_path = "checkpoints/baseline_unet_coco.pth"
    BASELINE_OUTPUT_CHANNELS = 1    

    # Build model
    model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)

    parser = argparse.ArgumentParser()
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--output_dir", type=str, default="checkpoints/")
    parser.add_argument("--viz_interval", type=int, default=10, help="Visualize every N epochs")
    args = parser.parse_args()

    # Train or load model
    train_baseline(model, args)