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

    
    with torch.no_grad():
        for batch in tqdm(val_dataloader, desc="Validating"):
            clean_images, _, gt_mask = batch
            clean_images, gt_mask = clean_images.to(device), gt_mask.to(device)
            
            outputs = model(clean_images)          
            loss = criterion(outputs, gt_mask)

            iou_score = jaccard_index(
                outputs, 
                gt_mask
            )
            psnr = psnr_metric(
                outputs, 
                gt_mask
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)


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

        val_loss, val_iou, val_psnr = validate(model, val_dataloader, device)
        print(f"Validation loss: {val_loss:.4f}")

        wandb.log({
            "val/loss": val_loss,
            "val/iou": val_iou,
            "val/psnr": val_psnr,
            "val/epoch": epoch
        })


    torch.save(model.state_dict(), output_dir_with_id)
    print(f"Final model saved to {output_dir_with_id}")
    


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
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--warmup_steps", type=int, default=500, help="Number of warmup steps for LR scheduler")
    parser.add_argument("--output_dir", type=str, default="checkpoints/baseline_unet_coco.pth")
    args = parser.parse_args()

    # Train or load model
    train_baseline(model, args)