import torch
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio
from tqdm.auto import tqdm
from train_diff import augment_batch
import os
from train_diff import CoarseOxfordIIITPet
from diffusers import UNet2DModel, DDPMScheduler


def evaluate_model(args):

    test_dataset = CoarseOxfordIIITPet(
        os.path.join(args.data_root_dir, "val")
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        drop_last=False
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet2DModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device)
    peak_signal_noise_ratio = PeakSignalNoiseRatio().to(device)

    noise_scheduler = DDPMScheduler(num_train_timesteps=args.num_train_timesteps)

    eval_image_transforms = A.Compose([
        A.Resize(height=args.image_size, width=args.image_size),
    ], additional_targets={"coarse_mask": "mask", "gt_mask": "mask"})

    iou_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Processing batches"):
            clean_images, coarse_mask, gt_mask = augment_batch(batch, eval_image_transforms)
            clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)

            # Start with pure noise for the mask
            noisy_masks = torch.randn_like(coarse_mask)

            for t in tqdm(noise_scheduler.timesteps, desc="Sampling"):
                
                # 1. Concatenate the noisy mask and the condition (RGB image)
                # Input shape: (batch_size, 4, H, W)
                model_input = torch.cat([noisy_masks, coarse_mask, clean_images], dim=1)
                # model_input = torch.cat([noisy_masks, clean_images], dim=1)
                
                # 2. Predict the noise
                noise_pred = model(model_input, t).sample

                # 3. Use the scheduler to "denoise" one step
                scheduler_output = noise_scheduler.step(noise_pred, t, noisy_masks)
                noisy_masks = scheduler_output.prev_sample

            # Update the coarse mask
            # noisy_masks = coarse_mask - noisy_masks

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

            iou_score = jaccard_index(
                predicted_masks, 
                gt_mask
            )
            psnr = peak_signal_noise_ratio(
                predicted_masks, 
                gt_mask
            )
            iou_scores.append(iou_score)
            psnr_scores.append(psnr)
            print(f"IOU score: {iou_score}, PSNR score: {psnr}")

    print(f"Average IOU score: {sum(iou_scores) / len(iou_scores)}")
    print(f"Average PSNR score: {sum(psnr_scores) / len(psnr_scores)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--num_train_timesteps", type=int, default=1000)
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()
    evaluate_model(args)