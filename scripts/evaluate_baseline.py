import torch
import argparse
from torch.utils.data import DataLoader
import albumentations as A
from torchmetrics import JaccardIndex, PeakSignalNoiseRatio
from tqdm.auto import tqdm
from train_diff import augment_batch
import os
from train_diff import CoarseOxfordIIITPet
from model.baseline_unet import UNetSmall


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

    BASELINE_OUTPUT_CHANNELS = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()

    jaccard_index = JaccardIndex(task="binary", threshold=0.5).to(device)
    peak_signal_noise_ratio = PeakSignalNoiseRatio().to(device)

    eval_image_transforms = A.Compose([
        A.Resize(height=args.image_size, width=args.image_size),
    ], additional_targets={"coarse_mask": "mask", "gt_mask": "mask"})

    iou_scores = []
    psnr_scores = []
    
    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Processing batches"):
            clean_images, coarse_mask, gt_mask = augment_batch(batch, eval_image_transforms)
            clean_images, coarse_mask, gt_mask = clean_images.to(device), coarse_mask.to(device), gt_mask.to(device)

            predicted_mask = model(clean_images)

            # --- Un-normalize all images for saving ---
            # Un-normalize from [-1, 1] to [0, 1]
            predicted_masks = (predicted_mask * 0.5 + 0.5).clamp(0, 1)
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
    parser.add_argument("--data_root_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--image_size", type=int, default=256)
    args = parser.parse_args()
    evaluate_model(args)