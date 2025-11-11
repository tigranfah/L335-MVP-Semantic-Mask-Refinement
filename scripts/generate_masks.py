import os
import torch
from tqdm.auto import tqdm
from scripts.data_utils import get_loaders
from model.baseline_unet import UNetSmall
from scripts.train_baseline import train_baseline   


def generate_and_save_masks(model, dataloader, dataset_dir, split_name, device):
    os.makedirs(os.path.join(dataset_dir, split_name, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split_name, "coarse_masks"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, split_name, "gt_masks"), exist_ok=True)

    model.eval()
    idx = 0
    with torch.no_grad():
        for images, gt_masks in tqdm(dataloader, desc=f"Generating {split_name} masks"):
            images = images.to(device)
            coarse_logits = model(images)
            # coarse_masks = torch.sigmoid(coarse_logits)
            coarse_masks = (torch.sigmoid(coarse_logits) > 0.5).float()
            for i in range(images.shape[0]):
                torch.save(images[i].cpu(), os.path.join(dataset_dir, split_name, "images", f"{idx:06d}.pt"))
                torch.save(coarse_masks[i].cpu(), os.path.join(dataset_dir, split_name, "coarse_masks", f"{idx:06d}.pt"))
                torch.save(gt_masks[i].cpu(), os.path.join(dataset_dir, split_name, "gt_masks", f"{idx:06d}.pt"))
                idx += 1
    print(f"Saved {idx} samples for {split_name}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "checkpoints/baseline_unet.pth"
    dataset_dir = "data/oxcoarse"
    image_size = 256
    batch_size = 32
    BASELINE_OUTPUT_CHANNELS = 1


    loaders = get_loaders(image_size=image_size, batch_size=batch_size)

    model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)
    model = train_baseline(model, loaders["train"], device, model_path=model_path)

    generate_and_save_masks(model, loaders["train"], dataset_dir, "train", device)
    generate_and_save_masks(model, loaders["dev"], dataset_dir, "dev", device)


if __name__ == "__main__":
    main()
