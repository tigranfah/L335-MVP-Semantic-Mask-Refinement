import os
import torch
import matplotlib.pyplot as plt

def visualize_sample(dataset_dir="data/oxcoarse", split="dev", idx=0, save_dir="visualizations"):
    """
    Save a visualization comparing image, coarse mask, and GT mask.
    """
    # File paths
    img_path = os.path.join(dataset_dir, split, "images", f"{idx:06d}.pt")
    coarse_path = os.path.join(dataset_dir, split, "coarse_masks", f"{idx:06d}.pt")
    gt_path = os.path.join(dataset_dir, split, "gt_masks", f"{idx:06d}.pt")

    if not (os.path.exists(img_path) and os.path.exists(coarse_path) and os.path.exists(gt_path)):
        raise FileNotFoundError(f"Missing files for index {idx} in split '{split}'")

    # Load tensors
    image = torch.load(img_path)
    coarse_mask = torch.load(coarse_path)
    gt_mask = torch.load(gt_path)

    # Convert image from [-1, 1] → [0, 1]
    image_vis = (image * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1)

    # Convert masks to numpy for visualization
    coarse_vis = coarse_mask.squeeze().detach().numpy()
    gt_vis = gt_mask.squeeze().detach().numpy()

    # Create save directory
    os.makedirs(save_dir, exist_ok=True)

    # Plot
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(image_vis)
    axs[0].set_title("Image")
    axs[0].axis("off")

    axs[1].imshow(coarse_vis, cmap="gray")
    axs[1].set_title("Coarse Mask")
    axs[1].axis("off")

    axs[2].imshow(gt_vis, cmap="gray")
    axs[2].set_title("Ground Truth Mask")
    axs[2].axis("off")

    plt.tight_layout()

    # Save instead of showing
    save_path = os.path.join(save_dir, f"{split}_{idx:06d}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved visualization: {save_path}")


if __name__ == "__main__":
    visualize_sample(dataset_dir="data/oxcoarse", split="dev", idx=0)
