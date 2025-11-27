import torch
from PIL import Image
from torchvision import transforms

def random_blob_image(
    height=256,
    width=256,
    num_blobs=10,
    blob_radius=10,
    device="cpu"
):
    # Base empty image
    img = torch.zeros((1, 1, height, width), device=device)

    for _ in range(num_blobs):
        # Random center
        cx = torch.randint(0, width, (1,))
        cy = torch.randint(0, height, (1,))
        
        # Random amplitude for blob
        amp = torch.rand(1).item() * 1.0
        
        # Create a 2D Gaussian kernel
        y = torch.arange(height, device=device).view(-1, 1)
        x = torch.arange(width, device=device).view(1, -1)

        blob = torch.exp(-((x - cx) ** 2 + (y - cy) ** 2) / (2 * blob_radius ** 2))
        blob = blob * amp
        
        img += blob

    # Normalize to [0,1]
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)

    return (img.squeeze() >= 0.5)  # shape: (H, W)

def save_tensor_png(tensor, path):
    """
    tensor: (H, W) or (1, H, W), values in [0,1] or [0,255]
    """
    if tensor.dim() == 3:
        tensor = tensor.squeeze(0)

    # Convert to uint8
    array = (tensor.clamp(0, 1) * 255).byte().cpu().numpy()

    # Create grayscale image
    img = Image.fromarray(array, mode="L")
    img.save(path)


# img = Image.open("experiments/city.png").convert("RGB")
# transform = transforms.ToTensor()   # converts to float tensor in [0,1], shape (C,H,W)
# tensor_img = torch.load("experiments/003625.pt") #transform(img)

def block_image(img, num_blobs, blob_radius,device):
    blob = random_blob_image(img.shape[1], img.shape[2], num_blobs=20, blob_radius=20).to(device)
    blob_img = blob.logical_not().unsqueeze(0) * img
    return img

# save_tensor_png(blob_img, "experiments/image_blob.png")
# save_tensor_png(blob, "experiments/blob.png")
# save_tensor_png(img, "experiments/003625.png")