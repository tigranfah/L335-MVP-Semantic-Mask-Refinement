import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.datasets import OxfordIIITPet
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset
from tqdm.auto import tqdm
from PIL import Image
import os
from matplotlib import pyplot as plt


if __name__ == "__main__":

    IMAGE_SIZE = 256

    transform_params = smp.encoders.get_preprocessing_params("resnet34")

    # Transforms for the RGB image
    image_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.PILToTensor()
    ])
    
    # Transform for model input (with normalization)
    model_input_transform = transforms.Compose([
        transforms.Normalize(
            transform_params["mean"], transform_params["std"]
        )
    ])

    # Transform for segmentation masks
    mask_transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.PILToTensor(),  # Converts PIL to tensor without normalizing
    ])

    dataset = OxfordIIITPet(
        root="./data",
        split="trainval", 
        download=True, 
        target_types="segmentation",
        transform=image_transform,
        target_transform=mask_transform
    )

    batch_size = 64
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    loaded_model = smp.from_pretrained("tigranfah/oxford-pet-segmentation")
    loaded_model.eval()

    sample_index = 0

    dataset_dir = "data/oxcoarse"
    os.makedirs(os.path.join(dataset_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "coarse_masks"), exist_ok=True)
    os.makedirs(os.path.join(dataset_dir, "gt_masks"), exist_ok=True)

    for batch in tqdm(dataloader, desc="Generating and saving samples"):
        images, gt_masks = batch

        # Images are already tensors from the dataset, now normalize for model input
        normalized_images = model_input_transform(images.float())
        with torch.no_grad():
            coarse_masks = loaded_model(normalized_images)
            coarse_masks = coarse_masks.sigmoid()

        # Iterate over each item in the batch
        for i in range(images.shape[0]):
            # Get the individual tensor for this sample
            img_tensor = images[i]
            coarse_mask_tensor = coarse_masks[i]
            gt_mask_tensor = gt_masks[i]
            
            # Define the output file path using a zero-padded index
            file_name = f"{sample_index:06d}.pt"
            
            # Save each tensor individually
            # Use args.output_dir
            torch.save(img_tensor, os.path.join(dataset_dir, "images", file_name))
            torch.save(coarse_mask_tensor, os.path.join(dataset_dir, "coarse_masks", file_name))
            torch.save(gt_mask_tensor, os.path.join(dataset_dir, "gt_masks", file_name))
            
            sample_index += 1