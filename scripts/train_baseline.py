import os
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from model.baseline_unet import UNetSmall
from scripts.data_utils import get_loaders


def train_baseline(
    model,
    train_loader,
    device,
    model_path="checkpoints/baseline_unet.pth",
    num_epochs=100,
    lr=1e-4,
):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if os.path.exists(model_path):
        print(f"Loading pretrained model from {model_path}")
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully!")
        return model

    print("Training new baseline model...")
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)

            # print(f"\n[DEBUG] Epoch {epoch+1}: Unique values in raw masks:", torch.unique(masks))
            targets = (masks + 1) / 2  # [-1,1] → [0,1]
            # print(f"[DEBUG] Epoch {epoch+1}: Unique values in normalized targets:", torch.unique(targets))
            # [DEBUG] Epoch 1: Unique values in raw masks: tensor([-1.,  1.], device='cuda:0')         
            # [DEBUG] Epoch 1: Unique values in normalized targets: tensor([0., 1.], device='cuda:0')  
            # Epoch 1/100:   2%|▊                                       | 1/46 [00:01<01:24,  1.89s/it]
            # [DEBUG] Epoch 1: Unique values in raw masks: tensor([-1.,  1.], device='cuda:0')         
            # [DEBUG] Epoch 1: Unique values in normalized targets: tensor([0., 1.], device='cuda:0')
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs} | Loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    return model


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 256
    batch_size = 32
    model_path = "checkpoints/baseline_unet.pth"
    BASELINE_OUTPUT_CHANNELS = 1    

    # Get dataloaders
    loaders = get_loaders(image_size=image_size, batch_size=batch_size)
    train_loader = loaders["train"]

    # Build model
    model = UNetSmall(in_ch=3, out_ch=BASELINE_OUTPUT_CHANNELS).to(device)

    # Train or load model
    train_baseline(model, train_loader, device, model_path)