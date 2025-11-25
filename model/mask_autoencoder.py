import torch
import torch.nn as nn


class MaskEncoder(nn.Module):
    """Encoder for segmentation masks"""
    
    def __init__(self, num_input_channels: int, base_channel_size: int, 
                 latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
            num_input_channels: Number of input channels (1 for binary masks)
            base_channel_size: Base number of channels
            latent_dim: Dimensionality of latent representation z
            act_fn: Activation function
        """
        super().__init__()
        c_hid = base_channel_size
        
        # For 256x256 input
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1, stride=2),  # 256->128
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2*c_hid, kernel_size=3, padding=1, stride=2),  # 128->64
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2*c_hid, 4*c_hid, kernel_size=3, padding=1, stride=2),  # 64->32
            act_fn(),
            nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(4*c_hid, 8*c_hid, kernel_size=3, padding=1, stride=2),  # 32->16
            act_fn(),
            nn.Conv2d(8*c_hid, 8*c_hid, kernel_size=3, padding=1, stride=2),  # 16->8
            act_fn(),
            nn.Flatten(),
            nn.Linear(8*c_hid*8*8, latent_dim)
        )
    
    def forward(self, x):
        return self.net(x)


class MaskDecoder(nn.Module):
    """Decoder for segmentation masks"""
    
    def __init__(self, num_output_channels: int, base_channel_size: int, 
                 latent_dim: int, act_fn: object = nn.GELU):
        """
        Args:
            num_output_channels: Number of output channels (1 for binary masks)
            base_channel_size: Base number of channels
            latent_dim: Dimensionality of latent representation z
            act_fn: Activation function
        """
        super().__init__()
        c_hid = base_channel_size
        
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 8*c_hid*8*8),
            act_fn()
        )
        
        # For 256x256 output
        self.net = nn.Sequential(
            nn.ConvTranspose2d(8*c_hid, 8*c_hid, kernel_size=3, output_padding=1, 
                              padding=1, stride=2),  # 8->16
            act_fn(),
            nn.Conv2d(8*c_hid, 8*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(8*c_hid, 4*c_hid, kernel_size=3, output_padding=1, 
                              padding=1, stride=2),  # 16->32
            act_fn(),
            nn.Conv2d(4*c_hid, 4*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(4*c_hid, 2*c_hid, kernel_size=3, output_padding=1, 
                              padding=1, stride=2),  # 32->64
            act_fn(),
            nn.Conv2d(2*c_hid, 2*c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2*c_hid, c_hid, kernel_size=3, output_padding=1, 
                              padding=1, stride=2),  # 64->128
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_output_channels, kernel_size=3, 
                              output_padding=1, padding=1, stride=2),  # 128->256
        )
    
    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 8, 8)
        x = self.net(x)
        return x


class MaskAutoencoder(nn.Module):
    """
    Autoencoder for binary segmentation masks.
    
    Input: (B, 1, H, W) in range [-1, 1]
    Output: (B, 1, H, W) logits (before sigmoid)
    """
    
    def __init__(self, base_channel_size: int = 32, latent_dim: int = 128,
                 num_input_channels: int = 1, num_output_channels: int = 1):
        """
        Args:
            base_channel_size: Base number of channels in encoder/decoder
            latent_dim: Dimensionality of bottleneck latent vector
            num_input_channels: Number of input channels (1 for binary masks)
            num_output_channels: Number of output channels (1 for binary masks)
        """
        super().__init__()
        self.latent_dim = latent_dim
        self.num_input_channels = num_input_channels
        self.num_output_channels = num_output_channels
        
        self.encoder = MaskEncoder(num_input_channels, base_channel_size, latent_dim)
        self.decoder = MaskDecoder(num_output_channels, base_channel_size, latent_dim)
    
    def forward(self, x):
        """
        Forward pass through autoencoder.
        
        Args:
            x: Input masks (B, 1, H, W) in range [-1, 1]
            
        Returns:
            logits: Output logits (B, 1, H, W) - apply sigmoid for probabilities
        """
        z = self.encoder(x)
        logits = self.decoder(z)
        return logits
    
    def encode(self, x):
        """Encode masks to latent representation"""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to mask logits"""
        return self.decoder(z)
    
    def reconstruct(self, x, threshold=0.5):
        """
        Reconstruct masks and return binary predictions.
        
        Args:
            x: Input masks (B, 1, H, W) in range [-1, 1]
            threshold: Threshold for binary prediction
            
        Returns:
            Binary masks (B, 1, H, W) with values {0, 1}
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.sigmoid(logits)
            binary = (probs > threshold).float()
        return binary


# # Test the model
# if __name__ == "__main__":
    # # Test with 256x256 masks
    # model = MaskAutoencoder(base_channel_size=32, latent_dim=128)
    
    # # Create dummy input
    # x = torch.randn(4, 1, 256, 256)  # Batch of 4 masks
    
    # print("Input shape:", x.shape)
    
    # # Forward pass
    # logits = model(x)
    # print("Output shape:", logits.shape)
    
    # # Encode
    # z = model.encode(x)
    # print("Latent shape:", z.shape)
    
    # # Reconstruct
    # recon = model.reconstruct(x)
    # print("Reconstruction shape:", recon.shape)
    
    # # Count parameters
    # total_params = sum(p.numel() for p in model.parameters())
    # print(f"Total parameters: {total_params:,}")