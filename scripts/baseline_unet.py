import torch
import torch.nn as nn

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, base=64):
        super().__init__()
        # Encoder
        self.enc1 = self._conv_block(in_ch, base)
        self.enc2 = self._conv_block(base, base * 2)
        self.enc3 = self._conv_block(base * 2, base * 4)
        
        # Bottleneck
        self.bottleneck = self._conv_block(base * 4, base * 8)
        
        # Decoder
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, kernel_size=2, stride=2)
        self.dec3 = self._conv_block(base * 8, base * 4)
        
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, kernel_size=2, stride=2)
        self.dec2 = self._conv_block(base * 4, base * 2)
        
        self.up1 = nn.ConvTranspose2d(base * 2, base, kernel_size=2, stride=2)
        self.dec1 = self._conv_block(base * 2, base)
        
        # Final output
        self.final = nn.Conv2d(base, out_ch, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)

    def _conv_block(self, in_ch, out_ch, dropout=0.1):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout) if dropout > 0 else nn.Identity(),
        )

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)        # base channels
        e2 = self.enc2(self.pool(e1))   # base*2
        e3 = self.enc3(self.pool(e2))   # base*4
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # base*8
        
        # Decoder with skip connections
        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        
        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        
        return self.final(d1)