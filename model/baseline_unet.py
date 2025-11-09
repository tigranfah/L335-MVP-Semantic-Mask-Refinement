import torch
import torch.nn as nn

class UNetSmall(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64):
        super().__init__()
        self.enc1 = self._conv_block(in_ch, base)
        self.enc2 = self._conv_block(base, base * 2)
        self.enc3 = self._conv_block(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec2 = self._conv_block(base * 4 + base * 2, base * 2)
        self.dec1 = self._conv_block(base * 2 + base, base)
        self.final = nn.Conv2d(base, out_ch, kernel_size=1)

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        d2 = self.up(e3)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.final(d1)