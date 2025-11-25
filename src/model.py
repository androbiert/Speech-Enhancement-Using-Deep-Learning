# src/model.py
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Conv1D Enhancer (time-domain)
# -------------------------
class Conv1DEnhancer(nn.Module):
    def __init__(self, channels=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(32, 16, kernel_size=9, padding=4),
            nn.ReLU(),
            nn.Conv1d(16, 1, kernel_size=9, padding=4)
        )

    def forward(self, x):
        # x: [B, 1, T]
        return self.net(x)

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class UNet1D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, base_channels=64):
        super(UNet1D, self).__init__()

        C = base_channels

        # Encoder
        self.e1 = ConvBlock(in_channels, C)
        self.pool1 = nn.MaxPool1d(2)

        self.e2 = ConvBlock(C, C * 2)
        self.pool2 = nn.MaxPool1d(2)

        self.e3 = ConvBlock(C * 2, C * 4)
        self.pool3 = nn.MaxPool1d(2)

        self.e4 = ConvBlock(C * 4, C * 8)
        self.pool4 = nn.MaxPool1d(2)

        # Bottleneck
        self.bottleneck = ConvBlock(C * 8, C * 16)

        # Decoder
        self.up4 = nn.ConvTranspose1d(C * 16, C * 8, kernel_size=2, stride=2)
        self.d4 = ConvBlock(C * 16, C * 8)

        self.up3 = nn.ConvTranspose1d(C * 8, C * 4, kernel_size=2, stride=2)
        self.d3 = ConvBlock(C * 8, C * 4)

        self.up2 = nn.ConvTranspose1d(C * 4, C * 2, kernel_size=2, stride=2)
        self.d2 = ConvBlock(C * 4, C * 2)

        self.up1 = nn.ConvTranspose1d(C * 2, C, kernel_size=2, stride=2)
        self.d1 = ConvBlock(C * 2, C)

        # Output
        self.out = nn.Conv1d(C, out_channels, kernel_size=1)

    def _crop_and_concat(self, upsuper, bypass):
        if upsuper.size(-1) != bypass.size(-1):
            diff = bypass.size(-1) - upsuper.size(-1)
            bypass = bypass[:, :, diff // 2: diff // 2 + upsuper.size(-1)]
        return torch.cat((upsuper, bypass), dim=1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool1(e1))
        e3 = self.e3(self.pool2(e2))
        e4 = self.e4(self.pool3(e3))

        b = self.bottleneck(self.pool4(e4))

        d4 = self.up4(b)
        d4 = self._crop_and_concat(d4, e4)
        d4 = self.d4(d4)

        d3 = self.up3(d4)
        d3 = self._crop_and_concat(d3, e3)
        d3 = self.d3(d3)

        d2 = self.up2(d3)
        d2 = self._crop_and_concat(d2, e2)
        d2 = self.d2(d2)

        d1 = self.up1(d2)
        d1 = self._crop_and_concat(d1, e1)
        d1 = self.d1(d1)

        return self.out(d1)


# -------------------------
# ResNetEnhancer (stub) - works on spectrogram magnitude input assumed [B,1,F,T]
# -------------------------
class ResNetEnhancer(nn.Module):
    def __init__(self, in_channels=1, base=16):
        super().__init__()
        # VERY simple residual conv stack for magnitude spectrograms (placeholder)
        self.conv1 = nn.Conv2d(in_channels, base, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(base)
        self.conv2 = nn.Conv2d(base, base, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(base)
        self.conv3 = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x):
        # x: [B, 1, F, T]
        y = F.relu(self.bn1(self.conv1(x)))
        y = F.relu(self.bn2(self.conv2(y)))
        y = self.conv3(y)
        return y

# -------------------------
# Simple DCCRN-like model
# -------------------------
class DCCRN(nn.Module):
    def __init__(self, n_fft=512, hop_length=256, base_channels=16):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        # small UNet-like conv2d that takes 2 channels (real/imag) as input
        self.enc1 = nn.Sequential(nn.Conv2d(2, base_channels, 3, padding=1), nn.ReLU())
        self.enc2 = nn.Sequential(nn.Conv2d(base_channels, base_channels*2, 3, padding=1), nn.ReLU())
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(base_channels*2, base_channels, kernel_size=2, stride=2)
        self.dec1 = nn.Sequential(nn.Conv2d(base_channels*2, base_channels, 3, padding=1), nn.ReLU())
        self.out_conv = nn.Conv2d(base_channels, 2, 1)  # output real+imag

    def forward(self, x):
        """
        x: waveform [B,1,T]
        returns: stft-like tensor [B, 2, F, T_frames]
        """
        # compute stft (real + imag)
        window = torch.hann_window(self.n_fft, device=x.device)
        stft = torch.stft(x.squeeze(1), n_fft=self.n_fft, hop_length=self.hop_length, window=window, return_complex=False, center=True)
        # shape [B, F, T, 2] -> permute to [B, 2, F, T]
        stft = stft.permute(0, 3, 1, 2)
        # pass through small conv network
        e1 = self.enc1(stft)
        e2 = self.enc2(e1)
        p = self.pool(e2)
        u = self.up2(p)
        # if mismatch in size, crop/pad
        if u.size(-2) != e1.size(-2) or u.size(-1) != e1.size(-1):
            u = F.interpolate(u, size=(e1.size(-2), e1.size(-1)), mode='bilinear', align_corners=False)
        d = torch.cat([u, e1], dim=1)
        d = self.dec1(d)
        out = self.out_conv(d)
        # out shape [B,2,F,T] (real+imag)
        return out
