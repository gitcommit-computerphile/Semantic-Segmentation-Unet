# Dependencies
!pip install torch torchvision numpy opencv-python

import torch
import torch.nn as nn
import torch.nn.functional as F

# Custom Segmentation Model Code
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool=True):
        super().__init__()
        self.pool = pool
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),  # SAME
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.down = nn.MaxPool2d(2, 2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x_down = self.down(x)
        return x, x_down


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(out_channels + skip_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.upsample(x)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class SegmentationModel(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()

        # Encoder
        self.enc1 = ConvBlock(in_channels, 64)
        self.enc2 = ConvBlock(64, 128)
        self.enc3 = ConvBlock(128, 256)
        self.enc4 = ConvBlock(256, 512)

        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024, pool=False)

        # Decoder
        self.up4 = UpBlock(1024, 512, 512)
        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        # Final prediction layer
        self.final_conv = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # Encode
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)

        _, x = self.bottleneck(x)

        # Decode
        x = self.up4(x, s4)
        x = self.up3(x, s3)
        x = self.up2(x, s2)
        x = self.up1(x, s1)

        # Final prediction
        x = self.final_conv(x)
        return x


# Training Loop (Binary Segmentation)
import torch.optim as optim

model = SegmentationModel(in_channels=3, num_classes=1).cuda()
criterion = nn.BCEWithLogitsLoss()  # Use nn.CrossEntropyLoss() for multi-class
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for epoch in range(num_epochs):
    model.train()
    for images, masks in dataloader:  # masks shape: [B, H, W] or [B, 1, H, W]
        images = images.cuda()
        masks = masks.cuda().float()  # for BCE

        outputs = model(images).squeeze(1)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
