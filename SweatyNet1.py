import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class SweatyNet1(nn.Module):
    def __init__(self):
        print("Model initialized")
        super(SweatyNet1, self).__init__()

        # -- Encoder --
        # ConvBlock1
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 8, 3, stride=1, padding=1),  # N x 3 x H x W --> N x 8 x H x W
            nn.ReLU(),
            nn.BatchNorm2d(8)
        )

        # ConvBlock2
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(8, 16, 3, stride=1, padding=1),  # N x 8 x (H/2) x (W/2) --> N x 16 x (H/2) x (H/2)
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # N x 16 x (H/2) x (W/2) --> N x 16 x (H/2) x (H/2)
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )

        # ConvBlock3
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(24, 32, 3, stride=1, padding=1),  # N x 24 x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # N x 32 x (H/4) x (W/4) --> N x 32 x (H/4) x (W/4)
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        # ConvBlock4
        self.conv_block4 = nn.Sequential(
            nn.Conv2d(56, 64, 3, stride=1, padding=1),  # N x 56 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N x 64 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 64, 3, stride=1, padding=1),  # N x 64 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # ConvBlock5
        self.conv_block5 = nn.Sequential(
            nn.Conv2d(120, 128, 3, stride=1, padding=1),  # N x 120 x (H/16) x (W/16) --> N x 128 x (H/16) x (W/16)
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # N x 128 x (H/16) x (W/16) --> N x 128 x (H/16) x (W/16)
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 128, 3, stride=1, padding=1),  # N x 128 x (H/16) x (W/16) --> N x 128 x (H/16) x (W/16)
            nn.ReLU(),
            nn.BatchNorm2d(128),

            nn.Conv2d(128, 64, 3, stride=1, padding=1),  # N x 128 x (H/16) x (W/16) --> N x 64 x (H/16) x (W/16)
            nn.ReLU(),
            nn.BatchNorm2d(64),
        )

        # -- Decoder --
        self.conv_block6 = nn.Sequential(
            nn.Conv2d(184, 64, 1, stride=1, padding=0),  # N x 184 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)
            nn.ReLU(),
            nn.BatchNorm2d(64),

            nn.Conv2d(64, 32, 3, stride=1, padding=1),  # N x 64 x (H/8) x (W/8) --> N x 32 x (H/8) x (W/8)
            nn.ReLU(),
            nn.BatchNorm2d(32),

            nn.Conv2d(32, 32, 3, stride=1, padding=1),  # N x 32 x (H/8) x (W/8) --> N x 32 x (H/8) x (W/8)
            nn.ReLU(),
            nn.BatchNorm2d(32),
        )

        self.conv_block7 = nn.Sequential(
            nn.Conv2d(88, 16, 1, stride=1, padding=0),  # N x 88 x (H/4) x (W/4) --> N x 16 x (H/4) x (W/4)
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 16, 3, stride=1, padding=1),  # N x 16 x (H/4) x (W/4) --> N x 16 x (H/4) x (W/4)
            nn.ReLU(),
            nn.BatchNorm2d(16),

            nn.Conv2d(16, 1, 3, stride=1, padding=1),  # N x 16 x (H/4) x (W/4) --> N x 1 x (H/4) x (W/4)
            nn.ReLU(),
            nn.BatchNorm2d(1),
        )

    def forward(self, x):
        # -- Encode --
        block1_out = self.conv_block1(x)  # N x 3 x H x W --> N x 8 x H x W

        mp1_out = nn.MaxPool2d(2, 2)(block1_out)  # N x 8 x H x W --> N x 8 x (H/2) x (W/2)
        block2_out = self.conv_block2(mp1_out)  # N x 8 x (H/2) x (W/2) --> N x 16 x (H/2) x (W/2)

        concat1 = torch.cat([mp1_out, block2_out], 1)  # N x 24 x (H/2) x (W/2)

        mp2_out = nn.MaxPool2d(2, 2)(concat1)  # N x 24 x (H/2) x (W/2) --> N x (8 + 16) x (H/4) x (W/4)
        block3_out = self.conv_block3(mp2_out)  # N x 24 x (H/4) x (H/4) --> N x 32 x (H/4) x (H/4)

        concat2 = torch.cat([mp2_out, block3_out], 1)  # N x (24 + 32) x (H/4) x (W/4)

        mp3_out = nn.MaxPool2d(2, 2)(concat2)  # N x 56 x (H/4) x (W/4) --> N x 56 x (H/8) x (W/8)
        block4_out = self.conv_block4(mp3_out)  # N x 56 x (H/8) x (W/8) --> N x 64 x (H/8) x (W/8)

        concat3 = torch.cat([mp3_out, block4_out], 1)  # N x (56 + 64) x (H/8) x (W/8)

        mp4_out = nn.MaxPool2d(2, 2)(concat3)  # N x 120 x (H/8) x (W/8) --> N x 120 x (H/16) x (W/16)
        block5_out = self.conv_block5(mp4_out)  # N x 120 x (H/16) x (W/16) --> N x 64 x (H/16) x (W/16)

        # -- Decode --
        up1_out = interpolate(block5_out, scale_factor=2, mode='bilinear', align_corners=True)  # N x 64 x (H/8) x (W/8)

        concat4 = torch.cat([concat3, up1_out], 1)  # N x (120 + 64) x (H/8) x (W/8)

        block6_out = self.conv_block6(concat4)  # N x 184 x (H/8) x (W/8) --> N x 32 x (H/8) x (W/8)

        up2_out = interpolate(block6_out, scale_factor=2, mode='bilinear', align_corners=True)  # N x 32 x (H/4) x (W/4)

        concat5 = torch.cat([concat2, up2_out], 1)  # N x (56 + 32) x (H/4) x (W/4)

        block7_out = self.conv_block7(concat5)  # N x 88 x (H/4) x (W/4) --> N x 1 x (H/4) x (W/4)

        return block7_out
