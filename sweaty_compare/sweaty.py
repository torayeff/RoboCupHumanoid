import torch
import torch.nn as nn
from torch.nn.functional import interpolate


class SweatyNet1(nn.Module):
    def __init__(self):
        super().__init__()

        self.layer1 = nn.Sequential(
                nn.Conv2d(3, 8, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(8)
            )

        self.max_pool_1 = nn.MaxPool2d(2, stride=2)

        self.layer2 = nn.Sequential(
                nn.Conv2d(8, 16, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16),

                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16)
            )

        self.max_pool_2 = nn.MaxPool2d(2, stride=2)

        self.layer3 = nn.Sequential(
                nn.Conv2d(24, 32, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 32, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32)
            )

        self.max_pool_3 = nn.MaxPool2d(2, stride=2)

        self.layer4 = nn.Sequential(
                nn.Conv2d(56, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64)
            )

        self.max_pool_4 = nn.MaxPool2d(2, stride=2)

        self.layer5 = nn.Sequential( 
                nn.Conv2d(120, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),

                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),

                nn.Conv2d(128, 128, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),

                nn.Conv2d(128, 64, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
            )

        self.layer6 = nn.Sequential(
                nn.Conv2d(184, 64, 1), 
                nn.ReLU(),
                nn.BatchNorm2d(64),

                nn.Conv2d(64, 32, 3, padding=1), 
                nn.ReLU(),
                nn.BatchNorm2d(32),

                nn.Conv2d(32, 32, 3, padding=1), 
                nn.ReLU(),
                nn.BatchNorm2d(32)
            )


        self.layer7 = nn.Sequential(
                nn.Conv2d(88, 16, 1), 
                nn.ReLU(), 
                nn.BatchNorm2d(16), 

                nn.Conv2d(16, 16, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(16), 

                nn.Conv2d(16, 1, 3, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(1)
            )


    def forward(self, x):
        x = self.layer1(x)

        out_pool = self.max_pool_1(x)
        x = self.layer2(out_pool)
        x = torch.cat((out_pool, x), 1)

        out_pool = self.max_pool_2(x)
        x = self.layer3(out_pool)
        o_1 = torch.cat((out_pool, x), 1)

        out_pool = self. max_pool_3(o_1)
        x = self.layer4(out_pool)
        o_2 = torch.cat((out_pool, x), 1)

        out_pool = self.max_pool_4(o_2)
        x = self.layer5(out_pool)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat((o_2, x), 1)
        x = self.layer6(x)
        x = interpolate(x, scale_factor=2, mode='bilinear', align_corners=True)

        x = torch.cat((o_1, x), 1)

        out = self.layer7(x)

        return out