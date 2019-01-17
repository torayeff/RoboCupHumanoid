import matplotlib.pyplot as plt
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as transforms
import matplotlib.cm as cm
from utils import SoccerBallDataset
from skimage.feature import peak
import torch.nn as nn
%matplotlib inline
%load_ext autoreload
%autoreload 2

class SweatyNet1(nn.Mod):
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
	    		nn.Conv2d(16, 32, 3, padding=1),
	    		nn.ReLU(),
	    		nn.BatchNorm2d(32),

	    		nn.Conv2d(32, 32, 3, padding=1),
	    		nn.ReLU(),
	    		nn.BatchNorm2d(32)
	    	)

	    self.max_pool_3 = nn.MaxPool2d(2, stride=2)

	    self.layer4 = nn.Sequential(
	    		nn.Conv2d(32, 64, 3, padding=1),
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
	    		nn.Conv2d(64, 128, 3, padding=1),
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

	    self.upsampling_1 = nn.UpsamplingBilinear2d(scale_factor=2)

	    self.layer6 = nn.Sequential(
	    		nn.Conv2d(64, 64, 1), 
	    		nn.ReLU(),
	    		nn.BatchNorm2d(64),

	    		nn.Conv2d(64, 32, 3, padding=1), 
	    		nn.ReLU(),
	    		nn.BatchNorm2d(32),

	    		nn.Conv2d(32, 32, 3, padding=1), 
	    		nn.ReLU(),
	    		nn.BatchNorm2d(32)
	    	)

	    self.upsampling_2 = nn.UpsamplingBilinear2d(scale_factor=2)

	    self.layer7 = nn.Sequential(
	    		nn.Conv2d(32, 16, 1), 
	    		nn.ReLU(), 
	    		nn.BatchNorm2d(16), 

	    		nn.Conv2d(16, 16, 3, padding=1),
	    		nn.ReLU(),
	    		nn.BatchNorm2d(16), 

	    		nn.Conv2d(16, 8, 3, padding=1),
	    		nn.ReLU(),
	    		nn.BatchNorm2d(8)
	    	)

	def forward(self, x):
		x = self.layer1(x)

		out_pool = self.max_pool_1(x)
		x = self.layer2(out_pool)
		x = out_pool + x

		out_pool = self.max_pool_2(x)
		x = self.layer3(out_pool)
		o_1 = out_pool + x

		out_pool = self. max_pool_3(o_1)
		x = self.layer4(out_pool)
		o_2 = out_pool + x

		out_pool = self.max_pool_4(o_2)
		x = self.layer5(out_pool)
		x = self.upsampling_1(x)

		x = o_2 + x
		x = self.layer6(x)
		x = self.upsampling_2(x)

		x = o_1 + x

		out = self.layer7(x)

		return out









