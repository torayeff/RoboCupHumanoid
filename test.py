import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

downsample = 4
batch_size = 4

model = SweatyNet1()
# model.load_state_dict(torch.load("pretrained_models/epoch_20.model", map_location='cpu'))
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

trainset = utils.SoccerBallDataset("data/train/data.csv", "data/train", downsample=downsample)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

model = SweatyNet1()
model.load_state_dict(torch.load("pretrained_models/epoch_100.model", map_location='cpu'))
model.eval()
metrics = utils.evaluate_model(model, device, trainset, verbose=True)
print(metrics)
