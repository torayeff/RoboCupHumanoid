import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from skimage.feature.peak import peak_local_max as peak_local_max

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

downsample = 4
batch_size = 4

model = SweatyNet1()
model.load_state_dict(torch.load("pretrained_models/epoch_50_2050.model", map_location='cpu'))
model.to(device)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())

trainset = utils.SoccerBallDataset("data/lab1/data.csv", "data/lab1", downsample=downsample)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

model.eval()
metrics = utils.evaluate_model(model, device, trainset, verbose=True, debug=False)
#
rc = metrics['tps']/(metrics['tps'] + metrics['fns'])
fdr = metrics['fps']/(metrics['fps'] + metrics['tps'])
#
print("RC: {}, FDR: {}".format(rc, fdr))
