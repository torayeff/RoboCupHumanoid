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

# model = SweatyNet1()
# model.load_state_dict(torch.load("pretrained_models/epoch_100.model", map_location='cpu'))
# model.to(device)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters())

trainset = utils.SoccerBallDataset("data/train_images/data.csv", "data/train_images", downsample=downsample)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)

# model.eval()
model = None
metrics = utils.evaluate_model(model, device, trainset, verbose=True, debug=True)
#
# rc = metrics['tps']/(metrics['tps'] + metrics['fns'])
# fdr = metrics['fps']/(metrics['fps'] + metrics['tps'])
#
# print("RC: {}, FDR: {}".format(rc, fdr))
