import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1
import time
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--load', default='', help='folder where you can load a pretrained model')
parser.add_argument('--convLstm', default='no', help='to use or not convLstm')

opt = parser.parse_args()


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = SweatyNet1()
model.to(device)

if opt.load != '':
    print("Loading Sweaty")
    model.load_state_dict(torch.load(opt.load))

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

trainset = utils.SoccerBallDataset("data/train_images/data.csv", "data/train_images", downsample=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

print("# examples: ", len(trainset))

epochs = 100
print("Starting training for {}...".format(epochs))

for epoch in range(epochs):
    epoch_loss = 0
    tic = time.time()
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()

        images = data['image'].float().to(device)
        signals = data['signal'].float().to(device)

        outputs = model(images)

        loss = criterion(signals, outputs)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "pretrained_models/joan/epoch_{}.model".format(epoch + 1))

    epoch_loss /= len(trainset)
    epoch_time = time.time() - tic
    print("Epoch: {}, loss: {}, time: {:.5f} seconds".format(epoch + 1, epoch_loss, epoch_time))
