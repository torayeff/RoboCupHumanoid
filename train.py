import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1
import time
import argparse
from conv_gru import ConvGruCell
from torch.optim import lr_scheduler


parser = argparse.ArgumentParser()

parser.add_argument('--load', default='', help='path to pretrained Sweaty model')
parser.add_argument('--convLstm', type=int, default=0, help='flag for conv-gru layer. By default it does not use it')
parser.add_argument('--epochs', type=int, default=100,  help='total number of epochs')
parser.add_argument('--batch_size', type=int, default=4,  help='batch size')

opt = parser.parse_args()

epochs = opt.epochs
batch_size = opt.batch_size
use_lstm = opt.convLstm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = SweatyNet1()
model.to(device)

if opt.load != '':
    print("Loading Sweaty")
    model.load_state_dict(torch.load(opt.load))

criterion = nn.MSELoss()

shuffle = True

if use_lstm == 0:
    optimizer = torch.optim.Adam(model.parameters())
else:
    print("Initializing conv-gru cell...")
    convGruModel = ConvGruCell(1, 1, device=device)
    convGruModel.to(device)
    parameters = list(model.parameters()) + list(convGruModel.parameters())
    optimizer = torch.optim.Adam(parameters)
    h_t = None

exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

trainset = utils.SoccerBallDataset("data/train_images/data.csv", "data/train_images", downsample=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=shuffle, num_workers=2)

print("# examples: ", len(trainset))

print("Starting training for {} epochs...".format(epochs))

for epoch in range(epochs):
    epoch_loss = 0
    tic = time.time()
    exp_lr_scheduler.step()

    for i, data in enumerate(trainloader):
        optimizer.zero_grad()

        images = data['image'].float().to(device)
        signals = data['signal'].float().to(device)

        outputs = model(images)

        if use_lstm != 0:
            outputs = convGruModel(outputs, h_t)
            h_t = outputs

        loss = criterion(signals, outputs)

        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        torch.save(model.state_dict(), "pretrained_models/joan/epoch_{}_stepLrNoBR.model".format(epoch + 1))

    epoch_loss /= len(trainset)
    epoch_time = time.time() - tic
    print("Epoch: {}, loss: {}, time: {:.5f} seconds".format(epoch + 1, epoch_loss, epoch_time))


utils.evaluate_model(model, device, trainset)
