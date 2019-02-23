import argparse
import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1
from conv_gru import ConvGruCell
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True, help='path to pretrained Sweaty model')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')

    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch_size

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Initializing conv-gru cell...")
    model, convGruModel = init_sweaty_gru(device, opt.load)

    parameters = list(model.parameters()) + list(convGruModel.parameters())
    optimizer = torch.optim.Adam(parameters)
    h_t = None

def init_sweaty_gru(device, load_path):
    model = SweatyNet1()
    model.to(device)
    print(model)
    if load_path != '':
        print("Loading Sweaty")
        model.load_state_dict(torch.load(load_path))
    else:
        raise Exception('Fine tuning the model, there should be a loading path.')

    convGruModel = ConvGruCell(1, 1, device=device)
    convGruModel.to(device)

    return model, convGruModel


def train_sweatyGru(criterion, device, epochs, model, optimizer, trainloader, trainset):
    print("Starting training for {} epochs...".format(epochs))
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

if __name__=='__main__':
    main()
