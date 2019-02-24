import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1
import time
import argparse
from torch.optim import lr_scheduler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', default='', help='path to pretrained Sweaty model')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size')
    opt = parser.parse_args()
    epochs = opt.epochs
    batch_size = opt.batch_size
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    model = init_sweaty(device, opt.load)

    criterion, optimizer, trainloader, trainset = init_training_configs(batch_size, model)

    train_sweaty(criterion, device, epochs, model, optimizer, trainloader, trainset)

    threshhold = utils.get_abs_threshold(trainset)
    utils.evaluate_sweaty_model(model, device, trainset, threshhold)


def init_training_configs(batch_size, model):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainset = utils.SoccerBallDataset("data/train_images/data.csv", "data/train_images", downsample=4)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    print("# examples: ", len(trainset))
    return criterion, optimizer, trainloader, trainset


# TODO: Share it
def init_sweaty(device, load_path):
    model = SweatyNet1()
    model.to(device)
    print(model)
    if load_path != '':
        print("Loading Sweaty")
        model.load_state_dict(torch.load(load_path))
    return model


def train_sweaty(criterion, device, epochs, model, optimizer, trainloader, trainset):
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
