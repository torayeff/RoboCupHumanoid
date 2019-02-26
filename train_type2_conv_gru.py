import torch
import torch.nn as nn
import utils as utils
from sweaty_net_2_outputs import SweatyNet1
from conv_gru import ConvGruCellPreConv
import time
import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', required=True, help='path to pretrained Sweaty model')
    parser.add_argument('--epochs', type=int, default=100, help='total number of epochs')
    parser.add_argument('--batch_size', type=int, default=15, help='batch size')
    parser.add_argument('--alpha', type=int, default=1000, help='batch size')
    parser.add_argument('--model_name', type=str, default="model", help='model name')

    opt = parser.parse_args()

    epochs = opt.epochs
    batch_size = opt.batch_size
    model_name = opt.model_name + str(opt.alpha)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    print("Initializing conv-gru cell...")
    sweaty, convGruModel = init_sweaty_gru(device, opt.load)

    criterion, trainloader, trainset = init_training_configs(batch_size, opt.alpha)
    train_sweatyGru(criterion, device, epochs, sweaty, convGruModel, trainloader, trainset, model_name)

    threshhold = utils.get_abs_threshold(trainset)
    utils.evaluate_sweaty_gru_model(sweaty, convGruModel, device, trainset, threshhold)


def init_training_configs(batch_size, alpha):
    criterion = nn.MSELoss()
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainset = utils.SoccerBallDataset("data/train/data.csv", "data/train", downsample=4, alpha=alpha)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("# examples: ", len(trainset))
    return criterion, trainloader, trainset


def init_sweaty_gru(device, load_path):
    model = SweatyNet1()
    model.to(device)
    print(model)
    if load_path != '':
        print("Loading Sweaty")
        model.load_state_dict(torch.load(load_path))
    else:
        raise Exception('Fine tuning the model, there should be a loading path.')

    convGruModel = ConvGruCellPreConv(89, 1, device=device)
    convGruModel.to(device)

    return model, convGruModel


def train_sweatyGru(criterion, device, epochs, sweaty, conv_gru, trainloader, trainset, model_name):
      # freeze sweaty
    for param in sweaty.parameters():
        param.requires_grad = False

    optimizer = torch.optim.Adam(conv_gru.parameters())

    print("Starting training for {} epochs...".format(epochs))
    for epoch in range(epochs):
        epoch_loss = 0
        tic = time.time()

        if epoch == 20:
            # unfreeze Sweaty
            for param in sweaty.parameters():
                param.requires_grad = True

            parameters = list(sweaty.parameters()) + list(conv_gru.parameters())
            optimizer = torch.optim.Adam(parameters)

        for i, data in enumerate(trainloader):
            optimizer.zero_grad()

            images = data['image'].float().to(device)
            signals = data['signal'].float().to(device)

            sweaty_features, skip_outputs = sweaty(images)

            input_for_gru = torch.cat([sweaty_features, skip_outputs], 1)

            hidden_state = conv_gru(input_for_gru)

            loss = criterion(signals, hidden_state)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            torch.save(sweaty.state_dict(), "pretrained_models/joan/{}_epoch_{}_sweaty.model".format(model_name, epoch + 1))
            torch.save(conv_gru.state_dict(), "pretrained_models/joan/{}_epoch_{}_gru.model".format(model_name, epoch + 1))

        epoch_loss /= len(trainset)
        epoch_time = time.time() - tic
        print("Epoch: {}, loss: {}, time: {:.5f} seconds".format(epoch + 1, epoch_loss, epoch_time))


if __name__=='__main__':
    main()
