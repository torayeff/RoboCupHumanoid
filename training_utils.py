"""
General methods used during model trainings.
TODO: Refactoring...
"""
import sweaty_net_2_outputs
from conv_gru import ConvGruCellPreConv
import torch
from torch import nn
import utils as utils


def init_type2_sweaty_gru(device, load_path, load_gru=''):
    model = sweaty_net_2_outputs.SweatyNet1()
    model.to(device)
    if load_path != '':
        print("Loading Sweaty")
        model.load_state_dict(torch.load(load_path))
    else:
        raise Exception('Fine tuning the model, there should be a loading path.')

    convGruModel = ConvGruCellPreConv(89, 1, device=device)
    convGruModel.to(device)

    if load_gru != '':
        print('Loading gru, continuing training ...')
        convGruModel.load_state_dict(torch.load(load_gru))

    return model, convGruModel


def init_sweaty(device, load_path):
    from SweatyNet1 import SweatyNet1
    model = SweatyNet1()
    model.to(device)
    print(model)
    if load_path != '':
        print("Loading Sweaty")
        model.load_state_dict(torch.load(load_path))
    return model


def init_training_configs_for_conv_gru(batch_size, alpha):
    criterion = nn.MSELoss()
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainset = utils.SoccerBallDataset("data/train/data.csv", "data/train", downsample=4, alpha=alpha)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=False, num_workers=2)
    print("# examples: ", len(trainset))
    return criterion, trainloader, trainset
