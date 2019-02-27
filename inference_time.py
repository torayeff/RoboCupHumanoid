import torch
import utils as utils
from SweatyNet1 import SweatyNet1
import conv_gru as gru
import argparse
from time import time
import sweaty_net_2_outputs as s2


def inference_sweaty(sweaty, trainset, device):
    sweaty.eval()
    times = []
    for i, data in enumerate(trainset):
        if i == 10:
            break
        tic = time()
        image = data['image'].unsqueeze(0).float().to(device)

        with torch.no_grad():
            sweaty_output = sweaty(image)
            times.append(time()-tic)

    avg_time = sum(times)/len(times)
    print("Inference time in {} for sweaty is {}".format(device, avg_time))


def inference_gru(sweaty, conv_gru, trainset, device):
    sweaty.eval()
    times = []
    for i, data in enumerate(trainset):
        if i == 10:
            break
        tic = time()
        image = data['image'].unsqueeze(0).float().to(device)

        with torch.no_grad():
            sweaty_output = sweaty(image)
            hidden_state = conv_gru(sweaty_output)

            times.append(time() - tic)

    avg_time = sum(times) / len(times)
    print("Inference time in {} for sweaty gru is {}".format(device, avg_time))


def inference_type2_gru(sweaty, conv_gru, trainset, device):
    sweaty.eval()
    times = []
    for i, data in enumerate(trainset):
        if i == 10:
            break
        tic = time()
        image = data['image'].unsqueeze(0).float().to(device)

        with torch.no_grad():
            sweaty_output, skip_outputs = sweaty(image)
            input_for_gru = torch.cat([sweaty_output, skip_outputs], 1)
            hidden_state = conv_gru(input_for_gru)

            times.append(time() - tic)

    avg_time = sum(times) / len(times)
    print("Inference time in {} for sweaty gru type 2 is {}".format(device, avg_time))



def main():
    parser = argparse.ArgumentParser()

    parser.add_argument('--loadSweaty', required=True, help='path to pretrained Sweaty model')
    parser.add_argument('--loadGru', default='', help='path to pretrained Gru cell')
    parser.add_argument('--downsample', type=int, default=4,  help='downsample')
    parser.add_argument('--alpha', type=int, default=1000, help='multiplication factor for the teacher signals')
    parser.add_argument('--seq_len', type=int, default=10, help='length of the sequence')
    parser.add_argument('--gruType', type=int, default=1, help='length of the sequence')
    parser.add_argument('--cpu', type=int, default=0, help='run inference time in cpu')

    opt = parser.parse_args()

    gruType = opt.gruType
    loadGru = opt.loadGru

    if opt.cpu != 0:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")

    print("Running inference time on ", device)

    trainset = utils.SoccerBallDataset("data/train/data.csv", "data/train", downsample=4, alpha=opt.alpha)

    if loadGru == '':
        sweaty = SweatyNet1()
        sweaty.to(device)
        sweaty.load_state_dict(torch.load(opt.loadSweaty))
        inference_sweaty(sweaty, trainset, device)

    elif loadGru != '' and gruType == 0:
        print("Loading sweaty and gru for type 1")
        sweaty = SweatyNet1()
        sweaty.to(device)
        sweaty.load_state_dict(torch.load(opt.loadSweaty))

        conv_gru = gru.ConvGruCell(1,1, device=device)
        conv_gru.to(device)
        conv_gru.load_state_dict(torch.load(loadGru))
        inference_gru(sweaty, conv_gru, trainset, device)

    else:
        print("Loading sweaty and gru for type 2")
        sweaty = s2.SweatyNet1()
        sweaty.to(device)
        sweaty.load_state_dict(torch.load(opt.loadSweaty))

        conv_gru = gru.ConvGruCell(89, 1, device=device)
        conv_gru.to(device)
        conv_gru.load_state_dict(torch.load(loadGru))
        inference_type2_gru(sweaty, conv_gru, trainset, device)


if __name__=='__main__':
    main()





