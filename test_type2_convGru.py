import torch
import utils as utils
from sweaty_net_2_outputs import SweatyNet1
from conv_gru import ConvGruCellPreConv
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--loadSweaty', required=True, help='path to pretrained Sweaty model')
parser.add_argument('--loadGru', required=True, help='path to pretrained Gru cell')
parser.add_argument('--testSet', required=True, help='dataroot of the test set')
parser.add_argument('--trainSet', required=True, help='dataroot of the train set')
parser.add_argument('--batch_size', type=int, default=4,  help='batch size')
parser.add_argument('--downsample', type=int, default=4,  help='downsample')
parser.add_argument('--p', type=float, default=0.7, help='percentage of abs threshold')
parser.add_argument('--alpha', type=int, default=1000, help='multiplication factor for the teacher signals')
parser.add_argument('--seq_len', type=int, default=10, help='length of the sequence')


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

opt = parser.parse_args()

downsample = opt.downsample
batch_size = opt.batch_size
testset = opt.testSet
trainset = opt.trainSet
pretrained_model = opt.loadSweaty
gru_cell = opt.loadGru


sweaty = SweatyNet1()
sweaty.load_state_dict(torch.load(pretrained_model))
sweaty.eval()

convGru = ConvGruCellPreConv(89, 1, device=device)
convGru.load_state_dict(torch.load(gru_cell))

testset = utils.SoccerBallDataset(testset + "data.csv", testset, downsample=downsample, alpha= opt.alpha)
trainset = utils.SoccerBallDataset(trainset + "data.csv", trainset, downsample=downsample, alpha= opt.alpha)

sweaty.eval()
convGru.eval()


threshold = utils.get_abs_threshold(trainset, opt.p)
metrics = utils.evaluate_type2_sweaty_gru_model(sweaty, convGru, device, testset, threshold, verbose=True, seq_len=opt.seq_len)


rc = metrics['tps']/(metrics['tps'] + metrics['fns'])
fdr = metrics['fps']/(metrics['fps'] + metrics['tps'])

print("RC: {}, FDR: {}".format(rc, fdr))
