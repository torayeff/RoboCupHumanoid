import torch
import utils as utils
from SweatyNet1 import SweatyNet1
from conv_gru import ConvGruCell
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--loadSweaty', required=True, help='path to pretrained Sweaty model')
parser.add_argument('--loadGru', required=True, help='path to pretrained Gru cell')
parser.add_argument('--testSet', required=True, help='dataroot of the test set')
parser.add_argument('--trainSet', required=True, help='dataroot of the train set')
parser.add_argument('--batch_size', type=int, default=4,  help='batch size')
parser.add_argument('--downsample', type=int, default=4,  help='downsample')


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

convGru = ConvGruCell(1, 1, device=device)
convGru.load_state_dict(torch.load(gru_cell))

testset = utils.SoccerBallDataset(testset + "data.csv", testset, downsample=downsample)
trainset = utils.SoccerBallDataset(trainset + "data.csv", trainset, downsample=downsample)

sweaty.eval()
convGru.eval()

threshold = utils.get_abs_threshold(trainset)
metrics = utils.evaluate_sweaty_gru_model(sweaty, convGru, device, testset, threshold, verbose=True)


rc = metrics['tps']/(metrics['tps'] + metrics['fns'])
fdr = metrics['fps']/(metrics['fps'] + metrics['tps'])

print("RC: {}, FDR: {}".format(rc, fdr))
