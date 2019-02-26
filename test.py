import torch
import utils as utils
from SweatyNet1 import SweatyNet1
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--load', required=True, help='path to pretrained Sweaty model')
parser.add_argument('--convLstm', type=int, default=0, help='flag for conv-gru layer. By default it does not use it')
parser.add_argument('--testSet', required=True, help='dataroot of the test set')
parser.add_argument('--trainSet', required=True, help='dataroot of the train set')
parser.add_argument('--batch_size', type=int, default=4,  help='batch size')
parser.add_argument('--downsample', type=int, default=4,  help='downsample')
parser.add_argument('--alpha', type=int, default=1000, help='batch size')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

opt = parser.parse_args()

downsample = opt.downsample
batch_size = opt.batch_size
testset = opt.testSet
trainset = opt.trainSet
pretrained_model = opt.load

model = SweatyNet1()
model.load_state_dict(torch.load(pretrained_model))
model.to(device)


testset = utils.SoccerBallDataset(testset + "data.csv", testset, downsample=downsample, alpha= opt.alpha)
trainset = utils.SoccerBallDataset(trainset + "data.csv", trainset, downsample=downsample, alpha=opt.alpha)

model.eval()
threshold = utils.get_abs_threshold(trainset)
metrics = utils.evaluate_sweaty_model(model, device, testset, threshold, verbose=True, debug=False)

rc = metrics['tps']/(metrics['tps'] + metrics['fns'])
fdr = metrics['fps']/(metrics['fps'] + metrics['tps'])

print("RC: {}, FDR: {}".format(rc, fdr))
