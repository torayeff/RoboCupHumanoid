from conv_gru import ConvGruCell
import torch
from SweatyNet1 import SweatyNet1
import utils
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import peak
import matplotlib.cm as cm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
modelPath = ''

model = SweatyNet1()
model.to(device)

if modelPath != '':
    print("Loading Sweaty")
    model.load_state_dict(torch.load(modelPath, map_location='cpu'))
print(model)

conv_gru = ConvGruCell(1, 1, device)

trainset = utils.SoccerBallDataset("data/train/data.csv", "data/train", downsample=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=2, shuffle=True, num_workers=2)

idx = 51
image = trainset[idx]['image']
signal = np.array(trainset[idx]['signal'].squeeze())

output = model(image.unsqueeze(0).float().to(device), None, None)
print(output.size())
output = conv_gru(output)

output_signal = np.array(output.cpu().squeeze().detach())

plt.title("RGB Image")
plt.imshow(np.array(image).transpose(1, 2, 0))
plt.show()

plt.title("Teacher Signal")
plt.imshow(signal, cmap=cm.jet)
plt.show()


plt.title("Output Signal")
plt.imshow(output_signal, cmap=cm.jet)
plt.show()