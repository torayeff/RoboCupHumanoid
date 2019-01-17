import torch
import torch.nn as nn
import utils as utils
from SweatyNet1 import SweatyNet1

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

model = SweatyNet1()
model.to(device)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters())

trainset = utils.SoccerBallDataset("data/train/data.csv", "data/train", downsample=4)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

epochs = 1

for epoch in range(epochs):
    for i, data in enumerate(trainloader):
        optimizer.zero_grad()

        images = data['image'].float()
        signals = data['signal'].float()

        outputs = model(images)

        loss = criterion(signals, outputs)

        loss.backward()
        optimizer.step()

        print("Loss: {}".format(loss.item()))

