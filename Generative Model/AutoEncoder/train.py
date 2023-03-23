import torch
import torchvision
import torch.nn.functional as F
from torch import nn, optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from model import AutoEncoder

#Hyper parameter
CFG = {
    'MODEL' : 'AutoEncdoer',
    'EPOCH' : 10,
    'BATCH_SIZE' : 64,
    'lr' : 0.0005
}

#read dataset
trainset = datasets.FashionMNIST(
    root= './data/',
    train = True,
    download= True,
    transform= transforms.ToTensor()
)

train_loader = DataLoader(
    dataset = trainset,
    batch_szie = CFG['BATCH_SIZE'],
    shuffle = True
)


#CPU check
DEVICE = torch.device("cuda" if USE_CUDA else "cpu")


#Model Define
autoencoder = AutoEncoder().to(DEVICE)
optimizer = optim.Adam(autoencoder.parameters(), lr = CFG['lr'])
criterion = nn.MSELoss()


#train
def train(model, data, criterion, optimizer):
    model.train()
    for step, (x, label) in enumerate(data):
        x = x.view(-1, 28*28).to(DEVICE)
        y = x.view(-1, 28*28).to(DEVICE)
        label = label.to(DEVICE)

        encoded, decoded = model(x)
        loss = criterion(decoded, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

for epoch in range(CFG['EPOCH'] + 1):
    train(autoencoder, train_loader, criterion=criterion, optimizer=optimizer)


        
