import torch
from torch import nn 
from torch import optim
from model import make_model

from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

import os
from tqdm import tqdm
import numpy as np

# config
PATH = './data/'
BATCH_SIZE= 32
SEED = 42
EPOCH = 100
lr = 0.01
NUM_CLASS = 10
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
np.random.seed(SEED)
torch.manual_seed(SEED) 

#data path make
if not os.path.exists(PATH):
    os.makedirs(PATH)

#transformation
transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(224)
])
#load dataset
train_data = datasets.CIFAR10(PATH, train=True, download=True, transform=transformation)
val_data = datasets.CIFAR10(PATH, train=False, download=True, transform=transformation)
#check
print(len(train_data))
print(len(val_data))

#data loader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=True)


#train
model = make_model('Densenet121', num_class=NUM_CLASS)
model.to(DEVICE)

loss_fun = nn.CrossEntropyLoss(reduction='sum')
optimzer = optim.Adam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(optimzer, mode='min', factor=0.1, patience=8)

for i in tqdm(range(EPOCH)):
    model.train()
    train_loss = 0
    for idx, (train_x, train_y) in enumerate(train_loader):
        train_x = train_x.to(DEVICE)
        train_y = train_y.to(DEVICE)

        optimzer.zero_grad()
        y_ = model(train_x)
        loss = loss_fun(y_, train_y)
        train_loss += loss.item() * train_x.size(0)
        loss.backward()
        optimzer.step()
    
    train_loss_t = train_loss/ len(train_loader.dataset)

    #val
    model.eval()
    val_loss = 0
    for idx, (val_x, val_y) in enumerate(val_loader):
        val_x = val_x.to(DEVICE)
        val_y = val_y.to(DEVICE)

        y_ = model(val_x)
        loss = loss_fun(y_, val_y)
        val_loss += loss.item() * val_x.size(0)
    
    epoch_loss = val_loss / len(val_loader.dataset)
    lr_scheduler.step(epoch_loss)

    print(f"{i+1}_train_loss :{train_loss_t}, {i+1}_val_loss :{epoch_loss} ")



        

