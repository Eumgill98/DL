import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2

from tqdm import tqdm
import random

"""
For multi label classifications task 

"""

#Fix Seed
def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

#Custom Dataset
class CustomDataset(Dataset):
    def __init__(self, path, label_list=None, transforms=None):
        self.path = path
        self.label_list = label_list
        self.transforms = transforms

    def __getitem__(self, index):
        data_path = self.path[index]
        image = cv2.imread(data_path)

        #If transforms parameter is on
        if self.transforms is not None:
            image = self.transforms(image = image)['image']
        
        if self.label is not None:
            label = torch.FloatTensor(self.label_list[index])

            return image, label
        else:
            return image
    
    def __len__(self):
        return len(self.path)

    def get_dataloader(self, batch_size, shuffle = False):
        return DataLoader(self, batch_size = batch_size, shuffle=shuffle)


#model validation
def validation(model, criterion, val_loader, device):
    model.eval()
    val_loss = []
    val_acc = []

    with torch.no_grad():
        for imgs, labels in tqdm(iter(val_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            y_hat = model(imgs)

            loss = criterion(y_hat, labels)

            y_hat = y_hat.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            preds  = y_hat > 0.5
            batch_acc = (labels == preds).mean()

            val_acc.append(batch_acc)
            val_loss.append(loss.item())

        _val_loss = np.mean(val_loss)
        _val_acc = np.mean(val_acc)
        
    return _val_loss, _val_acc

#model train
def train(model, optimizer, train_loader, val_loader, scheduler, device, epochs, name):
    model.to(device)
    criterion = nn.BCELoss().to(device)

    best_val_acc = 0
    best_model = None

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = []
        for imgs, labels in tqdm(iter(train_loader)):
            imgs = imgs.float().to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(imgs)
            loss = criterion(output, labels)
            
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        _val_loss, _val_acc = validation(model, criterion, val_loader, device)
        _train_loss = np.mean(train_loss)
        print(f'Epoch [{epoch}], Train Loss : [{_train_loss:.5f}] Val Loss : [{_val_loss:.5f}] Val ACC : [{_val_acc:.5f}]')


        if scheduler is not None:
            scheduler.step(_val_acc)

        if best_val_acc < _val_acc:
            best_val_acc = _val_acc
            best_model = model
        

    return best_model


#model predict
def predict(model, test_loader, device):
    model.to(device)
    model.eval()
    predictions = []

    with torch.no_grad():
        for imgs in tqdm(iter(test_loader)):
            imgs = imgs.float().to(device)

            y_hat = model(imgs)

            y_hat = y_hat.cpu().detach().numpy()
            y_hat = y_hat > 0.5
            y_hat = y_hat.astype(int)
            predictions += y_hat.tolist()

    return predictions