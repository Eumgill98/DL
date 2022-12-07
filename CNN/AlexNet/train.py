from model import AlexNet
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


import gc
gc.collect()
torch.cuda.empty_cache()


from tqdm import tqdm

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#dataset setting 
meanR = 0.4467106
meanG = 0.43980986
meanB = 0.40664646

stdR = 0.22414584
stdG = 0.22148906
stdB = 0.22389975

if __name__ == '__main__':
    batch_size = 1
    train_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(227),
                transforms.RandomHorizontalFlip(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
    ])

    test_transformer = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([meanR, meanG, meanB], [stdR, stdG, stdB]),
                transforms.Resize(227)
    ])

    train_data = datasets.STL10(root='./data', download = True, split='train', transform=train_transform)
    test_data = datasets.STL10(root = './data', download = True, split='test', transform=test_transformer)

    train_loader = DataLoader(train_data, batch_size = batch_size)
    test_loader = DataLoader(test_data, batch_size = batch_size)


    model = AlexNet()

    if torch.cuda.is_available():
        model.cuda()

    sgd = optim.SGD(model.parameters(), lr=1e-1)
    loss_f = nn.CrossEntropyLoss()
    epoch = 100


    for i in tqdm(range(epoch)):
        #train
        model.train()
        for idx, (train_x, train_y) in enumerate(train_loader):
            train_x = train_x.to(DEVICE)
            train_y = train_y.to(DEVICE)

            sgd.zero_grad()
            y_hat = model(train_x)
            loss = loss_f(y_hat, train_y)
            loss.backward()
            sgd.step()

        #test
        model.eval()
        running_loss = 0
        for idx, (test_x, test_y) in enumerate(test_loader):
            test_x = test_x.to(DEVICE)
            test_y = test_y.to(DEVICE)

            y_hat = model(test_x)
            loss = loss_f(y_hat, test_y)
            running_loss += loss.item() * test_x.size(0)
            
        epoch_loss = running_loss / len(test_loader.dataset)

        print(f"{i+1}_epoch_loss : ", epoch_loss)
        torch.save(model, './save/STL10_{:.2f}.pkl'.format(epoch_loss))

