from model import LeNet5
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets.mnist import MNIST
from torchvision import transforms 
from torch.utils.data import DataLoader

from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


if __name__ == '__main__':
    batch_size = 256
    transform = transforms.Compose([transforms.Resize((32, 32)),
                                 transforms.ToTensor()])
    train_data = MNIST(root='./data/MNIST/train', download =True, train=True, transform=transform)
    test_data = MNIST(root='./data/MNIST/test',download =True, train=False, transform=transform)

    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    model = LeNet5()

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
        torch.save(model, './save/mnist_{:.2f}.pkl'.format(epoch_loss))