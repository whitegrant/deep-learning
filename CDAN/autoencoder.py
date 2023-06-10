import os
import torch
import pickle
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

plt.style.use('ggplot')


# input: (otype, ex, k, price, vol, oi, iv, delta, gamma, theta, vega)

class AutoEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
         
        # 11 -> 11 -> 6 -> 3
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(11, 11),
            torch.nn.ReLU(),
            torch.nn.Linear(11, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 3)
        )
         
        # 3 -> 6 -> 11 -> 11
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(3, 6),
            torch.nn.ReLU(),
            torch.nn.Linear(6, 11),
            torch.nn.ReLU(),
            torch.nn.Linear(11, 11)
            # may want to end with a sigmoid in future to scale between 0 and 1
        )
 
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


if __name__ == "__main__":
    # parameters
    num_epochs = 25

    # GPU
    device = torch.device('mps')

    # load data
    train_data = []
    test_data = []
    path = 'data/tensors/'
    for file in os.listdir(path):
        if file[-3:] == '.pt':
            data_date = date(int(file[:4]), int(file[5:7]), int(file[8:10]))

            # training set
            if date(2020, 3, 3) <= data_date <= date(2021, 2, 2):
                train_data.append(torch.load(path + file).float())

            # testing set
            if date(2021, 2, 3) <= data_date <= date(2021, 3, 2):
                test_data.append(torch.load(path + file).float())
    train_data = torch.cat(train_data)
    test_data = torch.cat(test_data)

    # Dataset class
    class MyDataset(Dataset):
        def __init__(self, data, targets):
            self.data = data
            self.targets = targets
            
        def __getitem__(self, index):
            x = self.data[index]
            y = self.targets[index]
            
            return x, y
        
        def __len__(self):
            return len(self.data)

    # data loaders
    train_dataset = MyDataset(train_data, train_data)
    train_loader = DataLoader(train_dataset, batch_size=4096, shuffle=False, pin_memory=True)
    test_dataset = MyDataset(test_data, test_data)
    test_loader = DataLoader(test_dataset, batch_size=4096, shuffle=False, pin_memory=True)

    # initialize
    train_loss = []
    test_loss = []

    # initialize model, loss function, and optimizer
    model = AutoEncoder()
    model.to(device)
    objective = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    # train and test
    for epoch in range(num_epochs):
        print('epoch', epoch + 1)
        # initialize
        temp_train_loss = []
        temp_test_loss = []

        # train
        for batch, (x, y_truth) in enumerate(train_loader):
            # get prediction
            x, y_truth = x.to(device), y_truth.to(device)
            y_hat = model(x)
        
            # zero gradients
            optimizer.zero_grad()
                    
            # backward
            loss = objective(y_hat, y_truth)
            loss.backward()
            
            # save error
            temp_train_loss.append(loss.item())

            # step
            optimizer.step()

        # save error
        train_loss.append(np.mean(temp_train_loss))

        # test
        for batch, (x, y_truth) in enumerate(test_loader):
            # get prediction
            x, y_truth = x.to(device), y_truth.to(device)
            y_hat = model(x)
            
            # save error
            loss = objective(y_hat, y_truth)
            temp_test_loss.append(loss.item())

            # step
            optimizer.step()

        # save error
        test_loss.append(np.mean(temp_test_loss))

    # plot
    domain = np.arange(1, num_epochs + 1)
    plt.plot(domain, train_loss, label='training loss')
    plt.plot(domain, test_loss, label='testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Autoencoder Loss')
    plt.legend()
    plt.show()

    # save model with weights
    torch.save(model, 'model.pth')
