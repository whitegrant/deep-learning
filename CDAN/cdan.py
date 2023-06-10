import torch
import pickle
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

plt.style.use('ggplot')


class CDAN(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.net = torch.nn.Sequential(
            torch.nn.Linear(3, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 8),
            torch.nn.ReLU(),
            torch.nn.Linear(8, 4),
            torch.nn.ReLU(),
            torch.nn.Linear(4, 1),
            torch.nn.Sigmoid()
        )
 
    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    # parameters
    num_epochs = 50

    # GPU
    device = torch.device('mps')

    # load data
    train_data = torch.load('data/embedded/train_data_mean.pt')
    train_targets = torch.load('data/embedded/train_targets_w.pt').unsqueeze(1)
    train_n = len(train_data)
    test_data = torch.load('data/embedded/test_data_mean.pt')
    test_targets = torch.load('data/embedded/test_targets_w.pt').unsqueeze(1)
    test_n = len(test_data)

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
    train_dataset = MyDataset(train_data, train_targets)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=False, pin_memory=True)
    test_dataset = MyDataset(test_data, test_targets)
    test_loader = DataLoader(test_dataset, batch_size=19, shuffle=False, pin_memory=True)

    # initialize
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    # initialize model, loss function, and optimizer
    model = CDAN()
    model.to(device)
    objective = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # train and test
    for epoch in range(num_epochs):
        print('epoch', epoch + 1)
        # initialize
        temp_train_loss = []
        temp_train_acc = 0
        temp_test_loss = []
        temp_test_acc = 0

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
            
            # save error and accuracy
            temp_train_loss.append(loss.item())
            guess = torch.where(y_hat > .5, 1., 0.)
            temp_train_acc += (guess == y_truth).sum().item()

            # step
            optimizer.step()

        # save error and accuracy
        train_loss.append(np.mean(temp_train_loss))
        train_acc.append(temp_train_acc / train_n)

        # test
        for batch, (x, y_truth) in enumerate(test_loader):
            # get prediction
            x, y_truth = x.to(device), y_truth.to(device)
            y_hat = model(x)
            
            # save error and accuracy
            temp_test_loss.append(loss.item())
            guess = torch.where(y_hat > .5, 1., 0.)
            temp_test_acc += (guess == y_truth).sum().item()

        # save error and accuracy
        test_loss.append(np.mean(temp_test_loss))
        test_acc.append(temp_test_acc / test_n)

    # plot loss
    domain = np.arange(1, num_epochs + 1)
    plt.plot(domain, train_loss, label='training loss')
    plt.plot(domain, test_loss, label='testing loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.title('Loss')
    plt.legend()
    plt.show()

    # plot accuracy
    plt.plot(domain, train_acc, label='training accuracy')
    plt.plot(domain, test_acc, label='testing accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()
