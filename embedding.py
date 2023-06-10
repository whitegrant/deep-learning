import torch
import pickle
import numpy as np
from autoencoder import AutoEncoder


# input: (otype, ex, k, price, vol, oi, iv, delta, gamma, theta, vega)


# parameters
ret_idx = 2
# pool_func = lambda x: torch.sum(x, 0)
pool_func = lambda x: torch.max(x, 0).values

# initialize
train_data = []
train_targets = []
test_data = []
test_targets = []
path = 'data/tensors/'

# load data
all_dates = np.load('data/dates.npy', allow_pickle=True)
train_dates = all_dates[-273:-40]   # 3/3/2020 - 2/2/2021
test_dates = all_dates[-40:-21]     # 2/3/2021 - 3/2/2021
with open('data/returns.pickle', 'rb') as file:
    returns = pickle.load(file)

# GPU
device = torch.device('mps')

# load encoder
encoder = torch.load('encoder.pth').to(device)

# training set
for date in train_dates:
    file = str(date)

    # data
    temp_data = torch.load(path + file + '.pt').float().to(device)
    enc_data = encoder.encode(temp_data).cpu()
    pool_data = pool_func(enc_data)
    train_data.append(pool_data)

    # targets
    train_targets.append(np.float32(returns[file][ret_idx]))
train_data = torch.stack(train_data)
train_targets = torch.tensor(train_targets)
train_targets = torch.where(train_targets > 0., 1., 0.)

# testing set
for date in test_dates:
    file = str(date)

    # data
    temp_data = torch.load(path + file + '.pt').float().to(device)
    enc_data = encoder.encode(temp_data).cpu()
    pool_data = pool_func(enc_data)
    test_data.append(pool_data)

    # targets
    test_targets.append(np.float32(returns[file][ret_idx]))
test_data = torch.stack(test_data)
test_targets = torch.tensor(test_targets)
test_targets = torch.where(test_targets > 0., 1., 0.)

# save data
torch.save(train_data, 'data/embedded/train_data_max.pt')
torch.save(train_targets, 'data/embedded/train_targets_m.pt')
torch.save(test_data, 'data/embedded/test_data_max.pt')
torch.save(test_targets, 'data/embedded/test_targets_m.pt')
