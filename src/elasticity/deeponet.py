"""
Reference
----------
author:   Zongyi Li and Daniel Zhengyu Huang
source:   https://raw.githubusercontent.com/zongyi-li/Geo-FNO
reminder: slightly modified, e.g., file path, better output format, etc.
"""

import torch
import numpy as np
from util.util_deeponet import *
from util.utilities import *
from torch.optim import Adam
from timeit import default_timer


################################################################
# configs
################################################################
PATH = 'data/elasticity/'
input_data  = np.load(PATH+"Random_UnitCell_XY_10.npy")
output_data = np.load(PATH+"Random_UnitCell_sigma_10.npy")
N_p, N_dim, _ = input_data.shape

ntrain = 1000
ntest = 200 
ndata = ntrain + ntest

learning_rate = 0.001
epochs = 1000
step_size = 100
gamma = 0.5

N_neurons = 256
layers = 5

################################################################
# training and evaluation
################################################################
def main(x_train, y_train, x_test, y_test):
    ################################################################
    # training and evaluation
    ################################################################

    batch_size = 16
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train, y_train), 
        batch_size=batch_size, shuffle=True, 
        generator=torch.Generator(device=device))
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_test, y_test), 
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device))
    
    model = DeepONet(N_p*N_dim, 2, layers,  layers+1, N_neurons) 
    print("# Model parameters = ", count_params(model))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    myloss = torch.nn.MSELoss(reduction='sum')
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            loss = myloss(out , y)
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()
        
        test_l2 = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                loss = myloss(out , y)
                test_l2 += loss.item()

        # torch.save(model, "DeepONet.model")
        scheduler.step()

        train_l2/= ntrain
        test_l2/= ntest

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, train_l2, test_l2))
    
    # Return final results
    return train_l2, test_l2


if __name__ == "__main__":
    ################################################################
    # load data and data normalization
    ################################################################
    # input includes (x1,y1, x2, y2 ... xp, yp) for branch and (x_i,y_i) for trunk
    x_data = np.zeros((ndata * N_p, N_p*N_dim + N_dim), dtype = np.float32)
    # output includes sigma_i
    y_data = np.zeros(ndata * N_p, dtype = np.float32)
    for i in range(ndata):
        for j in range(N_p):
            x_data[j + i*N_p, 0:N_dim*N_p]                = input_data[:,:,i].reshape((-1))
            x_data[j + i*N_p, N_dim*N_p:N_dim*N_p+N_dim]  = input_data[j,:,i]
            y_data[j + i*N_p]                             = output_data[j, i]

    x_train  = torch.from_numpy(x_data[0:ntrain*N_p, :])
    y_train  = torch.from_numpy(y_data[0:ntrain*N_p]).unsqueeze(-1)
    x_test   = torch.from_numpy(x_data[ntrain*N_p:(ntrain + ntest)*N_p, :])
    y_test   = torch.from_numpy(y_data[ntrain*N_p:(ntrain + ntest)*N_p]).unsqueeze(-1)

    x_normalizer = UnitGaussianNormalizer(x_train)
    y_normalizer = UnitGaussianNormalizer(y_train)
    x_train      = x_normalizer.encode(x_train)
    y_train      = y_normalizer.encode(y_train)
    x_test       = x_normalizer.encode(x_test)
    y_test       = y_normalizer.encode(y_test)


    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = [] 
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2 = main(x_train, y_train, x_test, y_test)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
    print("=== Finish ===")
    for i in range(5):
        print("[Round {}] Train_L2: {:>4e} Test_L2: {:>4e}"
                .format(i+1, train_l2_res[i], test_l2_res[i]))
