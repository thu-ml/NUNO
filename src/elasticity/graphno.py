"""
Reference
----------
author:   Zongyi Li and Daniel Zhengyu Huang
source:   https://raw.githubusercontent.com/zongyi-li/Geo-FNO
reminder: slightly modified, e.g., file path, better output format, etc.
"""

import sklearn.metrics
import torch.nn.functional as F
from torch_geometric.data import Data, DataLoader
from util.utilities import *
from util.util_graph import NNConv_old
from timeit import default_timer


class KernelNN3(torch.nn.Module):
    def __init__(self, width_node, width_kernel, depth, ker_in, in_width=1, out_width=1):
        super(KernelNN3, self).__init__()
        self.depth = depth

        self.fc1 = torch.nn.Linear(in_width, width_node)

        kernel1 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv1 = NNConv_old(width_node, width_node, kernel1, aggr='mean')
        kernel2 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv2 = NNConv_old(width_node, width_node, kernel2, aggr='mean')
        kernel3 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv3 = NNConv_old(width_node, width_node, kernel3, aggr='mean')
        kernel4 = DenseNet([ker_in, width_kernel // 2, width_kernel, width_node**2], torch.nn.ReLU)
        self.conv4 = NNConv_old(width_node, width_node, kernel4, aggr='mean')

        self.fc2 = torch.nn.Linear(width_node, 128)
        self.fc3 = torch.nn.Linear(128, 1)

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = self.fc1(x)

        x = self.conv1(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv2(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv3(x, edge_index, edge_attr)
        x = F.gelu(x)
        x = self.conv4(x, edge_index, edge_attr)

        x = self.fc2(x)
        x = F.gelu(x)
        x = self.fc3(x)
        return x


################################################################
# configs
################################################################
PATH = 'data/elasticity/'
INPUT_PATH = PATH+'Random_UnitCell_XY_10.npy'
OUTPUT_PATH = PATH+'Random_UnitCell_sigma_10.npy'

ntrain = 1000
ntest = 200
batch_size = 1
learning_rate = 0.001

epochs = 201
step_size = 50
gamma = 0.5

# GNO
radius = 0.2
width = 32
ker_width = 128
depth = 4

edge_features = 4
node_features = 2


################################################################
# training and evaluation
################################################################
def main(data_train, data_test):
    train_loader = DataLoader(data_train, batch_size=batch_size, 
        shuffle=True, generator=torch.Generator(device=device))
    test_loader = DataLoader(data_test, batch_size=batch_size, 
        shuffle=False, generator=torch.Generator(device=device))
    # test_loader2 = DataLoader(data_test, batch_size=1, shuffle=False)

    model = KernelNN3(width, ker_width,depth,edge_features,in_width=node_features).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    print(count_params(model))

    myloss = LpLoss(size_average=False)
    ttrain = np.zeros((epochs, ))
    ttest = np.zeros((epochs,))
    model.train()
    t0 = default_timer()
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0.0
        train_l2 = 0.0
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)

            l2 = myloss(out.view(batch_size, -1), batch.y.view(batch_size, -1))
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step()
        t2 = default_timer()

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                out = model(batch)
                test_l2 += myloss(out.view(batch_size, -1), batch.y.view(batch_size, -1)).item()

        ttrain[ep] = train_l2/ntrain
        ttest[ep] = test_l2/ntest

        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, train_l2/ntrain, test_l2/ntest))
    
    # Return final results
    return ttrain[-1], ttest[-1], t2-t0



if __name__ == "__main__":
    ################################################################
    # load data and data normalization
    ################################################################
    t1 = default_timer()
    input = np.load(INPUT_PATH)
    input = torch.tensor(input, dtype=torch.float).permute(2,0,1)
    # input (n, x, 2)

    output = np.load(OUTPUT_PATH)
    output = torch.tensor(output, dtype=torch.float).permute(1,0)
    # output (n, x)

    x_train = input[:ntrain]
    y_train = output[:ntrain]
    x_test = input[-ntest:]
    y_test = output[-ntest:]

    ################################################################
    # construct graphs
    ################################################################
    def get_graph_ball(mesh, radius=0.1):
        pwd = sklearn.metrics.pairwise_distances(mesh, mesh)  # (mesh_n, grid_n)
        edge_index = np.vstack(np.where(pwd <= radius))
        edge_attr = mesh[edge_index.T].reshape(-1, 4)
        return torch.tensor(edge_index, dtype=torch.long), edge_attr

    def get_graph_gaussian(mesh, sigma=0.1):
        pwd = sklearn.metrics.pairwise_distances(mesh.cpu(), mesh.cpu())  # (mesh_n, grid_n)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        edge_index = np.vstack(np.where(sample))
        edge_attr = mesh[edge_index.T].reshape(-1, 4)
        return torch.tensor(edge_index, dtype=torch.long), edge_attr

    data_train = []
    for j in range(ntrain):
        edge_index, edge_attr = get_graph_gaussian(x_train[j], radius)
        data_train.append(Data(x=x_train[j], y=y_train[j], edge_index=edge_index, edge_attr=edge_attr))

    data_test = []
    for j in range(ntest):
        edge_index, edge_attr = get_graph_gaussian(x_test[j], radius)
        data_test.append(Data(x=x_test[j], y=y_test[j], edge_index=edge_index, edge_attr=edge_attr))

    print(edge_index.shape, edge_attr.shape)

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time = main(data_train, data_test)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
    print("=== Finish ===")
    for i in range(5):
        print("[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}"
                .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i]))
