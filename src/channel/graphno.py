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
        self.fc3 = torch.nn.Linear(128, out_width)

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
PATH = 'data/channel/'
INPUT_PATH = PATH + 'Random_UnitCell_XY_10.npy'
OUTPUT_PATH = PATH + 'Random_UnitCell_sigma_10.npy'

ntrain = 1000
ntest = 200
ntotal = ntrain + ntest
batch_size = 1
learning_rate = 0.001

n_points = 3809
T_in = 15   # input: [0, 0.15)
T = 30      # output: [0.15, 0.30)
output_dim = 3

epochs = 201
iterations = epochs*(ntrain//batch_size)

# GNO
radius = 0.05
width = 32
ker_width = 128
depth = 4

edge_features = 4
node_features = 45


################################################################
# training and evaluation
################################################################
def main(data_train, data_test):
    train_loader = DataLoader(data_train, batch_size=batch_size, 
        shuffle=True)
    test_loader = DataLoader(data_test, batch_size=batch_size, 
        shuffle=False)

    model = KernelNN3(width, ker_width, depth, edge_features, 
        in_width=node_features, out_width=node_features).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=iterations)
    print(count_params(model))

    myloss = MultiLpLoss(size_average=False)
    ttrain = np.zeros((epochs,))
    ttest = np.zeros((epochs,))
    t0 = default_timer()
    for ep in range(epochs):
        t1 = default_timer()
        train_l2 = 0.0
        model.train()
        for batch in train_loader:
            batch = batch.to(device)

            optimizer.zero_grad()
            out = model(batch)

            out = out.reshape(batch_size, n_points, T-T_in, output_dim)
            l2 = myloss(out, batch.y.reshape_as(out))
            l2.backward()

            optimizer.step()
            train_l2 += l2.item()

        scheduler.step()
        t2 = default_timer()

        model.eval()
        test_u_l2 = 0.0
        test_v_l2 = 0.0
        test_p_l2 = 0.0
        with torch.no_grad():
            for batch in test_loader:
                batch = batch.to(device)

                out = model(batch)
                y = batch.y.reshape(batch_size, n_points, T-T_in, output_dim)
                out = out.reshape_as(y)
                test_u_l2 += myloss(out[..., 0], y[..., 0], multi_channel=False).item()
                test_v_l2 += myloss(out[..., 1], y[..., 1], multi_channel=False).item()
                test_p_l2 += myloss(out[..., 2], y[..., 2], multi_channel=False).item()

        ttrain[ep] = train_l2/ntrain
        test_u_l2/=ntest
        test_v_l2/=ntest
        test_p_l2/=ntest
        ttest[ep] = (test_u_l2 + test_v_l2 + test_p_l2)/3

        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, ttrain[ep], ttest[ep]))
    
    # Return final results
    return ttrain[-1], ttest[-1], t2-t0, test_u_l2, test_v_l2, test_p_l2



if __name__ == "__main__":
    ################################################################
    # load data and data normalization
    ################################################################
    torch.set_default_tensor_type('torch.FloatTensor')
    t1 = default_timer()
    input_xy = np.load(PATH + 'Channel_Flow_XY.npy')
    input_xy = torch.from_numpy(input_xy).float()
    # shape (n_points, 2)
    input_data = np.load(PATH + 'Channel_Flow_Velocity_Pressure.npy')
    input_data = torch.from_numpy(input_data).float()
    # shape (ntotal, n_points, T+1, 3)

    x_train = input_data[:ntrain, :, :T_in, :].reshape(ntrain, n_points, -1)
    y_train = input_data[:ntrain, :, T_in:T, :].reshape(ntrain, n_points, -1)
    x_test = input_data[-ntest:, :, :T_in, :].reshape(ntest, n_points, -1)
    y_test = input_data[-ntest:, :, T_in:T, :].reshape(ntest, n_points, -1)

    ################################################################
    # construct graphs
    ################################################################
    def get_graph_ball(mesh, radius=0.1):
        pwd = sklearn.metrics.pairwise_distances(mesh, mesh)  # (mesh_n, grid_n)
        edge_index = np.vstack(np.where(pwd <= radius))
        edge_attr = mesh[edge_index.T].reshape(-1, 4)
        return torch.tensor(edge_index, dtype=torch.long), edge_attr

    def get_graph_gaussian(mesh, sigma=0.1):
        pwd = sklearn.metrics.pairwise_distances(mesh, mesh)  # (mesh_n, grid_n)
        rbf = np.exp(-pwd ** 2 / sigma ** 2)
        sample = np.random.binomial(1, rbf)
        edge_index = np.vstack(np.where(sample))
        edge_attr = mesh[edge_index.T].reshape(-1, 4)
        return torch.tensor(edge_index, dtype=torch.long), edge_attr

    edge_index, edge_attr = get_graph_gaussian(input_xy, radius)
    print(edge_index.shape, edge_attr.shape)

    data_train = []
    for j in range(ntrain):
        data_train.append(Data(x=x_train[j], y=y_train[j], edge_index=edge_index, edge_attr=edge_attr))

    data_test = []
    for j in range(ntest):
        data_test.append(Data(x=x_test[j], y=y_test[j], edge_index=edge_index, edge_attr=edge_attr))

    t2 = default_timer()
    print('preprocessing finished, time used:', t2-t1)

    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    test_u_l2_res = []
    test_v_l2_res = []
    test_p_l2_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time, test_u_l2, test_v_l2, test_p_l2 = \
            main(data_train, data_test)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
        test_u_l2_res.append(test_u_l2)
        test_v_l2_res.append(test_v_l2)
        test_p_l2_res.append(test_p_l2)
    print("=== Finish ===")
    for i in range(5):
        print('''[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}
            \tu_L2: {:>4e} v_L2: {:>4e} p_L2: {:>4e}'''
            .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i], 
            test_u_l2_res[i], test_v_l2_res[i], test_p_l2_res[i]))
