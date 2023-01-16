import torch
import numpy as np
from util.util_deeponet import *
from util.utilities import set_random_seed, SEED_LIST, device
from torch.optim import Adam
from timeit import default_timer


################################################################
# configs
################################################################
PATH = 'data/channel/'
input_xy  = np.load(PATH + 'Channel_Flow_XY.npy')
input_data = np.load(PATH + 'Channel_Flow_Velocity_Pressure.npy')
_, n_dim = input_xy.shape
ntotal, n_points, _, output_dim = input_data.shape

ntrain = 1000
ntest = 200 
ntotal = ntrain + ntest

T_in = 15   # input: [0, 0.15)
T = 30      # output: [0.15, 0.30)

batch_size = 2048
learning_rate = 0.0001
epochs = 101
patience = epochs // 20

N_neurons = 64
layers = 5

################################################################
# training and evaluation
################################################################
def main(branch_train, truck_train, y_train, branch_test, truck_test, y_test):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(branch_train, truck_train, y_train), 
        batch_size=batch_size, shuffle=True, 
        generator=torch.Generator(device=device))
    test_loader  = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(branch_test, truck_test, y_test), 
        batch_size=batch_size, shuffle=False,
        generator=torch.Generator(device=device))
    
    model = MultiDeepONet(n_points * T_in * output_dim, n_dim, layers, layers,
        N_neurons, output_dim=(T-T_in)*output_dim) 
    print("# Model parameters = ", count_params(model))
    model.to(device)

    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=patience)

    testloss = LpLoss(size_average=False)
    myloss = torch.nn.MSELoss(reduction='sum')
    y_normalizer.cuda()
    t0 = default_timer()
    for ep in range(epochs):
        t1 = default_timer()
        train_mse = 0
        model.train()
        for branch, truck, y in train_loader:
            branch = x_data[branch]
            x = torch.concat((branch, truck), dim=-1)
            # x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = myloss(out, y)
            loss.backward()
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            optimizer.step()
            train_mse += loss.item()
        
        test_mse = 0
        model.eval()
        with torch.no_grad():
            for branch, truck, y in test_loader:
                branch = x_data[branch]
                x = torch.concat((branch, truck), dim=-1)
                # x, y = x.to(device), y.to(device)

                out = model(x)
                loss = myloss(out, y)
                out = y_normalizer.decode(out)
                y = y_normalizer.decode(y)
                test_mse += loss.item()

        scheduler.step(train_mse)

        train_mse/=(ntrain*n_points)
        test_mse/=(ntest*n_points)

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s MSE: {:>4e} Test_MSE: {:>4e}"
                .format(ep, t2-t1, train_mse, test_mse))
    
    # Final test (in cpu, cause gpu run out of memory)
    test_u_l2 = 0
    test_v_l2 = 0
    test_p_l2 = 0
    model = model.cpu()
    y_normalizer.cpu()
    model.eval()
    with torch.no_grad():
        for i in range(ntest):
            branch = x_data[(ntrain+i):(ntrain+i+1), :].cpu().repeat(n_points, 1)
            truck = truck_test[i*n_points:(i+1)*n_points, :].cpu()
            x = torch.concat((branch, truck), dim=-1)
            y = y_test[i*n_points:(i+1)*n_points, :].cpu()

            out = model(x)
            out = y_normalizer.decode(out)
            y = y_normalizer.decode(y)

            y = y.reshape(1, n_points, T-T_in, output_dim)
            out = out.reshape_as(y)
            test_u_l2 += testloss(out[..., 0], y[..., 0]).item()
            test_v_l2 += testloss(out[..., 1], y[..., 1]).item()
            test_p_l2 += testloss(out[..., 2], y[..., 2]).item()

    test_u_l2/=ntest
    test_v_l2/=ntest
    test_p_l2/=ntest
    test_l2=(test_u_l2+test_v_l2+test_p_l2)/3

    # Return final results
    return np.nan, test_l2, t2-t0, test_u_l2, test_v_l2, test_p_l2


if __name__ == "__main__":
    ################################################################
    # load data and data normalization
    ################################################################
    # input includes (u | T \in [0, 0.15)) for branch and (x_i, y_i) for trunk
    branch_data = np.zeros((ntotal * n_points), dtype=np.int32)
    truck_data = np.zeros((ntotal * n_points, n_dim), dtype=np.float32)
    x_data = input_data[:, :, :T_in, :].reshape(ntotal, -1)
    # output includes (u | T \in [0.15, 0.30))
    y_data = np.zeros((ntotal * n_points, (T-T_in) * output_dim), dtype=np.float32)
    for i in range(ntotal):
        branch_data[i*n_points:(i+1)*n_points] = i
        truck_data[i*n_points:(i+1)*n_points, :] = input_xy
        y_data[i*n_points:(i+1)*n_points, :] = \
            input_data[i, :, T_in:T, :].reshape(n_points, -1)

    x_data = torch.from_numpy(x_data).cuda().float()
    x_normalizer = UnitGaussianNormalizer(x_data[:ntrain])
    x_data = x_normalizer.encode(x_data)
    
    branch_train = torch.from_numpy(branch_data[0:ntrain*n_points]).cuda().long()
    truck_train = torch.from_numpy(truck_data[0:ntrain*n_points]).cuda().float()
    y_train = torch.from_numpy(y_data[0:ntrain*n_points]).cuda().float()
    branch_test = torch.from_numpy(branch_data[ntrain*n_points:(ntrain+ntest)*n_points]).cuda().long()
    truck_test = torch.from_numpy(truck_data[ntrain*n_points:(ntrain+ntest)*n_points]).cuda().float()
    y_test = torch.from_numpy(y_data[ntrain*n_points:(ntrain+ntest)*n_points]).cuda().float()

    y_normalizer = UnitGaussianNormalizer(y_train)
    y_train      = y_normalizer.encode(y_train)
    y_test       = y_normalizer.encode(y_test)

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
            main(branch_train, truck_train, y_train, branch_test, truck_test, y_test)
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
