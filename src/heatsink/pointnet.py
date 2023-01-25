"""
Reference
----------
author:   fxia22
source:   https://github.com/fxia22/pointnet.pytorch
reminder: slightly modified, e.g., file path, better output format, etc.
"""
from timeit import default_timer
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from util.utilities import *


class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1,self.k*self.k).repeat(batchsize,1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNetfeat(nn.Module):
    def __init__(self, global_feat = True, feature_transform = False, k=3):
        super(PointNetfeat, self).__init__()
        self.stn = STNkd(k=k)
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = STNkd(k=64)

    def forward(self, x):
        n_pts = x.size()[2]
        trans = self.stn(x)
        x = x.transpose(2, 1)
        x = torch.bmm(x, trans)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2,1)
            x = torch.bmm(x, trans_feat)
            x = x.transpose(2,1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, n_pts)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


class PointNetDenseCls(nn.Module):
    def __init__(self, in_k=4, out_k=1, feature_transform=False):
        super(PointNetDenseCls, self).__init__()
        self.in_k = in_k
        self.out_k = out_k
        self.feature_transform=feature_transform
        self.feat = PointNetfeat(
            k=in_k, global_feat=False, feature_transform=feature_transform)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, self.out_k, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(128)

    def forward(self, x):
        batchsize = x.size()[0]
        n_pts = x.size()[2]
        # x, trans, trans_feat = self.feat(x)
        x, _, _ = self.feat(x)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = x.transpose(2, 1).contiguous()
        # x = F.log_softmax(x.view(-1,self.k), dim=-1)
        x = x.view(batchsize, n_pts, self.out_k)
        return x


################################################################
# configs
################################################################
PATH_XYZ = 'data/heatsink/Heatsink_Output_XYZ.npy'
PATH_U = 'data/heatsink/Heatsink_Output_Function.npy'

ntrain = 900
ntest = 100
ntotal = ntrain + ntest
n_points = 19517

batch_size = 2
learning_rate = 0.001
epochs = 501
patience = epochs // 20

input_dim = 3
output_dim = 1

################################################################
# training and evaluation
################################################################
def main(train_a, train_u, test_a, test_u):
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(train_a, train_u), 
        batch_size=batch_size, shuffle=True,
        generator=torch.Generator(device=device)
    )
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(test_a, test_u), 
        batch_size=batch_size,
        shuffle=False,
        generator=torch.Generator(device=device)
    )

    model = PointNetDenseCls(in_k=input_dim+3, 
        out_k=output_dim).cuda()
    print(count_params(model))

    optimizer = Adam(model.parameters(), 
        lr=learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, patience=patience)

    myloss = LpLoss(size_average=False)
    t0 = default_timer()
    y_normalizer.cuda()
    for ep in range(epochs):
        model.train()
        t1 = default_timer()
        train_l2 = 0
        for x, y in train_loader:
            x, y = x.cuda(), y.cuda()
            x = torch.concat((input_xyz, x), dim=-1)
            x = x.permute(0, 2, 1)

            optimizer.zero_grad()
            out = model(x)
            out = y_normalizer.decode(out)

            loss = myloss(out, y)
            loss.backward()

            optimizer.step()
            train_l2 += loss.item()

        scheduler.step(train_l2)

        model.eval()
        test_l2 = 0.0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.cuda(), y.cuda()
                x = torch.concat((input_xyz, x), dim=-1)
                x = x.permute(0, 2, 1)
                out = model(x)
                out = y_normalizer.decode(out)
                test_l2 += myloss(out, y).item()

        train_l2 /= ntrain
        test_l2 /= ntest

        t2 = default_timer()
        print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
                .format(ep, t2-t1, train_l2, test_l2))

    # Return final results
    return train_l2, test_l2, t2-t0, test_l2


if __name__ == "__main__":
    ################################################################
    # load data and preprocessing
    ################################################################
    input_xyz = np.load(PATH_XYZ)           # shape: (19517, 3)
    input_point_cloud = np.load(PATH_U)     # shape: (1000, 19517, 5)
    input_point_cloud = input_point_cloud[:ntotal]

    input_xyz = torch.from_numpy(input_xyz).cuda().float()
    input_xyz = input_xyz.unsqueeze(0).repeat([batch_size, 1, 1])

    input_point_cloud = torch.from_numpy(input_point_cloud).float()

    train_a = input_point_cloud[:ntrain, :, 1:4]
    test_a = input_point_cloud[-ntest:, :, 1:4]

    train_u = input_point_cloud[:ntrain, :, 0:1]
    test_u = input_point_cloud[-ntest:, :, 0:1]

    a_normalizer = UnitGaussianNormalizer(train_a)
    train_a = a_normalizer.encode(train_a)
    test_a = a_normalizer.encode(test_a)

    y_normalizer = UnitGaussianNormalizer(train_u)

    ################################################################
    # re-experiment with different random seeds
    ################################################################
    train_l2_res = []
    test_l2_res = []
    time_res = []
    test_T_l2_res = []
    for i in range(5):
        print("=== Round %d ==="%(i+1))
        set_random_seed(SEED_LIST[i])
        train_l2, test_l2, time, test_T_l2 = \
            main(train_a, train_u, test_a, test_u)
        train_l2_res.append(train_l2)
        test_l2_res.append(test_l2)
        time_res.append(time)
        test_T_l2_res.append(test_T_l2)
    print("=== Finish ===")
    for i in range(5):
        print('''[Round {}] Time: {:.1f}s Train_L2: {:>4e} Test_L2: {:>4e}
            \tT_L2: {:>4e}'''
            .format(i+1, time_res[i], train_l2_res[i], test_l2_res[i], 
            test_T_l2_res[i]))
