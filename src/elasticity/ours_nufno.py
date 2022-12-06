from timeit import default_timer
from util.utilities3 import *
from torch.optim import Adam
from ..nufno.nufno_2d import NUFNO2d
from .kdtree import KDTree


torch.manual_seed(0)
np.random.seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True


################################################################
# configs
################################################################
Ntotal = 2000
ntrain = 1000
ntest = 200

batch_size = 20
learning_rate = 0.001

epochs = 501
step_size = 50
gamma = 0.5

modes = 12
width = 32
# KD-Tree domain splitting
n_subdomains = 8


################################################################
# load data and data preprocessing (KD-Tree)
################################################################
PATH = 'data/elasticity/'
PATH_Sigma = PATH+'Random_UnitCell_sigma_10.npy'
PATH_XY = PATH+'Random_UnitCell_XY_10.npy'
PATH_rr = PATH+'Random_UnitCell_rr_10.npy'
# Wether to save preprocessing result
PATH_ind = PATH+'Preprocess_ind.npy'
PATH_sep = PATH+'Preprocess_sep.npy'
SAVE_preprocess = False
LOAD_preprocess = True

input_rr = np.load(PATH_rr)
input_s = np.load(PATH_Sigma)
input_xy = np.load(PATH_XY)

if LOAD_preprocess:
    input_ind = np.load(PATH_ind)
    input_sep = np.load(PATH_sep)
else:
    print("Start KD-Tree splitting...")
    point_clouds = np.transpose(input_xy, (2, 0, 1))
    point_clouds = np.concatenate(
        (point_clouds[:ntrain], point_clouds[-ntest:]), 
        axis=0
    )
    t1 = default_timer()
    input_ind = []
    input_sep = []
    for i in range(len(point_clouds)):
        point_cloud = point_clouds[i].tolist()
        tree= KDTree(
            point_cloud, 2, n_subdomains=n_subdomains, 
            n_blocks=8, return_indices=True
        )
        subdomain_indices = tree.get_subdomain_indices()
        input_ind.append(np.concatenate(subdomain_indices))
        subdomain_separator = [0] * (n_subdomains + 1)
        for j in range(n_subdomains):
            subdomain_separator[j+1] = subdomain_separator[j] + \
                subdomain_indices[j].shape[0]
        input_sep.append(subdomain_separator)
    t2 = default_timer()
    print("Finish KD-Tree preprocessing, time elapsed: {:.1f}s".format(t2-t1))
    if SAVE_preprocess:
        np.save(PATH_ind, input_ind)
        np.save(PATH_sep, input_sep)

################################################################
# dataset preparation
################################################################
input_rr = torch.tensor(input_rr, dtype=torch.float).permute(1,0).unsqueeze(-1)
input_s = torch.tensor(input_s, dtype=torch.float).permute(1,0).unsqueeze(-1)
input_xy = torch.tensor(input_xy, dtype=torch.float).permute(2,0,1)
input_ind = torch.tensor(input_ind, dtype=torch.long)
input_sep = torch.tensor(input_sep, dtype=torch.int)

train_rr = input_rr[:ntrain]
test_rr = input_rr[-ntest:]
train_s = input_s[:ntrain]
test_s = input_s[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]
train_ind = input_ind[:ntrain]
test_ind = input_ind[-ntest:]
train_sep = input_sep[:ntrain]
test_sep = input_sep[-ntest:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        train_rr, train_s, train_xy, 
        train_ind, train_sep
    ), 
    batch_size=batch_size, shuffle=True,
    generator=torch.Generator(device=device)
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        test_rr, test_s, test_xy,
        test_ind, test_sep
    ), 
    batch_size=batch_size, shuffle=False,
    generator=torch.Generator(device=device)
)

################################################################
# training and evaluation
################################################################
model = NUFNO2d(
    train_rr.shape[1], train_s.shape[1],
    modes, modes, width, n_subdomains=8
)
print("Model size: %d"%count_params(model))

params = list(model.parameters())
optimizer = Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
N_sample = 1000
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    train_reg = 0
    for rr, sigma, loc, ind, sep in train_loader:
        optimizer.zero_grad()
        out = model(rr, loc=loc, ind=ind, sep=sep)

        loss = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, sigma, loc, ind, sep in test_loader:
            out = model(rr, loc=loc, ind=ind, sep=sep)
            test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
            .format(ep, t2-t1, train_l2, test_l2))

    if ep%100==0:
        XY = loc[-1].squeeze().detach().cpu().numpy()
        truth = sigma[-1].squeeze().detach().cpu().numpy()
        pred = out[-1].squeeze().detach().cpu().numpy()

        # lims = dict(cmap='RdBu_r', vmin=truth.min(), vmax=truth.max())
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        # ax[0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
        # ax[1].scatter(XY[:, 0], XY[:, 1], 100, pred, edgecolor='w', lw=0.1, **lims)
        # ax[2].scatter(XY[:, 0], XY[:, 1], 100, truth - pred, edgecolor='w', lw=0.1, **lims)
        # fig.show()