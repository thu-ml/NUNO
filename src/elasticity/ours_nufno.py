from timeit import default_timer
from tqdm import tqdm
from util.utilities3 import *
from torch.optim import Adam
from ..nufno.nufno_2d import NUFNO
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

epochs = 1001
step_size = 100
gamma = 0.5

modes = 16
width = 64
# KD-Tree domain splitting
n_subdomains = 8
oversampling_ratio = 1.5


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
PATH_subdomain_info = PATH+'Preprocess_subdomain_info.npy'
SAVE_preprocess = True
LOAD_preprocess = False

input_rr = np.load(PATH_rr)
input_rr = np.transpose(input_rr, (1, 0))
input_sigma = np.load(PATH_Sigma)
input_sigma = np.transpose(input_sigma, (1, 0))
input_xy = np.load(PATH_XY)
input_xy = np.transpose(input_xy, (2, 0, 1))
input_xy = np.concatenate(
    (input_xy[:ntrain], input_xy[-ntest:]), 
    axis=0
)

if LOAD_preprocess:
    input_ind = np.load(PATH_ind)
    input_sep = np.load(PATH_sep)
    input_subdomain_info = np.load(PATH_subdomain_info)
else:
    print("Start KD-Tree splitting...")
    point_clouds = input_xy
    input_ind = []
    input_sep = []
    input_subdomain_info = []
    t1 = default_timer()
    for i in tqdm(range(len(point_clouds)), leave=False):
        point_cloud = point_clouds[i].tolist()
        tree= KDTree(
            point_cloud, dim=2, n_subdomains=n_subdomains, 
            n_blocks=8, return_indices=True
        )
        tree.sort_nodes_by_n_points()
        subdomain_indices = tree.get_subdomain_indices()
        input_ind.append(np.concatenate(subdomain_indices))
        subdomain_separator = [0] * (n_subdomains + 1)
        for j in range(n_subdomains):
            subdomain_separator[j+1] = subdomain_separator[j] + \
                subdomain_indices[j].shape[0]
        input_sep.append(subdomain_separator)
        # Gather subdomain info
        bboxes = tree.get_subdomain_bounding_boxes()
        info = []
        for j in range(n_subdomains):
            bbox = bboxes[j]
            n_points = subdomain_indices[j].shape[0]
            # Calculate the grid size, where the aspect ratio of the discrete grid 
            # remains the same as the that of the original subdomain (bbox)
            grid_size_x = np.sqrt(n_points * oversampling_ratio * \
                (bbox[0][1] - bbox[0][0]) / (bbox[1][1] - bbox[1][0]))
            grid_size_y = grid_size_x * (bbox[1][1] - bbox[1][0]) / (bbox[0][1] - bbox[0][0])
            grid_size_x, grid_size_y = max(int(np.round(grid_size_x)), 2), max(int(np.round(grid_size_y)), 2)

            info.append((bbox[0][0], bbox[1][0], 
                bbox[0][1] - bbox[0][0], bbox[1][1] - bbox[1][0], grid_size_x, grid_size_y))

        input_subdomain_info.append(info)
    t2 = default_timer()
    print("Finish KD-Tree splitting, time elapsed: {:.1f}s".format(t2-t1))

    input_ind = np.array(input_ind)
    input_sep = np.array(input_sep)
    input_subdomain_info = np.array(input_subdomain_info)

    if SAVE_preprocess:
        np.save(PATH_ind, input_ind)
        np.save(PATH_sep, input_sep)
        np.save(PATH_subdomain_info, input_subdomain_info)

################################################################
# dataset preparation
################################################################
input_rr = torch.tensor(input_rr, dtype=torch.float)
input_xy = torch.tensor(input_xy, dtype=torch.float)
input_sigma = torch.tensor(input_sigma, dtype=torch.float).unsqueeze(-1)
input_ind = torch.tensor(input_ind, dtype=torch.long)
input_sep = torch.tensor(input_sep, dtype=torch.long)
input_subdomain_info = torch.tensor(input_subdomain_info, dtype=torch.int)

train_rr = input_rr[:ntrain]
test_rr = input_rr[-ntest:]
train_xy = input_xy[:ntrain]
test_xy = input_xy[-ntest:]
train_sigma = input_sigma[:ntrain]
test_sigma = input_sigma[-ntest:]
train_ind = input_ind[:ntrain]
test_ind = input_ind[-ntest:]
train_sep = input_sep[:ntrain]
test_sep = input_sep[-ntest:]
train_subdomain_info = input_subdomain_info[:ntrain]
test_subdomain_info = input_subdomain_info[-ntest:]

train_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        train_rr, train_xy, train_sigma, 
        train_ind, train_sep, train_subdomain_info
    ), 
    batch_size=batch_size, shuffle=True,
    generator=torch.Generator(device=device)
)
test_loader = torch.utils.data.DataLoader(
    torch.utils.data.TensorDataset(
        test_rr, test_xy, test_sigma,
        test_ind, test_sep, test_subdomain_info
    ), 
    batch_size=batch_size, shuffle=False,
    generator=torch.Generator(device=device)
)

################################################################
# training and evaluation
################################################################
model = NUFNO(modes, width)
print("Model size: %d"%count_params(model))

params = list(model.parameters())
optimizer = Adam(params, lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

myloss = LpLoss(size_average=False)
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0.0
    for rr, xy, sigma, ind, sep, s_info in train_loader:
        optimizer.zero_grad()
        out = model(rr, xy)

        loss = myloss(out.view(batch_size, -1), sigma.view(batch_size, -1))
        loss.backward()
        optimizer.step()
        train_l2 += loss.item()

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for rr, xy, sigma, ind, sep, s_info in test_loader:
            out = model(rr, xy)
            test_l2 += myloss(out.view(batch_size, -1), sigma.view(batch_size, -1)).item()

    train_l2 /= ntrain
    test_l2 /= ntest

    t2 = default_timer()
    print("[Epoch {}] Time: {:.1f}s L2: {:>4e} Test_L2: {:>4e}"
            .format(ep, t2-t1, train_l2, test_l2))

    if ep%100==0:
        pass
        # XY = loc[-1].squeeze().detach().cpu().numpy()
        # truth = sigma[-1].squeeze().detach().cpu().numpy()
        # pred = out[-1].squeeze().detach().cpu().numpy()

        # lims = dict(cmap='RdBu_r', vmin=truth.min(), vmax=truth.max())
        # fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4))
        # ax[0].scatter(XY[:, 0], XY[:, 1], 100, truth, edgecolor='w', lw=0.1, **lims)
        # ax[1].scatter(XY[:, 0], XY[:, 1], 100, pred, edgecolor='w', lw=0.1, **lims)
        # ax[2].scatter(XY[:, 0], XY[:, 1], 100, truth - pred, edgecolor='w', lw=0.1, **lims)
        # fig.show()