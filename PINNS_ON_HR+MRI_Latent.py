import random

import numpy as np
import torch

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import time
from os.path import exists

import matplotlib
import matplotlib.pyplot as plt
import pandas
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from constants import *
from datapreprocess import *
from losses import loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall, \
    loss_data_single_concat_sparse_latent_embed_coord, \
    criterion_pde_single_concat_sparse_latent_embed_coord

matplotlib.use("Agg")  # headless plotting for Slurm
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
# from model_dependent import NetUVWWithEmbed
from New_Model import PINN

from sparse_data_encoder import InletEncoder

import os

if os.name == "nt":
    FILE_PATH = os.path.join(split(os.path.dirname(__file__))[0], "pinns_files")
else:
    FILE_PATH = os.path.join(split(os.path.dirname(__file__))[0], "pinns_files")

if not exists(FILE_PATH):
    raise ValueError(f"{FILE_PATH} does not exist")

RESULT_DIR = join(FILE_PATH, "Results")
weights_dir = join(FILE_PATH, "Weights")
MRI_Encoder_weights = os.path.join(FILE_PATH, "ckpts/sparse_pointnet_vae_k32_N4000.pt")
os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
MRI_FILE = os.path.join(FILE_PATH, "CL085_MRI_Left2.csv")
MESH_FILE = os.path.join(FILE_PATH, "lr_data.csv")  # "lr_data.csv"
WALL_FILE = os.path.join(FILE_PATH, "vessel_wall.csv")
INLET_FILE = os.path.join(FILE_PATH, "inlet_HR.csv")
CFD_FILE = os.path.join(FILE_PATH, "CFD.csv")
OUTLET1_FILE = os.path.join(FILE_PATH, "ica.csv")
OUTLET2_FILE = os.path.join(FILE_PATH, "eca.csv")
INLET_Normals = os.path.join(FILE_PATH, "inlet_normals.csv")  # pts,normals, and area

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TEMPORARY ONLY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # perf hint

print("Loading mesh:", MESH_FILE, flush=True)
csv_mesh = pandas.read_csv(MESH_FILE)
mesh_xyz = csv_mesh.iloc[:, : 3].to_numpy().astype(np.float32)

coord, vel, L0, L1, L_gt, U_gt = preprocess_carotid(MESH_FILE, res=2)

coord = (coord - L0) / L_gt
N = coord.shape[0]

print("n_points of the mesh:", N, flush=True)

input_t1 = coord
data_num_1 = len(input_t1)

vel_gt = vel
gt = vel_gt / U_gt

x = torch.tensor(input_t1[:, 0].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
y = torch.tensor(input_t1[:, 1].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)
z = torch.tensor(input_t1[:, 2].reshape(-1, 1), dtype=torch.float32, requires_grad=True).to(device)

train_gt = torch.tensor(gt.reshape(-1, 3), dtype=torch.float32).to(device)
zero = torch.unsqueeze(torch.tensor(np.zeros(data_num_1), dtype=torch.float32, requires_grad=True), -1).to(device)

viscosity = 4e-2
Re = 1 / viscosity
Re = U_gt * L_gt * Re
# selecting some random points for data loss
df_mesh = pandas.DataFrame(csv_mesh)
data_points = df_mesh.sample(n=internal_size, replace=False, random_state=42)

print("loading sparse data:", MRI_FILE, flush=True)
# lets repalace it with a higher resolution version
# csv_MRI = pandas.read_csv(MRI_FILE)
# MRI_xyz = csv_MRI.iloc[:, 3: 6].to_numpy().astype(np.float32)
# MRI_uvw = csv_MRI.iloc[:, : 3].to_numpy().astype(np.float32)
csv_HR = pandas.read_csv(MESH_FILE)
MRI_xyz = csv_HR.iloc[:, : 3].to_numpy().astype(np.float32)
MRI_uvw = csv_HR.iloc[:, 3: 6].to_numpy().astype(np.float32)

Sparse_vector = np.concatenate((MRI_xyz, MRI_uvw), axis=1)
print("Final shape:", Sparse_vector.shape)

print("Loading wall:", WALL_FILE, flush=True)
csv_wall = pandas.read_csv(WALL_FILE)
wall_points = df_mesh.sample(n=wall_points, replace=False,
                             random_state=42)  # this line can be removed as we are not sampling from the wall but the mesh. We tried sampling from wall points to be sed at bc condition but the solution diverged!
wall_xyz = wall_points.iloc[:, : 3].to_numpy().astype(np.float32)
wall_uvw = wall_points.iloc[:, 3: 6].to_numpy().astype(np.float32)

wall_xyz = (wall_xyz - L0) / L_gt
xb = wall_xyz[:, 0:1]
yb = wall_xyz[:, 1:2]
zb = wall_xyz[:, 2:3]

print("Loading inlet:", INLET_Normals, flush=True)
# csv_inlet = pandas.read_csv(INLET_Normals)
# inlet_xyz_ = csv_inlet.iloc[:, : 3].to_numpy().astype(np.float32)
# inlet_normals = csv_inlet.iloc[:, 3: 6].to_numpy().astype(np.float32)
# inlet_areas = csv_inlet.iloc[:, 6: 7].to_numpy().astype(np.float32)
#
# inlet_xyz_ = (inlet_xyz_ - L0) / L_gt
# xb_in = inlet_xyz_[:, 0:1]
# yb_in = inlet_xyz_[:, 1:2]
# zb_in = inlet_xyz_[:, 2:3]

print("data_points:", data_points.shape, flush=True)
# print("wall:", wall_xyz.shape, flush=True)
# data_points = data_points[~((data_points["Points:0"] in xb[:,0]) & (df_mesh["Points:1"] in yb[:,0]) & (df_mesh["Points:2"] in zb[:,0]))]


# 2) Masks on the FULL mesh (same length)
data_coords = pd.MultiIndex.from_frame(data_points.iloc[:, :3].round(6))
wall_coords = pd.MultiIndex.from_frame(wall_points.iloc[:, :3].round(6))
data_points = data_points[~data_coords.isin(wall_coords)].copy()

x_data = data_points["Points:0"].values.reshape(-1, 1)
y_data = data_points["Points:1"].values.reshape(-1, 1)
z_data = data_points["Points:2"].values.reshape(-1, 1)

u_data = data_points["U:0"].values.reshape(-1, 1)
v_data = data_points["U:1"].values.reshape(-1, 1)
w_data = data_points["U:2"].values.reshape(-1, 1)

x_data = (x_data - L0) / L_gt
y_data = (y_data - L0) / L_gt
z_data = (z_data - L0) / L_gt

data_xyz = np.concatenate((x_data, y_data, z_data), axis=1)

u_data = u_data / U_gt
v_data = v_data / U_gt
w_data = w_data / U_gt

print("Saving normalized and unnormalized")
data_points.to_csv(join(FILE_PATH, "internal_random_data_un_normalized.csv"), index=False)
normalized_data_points = pd.DataFrame(
    [[x_[0], y_[0], z_[0], u_, v_, w_] for x_, y_, z_, u_, v_, w_ in
     zip(x_data, y_data, z_data, u_data, v_data, w_data)],
    columns=data_points.columns)
normalized_data_points.to_csv(join(FILE_PATH, "internal_random_data_normalized.csv"), index=False)
print("Saving normalized and unnormalized --> Done")

# read, sample and normalize CFD points for data fidelity loss
# CFD_points = pandas.read_csv(CFD_FILE)
# CFD_points = pandas.DataFrame(CFD_points)
# CFD_points = CFD_points.sample(n=internal_size, replace=False, random_state=42)
#
# CFD_xyz = CFD_points.iloc[:, : 3].to_numpy().astype(np.float32)
# CFD_uvw = CFD_points.iloc[:, 3: 6].to_numpy().astype(np.float32)

# xcfd = CFD_xyz[:, 0:1]
# ycfd = CFD_xyz[:, 1:2]
# zcfd = CFD_xyz[:, 2:3]
#
# ucfd = CFD_uvw[:, 0:1]
# vcfd = CFD_uvw[:, 1:2]
# wcfd = CFD_uvw[:, 2:3]

# xcfd_max = xcfd.max()
# ycfd_max = ycfd.max()
# zcfd_max = zcfd.max()
#
# ucfd_max = ucfd.max()
# vcfd_max = vcfd.max()
# wcfd_max = wcfd.max()
#
# x_cfd = (xcfd - L0) / L_gt
# y_cfd = (ycfd - L0) / L_gt
# z_cfd = (zcfd - L0) / L_gt

# CFD_XYZ = np.concatenate((x_cfd, y_cfd, z_cfd), axis=1)

# u_cfd = ucfd / U_gt
# v_cfd = vcfd / U_gt
# w_cfd = wcfd / U_gt

# CFD_UVW = np.concatenate((u_cfd, v_cfd, w_cfd), axis=1)

# Masks on the FULL mesh (same length)
# CFD_coords = pd.MultiIndex.from_frame(CFD_points.iloc[:, :3].round(6))
# wall_coords = pd.MultiIndex.from_frame(wall_points.iloc[:, :3].round(6))
# CFD_points = CFD_points[~CFD_coords.isin(wall_coords)].copy()

print("Loading outlet1:", OUTLET1_FILE, flush=True)
csv_outlet1 = pandas.read_csv(OUTLET1_FILE)

outlet1_xyz = csv_outlet1.iloc[:, : 3].to_numpy().astype(np.float32)
outlet1_xyz = (outlet1_xyz - L0) / L_gt
outlet1_x = outlet1_xyz[:, 0:1]
outlet1_y = outlet1_xyz[:, 1:2]
outlet1_z = outlet1_xyz[:, 2:3]

print("Loading outlet2:", OUTLET2_FILE, flush=True)
csv_outlet2 = pandas.read_csv(OUTLET2_FILE)

outlet2_xyz = csv_outlet2.iloc[:, : 3].to_numpy().astype(np.float32)
outlet2_xyz = (outlet2_xyz - L0) / L_gt
outlet2_x = outlet2_xyz[:, 0:1]
outlet2_y = outlet2_xyz[:, 1:2]
outlet2_z = outlet2_xyz[:, 2:3]

# selecting collocation points to be constraint by PDE excluding data points
mesh = csv_mesh  # full mesh DataFrame, shape (81164, 6)
# data = data_points  # data DataFrame, shape (1000, 6)
# data = CFD_points

# 1) Build MultiIndex from the first 3 columns (x, y, z)
mesh_key = mesh.iloc[:, :3].round(6)
# data_key = data.iloc[:, :3].round(6)
outlet1_key = csv_outlet1.iloc[:, :3].round(6)
outlet2_key = csv_outlet2.iloc[:, :3].round(6)
wall_key = wall_points.iloc[:, :3].round(6)

mesh_coords = pd.MultiIndex.from_frame(mesh_key)
# data_coords = pd.MultiIndex.from_frame(data_key)
outlet1_coords = pd.MultiIndex.from_frame(outlet1_key)
outlet2_coords = pd.MultiIndex.from_frame(outlet2_key)
wall_coords = pd.MultiIndex.from_frame(wall_key)

# 2) Masks on the FULL mesh (same length)
mask_data = ~mesh_coords.isin(data_coords)  # not in data
mask_wall = ~mesh_coords.isin(wall_coords)  # not in wall
mask_outlet1 = ~mesh_coords.isin(outlet1_coords)  # not in outlet1
mask_outlet2 = ~mesh_coords.isin(outlet2_coords)  # not in outlet2

# 3) Combine masks and filter mesh once
final_mask = mask_data & mask_wall & mask_outlet1 & mask_outlet2

collocation_points = mesh[final_mask].copy()

# 4) Extract xyz / uvw from collocation_points
coll_xyz_ = collocation_points.iloc[:, :3].to_numpy().astype(np.float32)
coll_uvw_ = collocation_points.iloc[:, 3:6].to_numpy().astype(np.float32)

coll_xyz_ = (coll_xyz_ - L0) / L_gt
x_coll = coll_xyz_[:, 0:1]
y_coll = coll_xyz_[:, 1:2]
z_coll = coll_xyz_[:, 2:3]

coll_uvw_ = (coll_uvw_ - L0) / L_gt
u = coll_uvw_[:, 0:1]
v = coll_uvw_[:, 1:2]
w = coll_uvw_[:, 2:3]
"""
print("Loading inlet:", INLET_FILE, flush=True)

csv_inlet = pandas.read_csv(INLET_FILE)
inlet_xyz = csv_inlet.iloc[:, : 3].to_numpy().astype(np.float32)
inlet_uvw = csv_inlet.iloc[:, 3: 6].to_numpy().astype(np.float32)

xb_in = inlet_xyz[:, 0:1]
yb_in = inlet_xyz[:, 1:2]
zb_in = inlet_xyz[:, 2:3]

xb_in = xb_in / x_max
yb_in = yb_in / y_max
zb_in = zb_in / z_max

ub_in = inlet_uvw[:, 0:1]
vb_in = inlet_uvw[:, 1:2]
wb_in = inlet_uvw[:, 2:3]

ub_in = ub_in / u_max
vb_in = vb_in / v_max
wb_in = wb_in / w_max
# ub_in = (ub_in - u_min_d) / (u_max_d - u_min_d)
# vb_in = (vb_in - v_min_d) / (v_max_d - v_min_d)
# wb_in = (wb_in - w_min_d) / (w_max_d - w_min_d)

print("n_points at inlet:", xb_in.shape[0], flush=True)


# xb_in = (xb_in - x_min_d) / (x_max_d - x_min_d)
# yb_in = (yb_in - y_min_d) / (y_max_d - y_min_d)
# zb_in = (zb_in - z_min_d) / (z_max_d - z_min_d)
"""


# DEfining MRI Encoder to concat to the input
def load_sparse_encoder_and_latent(sparse_to_compress: np.ndarray, k: int) -> torch.Tensor:
    enc = InletEncoder(k=k).to(device)
    ckpt = torch.load(MRI_Encoder_weights, map_location=device)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        ckpt = ckpt["model_state_dict"]
    if isinstance(ckpt, dict):
        enc_state = {kk.replace("enc.", "", 1): vv for kk, vv in ckpt.items() if kk.startswith("enc.")}
        enc.load_state_dict(enc_state if enc_state else ckpt, strict=True)
    enc.eval()
    with torch.no_grad():
        coords_t = torch.from_numpy(sparse_to_compress).unsqueeze(0).to(device)  # [1,N,6]
        mu, lv = enc(coords_t)  # [1,k], [1,k]
        z = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    return z.squeeze(0)  # [k]


# Instantiate with the *checkpoint's* k

sparse_latent = load_sparse_encoder_and_latent(Sparse_vector, k=32).to(device)  # [K2]

# k3 = sparse_latent.numel()


#
# if concat_latent_sparse:
#     if embed_coords:
#         input_n = k3 + k3  # the 3 coord dimension has been embedded to k3 dimension
#     else:
#         input_n = 3 + k3
# else:
#     input_n = 3

# ----------------------------
# Training (CPU dataset GPU compute)
# ----------------------------


scaler = GradScaler(enabled=use_amp)

boundary_loss_list = []
internal_data_loss_list = []
pde_equation_loss_list = []


def grad_ok(net, name):
    g = 0.0
    n = 0
    for p in net.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            g += p.grad.detach().abs().mean().item()
            n += 1
    print(f"grad {name}: {g:.3e} (from {n} tensors)")


def train(x, y, z):
    # CPU tensors for dataset
    if isinstance(x, np.ndarray):
        x_cpu = torch.from_numpy(x).float()
        y_cpu = torch.from_numpy(y).float()
        z_cpu = torch.from_numpy(z).float()
    elif torch.is_tensor(x):
        # ensure CPU + float
        x_cpu = x.detach().cpu().float()
        y_cpu = y.detach().cpu().float()
        z_cpu = z.detach().cpu().float()
    else:
        raise TypeError(f"x/y/z must be numpy arrays or torch tensors, got {type(x)}")

    dataset = TensorDataset(x_cpu, y_cpu, z_cpu)
    sampler = RandomSampler(dataset, replacement=True, num_samples=collocation_size)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True, drop_last=True)
    # for (x_in_cpu, y_in_cpu, z_in_cpu) in loader:
    #     # move ONLY the batch to GPU
    #     x_in = x_in_cpu.to(device, non_blocking=True).reshape(-1,1)
    #     y_in = y_in_cpu.to(device, non_blocking=True).reshape(-1,1)
    #     z_in = z_in_cpu.to(device, non_blocking=True).reshape(-1,1)

    # small BC/sparse tensors — prepare GPU copies ONCE
    # xb_t = torch.from_numpy(xb).float().to(device)
    # yb_t = torch.from_numpy(yb).float().to(device)
    # zb_t = torch.from_numpy(zb).float().to(device)
    # sparse_wall = sparse_latent.view(1, -1).to(device).expand(xb_t.size(0), -1)

    # xb_in_t = torch.from_numpy(xb_in).float().to(device)
    # yb_in_t = torch.from_numpy(yb_in).float().to(device)
    # zb_in_t = torch.from_numpy(zb_in).float().to(device)
    # sparse_inlet = sparse_latent.view(1, -1).to(device).expand(xb_in_t.size(0), -1)
    # inlet_normals_t = torch.from_numpy(inlet_normals).float().to(device)
    # inlet_areas_t = torch.from_numpy(inlet_areas).float().to(device)
    # ub_in_t = torch.from_numpy(ub_in).float().to(device)
    # vb_in_t = torch.from_numpy(vb_in).float().to(device)
    # wb_in_t = torch.from_numpy(wb_in).float().to(device)

    xdata_t = torch.from_numpy(x_data).float().to(device)
    ydata_t = torch.from_numpy(y_data).float().to(device)
    zdata_t = torch.from_numpy(z_data).float().to(device)

    udata_t = torch.from_numpy(u_data).float().to(device)
    vdata_t = torch.from_numpy(v_data).float().to(device)
    wdata_t = torch.from_numpy(w_data).float().to(device)

    sparse_data = sparse_latent.view(1, -1).to(device).expand(xdata_t.size(0), -1)

    # udata_t = torch.from_numpy(u_data).float().to(device)
    # vdata_t = torch.from_numpy(v_data).float().to(device)
    # wdata_t = torch.from_numpy(w_data).float().to(device)

    # replace it with CFD values
    """
    xcfd_t = torch.from_numpy(x_cfd).float().to(device)
    ycfd_t = torch.from_numpy(y_cfd).float().to(device)
    zcfd_t = torch.from_numpy(z_cfd).float().to(device)

    u_cfd_t = torch.from_numpy(u_cfd ).float().to(device)
    v_cfd_t = torch.from_numpy(v_cfd ).float().to(device)
    w_cfd_t = torch.from_numpy(w_cfd ).float().to(device)
    """
    outlet1_x_t = torch.from_numpy(outlet1_x).float().to(device)
    outlet1_y_t = torch.from_numpy(outlet1_y).float().to(device)
    outlet1_z_t = torch.from_numpy(outlet1_z).float().to(device)

    sparse_outlet1 = sparse_latent.view(1, -1).expand(outlet1_x_t.size(0), -1)

    outlet2_x_t = torch.from_numpy(outlet2_x).float().to(device)
    outlet2_y_t = torch.from_numpy(outlet2_y).float().to(device)
    outlet2_z_t = torch.from_numpy(outlet2_z).float().to(device)

    sparse_outlet2 = sparse_latent.view(1, -1).expand(outlet2_x_t.size(0), -1)

    # net = NetUVWWithEmbed(input_x, h_n, n_blocks=6, out_dim=4, k=32).to(device)
    k = 32  # latent size (must match encoder)
    hidden = 64  # emb_s
    depth = 10
    # Inspirde by Korean group Net
    net = PINN(
        in_l=3,  # xyz
        out_l=4,  # u,v,w
        emb_s=hidden,
        d=depth,
        layer_mode="Tanh",
        enc_mode=False,  # no Fourier PE (start simple)
        concat=True,  # use sparse latent
        k=k
    ).to(device)

    def init_kaiming(m):
        if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight)

    # for net in (net_u, net_v, net_w, net_p):
    net.apply(init_kaiming)

    opt = torch.optim.AdamW(params=net.parameters(), lr=learning_rate,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # (A) Step **per epoch**:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    # t0 = time.time()
    for ep in range(1, epochs + 1):

        net.train()

        epoch_time = time.time()
        pde_equation_loss_total = boundary_condition_loss_total = internal_data_loss_total = 0.0
        nb = 0
        leq_epoch = ldata_epoch = lbc_epoch = 0

        for (x_in_cpu, y_in_cpu, z_in_cpu) in loader:
            # move ONLY the batch to GPU
            x_in = x_in_cpu.to(device, non_blocking=True)
            y_in = y_in_cpu.to(device, non_blocking=True)
            z_in = z_in_cpu.to(device, non_blocking=True)
            print(f"len x_in", len(x_in))

            sparse_pde = sparse_latent.view(1, -1).expand(x_in.size(0), -1)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda:0", enabled=use_amp):
                # PDE on collocation batch
                # ub_in_t = ub_in_t.reshape(-1, 1)
                # vb_in_t = vb_in_t.reshape(-1, 1)
                # wb_in_t = wb_in_t.reshape(-1, 1)

                xyz_in = torch.cat([x_in, y_in, z_in], dim=1)
                # xyz_wall = torch.cat([xb_t, yb_t, zb_t], dim=1)
                # xyz_inlet = torch.cat([xb_in_t, yb_in_t, zb_in_t], dim=1)
                # uvw_inlet = torch.cat([ub_in_t , vb_in_t, wb_in_t], dim=1)
                # outlet1_xyz = torch.cat([outlet1_x_t, outlet1_y_t, outlet1_z_t], dim=1)
                # outlet2_xyz = torch.cat([outlet2_x_t, outlet2_y_t, outlet2_z_t], dim=1)

                # lbc = loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall(net, xyz_wall,
                #                                                                                      sparse_wall
                #
                #                                                                                      )
                lbc = 0

                # trial
                """
                lbc=loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall_trial(net, xyz_wall,
                                                                                                     sparse_wall,
                                                                                                     xyz_inlet,
                                                                                                     inlet_normals_t,
                                                                                                     inlet_areas_t,
                                                                                                     sparse_inlet,
                                                                                                     outlet1, outlet2,
                                                                                                     sparse_outlet1,
                                                                                                     sparse_outlet2,
                                                                                                     q_target
                                                                                                     )
                """

                # lbc=loss_boundary_condition_parabolic_noslip_concat_latent_sparse_embed_coord_wall_inlet(net, xyz_wall,xyz_inlet,uvw_inlet,
                #                                                              sparse_wall,
                #                                                            sparse_inlet
                #                                                           )

                # xyz = torch.cat([xdata_t, ydata_t, zdata_t], dim=1)
                # udata_t = udata_t.reshape(-1, 1)
                # vdata_t = vdata_t.reshape(-1, 1)
                # wdata_t = wdata_t.reshape(-1, 1)
                # uvw = torch.cat([udata_t, vdata_t, wdata_t], dim=1)
                # replacing it with CFD points
                xyz = torch.cat([xdata_t, ydata_t, zdata_t], dim=1)
                """
                ucfd_t = u_cfd_t.reshape(-1, 1)
                vcfd_t = v_cfd_t.reshape(-1, 1)
                wcfd_t = w_cfd_t.reshape(-1, 1)
                """

                uvw = torch.cat([udata_t, vdata_t, wdata_t], dim=1)  # (B,3) or (N,3)

                sparse_data = sparse_latent.view(1, -1).expand(xyz.size(0), -1)
                ldata = loss_data_single_concat_sparse_latent_embed_coord(net, xyz, uvw, sparse_data)
                print(f"len xyz", len(xyz))
                # ldata = loss_data_single_concat_sparse_latent_embed_coord(net, xyz, uvw, sparse_data)

                if ldata < ldata_limit:
                    leq = criterion_pde_single_concat_sparse_latent_embed_coord(net, xyz_in.float(), sparse_pde, Re=Re)

                else:
                    leq = 0
                # leq = criterion_pde_single_concat_sparse_latent_embed_coord(net, xyz_in.float(), sparse_pde)

                # if lbc < lbc_limit:
                # lbc_show = lbc
                # lbc = 0
                # else:
                #   lbc_show = lbc

                # if ldata < ldata_limit:
                # ldata_show = ldata
                # ldata = 0
                # else:
                #  ldata_show = ldata

                # lbc = loss_boundary_condition(net,
                #                             # xb_t, yb_t, zb_t,
                #                            xb_in_t, yb_in_t, zb_in_t, ub_in_t, vb_in_t, wb_in_t)

                # ldata = loss_data(net_u, net_v, net_w, xdata_t, ydata_t, zdata_t, udata_t, vdata_t, wdata_t)

                loss = ldata + 0.1 * lbc + 0.01 * leq
                # loss = Lambda_PDE * leq + Lambda_BC * lbc + Lambda_INTERNAL * ldata
                # loss = ldata

                scaler.scale(loss).backward()

                # grad_ok(net_u, "u")
                # grad_ok(net_v, "v")
                # grad_ok(net_w, "w")
                # grad_ok(net_p, "p")
                # U.clip_grad_norm_(net_u.parameters(), 12)  # start generous
                # U.clip_grad_norm_(net_v.parameters(), 12)
                # U.clip_grad_norm_(net_w.parameters(), 12)
                # U.clip_grad_norm_(net_p.parameters(), 12)  # p a bit tighter

                scaler.step(opt)
                scaler.update()

                leq_epoch += leq.item() if isinstance(leq, torch.Tensor) else leq
                lbc_epoch += lbc.item() if isinstance(lbc, torch.Tensor) else lbc
                ldata_epoch += ldata.item() if isinstance(ldata, torch.Tensor) else ldata

        pde_equation_loss_total += (leq_epoch / len(loader))
        boundary_condition_loss_total += (lbc_epoch / len(loader))
        internal_data_loss_total += (ldata_epoch / len(loader))

        boundary_loss_list.append(lbc_epoch / len(loader))
        internal_data_loss_list.append(ldata_epoch / len(loader))
        pde_equation_loss_list.append(leq_epoch / len(loader))
        plt.figure()
        plt.plot(boundary_loss_list, label="boundary_condition")
        plt.plot(internal_data_loss_list, label="internal_data")
        plt.plot(pde_equation_loss_list, label="pde_equation")
        plt.legend()

        plt.savefig(join(FILE_PATH, "loss.jpg"))
        plt.close()
        nb += 1

        scheduler.step()
        print(
            f"Epoch {ep:04d} | Loss pde-equation: {pde_equation_loss_total / nb:.3e}  Loss Boundary Condition {boundary_condition_loss_total / nb:.3e}  "
            f"Internal data Loss {internal_data_loss_total / nb:.3e}  lr {opt.param_groups[0]['lr']:.2e}, epoch time: {time.time() - epoch_time:.2f}s ",
            flush=True)

        if ep > 0 and (ep % result_check == 0):
            print("Starting the evaluation")
            # net_u.eval()
            # net_v.eval()
            # net_w.eval()
            net.eval()

            # Save model weights
            checkpoint_path = join(weights_dir, f"model_weights_epoch_{ep}.pth")
            """torch.save(dict(state_dict=net.state_dict(),
                            x_max=x_max,
                            y_max=y_max,
                            z_max=z_max,
                            u_max=u_max,
                            v_max=v_max,
                            w_max=w_max
                            ), checkpoint_path)"""

            print("Saved model weights:", checkpoint_path)

            # CHUNKED, NO-GRAD INFERENCE
            N = coord.shape[0]

            x = coord[:, 0:1]
            y = coord[:, 1:2]
            z = coord[:, 2:3]

            # x = (x - x_min_d) / (x_max_d - x_min_d)
            # y = (y - y_min_d) / (y_max_d - y_min_d)
            # z = (z - z_min_d) / (z_max_d - z_min_d)

            u_pred = np.empty((N, 1), dtype=np.float32)
            v_pred = np.empty((N, 1), dtype=np.float32)
            w_pred = np.empty((N, 1), dtype=np.float32)

            B = 5000  # tune if needed
            # s1 = sparse_latent.view(1, -1).to(device)

            with torch.no_grad(), torch.amp.autocast("cuda:0", enabled=True):
                for s in range(0, N, B):
                    e = min(s + B, N)
                    xi = torch.from_numpy(x[s:e]).float().to(device)  # x,y,z are numpy arrays of shape (N,1)
                    yi = torch.from_numpy(y[s:e]).float().to(device)
                    zi = torch.from_numpy(z[s:e]).float().to(device)
                    sparse_ = sparse_latent.view(1, -1).to(device).expand(xi.size(0), -1)
                    coords_i = torch.cat((xi, yi, zi), 1)
                    # embeded_coords = embed_net(coords_i)
                    # si = s1.expand(e - s, -1)
                    # if concat_latent_sparse:
                    #    nin = torch.cat((embeded_coords, si), dim=1)
                    # else:
                    #    nin = coords_i

                    prediction = net(coords_i, sparse_).float().cpu().numpy()
                    u_pred[s:e], v_pred[s:e], w_pred[s:e] = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]
                    # u_pred[s:e] = net_u(coords_i).float().cpu().numpy()
                    # v_pred[s:e] = net_v(coords_i).float().cpu().numpy()
                    # w_pred[s:e] = net_w(coords_i).float().cpu().numpy()

            # WRITE A VTK FILE FROM CSV POINTS (PolyData .vtp)
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(N)
            # Rescale predictions to the original scaling
            u_pred = u_pred * U_gt
            v_pred = v_pred * U_gt
            w_pred = w_pred * U_gt
            # u_pred = (u_pred * (u_max_d - u_min_d)) + u_min_d
            # v_pred = (v_pred * (v_max_d - v_min_d)) + v_min_d
            # w_pred = (w_pred * (w_max_d - w_min_d)) + w_min_d

            for i, (xx, yy, zz) in enumerate(mesh_xyz):  # use ORIGINAL coords
                points.SetPoint(i, float(xx), float(yy), float(zz))

            verts = vtk.vtkCellArray()
            verts.Allocate(N)
            for i in range(N):
                verts.InsertNextCell(1)
                verts.InsertCellPoint(i)

            poly = vtk.vtkPolyData()
            poly.SetPoints(points)
            poly.SetVerts(verts)

            vel = np.hstack([u_pred, v_pred, w_pred]).astype(np.float32)
            vel_vtk = numpy_to_vtk(vel, deep=True)
            vel_vtk.SetNumberOfComponents(3)
            vel_vtk.SetName("flow")
            u_vtk = numpy_to_vtk(u_pred.astype(np.float32), deep=True)
            u_vtk.SetName("u")
            v_vtk = numpy_to_vtk(v_pred.astype(np.float32), deep=True)
            v_vtk.SetName("v")
            w_vtk = numpy_to_vtk(w_pred.astype(np.float32), deep=True)
            w_vtk.SetName("w")

            pd = poly.GetPointData()
            pd.AddArray(vel_vtk)

            pd.SetActiveVectors("flow")
            pd.AddArray(u_vtk)
            pd.AddArray(v_vtk)
            pd.AddArray(w_vtk)

            out_vtp = join(RESULT_DIR, f"pinns_result_Raw_HR_labels+wall_on_latent_sparse_embeded_coord{ep}.vtp")

            vtp_writer = vtk.vtkXMLPolyDataWriter()
            vtp_writer.SetFileName(out_vtp)
            vtp_writer.SetInputData(poly)
            vtp_writer.Write()
            print("Saved:", out_vtp)


# ----------------------------
# Main
# ----------------------------
if __name__ == "__main__":
    # tip: in your sbatch add:
    #   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    train(x, y, z)
