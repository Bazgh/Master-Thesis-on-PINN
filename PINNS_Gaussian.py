import torch
import random
import numpy as np
from scipy.spatial import cKDTree

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

import time
from os.path import exists

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas
import pandas as pd
import torch.nn as nn
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, TensorDataset, RandomSampler

from constants import *

matplotlib.use("Agg")  # headless plotting for Slurm
import vtk
from vtk.util.numpy_support import vtk_to_numpy
from vtk.util.numpy_support import numpy_to_vtk
from New_Model import *
from losses import loss_data_single, criterion_pde

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

os.makedirs(RESULT_DIR, exist_ok=True)
os.makedirs(weights_dir, exist_ok=True)
MRI_FILE = os.path.join(FILE_PATH, "CL085_MRI_Left2.csv")
MESH_FILE = os.path.join(FILE_PATH, "lr_data.csv")

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TEMPORARY ONLY
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True  # perf hint

print("Loading mesh:", MESH_FILE, flush=True)
csv_mesh = pandas.read_csv(MESH_FILE)

mesh_xyz = csv_mesh.iloc[:, : 3].to_numpy().astype(np.float32)
mesh_uvw = csv_mesh.iloc[:, 3: 6].to_numpy().astype(np.float32)
x = mesh_xyz[:, 0:1]
y = mesh_xyz[:, 1:2]
z = mesh_xyz[:, 2:3]

u = mesh_uvw[:, 0:1]
v = mesh_uvw[:, 1:2]
w = mesh_uvw[:, 2:3]

x_min, x_max = x.min(), x.max()
y_min, y_max = y.min(), y.max()
z_min, z_max = z.min(), z.max()

u_min, u_max = u.min(), u.max()
v_min, v_max = v.min(), v.max()
w_min, w_max = w.min(), w.max()


def norm(data,min,max):
    return (data-min)/(max-min)

# Normalize XYZ
x =norm(x,x_min,x_max)
y =norm(y,y_min,y_max)
z = norm(z,z_min,z_max)

# Normalize UVW
u =norm(u,u_min,u_max)
v =norm(v,v_min,v_max)
w =norm(w,w_min,w_max)

N = mesh_xyz.shape[0]
print(f"{u_max=}, {v_max=}, {w_max=}")
print(f"{x_max=}, {y_max=}, {z_max=}")
print("n_points of the mesh:", N, flush=True)

# selecting some random points for data loss
df_mesh = pandas.DataFrame(csv_mesh)
# data_points = df_mesh.sample(n=internal_size, replace=False, random_state=42)

# data_points_xyz = data_points.iloc[:, : 3].to_numpy().astype(np.float32)
# data_points_uvw = data_points.iloc[:, 3: 6].to_numpy().astype(np.float32)

print("loading sparse data:", MRI_FILE, flush=True)

csv_MRI = pandas.read_csv(MRI_FILE)
MRI_xyz = csv_MRI.iloc[:, 3: 6].to_numpy().astype(np.float32)
MRI_uvw = csv_MRI.iloc[:, : 3].to_numpy().astype(np.float32)
data_points = csv_MRI.sample(n=internal_size, replace=False, random_state=42)
data_points_xyz = data_points.iloc[:, : 3].to_numpy().astype(np.float32)
data_points_uvw = data_points.iloc[:, 3: 6].to_numpy().astype(np.float32)

x_data = data_points_xyz[:, 0:1]
y_data = data_points_xyz[:, 1:2]
z_data = data_points_xyz[:, 2:3]

u_data = data_points_uvw[:, 0:1]
v_data = data_points_uvw[:, 1:2]
w_data = data_points_uvw[:, 2:3]

# Normalize XYZ
x_data =norm(x_data,x_min,x_max)
y_data =norm(y_data,y_min,y_max)
z_data=norm(z_data,z_min,z_max)

# Normalize UVW
u_data =norm(u_data,u_min,u_max)
v_data =norm(v_data,v_min,v_max)
w_data =norm(w_data,w_min,w_max)

data_xyz = np.concatenate((x_data, y_data, z_data), axis=1)

print("Saving normalized and unnormalized")
data_points.to_csv(join(FILE_PATH, "internal_random_data_un_normalized.csv"), index=False)
normalized_data_points = pd.DataFrame(
    [[x_[0], y_[0], z_[0], u_, v_, w_] for x_, y_, z_, u_, v_, w_ in
     zip(x_data, y_data, z_data, u_data, v_data, w_data)],
    columns=data_points.columns)
normalized_data_points.to_csv(join(FILE_PATH, "internal_random_data_normalized.csv"), index=False)
print("Saving normalized and unnormalized --> Done")

# ----------------------------
# Training (CPU dataset GPU compute)
# ----------------------------


scaler = GradScaler(enabled=use_amp)

internal_data_loss_list = []
pde_loss_list = []


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
    x_cpu = torch.from_numpy(x).float()
    y_cpu = torch.from_numpy(y).float()
    z_cpu = torch.from_numpy(z).float()

    dataset = TensorDataset(x_cpu, y_cpu, z_cpu)
    sampler = RandomSampler(dataset, replacement=True, num_samples=collocation_size)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                        num_workers=0, pin_memory=True, drop_last=True)

    xdata_t = torch.from_numpy(x_data).float().to(device)
    ydata_t = torch.from_numpy(y_data).float().to(device)
    zdata_t = torch.from_numpy(z_data).float().to(device)

    udata_t = torch.from_numpy(u_data).float().to(device)
    vdata_t = torch.from_numpy(v_data).float().to(device)
    wdata_t = torch.from_numpy(w_data).float().to(device)

    xyz_all = torch.cat([xdata_t, ydata_t, zdata_t], dim=1)
    uvw_all = torch.cat([udata_t, vdata_t, wdata_t], dim=1)
    Ndata = xyz_all.shape[0]
    data_bs = min(2048, Ndata)

    net = Network(
        in_l=3,
        out_l=4,
        emb_s=256,
        d=8,
        layer_mode='Sin',
        _O=10,
        enc_mode=False,
        L=10
    ).to(device)

    def init_kaiming(m):
        if isinstance(m, nn.Linear): nn.init.kaiming_normal_(m.weight)

    # for net in (net_u, net_v, net_w, net_p):
    #net.apply(init_kaiming)

    opt = torch.optim.AdamW(params=net.parameters(), lr=learning_rate,
                            betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0)

    # (A) Step **per epoch**:
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-6)

    t0 = time.time()


    for ep in range(1, epochs + 1):

        net.train()

        epoch_time = time.time()
        pde_equation_loss_total = boundary_condition_loss_total = internal_data_loss_total = 0.0
        nb = 0
        lpde_epoch = ldata_epoch = lbc_epoch = 0

        for (x_in_cpu, y_in_cpu, z_in_cpu) in loader:
            # move ONLY the batch to GPU
            x_in = x_in_cpu.to(device, non_blocking=True)
            y_in = y_in_cpu.to(device, non_blocking=True)
            z_in = z_in_cpu.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with autocast("cuda:0", enabled=use_amp):
                xyz_in = torch.cat([x_in, y_in, z_in], dim=1)

                xyz = torch.cat([xdata_t, ydata_t, zdata_t], dim=1)
                udata_t = udata_t.reshape(-1, 1)
                vdata_t = vdata_t.reshape(-1, 1)
                wdata_t = wdata_t.reshape(-1, 1)
                uvw = torch.cat([udata_t, vdata_t, wdata_t], dim=1)
                """
                idx = torch.randint(0, Ndata, (data_bs,), device=device)
                xyz = xyz_all[idx]
                uvw = uvw_all[idx]
                """

                # Data loss (fast, AMP on)
                with autocast("cuda", enabled=use_amp):
                    ldata = loss_data_single(net, xyz, uvw)
                if ldata<ldata_limit:
                    # PDE loss (stable, AMP off)
                    with autocast("cuda", enabled=False):
                        lpde = criterion_pde(net, xyz_in.float())
                else:
                    lpde=0


                loss = ldata + lpde

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

                ldata_epoch += ldata.item() if isinstance(ldata, torch.Tensor) else ldata
                lpde_epoch += lpde.item() if isinstance(lpde, torch.Tensor) else lpde

        internal_data_loss_total += (ldata_epoch / len(loader))
        pde_equation_loss_total += lpde_epoch / len(loader)

        internal_data_loss_list.append(ldata_epoch / len(loader))
        pde_loss_list.append(lpde_epoch / len(loader))

        plt.figure()
        plt.plot(internal_data_loss_list, label="internal_data")
        plt.legend()

        plt.savefig(join(FILE_PATH, "loss.jpg"))
        plt.close()
        nb += 1

        scheduler.step()
        print(
            f"Epoch {ep:04d} | Loss pde-equation: {pde_equation_loss_total / nb:.3e}",
            f"Internal data Loss {internal_data_loss_total / nb:.3e}  lr {opt.param_groups[0]['lr']:.2e}, epoch time: {time.time() - epoch_time:.2f}s ",
            flush=True)

        if ep > 0 and (ep % result_check == 0):
            print("Starting the evaluation")

            net.eval()

            # Save model weights
            checkpoint_path = join(weights_dir, f"model_weights_epoch_{ep}.pth")
            torch.save(dict(state_dict=net.state_dict(),
                            x_max=x_max,
                            y_max=y_max,
                            z_max=z_max,
                            u_max=u_max,
                            v_max=v_max,
                            w_max=w_max
                            ), checkpoint_path)

            print("Saved model weights:", checkpoint_path)

            # CHUNKED, NO-GRAD INFERENCE

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

                    coords_i = torch.cat((xi, yi, zi), 1)

                    prediction = net(coords_i).float().cpu().numpy()
                    u_pred[s:e], v_pred[s:e], w_pred[s:e] = prediction[:, 0:1], prediction[:, 1:2], prediction[:, 2:3]

            # WRITE A VTK FILE FROM CSV POINTS (PolyData .vtp)
            points = vtk.vtkPoints()
            points.SetNumberOfPoints(N)
            # Rescale predictions to the original scaling

            u_pred = u_pred *(u_max-u_min)+u_min
            v_pred = v_pred * (v_max-v_min)+v_min
            w_pred = w_pred * (w_max-w_min)+w_min

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

            out_vtp = join(RESULT_DIR, f"HR_reconstruction{ep}.vtp")

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
