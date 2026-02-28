import glob

from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from vtk.util.numpy_support import numpy_to_vtk
from vtk.util.numpy_support import vtk_to_numpy

from constants import *
from datapreprocess import *
from sparse_data_encoder import InletEncoder

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # TEMPORARY ONLY


# Defining MRI Encoder to concat to the input
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


patients_LatentMRI = {}  # dict to store each patient's Latent

for file in os.listdir(mri_path):
    for f in os.listdir(os.path.join(mri_path, file)):
        if f.endswith(".csv"):
            data = pd.read_csv(os.path.join(mri_path, file, f))

            MRI_xyz = data.iloc[:, 3:6].to_numpy().astype(np.float32)
            MRI_uvw = data.iloc[:, :3].to_numpy().astype(np.float32)

            Sparse_vector = np.concatenate((MRI_xyz, MRI_uvw), axis=1)
            sparse_latent = load_sparse_encoder_and_latent(
                Sparse_vector, k=32
            ).to(device)

            latent_name = os.path.splitext(f)[0]  # id  for eaxmple"JW087"
            print(latent_name)
            patients_LatentMRI[latent_name] = sparse_latent.detach().cpu()

patients_Re = {}  # dict to store each patient's Re number

hr_files = [item for item in os.listdir(data_path) if item.startswith("hr_data_")]
sorted(
    f for f in glob.glob(os.path.join(data_path, "*.csv"))
    if os.path.basename(f).startswith("hr_data_")
)

for hr_path in hr_files:
    filename = os.path.basename(hr_path)
    full_path = os.path.join(data_path, filename)
    pid_ = filename.split("_")[2]  # JW044

    # IMPORTANT: preprocess_carotid expects the full path
    coord, vel, L0, L1, L_gt, U_gt = preprocess_carotid(full_path, res=2)

    viscosity = 4e-2
    Re = 1 / viscosity
    Re = U_gt * L_gt * Re

    patients_Re[pid_] = float(Re)

print("Computed Re for:", list(patients_Re.keys()))

# Get all CSVs (each one is a point cloud)
all_files = sorted(
    f for f in glob.glob(os.path.join(data_path, "*.csv"))
    if f.startswith(os.path.join(data_path, "hr_data_"))
)

rng = np.random.default_rng(42)
rng.shuffle(all_files)

n = len(all_files)
train_ratio, val_ratio = 0.5, 0.5

n_train = int(train_ratio * n)
n_val = int(val_ratio * n)
# ensure all files used
# n_test = n - n_train - n_val

train_files = all_files[:n_train]
val_files = all_files[n_train:n_train + n_val]


# test_files = all_files[n_train + n_val:]


# print(len(train_files), len(val_files), len(test_files))


class CustomDataset(Dataset):
    def __init__(self, data, sparse_dict, Re_dic):
        self.data = data  # list of (features_np, patient_id_str)
        self.sparse_dict = sparse_dict
        self.Re_dic = Re_dic

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        features, pid = self.data[idx]
        sparse_data = self.sparse_dict[pid]  # latent tensor on CPU
        Re = self.Re_dic[pid]

        return features.astype(np.float32), sparse_data, Re


def get_data(files):
    train_collocation = []
    train_data = []

    for file in files:
        filename = os.path.basename(file)
        pid = filename.split("_")[2]  # patient id string

        # preprocess
        coord, vel, L0, L1, L_gt, U_gt = preprocess_carotid(file, res=2)

        coord = (coord - L0) / L_gt

        gt = vel / U_gt  # [N,3] numpy

        # collocation points: use ALL coords (numpy)
        x = coord[:, 0:1].astype(np.float32)
        y = coord[:, 1:2].astype(np.float32)
        z = coord[:, 2:3].astype(np.float32)

        u = gt[:, 0:1].astype(np.float32)
        v = gt[:, 1:2].astype(np.float32)
        w = gt[:, 2:3].astype(np.float32)

        colloc_features = np.concatenate([x, y, z, u, v, w], axis=1)  # [N,6]
        # store as tuples (features_row, pid) OR (features_matrix, pid)
        # usually easiest is per-row for DataLoader batching:
        for row in colloc_features:
            train_collocation.append((row, pid))

        # data fidelity points: sample subset from CSV mesh (still numpy)
        df_mesh = pd.read_csv(file)
        data_points = df_mesh.sample(n=internal_size, replace=False, random_state=42)

        x_data = data_points["Points:0"].values.reshape(-1, 1)
        y_data = data_points["Points:1"].values.reshape(-1, 1)
        z_data = data_points["Points:2"].values.reshape(-1, 1)

        u_data = data_points["U:0"].values.reshape(-1, 1)
        v_data = data_points["U:1"].values.reshape(-1, 1)
        w_data = data_points["U:2"].values.reshape(-1, 1)

        # normalize
        x_data = ((x_data - L0) / L_gt).astype(np.float32)
        y_data = ((y_data - L0) / L_gt).astype(np.float32)
        z_data = ((z_data - L0) / L_gt).astype(np.float32)

        u_data = (u_data / U_gt).astype(np.float32)
        v_data = (v_data / U_gt).astype(np.float32)
        w_data = (w_data / U_gt).astype(np.float32)

        data_features = np.concatenate([x_data, y_data, z_data, u_data, v_data, w_data], axis=1)  # [M,6]
        for row in data_features:
            train_data.append((row, pid))

    return train_collocation, train_data


def get_walldata(files):
    wall_list = []

    for file in files:
        filename = os.path.basename(file)
        pid = filename.split("_")[2]  # JW044

        wall_filename = f"vessel_wall_{pid}_left.csv"
        wall_file = os.path.join(wall_path, wall_filename)

        if not os.path.exists(wall_file):
            print(f"Wall file not found for {pid}: {wall_file}")
            continue

        wall_df = pd.read_csv(wall_file)

        # ---- IMPORTANT: pick the correct columns for xyz in your wall CSV ----
        # If the wall CSV has columns like Points:0 Points:1 Points:2:
        if {"Points:0", "Points:1", "Points:2"}.issubset(wall_df.columns):
            wall_xyz = wall_df[["Points:0", "Points:1", "Points:2"]].to_numpy().astype(np.float32)
        else:
            # fallback: assume first 3 columns are xyz
            wall_xyz = wall_df.iloc[:, :3].to_numpy().astype(np.float32)

        # ---- Normalize wall coords EXACTLY like your training coords ----
        # Best: reuse preprocess_carotid normalization parameters from the same CFD file
        coord, vel, L0, L1, L_gt, U_gt = preprocess_carotid(file, res=2)

        x = ((wall_xyz[:, 0:1] - L0) / L_gt).astype(np.float32)
        y = ((wall_xyz[:, 1:2] - L0) / L_gt).astype(np.float32)
        z = ((wall_xyz[:, 2:3] - L0) / L_gt).astype(np.float32)

        wall_features = np.concatenate([x, y, z], axis=1)  # [Nwall, 3]

        for row in wall_features:
            wall_list.append((row, pid))

    return wall_list


train_collocation, train_data = get_data(train_files)
train_bc = get_walldata(train_files)
val_collocation, val_data = get_data(val_files)
val_bc = get_walldata(val_files)

print(f"{len(train_data)=}")
print(f"{len(val_data)=}")
train_collocation_dataset = CustomDataset(train_collocation, patients_LatentMRI, patients_Re)
train_data_dataset = CustomDataset(train_data, patients_LatentMRI, patients_Re)
train_bc_dataset = CustomDataset(train_bc, patients_LatentMRI, patients_Re)

val_collocation_dataset = CustomDataset(val_collocation, patients_LatentMRI, patients_Re)
val_data_dataset = CustomDataset(val_data, patients_LatentMRI, patients_Re)
val_bc_dataset = CustomDataset(val_bc, patients_LatentMRI, patients_Re)

train_collocation_dl = DataLoader(train_collocation_dataset, batch_size=collocation_batch_size, shuffle=True)
train_data_dl = DataLoader(train_data_dataset, batch_size=data_batch_size, shuffle=False)
train_bc_dl = DataLoader(train_bc_dataset, batch_size=wall_batch_size, shuffle=False)

val_collocation_dl = DataLoader(val_collocation_dataset, batch_size=collocation_batch_size, shuffle=False)
val_data_dl = DataLoader(val_data_dataset, batch_size=data_batch_size, shuffle=False)
val_bc_dl = DataLoader(val_bc_dataset, batch_size=wall_batch_size, shuffle=False)

# TODO: Work on test
# test_files = all_files[n_train + n_val:]
# print("The Test file is:", os.path.basename(test_files[0]))
# test_collocation, test_data = get_data(test_files)
# test_bc = get_walldata(test_files)
# test_collocation_dataset = CustomDataset(test_collocation, patients_LatentMRI, patients_Re)
# test_bc_dataset = CustomDataset(test_bc, patients_LatentMRI, patients_Re)
# test_data_dataset = CustomDataset(test_data, patients_LatentMRI, patients_Re)
# test_collocation_dl = DataLoader(test_collocation_dataset, batch_size=collocation_batch_size, shuffle=False)
# test_data_dl = DataLoader(test_data_dataset, batch_size=data_batch_size, shuffle=False)
# test_bc_dl = DataLoader(test_bc_dataset, batch_size=wall_batch_size, shuffle=False)
# for test_file in test_files:
#     file.name = test_file.replace(".csv", "")  # patient's ID
#     csv_mesh = pd.read_csv(test_file)
#     mesh_xyz_test = csv_mesh.iloc[:, : 3].to_numpy().astype(np.float32)
#     print(f"Testing file: {os.path.basename(test_file)}")
"""
if __name__ == '__main__':
    print(train_collocation_dl.dataset.__len__())
    iter_dl = iter(train_collocation_dl)
    while True:
        d,b = next(iter_dl)
        print(d.shape, b.shape)

    # for d, b in train_collocation_dl:
    #     print(d.shape, b.shape)
    # for d_, b_ in test_collocation_dl:
    #     print(d_.shape, b_.shape)
"""
