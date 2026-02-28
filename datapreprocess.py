import numpy as np
import torch
import scipy.io
import pandas as pd
from torch.autograd import Variable

torch.manual_seed(777)
torch.cuda.manual_seed(777)
np.random.seed(777)

device = 'cuda'


# load spatial averaged data
def preprocess_carotid(data_dir: str, res: int = 2):
    data = pd.read_csv(data_dir)

    x = data['Points:0'].to_numpy(dtype=np.float32)
    y = data['Points:1'].to_numpy(dtype=np.float32)
    z = data['Points:2'].to_numpy(dtype=np.float32)

    # Check if velocity columns exist
    if {'U:0', 'U:1', 'U:2'}.issubset(data.columns):
        u = data['U:0'].to_numpy(dtype=np.float32)
        v = data['U:1'].to_numpy(dtype=np.float32)
        w = data['U:2'].to_numpy(dtype=np.float32)
    else:
        # wall file: no velocity
        u = np.zeros_like(x)
        v = np.zeros_like(y)
        w = np.zeros_like(z)

    coord = np.stack((x, y, z), axis=-1)
    vel = np.stack((u, v, w), axis=-1)

    L0 = np.min(coord)
    L1 = np.max(coord)
    L = L1 - L0
    U = np.max(vel)

    return coord, vel, L0, L1, L, U