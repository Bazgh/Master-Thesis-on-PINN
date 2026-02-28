
import torch
import numpy as np

from torch import nn
from math import *


class adaemb(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L
        self.PE = nn.Parameter(torch.ones(L))

    def forward(self, x):
        embed = []
        embed.append(x)
        for i in range(self.L):
            for fn in [torch.sin, torch.cos]:
                embed.append(fn(2. * pi * i * self.PE[i] * x))

        return torch.cat(embed, -1)


class posemb(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.L = L

    def forward(self, x):
        embed = []
        embed.append(x)
        for i in range(self.L):
            for fn in [torch.sin, torch.cos]:
                embed.append(fn(2. * pi * i * x))

        return torch.cat(embed, -1)
class SineLayer(nn.Module):

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=10):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)

            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))


class TanhLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.Tanh())

    def forward(self, x):
        out = self.NN(x)

        return out


class ReLULayer(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.NN = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU())

    def forward(self, x):
        out = self.NN(x)

        return out

class EmbedInitialCoordLatentSpace(nn.Module):
    def __init__(self, in_dim=3, k=32):
        super().__init__()
        #self.ff = FourierFeatures(in_dim=in_dim, num_freq=num_freq, scale=scale)
        #self.include_raw = include_raw

        #ff_dim = 2 * num_freq
        #embed_in = (in_dim + ff_dim) if include_raw else ff_dim

        self.conv = nn.Sequential(
            nn.Linear(in_dim, 64), nn.ReLU(),
            nn.Linear(64, 128), nn.ReLU(),
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
        )
        self.fc = nn.Linear(128, k)

    def forward(self, x):
        #x_ff = self.ff(x)
        #x_in = torch.cat([x, x_ff], dim=-1) if self.include_raw else x_ff
        x = self.conv(x)
        x = self.fc(x)
        return x

class PINN(nn.Module):
    def __init__(self, in_l: int = 3, out_l: int = 4,
                 emb_s: int = 256, d: int = 8,
                 layer_mode: str = 'Sin',
                 _O: int = 10,
                 enc_mode: bool = True,
                 L: int = 10,
                 concat: bool = False,
                 k: int = 32):
        super().__init__()

        self.if_pos = enc_mode
        self.if_concat_sparse = concat
        self.depth = d
        self.init_w = False

        # coord embedding if concat
        if self.if_concat_sparse:
            self.coord_embed = EmbedInitialCoordLatentSpace(in_dim=3, k=k)
            in_l = 2 * k  # coord_embed gives k, sparse is k

        # positional encoding (applied AFTER concat or directly on xyz)
        if enc_mode:
            in_l = in_l * ((2 * L) + 1)
            self.PE = posemb(L)

        # choose backbone
        if layer_mode == 'Tanh':
            self.NN_init = TanhLayer(in_l, emb_s)
            self.NN = TanhLayer(emb_s, emb_s)
        elif layer_mode == 'ReLU':
            self.NN_init = ReLULayer(in_l, emb_s)
            self.NN = ReLULayer(emb_s, emb_s)
        elif layer_mode == 'Sin':
            self.NN_init = SineLayer(in_l, emb_s, True, True, _O)
            self.NN = SineLayer(emb_s, emb_s, omega_0=_O)
            self.init_w = True
        else:
            raise ValueError(f"Unknown layer_mode: {layer_mode}")

        self.NN_out = nn.Linear(emb_s, out_l)

        if self.init_w:
            with torch.no_grad():
                self.NN_out.weight.uniform_(-np.sqrt(6 / emb_s) / _O, np.sqrt(6 / emb_s) / _O)

    def forward(self, xyz, sparse=None):
        # xyz: (N,3), sparse: (N,k)

        if self.if_concat_sparse:
            if sparse is None:
                raise ValueError("sparse must be provided when concat=True")
            xyz_k = self.coord_embed(xyz)              # (N,k)
            inputs = torch.cat([xyz_k, sparse], dim=-1)  # (N,2k)
        else:
            inputs = xyz

        if self.if_pos:
            inputs = self.PE(inputs)

        out = self.NN_init(inputs)
        for _ in range(self.depth - 1):
            out = self.NN(out)
        return self.NN_out(out)
