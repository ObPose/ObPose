import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from numpy import pi
from utils.funcs import get_rot_matrix
from torch.distributions.categorical import Categorical
import pdb
import time

class Nerf(nn.Module):
    def __init__(self,device):
        super(Nerf,self).__init__()
        self.device = device
        self.z_dim = 32
        self.n_layers = 8
        self.cell_num = 24
        self.h_dim = 128
        self.n_freq_posenc = 10
        self.n_freq_posenc_views = 4
        self.embd_dim = 3*self.n_freq_posenc*2
        self.embd_dim_view = 3*self.n_freq_posenc_views*2
        # Density Prediction Layers
        self.fc_in = nn.Linear(self.embd_dim, self.h_dim)
        self.fc_z = nn.Linear(self.z_dim,self.h_dim)
        self.net = nn.ModuleList([
            nn.Linear(self.h_dim, self.h_dim) for i in range(self.n_layers - 1)
        ])
        self.fc_z_skips=nn.Linear(self.z_dim,self.h_dim)
        self.fc_p_skips = nn.Linear(self.embd_dim,self.h_dim)
        self.sigma_out = nn.Linear(self.h_dim,1)
        # RGB Prediction Layers
        self.fc_z_view = nn.Linear(self.z_dim,self.h_dim)
        self.rgb_view = nn.Linear(self.h_dim,self.h_dim)
        self.fc_view = nn.Linear(self.embd_dim_view,self.h_dim)
        self.rgb_out = nn.Linear(self.h_dim,3)
        self.net_view = nn.Linear(self.embd_dim_view+self.h_dim,self.h_dim)

    def transform_points(self, p, views=False):
        # Positional encoding
        # p should be in [-1,1]
        L = self.n_freq_posenc_views if views else self.n_freq_posenc
        out = torch.cat([torch.cat(
                         [torch.sin((2 ** i) * pi * p),
                          torch.cos((2 ** i) * pi * p)],
                          dim=-1) for i in range(L)], dim=-1)
        return out

    def forward(self,p_surface,num_surface,p_air,num_air,ray,z_what):
        #p_surface/air: N_surface-3
        #num_surface/air: BxK
        #ray: N_surface-3
        #z_what: B-K-32
        num_surface = num_surface.long()
        B,K,_= z_what.shape
        num_air = num_air.long()
        #num_air: BxK
        p_in = torch.cat([p_air,p_surface],dim=0)
        #p_in: N_all-3
        density_out,out = self.compute_sigma(p_in,z_what.view(B*K,-1).repeat(2,1),torch.cat([num_air,num_surface],dim=0))
        #density_out: N_all-1, out: N_all-128
        rgb_out = self.compute_rgb(out[num_air.sum():],z_what.view(B*K,-1),num_surface,ray)
        #rgb_out: N_surface-3
        density_out = 10.*torch.sigmoid(density_out)
        return rgb_out,density_out[:num_air.sum()],density_out[num_air.sum():]

    def compute_sigma(self,p_in,z_what,num_z):
        a = F.relu
        p = self.transform_points(p_in)
        #p: N_all-3xLx2
        out = self.fc_in(p)
        #out: N_all-128
        z_out = self.fc_z(z_what)
        #z_out: 2xBxK-128
        z_skip_out = self.fc_z_skips(z_what)
        #z_skip_out: 2xBxK-128
        z_out_padded = torch.repeat_interleave(z_out,num_z,dim=0)
        #z_out_padded: N_all-128
        z_skip_out_padded = torch.repeat_interleave(z_skip_out,num_z,dim=0)
        #z_skip_out_padded: N_all-128
        out = out+z_out_padded
        out = a(out)
        #out: N_all-128
        for idx, layer in enumerate(self.net):
            out = a(layer(out))
            if idx==3:
                out = out + z_skip_out_padded
                out = out + self.fc_p_skips(p)
        #out: N_all-128
        density_out = self.sigma_out(out)
        #density_out: N_all-1
        return density_out,out

    def compute_rgb(self,out,z_what,num_surface,ray):
        a = F.relu
        #out = self.rgb_view(out[num_air.sum():])
        out = self.rgb_view(out)
        #out: N_surface-128
        z_view_out = self.fc_z_view(z_what)
        #z_view_out: BxK-128
        z_view_out_padded = torch.repeat_interleave(z_view_out,num_surface,dim=0)
        #z_view_out_padded: N_surface-128
        out = out + z_view_out_padded
        #out: N_surface-128
        ray = self.transform_points(ray, views=True)
        #ray: N_surface-3xLx2
        out = out+self.fc_view(ray)
        out = a(out)
        #out: N_surface-128
        rgb_out = torch.sigmoid(self.rgb_out(out))
        #rgb_out: N_surface-3
        return rgb_out
