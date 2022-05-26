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

    def forward(self,p_surface,num_surface,ray,p_air,num_air,z_what,bb_loc,bb_rot_matrix,depth,obj_size,cam_proj_homo):
        #p_surface/air: N_surface-3
        #num_surface: BxK
        #ray: N_surface-3
        #z_what: B-K-32
        #bb_loc: B-K-3
        #bb_rot_matrix: B-K-3-3
        #depth: B-128-128
        #obj_size: B-K-3
        #cam_proj_homo: B-4-4
        num_surface = num_surface.long()
        B,K,_= z_what.shape
        #evaluate air points.
        with torch.no_grad():
            cell_size = 2./self.cell_num
            left_most_centre = -1. + cell_size/2.
            mesh_voxel = torch.meshgrid(torch.arange(self.cell_num,device=self.device),torch.arange(self.cell_num,device=self.device),torch.arange(self.cell_num,device=self.device))
            mesh_voxel = torch.stack(mesh_voxel,dim=3)
            #mesh_voxel:cn-cn-cn-3
            p_voxel = left_most_centre + mesh_voxel*cell_size
            #p_voxel: cn-cn-cn-3
            p_voxel = p_voxel[None,None].repeat(B,K,1,1,1,1) + (-1.+2.*torch.rand(B,K,self.cell_num,self.cell_num,self.cell_num,3,device=self.device))*cell_size/2.
            #p_voxel: B-K-cn-cn-cn-3
            p_voxel = p_voxel.view(B,K,self.cell_num**3,3)
            #p_voxel: B-K-cnxcnxcn-3
            num_voxels = (self.cell_num**3)*torch.ones([B*K],device=self.device).long()
            #gpu_used  = torch.cuda.max_memory_allocated()/1048576
            density_out_v,_ = self.compute_sigma(p_voxel.view(-1,3),z_what.view(B*K,-1),num_voxels)
            #density_out_v: BxKxcnxcnxcn-1
            #print("--- %s seconds ---" % (time.time() - start_time))
            prob_occup = torch.tanh(F.softplus(density_out_v)).view(B,K,-1)
            #prob_occup: B-K-cnxcnxcn
            p_voxel_world = bb_rot_matrix[:,:,None]@((p_voxel*(obj_size[:,:,None]/2.))[:,:,:,:,None])
            #obj_size: B-K-3 -> B-K-1-3
            #bb_rot_matrix: B-K-3-3 -> B-K-1-3-3; p_voxel: B-K-cnxcnxcn-3 -> B-K-cnxcnxcn-3-1
            #p_voxel_world: B-K-cnxcnxcn-3-1
            p_voxel_world = p_voxel_world[:,:,:,:,0] + bb_loc[:,:,None]
            #B-K-cnxcnxcn-3
            posi_map = (prob_occup>0.5)&(p_voxel_world[:,:,:,2]>-0.002)
            #posi_map: B-K-cnxcnxcn
            #print('Training starts, GPU usage in train start: ', torch.cuda.max_memory_allocated()/1048576)
            mask_desk = p_voxel_world[:,:,:,2]<-0.002
            #mask_desk: B-K-cnxcnxcn
            p_voxel_cam = cam_proj_homo[:,None,None,:3]@torch.cat([p_voxel_world,torch.ones([B,K,self.cell_num**3,1],device=self.device)],dim=3)[:,:,:,:,None]
            #p_voxel_cam: B-K-cnxcnxcn-3-1
            p_voxel_cam = p_voxel_cam[:,:,:,:,0]
            #p_voxel_cam: B-K-cnxcnxcn-3
            depth_voxel = p_voxel_cam[:,:,:,2:3]
            #depth_voxel: B-K-cnxcnxcn-1
            p_voxel_cam = torch.floor(p_voxel_cam[:,:,:,:2]/depth_voxel)
            #p_voxel_cam: B-K-cnxcnxcn-2
            #start_time = time.time()
            for_time = time.time()
            voxel_mask = torch.all(p_voxel_cam<=127,dim=-1)&torch.all(p_voxel_cam>=0,dim=-1)
            #voxel_mask: B-K-cnxcnxcn
            p_voxel_cam = p_voxel_cam.clamp(0,127)
            #p_voxel_cam: B-K-cnxcnxcn-2
            p_voxel_cam = p_voxel_cam[:,:,:,0]+p_voxel_cam[:,:,:,1]*128
            #p_voxel_cam: B-K-cnxcnxcn
            depth_obs = depth[:,None].repeat(1,K,1,1).view(B,K,-1)
            #depth: B-128-128 -> B-K-128-128 -> B-K-128x128
            depth_obs = torch.gather(depth_obs,2,p_voxel_cam.long())
            #depth_obs: B-K-cnxcnxcn
            mask_air = torch.logical_and(depth_voxel[:,:,:,0]<depth_obs,voxel_mask)
            #mask_air: B-K-cnxcnxcn

        #print("--- %s seconds for time---" % (time.time() - for_time))
        num_air = num_air.long()
        #num_air: BxK
        p_in = torch.cat([p_air,p_surface],dim=0)
        #p_in: N_all-3
        density_out,out = self.compute_sigma(p_in,z_what.view(B*K,-1).repeat(2,1),torch.cat([num_air,num_surface],dim=0))
        #density_out: N_all-1, out: N_all-128
        rgb_out = self.compute_rgb(out[num_air.sum():],z_what.view(B*K,-1),num_surface,ray)
        #rgb_out: N_surface-3

        return rgb_out,density_out[:num_air.sum()],density_out[num_air.sum():],\
               p_voxel_world,prob_occup,posi_map,mask_air

    def forward_bg(self,p_surface,num_surface,p_air,num_air,ray,z_what):
        #p_surface: N_surface-3
        #num_surface: B
        #p_air: N_air-3
        #num_air: B
        #ray: N_surface-3
        #z_what: B-32
        B,_ = z_what.shape
        p_in = torch.cat([p_air,p_surface],dim=0)
        #p_in: N_all-3
        num_in = torch.cat([num_air,num_surface],dim=0)
        #num_in: 2xB
        density_out,out = self.compute_sigma(p_in,z_what.repeat(2,1),num_in)
        #density_out: N_all-1, out: N_all-128
        rgb_out = self.compute_rgb(out[num_air.sum():],z_what,num_surface,ray)
        #rgb_out: N_surface-3
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
