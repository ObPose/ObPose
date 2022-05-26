import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as KL
from utils.funcs import to_gaussian
from modules.networks import MLP
import pdb
from attrdict import AttrDict

class Background(nn.Module):
    def __init__(self,nerf,device):
        super(Background, self).__init__()
        self.device = device
        #inference
        self.q_z_net = MLP(32,128,32*2)
        self.dec = nerf


    def forward(self,p_eval_surface,p_eval_air,ray,z):
        #---------------------INPUT---------------------#
        #p_eval_surface/air: B-N-3
        #ray: B-N-3
        #z: B-128
        B,N,_ = ray.size()
        #---------------------SCOPE---------------------#
        surface_num = N*torch.ones([B],device=self.device).long()
        air_num = N*torch.ones([B],device=self.device).long()
        rgb,logits_air,logits_surface = self.dec.forward(p_eval_surface.reshape(-1,3),surface_num,\
                                                            p_eval_air.reshape(-1,3),air_num,\
                                                            ray.reshape(-1,3),z[:,None])
        #rgb: BxN-3
        #logits_air: BxN-1
        #logits_surface: BxN-1
        rgb_out = rgb.view(B,N,3)
        logits_surface_out = logits_surface.view(B,N,1)
        logits_air_out = logits_air.view(B,N,1)
        out = AttrDict(
            rgb=rgb_out[:,None],#B-1-N-3
            logits_surface=logits_surface_out[:,None],#B-1-N-1
            logits_air=logits_air_out[:,None],#B-1-N-1
            z=z,#B-32
        )
        return out
