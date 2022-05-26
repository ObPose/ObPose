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
        self.q_rnn = nn.GRUCell(64,32)
        self.q_z_net = MLP(32,32,64)
        self.q_h_his = []
        self.q_h0 = nn.Parameter(torch.randn(1,32))
        self.z_his = []
        self.dec = nerf
        #prior
        self.p_h_his = []
        self.p_h0 = nn.Parameter(torch.randn(1,32))
        self.p_rnn = nn.GRUCell(32,32)
        self.p_z_net = MLP(32,32,64)

    def forward(self,p_eval_surface,p_eval_air,ray,bg_enc):
        #---------------------INPUT---------------------#
        #p_eval_surface: B-N-3
        #ray: B-N-3
        #bg_enc: B-128
        #depth: B-128-128
        B,N,_ = ray.size()
        #---------------------SCOPE---------------------#
        q_h = self.q_h_his[-1]
        #q_h: B-32
        q_h_n = self.q_rnn(bg_enc,q_h)
        #q_h_n: B-32
        self.q_h_his.append(q_h_n)
        z,q_z = to_gaussian(self.q_z_net(q_h_n))
        self.z_his.append(z)
        #z: B-32
        surface_num = N*torch.ones([B],device=self.device).long()
        air_num = N*torch.ones([B],device=self.device).long()
        rgb,logits_air,logits_surface = self.dec.forward_bg(p_eval_surface.reshape(-1,3),surface_num,\
                                                            p_eval_air.reshape(-1,3),air_num,\
                                                            ray.reshape(-1,3),z)
        #rgb: N_surface-3
        #logits_surface: N_surface-1
        rgb_out = rgb.view(B,N,3)
        logits_surface_out = logits_surface.view(B,N,1)
        logits_air_out = logits_air.view(B,N,1)
        
        ######################PRIOR######################
        if len(self.z_his) == 1:
            p_z = Normal(torch.zeros_like(z),torch.ones_like(z))
        else:
            p_h = self.p_h_his[-1]
            #p_h: B-32
            p_h_n = self.p_rnn(self.z_his[-2],p_h)
            #p_h_n: B-32
            self.p_h_his.append(p_h_n)
            _,p_z = to_gaussian(self.p_z_net(p_h_n))
        kl_bg = KL(q_z,p_z).sum(dim=1)
        #kl_bg: B
        out = AttrDict(
            rgb=rgb_out[:,None],#B-1-N-3
            logits_surface=logits_surface_out[:,None],#B-1-N-1
            logits_air=logits_air_out[:,None],#B-1-N-1
            z=z,#B-32
            kl=kl_bg,#B,
        )
        return out

    def init_his(self,B):
        q_h0 = self.q_h0.repeat(B,1)
        #q_h0: B-32
        self.q_h_his.append(q_h0)
        p_h0 = self.p_h0.repeat(B,1)
        #p_h0: B-32
        self.p_h_his.append(p_h0)

    def free_his(self):
        self.q_h_his = []
        self.p_h_his = []
        self.z_his = []
