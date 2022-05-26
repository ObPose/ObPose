import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as KL
from modules.networks import MLP
from attrdict import AttrDict
from utils.funcs import to_gaussian

import os
import numpy as np
import pdb
import time
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
class ObjectModule(nn.Module):
    def __init__(self,decoder,args,device):
        super(ObjectModule, self).__init__()
        self.args = args
        self.device = device
        self.K = 4
        #inference
        ##what
        self.q_what_rnn = nn.GRUCell(32,32)
        self.q_z_what_net = MLP(32,32,32*2)
        self.q_h_what_his = []
        self.z_what_his = []
        self.decoder = decoder
        #prior
        ##what
        self.p_h_what_his = []
        self.p_what_rnn = nn.GRUCell(32,32)
        self.p_z_what_net = MLP(32,32,32*2)


    def forward(self,p_eval_surface,p_eval_air,ray,slots):
        #p_eval_surface/air,ray: B-N-3
        B,N,_ = p_eval_surface.shape      
        #-------------------- ENCODE WHAT --------------------
        q_h_what = self.q_h_what_his[-1]
        #q_h_what: BxK-32
        q_h_what_n = self.q_what_rnn(slots.view(-1,32),q_h_what)
        #q_h_what_n: BxK-32
        self.q_h_what_his.append(q_h_what_n)
        z_what,q_z_what = to_gaussian(self.q_z_what_net(q_h_what_n.view(B,self.K,-1)))
        #z_what: B-K-32
        self.z_what_his.append(z_what)
        if len(self.z_what_his) == 1:
            p_z_what = Normal(torch.zeros_like(z_what),torch.ones_like(z_what))
        else:
            p_h_what = self.p_h_what_his[-1]
            #p_h_what: BxK-32
            p_h_what_n = self.p_what_rnn(self.z_what_his[-2].view(-1,32),p_h_what)
            #p_h_what_n: BxK-32
            self.p_h_what_his.append(p_h_what_n)
            _,p_z_what = to_gaussian(self.p_z_what_net(p_h_what_n.view(B,self.K,-1)))
        kl_what = KL(q_z_what,p_z_what).sum(dim=(1,2))
        #kl_what: B
        #-------------------- DECODE --------------------
        decode_time = time.time()
        #object
        surface_num = (N*torch.ones([B,self.K],device=self.device).long()).view(-1)
        air_num = surface_num.clone()
        p_eval_surface = p_eval_surface[:,None].repeat(1,self.K,1,1).reshape(-1,3)
        p_eval_air = p_eval_air[:,None].repeat(1,self.K,1,1).reshape(-1,3)
        ray = ray[:,None].repeat(1,self.K,1,1).reshape(-1,3)
        rgb,logits_air,logits_surface = self.decoder(p_eval_surface,surface_num,\
                                                     p_eval_air,air_num,
                                                     ray,z_what)
        #rgb: N_surface-3
        #logits_surface: N_surface-1
        #logits_air: N_air-1
        rgb_out = rgb.view(B,self.K,N,3)
        logits_surface_out = logits_surface.view(B,self.K,N,1)
        logits_air_out = logits_air.view(B,self.K,N,1)
        #-------------------- DECODE --------------------
        #outputs
        out = AttrDict(
            rgb=rgb_out,#B-K-N-3
            logits_surface=logits_surface_out,#B-K-N-1
            logits_air=logits_air_out,#B-K-N-1
            z_what=z_what,#B-K-C
            kl_what=kl_what,#B
        )
        return out

    def init_his(self,B):
        #what
        q_h0_what = torch.zeros([B*self.K,32],device=self.device)
        self.q_h_what_his.append(q_h0_what)
        #q_h0_what: BxK-32
        p_h0_what = torch.zeros([B*self.K,32],device=self.device)
        self.p_h_what_his.append(p_h0_what)
        #p_h0_what: BxK-32

    def free_his(self):
        #what
        self.q_h_what_his = []
        self.p_h_what_his = []
        self.z_what_his = []
