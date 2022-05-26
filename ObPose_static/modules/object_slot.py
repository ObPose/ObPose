import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.kl import kl_divergence as KL
from modules.networks import MLP,ClusterEncoderKPConv
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
        self.decoder = decoder

    def forward(self,p_eval_surface,p_eval_air,ray,slots):
        #p_eval_surface/air,ray: B-N-3
        #slots: B-K-D
        B,N,_ = p_eval_surface.shape 
        #-------------------- DECODE --------------------
        surface_num = (N*torch.ones([B,self.K],device=self.device).long()).view(-1)
        air_num = surface_num.clone()
        p_eval_surface = p_eval_surface[:,None].repeat(1,self.K,1,1).reshape(-1,3)
        p_eval_air = p_eval_air[:,None].repeat(1,self.K,1,1).reshape(-1,3)
        ray = ray[:,None].repeat(1,self.K,1,1).reshape(-1,3)
        #object
        rgb,logits_air,logits_surface = self.decoder(p_eval_surface,surface_num,\
                                                     p_eval_air,air_num,\
                                                     ray,slots)
        #rgb: N_surface-3
        #logits_surface: N_surface-1
        #logits_air: B-K-128x128-1
        rgb_out = rgb.view(B,self.K,N,3)
        logits_surface_out = logits_surface.view(B,self.K,N,1)
        logits_air_out = logits_air.view(B,self.K,N,1)

        #-------------------- DECODE --------------------
        #outputs
        out = AttrDict(
            rgb=rgb_out,#B-K-N-3
            logits_surface=logits_surface_out,#B-K-N-1
            logits_air=logits_air_out,#B-K-N-1
        )
        return out
