import torch
from torch import nn
from modules.networks import ClusterEncoderKPConv,PointEncoderKPConv,MLP
from modules.point_transformer_modules import UpSample
from modules.decoder_slot import Nerf
from modules.background_slot import Background
from modules.object_slot import ObjectModule

import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from utils.funcs import loc2extr,clamp_preserve_gradients

import os
import pdb
import time
import numpy as np
from attrdict import AttrDict
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

class ObPose(nn.Module):
    def __init__(self,device,args):
        super(ObPose, self).__init__()
        self.device = device
        self.args = args
        #encoding
        self.colour_dim = 8
        self.feature_dim = 32
        self.K = 4
        self.p_enc = PointEncoderKPConv()
        self.feature_head = MLP(16,32,self.feature_dim)
        self.position_net = nn.Linear(3,32,bias=False)
        self.enc_norm = torch.nn.LayerNorm(self.feature_dim)
        self.enc_mlp = MLP(self.feature_dim,self.feature_dim,self.feature_dim)
        #slot att.
        enc_dim = self.feature_dim
        self.inpt_norm = torch.nn.LayerNorm(enc_dim)
        self.slot_norm_fg = torch.nn.LayerNorm(enc_dim)
        self.res_norm_fg = torch.nn.LayerNorm(enc_dim)
        self.slot_norm_bg = torch.nn.LayerNorm(enc_dim)
        self.res_norm_bg = torch.nn.LayerNorm(enc_dim)
        self.slots_mu_fg = nn.init.xavier_uniform_(torch.zeros([1,1,enc_dim],device=self.device))
        self.slots_log_sigma_fg = nn.init.xavier_uniform_(torch.zeros([1,1,enc_dim],device=self.device))
        self.slots_mu_bg = nn.init.xavier_uniform_(torch.zeros([1,1,enc_dim],device=self.device))
        self.slots_log_sigma_bg = nn.init.xavier_uniform_(torch.zeros([1,1,enc_dim],device=self.device))
        self.project_k = nn.Linear(enc_dim,enc_dim)
        self.project_q_fg = nn.Linear(enc_dim,enc_dim)
        self.project_q_bg = nn.Linear(enc_dim,enc_dim)
        self.project_v_fg = nn.Linear(enc_dim,enc_dim)
        self.project_v_bg = nn.Linear(enc_dim,enc_dim)
        self.res_mlp_fg = MLP(enc_dim,enc_dim,enc_dim)
        self.slot_att_gru_fg = nn.GRUCell(enc_dim,enc_dim)
        self.res_mlp_bg = MLP(enc_dim,enc_dim,enc_dim)
        self.slot_att_gru_bg = nn.GRUCell(enc_dim,enc_dim)
        #decoder and modules
        self.nerf = Nerf(device)
        self.nerf_bg = Nerf(device)
        self.om = ObjectModule(self.nerf,args,device)
        self.bg = Background(self.nerf_bg,device)

    def forward(self,pcd,rgb,depth,camera_in,camera_ex):
        #-------------------- INPUT --------------------#
        #pcd: B-128-128-3
        #rgb: B-128-128-3
        #depth: B-128-128
        #camera_in: B-3-3
        #camera_ex: B-4-4
        #-------------------- INIT --------------------#
        timer = time.time()
        self.B = pcd.shape[0]
        pcd = pcd.to(self.device).float().view(self.B,128**2,3)
        rgb = rgb.to(self.device).float().view(self.B,128**2,3)
        depth = depth.to(self.device).float()
        camera_in = camera_in.to(self.device).float()
        camera_ex = camera_ex.to(self.device).float()
        loss_dict,out_dict = self.get_init()
        #-------------------- INFERENCE --------------------#
        inpt = torch.cat([pcd,rgb],dim=2).permute(0,2,1).contiguous()
        #inpt: B-6-N
        pcd2,enc,_ = self.p_enc(pcd,inpt)
        #enc: B-N2-D
        #pcd2: B-N2-3
        enc = self.feature_head(enc)+self.position_net(pcd2)
        #enc: B-N2-D
        enc = self.enc_mlp(self.enc_norm(enc))
        #enc: B-N2-D
        slots,bg_slots = self.slot_att(enc)
        #slots: B-K-D
        #bg_slots: B-D
        #-------------------- SAMPLE QUERY POINTS --------------------#
        camera_posi,cam_proj_inv,cam_proj_homo = self.compute_camera(camera_in,camera_ex)
        #camera_posi: B-3
        #cam_proj_inv: B-3-4
        #cam_proj_homo: B-4-4
        p_eval_surface,p_eval_air,ray,importance_air = self.sample_query_points(depth,cam_proj_inv,camera_posi)
        #p_eval_surface/air: B-N-3
        #ray: B-N-3
        #importance_air: B-N-1
        #-------------------- BACKGROUND --------------------#
        bg_i = self.bg(p_eval_surface,p_eval_air,ray,bg_slots)
        #-------------------- OBJECT --------------------#
        obj_i = self.om(p_eval_surface,p_eval_air,ray,\
                        slots)
        #-------------------- COMPOSE --------------------#
        sigma_air,sigma_surface = self.eval_sigma(obj_i.logits_air,obj_i.logits_surface,\
                                                  bg_i.logits_air,bg_i.logits_surface)
        #-------------------- LOSS --------------------#
        rgb_loss,sigma_air_loss,sigma_surface_loss = self.compute_loss(rgb,obj_i.rgb,bg_i.rgb,\
                                                                               sigma_surface,sigma_air,\
                                                                               importance_air)
        #log
        loss_dict.rgb_loss = rgb_loss.mean()
        loss_dict.sigma_air_loss = sigma_air_loss.mean()
        loss_dict.sigma_surface_loss = sigma_surface_loss.mean()
        #output
        out_dict_t = AttrDict()
        out_dict_t.bg_i = bg_i
        out_dict_t.obj_i = obj_i
        out_dict_t.sigma = (sigma_surface,sigma_air)
        out_dict_t.pcd2 = pcd2
        out_dict.append(out_dict_t)
        loss_dict.sum = loss_dict.rgb_loss+loss_dict.sigma_air_loss+loss_dict.sigma_surface_loss
        return loss_dict,out_dict

    def compute_camera(self,camera_in,camera_ex):
        #camera_in: B-3-3
        #camera_ex: B-4-4
        B = camera_in.shape[0]
        camera_posi = camera_ex[:,:3, 3]
        #camera_posi: B-3
        rot = camera_ex[:,:3,:3]
        #rot: B-3-3
        cam_ex = loc2extr(camera_posi,rot)
        #cam_ex: B-3-4
        cam_proj = camera_in@cam_ex
        #cam_proj: B-3-4
        cam_proj_homo = torch.zeros([B,4,4],device=self.device)
        #cam_proj_homo: B-4-4
        cam_proj_homo[:,3,3] = 1.
        #cam_proj_homo: B-4-4
        cam_proj_homo[:,:3] = cam_proj
        #cam_proj_homo: B-4-4
        cam_proj_inv = torch.inverse(cam_proj_homo)[:,0:3]
        #cam_proj_inv: B-3-4
        return camera_posi,cam_proj_inv,cam_proj_homo

    def sample_query_points(self,depth,camera_proj_inv,cam_posi):
        #depth: B-128-128
        #camera_proj_inv: B-3-4
        #cam_posi: B-3
        B = depth.shape[0]
        res = depth.shape[1]
        pixels = torch.meshgrid(torch.arange(0,res,device=self.device),torch.arange(0,res,device=self.device))
        pixels = torch.stack([pixels[0],pixels[1],torch.ones([res,res],device=self.device)],dim=-1).long()
        pixels = pixels.transpose(0,1)[None].repeat(B,1,1,1)
        #pixels: B-128-128-3
        #surface points
        delta = 0.01 # object surface thickness
        d_surface = depth+delta*torch.rand(B,res,res,device=self.device)
        #d_surface: B-128-128
        pixels_surface = pixels*d_surface[:,:,:,None]
        #pixels_surface: B-128-128-3
        pixels_surface = pixels_surface.reshape(B,-1,3).transpose(2,1)
        #pixels_surface: B-3-128x128
        pixels_homo = torch.cat([pixels_surface,torch.ones([B,1,res*res],device=self.device)],dim=1)
        #pixels_homo: B-4-128x128
        pixels_surface = camera_proj_inv@pixels_homo
        pixels_surface = pixels_surface.permute(0,2,1)
        #pixels_surface: B-128x128-3
        #air points
        near_far = torch.randint(0,2,(B,res,res),device=self.device)#0:near; 1:far
        #near_far: B-128-128
        d_air = near_far*depth*0.98*torch.rand(B,res,res,device=self.device) + (1.-near_far)*(depth*0.98+depth*0.02*torch.rand(B,res,res,device=self.device))
        #depth_air: B-128-128
        importance_air = near_far*0.5*(1./(0.98*depth)) + (1.-near_far)*0.5*(1./(0.02*depth))
        #importance_air: B-128-128
        importance_air = importance_air.view(B,res*res,1)
        #importance_air: B-128x128-1
        pixels_air = pixels*d_air[:,:,:,None]
        #pixels_air: B-128-128-3
        pixels_air = pixels_air.reshape(B,-1,3).transpose(2,1)
        #pixels_air: B-3-128x128
        pixels_air_homo = torch.cat([pixels_air,torch.ones([B,1,res*res],device=self.device)],dim=1)
        #pixels_air_homo: B-4-128x128
        pixels_air = camera_proj_inv@pixels_air_homo
        pixels_air = pixels_air.permute(0,2,1)
        #pixels_air: B-128x128-3
        #ray
        ray = pixels_surface - cam_posi[:,None]
        #ray: B-128x128-3
        ray = ray/torch.norm(ray, dim=-1,keepdim=True)
        #ray: B-128x128-3
        return pixels_surface,pixels_air,ray,importance_air

    def eval_sigma(self,obj_logits_air,obj_logits_surface,\
                   bg_logits_air,bg_logits_surface):
        #obj_logits_surface/air: B-K-N-1
        #bg_logits_surface/air: B-1-N-1
        B = obj_logits_air.shape[0]
        sigma_logits_surface = torch.cat([bg_logits_surface,obj_logits_surface],dim=1)
        #sigma_logits_surface: B-K_all-N-1
        sigma_joint_surface = torch.tanh(torch.sum(F.softplus(sigma_logits_surface),dim=1,keepdims=True))
        #sigma_joint_surface: B-1-N-1
        sigma_share_surface = torch.softmax(sigma_logits_surface,dim=1)
        #sigma_share_surface: B-K_all-N-1
        sigma_surface = sigma_joint_surface*sigma_share_surface
        #sigma_surface: B-K_all-N-1
        sigma_logits_air = torch.cat([bg_logits_air,obj_logits_air],dim=1)
        #sigma_logits_air: B-K_all-N-1
        sigma_joint_air = torch.tanh(torch.sum(F.softplus(sigma_logits_air),dim=1,keepdims=True))
        #sigma_joint_air: B-1-N-1
        return sigma_joint_air,sigma_surface

    def compute_loss(self,rgb,obj_rgb,bg_rgb,\
                     sigma_surface,sigma_air,\
                     obj_importance):
        #rgb: B-N-3
        #obj/bg_rgb: B-/K/1-N-3
        #sigma_surface: B-K_all-N-1
        #sigma_air: B-1-N-1
        #obj_importance: B-N-1
        B = rgb.shape[0]
        K_all = obj_rgb.shape[1]+1
        rgb_recon = torch.cat([bg_rgb,obj_rgb],dim=1)
        #rgb_recon: B-K_all-N-3
        distr_rgb = Normal(rgb_recon,0.1)
        rgb = rgb[:,None].repeat(1,K_all,1,1)
        #rgb: B-N-3 -> B-1-N-3 -> B-K_all-N-3
        lp_rgb = distr_rgb.log_prob(rgb)
        #lp_rgb: B-K_all-N-3
        log_mx = torch.log(sigma_surface+1e-8)+lp_rgb
        #log_mx: B-K_all-N-3
        rgb_loss = -torch.log(log_mx.exp().sum(dim=1)+1e-8)
        #rgb_loss: B-N-3
        rgb_loss = rgb_loss.sum(dim=(1,2))
        #rgb_loss: B
        ##sigma loss
        importance_weight = obj_importance[:,None]
        #importance_weight: B-1-N-1
        lp_sigma_air = 10.*sigma_air/importance_weight
        #lp_sigma_air: B-1-N-1
        sigma_air_loss = lp_sigma_air.sum()/B
        #sigma_air_loss: 1
        lp_sigma_surface = -torch.log(10.*sigma_surface.sum(dim=1)+1e-8)
        #lp_sigma_surface: B-N-1
        sigma_surface_loss = lp_sigma_surface.sum(dim=(1,2))
        #sigma_surface_loss: B
        return rgb_loss,sigma_air_loss,sigma_surface_loss

    def slot_att(self,embds):
        #embds: B-N-D
        B,N,D = embds.size()
        inpt_embds = self.inpt_norm(embds)
        k = self.project_k(inpt_embds)
        #k: B-N-D
        fg_v = self.project_v_fg(inpt_embds)
        #fg_v: B-N-D
        bg_v = self.project_v_bg(inpt_embds)
        #bg_v: B-N-D
        fg_slots_init = torch.randn((B,self.K,D),device=self.device)
        fg_slots = self.slots_mu_fg+self.slots_log_sigma_fg.exp()*fg_slots_init
        #fg_slots: B-K-D
        bg_slots_init = torch.randn((B,1,D),device=self.device)
        bg_slots = self.slots_mu_bg+self.slots_log_sigma_bg.exp()*bg_slots_init
        #bg_slots: B-1-D
        for _ in range(3):
            fg_slots_prev = fg_slots
            #fg_slots_prev: B-K-D
            bg_slots_prev = bg_slots
            #bg_slots_prev: B-1-D
            fg_slots = self.slot_norm_fg(fg_slots)
            #fg_slots: B-K-D
            bg_slots = self.slot_norm_bg(bg_slots)
            #bg_slots: B-1-D
            fg_q = self.project_q_fg(fg_slots).permute(0,2,1)
            #fg_q: B-K-D -> B-D-K
            bg_q = self.project_q_bg(bg_slots).permute(0,2,1)
            #bg_q: B-1-D -> B-D-1
            q = torch.cat([bg_q,fg_q],dim=2)
            #q: B-D-K+1
            attn_logits = k@q*(D**-0.5)
            #attn_logits: B-N-K+1
            attn = F.softmax(attn_logits,dim=2)
            #attn: B-N-K+1
            attn = attn + 1e-8
            attn_v = attn/attn.sum(dim=1,keepdim=True)
            attn_v = attn_v.permute(0,2,1)
            #attn_v: B-N-K+1 -> B-K+1-N
            #fg/bg_v: B-N-D
            fg_updates = attn_v[:,1:] @ fg_v
            #fg_updates: B-K-D
            bg_updates = attn_v[:,:1] @ bg_v
            #bg_updates: B-1-D
            fg_slots = self.slot_att_gru_fg(fg_updates.view(-1,D), fg_slots_prev.view(-1,D)).view(B,self.K,D)
            #fg_slots: B-K-D
            bg_slots = self.slot_att_gru_bg(bg_updates[:,0], bg_slots_prev[:,0]).view(B,1,D)
            #bg_slots: B-1-D
            fg_slots = fg_slots+self.res_mlp_fg(self.res_norm_fg(fg_slots))
            #fg_slots: B-K-D
            bg_slots = bg_slots+self.res_mlp_bg(self.res_norm_bg(bg_slots))
            #bg_slots: B-1-D
        return fg_slots,bg_slots[:,0]

    def get_init(self):
        loss_dict = AttrDict()
        loss_dict.rgb_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.sigma_air_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.sigma_surface_loss = torch.zeros([1],device=self.device).sum()
        #output
        out_dict = []
        return loss_dict,out_dict
