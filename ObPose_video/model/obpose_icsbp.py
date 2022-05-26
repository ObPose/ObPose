import torch
from torch import nn
from modules.networks import ClusterEncoderKPConv,PointEncoderKPConv,MLP
from modules.point_transformer_modules import UpSample
from modules.decoder_icsbp import Nerf
from modules.object_icsbp import ObjectModule

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
        self.camera_init()
        #encoding
        self.colour_dim = 8
        self.feature_dim = 32
        self.K = 5
        sigma_init = 1.0 / (self.K*np.log(2))
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        self.p_enc = PointEncoderKPConv()
        self.colour_head = MLP(16,32,self.colour_dim)
        self.feature_head = MLP(16,32,self.feature_dim)
        self.position_net = nn.Linear(3,16,bias=False)
        #decoder and modules
        self.nerf = Nerf(device)
        self.om = ObjectModule(self.nerf,args,device)

    def forward(self,pcd,rgb,depth):
        #-------------------- INPUT --------------------#
        #pcd: B-128-128-3
        #rgb: B-128-128-3
        #depth: B-128-128
        #-------------------- INIT --------------------#
        timer = time.time()
        self.B = pcd.shape[0]
        pcd = pcd.to(self.device).float().view(self.B,128**2,3)
        rgb = rgb.to(self.device).float().view(self.B,128**2,3)
        depth = depth.to(self.device).float()
        loss_dict,out_dict = self.get_init()
        #-------------------- INFERENCE --------------------#
        inpt = torch.cat([pcd,rgb],dim=2).permute(0,2,1).contiguous()
        #inpt: B-6-N
        pcd2,enc,scene_enc = self.p_enc(pcd,inpt)
        #enc: B-N2-32
        #scene_enc: B-128
        #pcd2: B-N2-3
        enc_colour = enc + self.position_net(pcd2)
        #enc_colour: B-N2-32
        colour_embd = self.colour_head(enc_colour)
        #colour_embd: B-N2-8
        feature_embd = self.feature_head(enc)
        #feature_embd: B-N2-32
        #-------------------- SAMPLE QUERY POINTS --------------------#
        p_eval_surface,p_eval_air,ray,importance_air = self.sample_query_points(depth)
        #p_eval_surface/air: B-N-3
        #ray: B-N-3
        #importance_air: B-N-1
        #-------------------- OBJECT --------------------#
        att_obj = self.obj_att(colour_embd)
        #att_obj: B-K-N2
        feature_obj = feature_embd[:,None].repeat(1,self.K,1,1)*att_obj[:,:,:,None]
        #feature_obj: B-K-N2-32
        slots = feature_obj.sum(dim=2)/(att_obj.sum(dim=2)+1e-8)[:,:,None]
        #slots: B-K-32
        obj_i = self.om(p_eval_surface,p_eval_air,ray,\
                        slots)
        #-------------------- COMPOSE --------------------#
        sigma_air,sigma_surface = self.eval_sigma(obj_i.logits_air,obj_i.logits_surface)
        #-------------------- LOSS --------------------#
        rgb_loss,sigma_air_loss,sigma_surface_loss = self.compute_loss(rgb,obj_i.rgb,\
                                                                               sigma_surface,sigma_air,\
                                                                               importance_air)
        #log
        loss_dict.rgb_loss = rgb_loss.mean()
        loss_dict.sigma_air_loss = sigma_air_loss.mean()
        loss_dict.sigma_surface_loss = sigma_surface_loss.mean()
        loss_dict.kl_obj_what = obj_i.kl_what.mean()
        #output
        out_dict_t = AttrDict()
        out_dict_t.obj_i = obj_i
        out_dict_t.sigma = (sigma_surface,sigma_air)
        out_dict_t.att_obj = att_obj
        out_dict_t.pcd2 = pcd2
        out_dict.append(out_dict_t)
        loss_dict.sum = loss_dict.rgb_loss+loss_dict.sigma_air_loss+loss_dict.sigma_surface_loss+\
                        loss_dict.kl_obj_what
        return loss_dict,out_dict

    def camera_init(self):
        misc = np.load(CUR_PATH+'/../constants/camera_info.npy',allow_pickle=True).item()
        cam_loc=torch.from_numpy(misc['camera_extrinsics']).to(self.device)
        #cam_loc: 4-4
        cam_in=torch.from_numpy(misc['camera_intrinsics']).to(self.device)
        #cam_in: 3-3
        self.trans = cam_loc[:3, 3]
        #trans: 3
        rot = cam_loc[:3,:3]
        #rot: 3-3
        cam_ex = loc2extr(self.trans,rot)
        #cam_ex: 3-4
        cam_proj = cam_in@cam_ex
        #cam_proj: 3-4
        self.cam_proj_homo = torch.cat([cam_proj,torch.cuda.FloatTensor([[0, 0, 0, 1]])],dim=0)
        #cam_proj_homo: 4-4
        self.cam_proj_inv = torch.inverse(self.cam_proj_homo)[0:3]
        #cam_proj_inv: 3-4

    def sample_query_points(self,depth):
        res = 128
        B=self.B
        pixels = torch.meshgrid(torch.arange(0,res,device=self.device),torch.arange(0,res,device=self.device))
        pixels = torch.stack([pixels[0],pixels[1],torch.ones([res,res],device=self.device)],dim=-1).long()
        pixels = pixels.transpose(0,1)[None].repeat(B,1,1,1)
        #pixels: B-128-128-3
        #surface points
        delta = 0.01 # object surface thickness 0.01
        d_surface = depth+delta*torch.rand(B,res,res,device=self.device)
        #d_surface: B-128-128
        pixels_surface = pixels*d_surface[:,:,:,None]
        #pixels_surface: B-128-128-3
        pixels_surface = pixels_surface.reshape(B,-1,3).transpose(2,1)
        #pixels_surface: B-3-128x128
        pixels_homo_surface = torch.cat([pixels_surface,torch.ones([B,1,res*res],device=self.device)],dim=1)
        #pixels_homo_surface: B-4-128x128
        cam_proj_inv_batched = self.cam_proj_inv[None].repeat(B,1,1)
        #cam_proj_inv_batched: B-3-4
        pixels_surface = cam_proj_inv_batched@pixels_homo_surface
        pixels_surface = pixels_surface.permute(0,2,1)
        #pixels_surface: B-128x128-3
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
        pixels_homo_air = torch.cat([pixels_air,torch.ones([B,1,res*res],device=self.device)],dim=1)
        #pixels_homo_air: B-4-128x128
        pixels_air = cam_proj_inv_batched@pixels_homo_air
        pixels_air = pixels_air.permute(0,2,1)
        #pixels_air: B-128x128-3
        ray = pixels_surface - self.trans
        #ray: B-128x128-3
        ray = ray/torch.norm(ray, dim=-1,keepdim=True)
        #ray: B-128x128-3
        return pixels_surface,pixels_air,ray,importance_air

    def eval_sigma(self,obj_logits_air,obj_logits_surface):
        #obj_logits_surface/air: B-K-N-1
        B = obj_logits_air.shape[0]
        sigma_logits_surface = obj_logits_surface
        #sigma_logits_surface: B-K-N-1
        sigma_joint_surface = torch.tanh(torch.sum(F.softplus(sigma_logits_surface),dim=1,keepdims=True))
        #sigma_joint_surface: B-1-N-1
        sigma_share_surface = torch.softmax(sigma_logits_surface,dim=1)
        #sigma_share_surface: B-K-N-1
        sigma_surface = sigma_joint_surface*sigma_share_surface
        #sigma_surface: B-K-N-1
        sigma_logits_air = obj_logits_air
        #sigma_logits_air: B-K-N-1
        sigma_joint_air = torch.tanh(torch.sum(F.softplus(sigma_logits_air),dim=1,keepdims=True))
        #sigma_joint_air: B-1-N-1
        return sigma_joint_air,sigma_surface

    def compute_loss(self,rgb,obj_rgb,\
                     sigma_surface,sigma_air,\
                     obj_importance):
        #rgb: B-N-3
        #obj_rgb: B-K-N-3
        #sigma_surface: B-K-N-1
        #sigma_air: B-1-N-1
        #obj_importance: B-N-1
        B = rgb.shape[0]
        K_all = obj_rgb.shape[1]
        rgb_recon = obj_rgb
        #rgb_recon: B-K-N-3
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

    def obj_att(self,colour_embd):
        #colour_embd: B-N-D
        B,N,D= colour_embd.size()
        log_s = torch.zeros([B,N],device=self.device)
        #log_s: B-N
        attn_mask = torch.zeros([B,self.K,N],device=self.device)
        for k in range(self.K-1):
            # Determine seed
            scope = log_s.exp()
            #scope: B-N
            rand_pixel = torch.empty(B,N,device=attn_mask.device)
            rand_pixel = rand_pixel.uniform_()
            #rand_pixel: B-N
            pixel_probs = rand_pixel*scope
            #pixel_probs: B-N
            rand_max = pixel_probs.argmax(dim=1,keepdim=True)
            #rand_max: B-1
            seed = torch.gather(colour_embd,1,rand_max[:,:,None].repeat(1,1,D))
            #seed: B-1-D
            distance = ((colour_embd-seed)**2).sum(dim=2)
            #distance: B-N
            alpha = torch.exp(- distance / self.log_sigma.exp())
            #alpha: B-N
            alpha = clamp_preserve_gradients(alpha,1e-6,1.-1e-6)
            # SBP update
            log_a = torch.log(alpha)
            #log_a: B-N
            log_neg_a = torch.log(1 - alpha)
            #log_neg_a: B-N
            log_m = log_s + log_a
            #log_m: B-N
            attn_mask[:,k] = log_m.exp()
            #attn_mask: B-K-N
            log_s = log_s + log_neg_a
            #log_s: B-N
        attn_mask[:,-1] = log_s.exp()
        return attn_mask

    def get_init(self):
        loss_dict = AttrDict()
        loss_dict.rgb_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.sigma_air_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.sigma_surface_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.kl_obj_what = torch.zeros([1],device=self.device).sum()
        #output
        out_dict = []
        return loss_dict,out_dict
