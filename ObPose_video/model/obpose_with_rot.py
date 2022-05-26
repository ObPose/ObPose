import torch
from torch import nn
from modules.networks import ClusterEncoderKPConv,PointEncoderKPConv,MLP
from modules.point_transformer_modules import UpSample
from modules.decoder import Nerf
from modules.background import Background
from modules.object_with_rot import ObjectModule

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
        self.camera_init()
        self.args = args
        #encoding
        self.colour_dim = 8
        self.feature_dim = 16
        self.K = 4
        sigma_init = 1.0 / (self.K*np.log(2))
        self.log_sigma = nn.Parameter(torch.tensor(sigma_init).log())
        self.p_enc = PointEncoderKPConv()
        self.what_encoder = ClusterEncoderKPConv()
        self.colour_head = MLP(16,32,self.colour_dim)
        self.feature_head = MLP(16,32,self.feature_dim)
        self.position_net = nn.Linear(3,16,bias=False)
        self.bg_rob_obj_head = MLP(self.colour_dim,32,2)
        self.up_sample_layer = UpSample()
        #decoder and modules
        self.nerf = Nerf(self.cam_proj_homo[:3],device)
        self.nerf_bg = Nerf(self.cam_proj_homo[:3],device)
        self.om = ObjectModule(self.what_encoder,self.nerf,args,device)
        self.bg = Background(self.nerf_bg,device)

    def forward(self,pcd_video,rgb_video,depth_video):
        #-------------------- INPUT --------------------#
        #pcd_video: B-L-128-128-3
        #rgb_video: B-L-128-128-3
        #depth_video: B-L-128-128
        #-------------------- INIT --------------------#
        self.B = pcd_video.shape[0]
        L = pcd_video.shape[1]
        pcd_video = pcd_video.to(self.device).float()
        rgb_video = rgb_video.to(self.device).float()
        depth_video = depth_video.to(self.device).float()
        loss_dict,out_dict = self.get_init()
        #-------------------- INFERENCE --------------------#
        for t in range(L):
            encode_time = time.time()
            pcd = pcd_video[:,t].reshape(self.B,-1,3).contiguous()
            #pcd: B-N-3
            rgb = rgb_video[:,t].reshape(self.B,-1,3).contiguous()
            #rgb: B-N-3
            depth = depth_video[:,t]
            #depth: B-128-128
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
            bg_rob_obj_prob = F.softmax(self.bg_rob_obj_head(colour_embd),dim=2).view(self.B,-1,2)
            #bg_rob_obj_prob: B-N2-2
            #-------------------- SAMPLE QUERY POINTS --------------------#
            p_eval_surface,p_eval_air,ray,importance_air = self.sample_query_points(depth)
            #p_eval_surface/air: B-N-3
            #ray: B-N-3
            #importance_air: B-N-1
            #-------------------- BACKGROUND --------------------#
            bg_i = self.bg(p_eval_surface,p_eval_air,ray,scene_enc)
            #-------------------- OBJECT --------------------#
            mask_obj = bg_rob_obj_prob[:,:,-1]
            #mask_obj: B-N2
            if t == 0:
                idle_states = torch.zeros([self.B,self.K],device=self.device)
                obj_seeds = torch.zeros([self.B,self.K,self.colour_dim],device=self.device)
            att_obj,remain_mask,obj_seeds,idle_states = self.obj_att(colour_embd,mask_obj,obj_seeds,idle_states,t)
            #att_obj: B-K-N2
            #remain_mask: B-N2
            #obj_seeds: B-K-D
            #idle_states: B-K
            if self.args.detach: 
                feature_obj = feature_embd[:,None].repeat(1,self.K,1,1)*att_obj[:,:,:,None].detach()
                #feature_obj: B-K-N2-32
            else:
                feature_obj = feature_embd[:,None].repeat(1,self.K,1,1)*att_obj[:,:,:,None]
                #feature_obj: B-K-N2-32
            obj_i = self.om(pcd2,\
                            p_eval_surface,p_eval_air,ray,\
                            feature_obj,att_obj[:,:,:,None],depth,bg_rob_obj_prob,idle_states,t)
            #-------------------- COMPOSE --------------------#
            sigma_air,sigma_surface = self.eval_sigma(obj_i.logits_air,obj_i.logits_surface,\
                                                      bg_i.logits_air,bg_i.logits_surface)
            #-------------------- LOSS --------------------#
            rgb_loss,sigma_air_loss,sigma_surface_loss,ic_loss = self.compute_loss(rgb,obj_i.rgb,bg_i.rgb,\
                                                                                   sigma_surface,sigma_air,\
                                                                                   importance_air,\
                                                                                   bg_rob_obj_prob,att_obj,pcd,pcd2,remain_mask)
            #log
            loss_dict.rgb_loss = loss_dict.rgb_loss+rgb_loss.mean()/L
            loss_dict.sigma_air_loss = loss_dict.sigma_air_loss+sigma_air_loss.mean()/L
            loss_dict.sigma_surface_loss = loss_dict.sigma_surface_loss+sigma_surface_loss.mean()/L
            loss_dict.kl_obj_what = loss_dict.kl_obj_what+obj_i.kl_what.mean()/L
            loss_dict.kl_obj_where = loss_dict.kl_obj_where+obj_i.kl_where.mean()/L
            loss_dict.kl_bg = loss_dict.kl_bg+bg_i.kl.mean()/L
            loss_dict.ic_loss = loss_dict.ic_loss+ic_loss.mean()/L
            loss_dict.trans_loss = loss_dict.trans_loss + obj_i.trans_loss.mean()/L
            #output
            out_dict_t = AttrDict()
            out_dict_t.bg_i = bg_i
            out_dict_t.obj_i = obj_i
            out_dict_t.sigma = (sigma_surface,sigma_air)
            out_dict_t.bg_rob_obj_prob = bg_rob_obj_prob
            out_dict_t.att_obj = att_obj
            out_dict_t.pcd2 = pcd2
            out_dict_t.p_eval_surface = p_eval_surface
            out_dict.append(out_dict_t)
            #print("--- %s seconds --- one step time" % (time.time() - encode_time))
        loss_dict.sum = loss_dict.rgb_loss+loss_dict.sigma_air_loss+loss_dict.sigma_surface_loss+\
                        loss_dict.kl_obj_what+loss_dict.kl_obj_where+loss_dict.kl_bg+\
                        loss_dict.ic_loss+loss_dict.trans_loss
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

    def eval_sigma(self,obj_logits_air,obj_logits_surface,\
                   bg_logits_air,bg_logits_surface):
        #obj_logits_surface: B-K-N-1
        #bg_logits_surface/air: B-1-N-1
        #obj_logits_air: B-1-N-1
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
                     importance,
                     bg_rob_obj_prob,att_obj,pcd,pcd2,remain_mask):
        #rgb: B-N-3
        #obj/bg_rgb: B-K_rob/K/1-N-3
        #sigma_surface/air: B-K_all-N-1
        #sigma_air: B-1-N-1
        #importance: B-N-1
        #bg_rob_obj_prob: B-N2-2
        #att_obj: B-sn-N2
        #pcd: B-N-3
        #pcd2: B-N2-3
        #remain_mask: B-N2
        B = rgb.shape[0]
        K_all = obj_rgb.shape[1]+1
        rgb_recon = torch.cat([bg_rgb,obj_rgb],dim=1)
        #rgb_recon: B-K_all-N-3
        if self.args.SG:
            sigma_surface_denom = sigma_surface.sum(dim=1,keepdim=True)
            sigma_surface_denom[sigma_surface_denom == 0] = 1e-4
            #sigma_surface_denom: B-1-N-1
            sigma_surface_normed = sigma_surface/sigma_surface_denom
            #sigma_surface_denom: B-K_all-N-1
            rgb_recon_sg = (rgb_recon*sigma_surface_normed).sum(dim=1)
            #rgb_recon_sg: B-N-3
            distr_rgb_sg = Normal(rgb_recon_sg,0.1)
            rgb_loss = -distr_rgb_sg.log_prob(rgb)
            #rgb_loss: B-N-3
        else:
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
        importance_weight = importance[:,None]
        #importance_weight: B-1-N-1
        lp_sigma_air = 10.*sigma_air/importance_weight
        #lp_sigma_air: B-1-N-1
        sigma_air_loss = lp_sigma_air.sum()/B
        #sigma_air_loss: 1
        lp_sigma_surface = -torch.log(10.*sigma_surface.sum(dim=1)+1e-8)
        #lp_sigma_surface: B-N-1
        sigma_surface_loss = lp_sigma_surface.sum()/B
        #sigma_surface_loss: 1
        ##ic loss
        cluster_prob = torch.cat([bg_rob_obj_prob[:,:,:1].permute(0,2,1),att_obj],dim=1)[:,:,:,None]
        #cluster_prob: B-K_all-N2-1
        cluster_prob = cluster_prob.reshape(B*K_all,-1,1).permute(0,2,1).contiguous()
        #cluster_prob: B-K_all-N2-1 -> B*K_all-N2-1 ->B*K_all-1-N2
        cluster_prob = self.up_sample_layer(pcd2.repeat_interleave(K_all,0),cluster_prob,pcd.repeat_interleave(K_all,0))
        #cluster_prob: B*K_all-1-N
        cluster_prob = cluster_prob.permute(0,2,1).reshape(B,K_all,-1,1)
        #cluster_prob: B*K_all-1-N -> B*K_all-N-1 -> B-K_all-N-1
        if self.args.SG:
            distr_rgb = Normal(rgb_recon,0.1)
            rgb = rgb[:,None].repeat(1,K_all,1,1)
            #rgb: B-N-3 -> B-1-N-3 -> B-K_all-N-3
            lp_rgb = distr_rgb.log_prob(rgb)
            #lp_rgb: B-K_all-N-3
        p_rgb_ic = lp_rgb.exp()*sigma_surface
        #p_rgb_ic: B-K_all-N-3
        p_sigma_ic = sigma_surface
        #p_rgb_ic: B-K_all-N-1
        rgb_mx_ic = cluster_prob*p_rgb_ic.detach()
        #rgb_mx_ic: B-K_all-N-3
        rgb_ic_loss = -torch.log(rgb_mx_ic.sum(dim=1)+1e-8)
        #rgb_ic_loss: B-N-3
        sigma_mx_ic = cluster_prob*p_sigma_ic.detach()
        #sigma_mx_ic: B-K_all-N-1
        sigma_ic_loss = -torch.log(sigma_mx_ic.sum(dim=1)+1e-8)
        #sigma_ic_loss: B-N-1
        ic_loss = (rgb_ic_loss.sum()+sigma_ic_loss.sum()+remain_mask.sum())/B
        #ic_loss: 1
        return rgb_loss,sigma_air_loss,sigma_surface_loss,ic_loss

    def obj_att(self,colour_embd,mask_obj,obj_seeds,idle_states,t):
        #colour_embd: B-N-D
        #mask_obj: B-N
        #obj_seeds: B-K-D
        #idle_states: B-K
        B,N,D= colour_embd.size()
        mask_obj = clamp_preserve_gradients(mask_obj,1e-6,1.-1e-6)
        log_s = torch.log(mask_obj)
        #log_s: B-N
        attn_mask = torch.zeros([B,self.K,N],device=self.device)
        for k in range(self.K):
            # Determine seed
            scope = log_s.exp()
            #scope: B-N
            if t == 0:
                rand_pixel = torch.empty(B,N,device=mask_obj.device)
                rand_pixel = rand_pixel.uniform_()
                #rand_pixel: B-N
                pixel_probs = rand_pixel*scope
                #pixel_probs: B-N
                rand_max = pixel_probs.argmax(dim=1,keepdim=True)
                #rand_max: B-1
                seed = torch.gather(colour_embd,1,rand_max[:,:,None].repeat(1,1,D))
                #seed: B-1-D
                obj_seeds[:,k] = seed[:,0]
                #obj_seeds: B-K-D
                scope_ber = Bernoulli(probs=scope)
                scope_bin = scope_ber.sample()
                #scope_bin: B-N
                idle_states[:,k] = scope_bin.sum(dim=1) == 0.
                #idle_states: B-K
            else:
                seed = obj_seeds[:,k][:,None]
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
            attn_mask[:,k] = log_m.exp()*(1.-idle_states[:,k][:,None])
            #attn_mask: B-K-N
            log_s = log_s + log_neg_a*(1.-idle_states[:,k][:,None])
            #log_s: B-N
        remain_mask = log_s.exp()
        #remain_mask: B-N
        return attn_mask,remain_mask,obj_seeds,idle_states

    def get_init(self):
        loss_dict = AttrDict()
        loss_dict.rgb_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.sigma_air_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.sigma_surface_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.kl_obj_what = torch.zeros([1],device=self.device).sum()
        loss_dict.kl_obj_where = torch.zeros([1],device=self.device).sum()
        loss_dict.kl_bg = torch.zeros([1],device=self.device).sum()
        loss_dict.ic_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.trans_loss = torch.zeros([1],device=self.device).sum()
        #output
        out_dict = []
        #bg
        self.bg.init_his(self.B)
        #object
        self.om.init_his(self.B)
        return loss_dict,out_dict

    def free_his(self):
        self.bg.free_his()
        self.om.free_his()
