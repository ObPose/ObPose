import torch
from torch import nn

import torch.nn.functional as F
from torch.distributions.normal import Normal
from utils.funcs import loc2extr

import os
import pdb
import time
import numpy as np
from attrdict import AttrDict
CUR_PATH = os.path.dirname(os.path.abspath(__file__))

class ObsurfFull(nn.Module):
    def __init__(self,device,args,obsurf):
        super(ObsurfFull, self).__init__()
        self.device = device
        self.args = args
        self.obsurf = obsurf
        

    def forward(self,pcd,rgb,depth,camera_in,camera_ex,iter_num):
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
        #-------------------- SAMPLE QUERY POINTS --------------------#
        camera_posi,cam_proj_inv,cam_proj_homo = self.compute_camera(camera_in,camera_ex)
        #camera_posi: B-3
        #cam_proj_inv: B-3-4
        #cam_proj_homo: B-4-4
        p_eval_surface,p_eval_air,ray,importance_air,rgb_target,ray_all = self.sample_query_points(depth,cam_proj_inv,camera_posi,rgb,self.args.ray_sample)
        #p_eval_surface/air: B-N-3
        #ray: B-N-3
        #importance_air: B-N-1
        #rgb_target: B-N-3
        #-------------------- INFERENCE --------------------#
        pixel_encoding, c=self.obsurf.encoder(rgb.permute(0,2,1).view(self.B,3,128,128),camera_posi,ray_all.view(self.B,128,128,3))

        surface_reps = self.obsurf.decode(p_eval_surface, c, rays=ray,
                                         pixel_features=pixel_encoding, camera_pos=camera_posi)
        reps = surface_reps

        value_ll = surface_reps.likelihood(rgb_target).sum(-1).mean(1)  # Sum over value dims

        empty_reps = self.obsurf.decode(p_eval_air, c, rays=ray,
                                       pixel_features=pixel_encoding, camera_pos=camera_posi)
        surface_density = surface_reps.global_presence()
        empty_density = empty_reps.global_presence()
        empty_points_weights = 1./importance_air
        empty_log_prob = (-empty_density * empty_points_weights[:,:,0])
        depth_ll = (empty_log_prob + torch.log(surface_density)).mean(1)
        
        slot_probs = reps.local_presence()
        overlap = slot_probs.sum(1) - slot_probs.max(1)[0]
        l1_penalty = (overlap * self.get_l1_factor(iter_num)).mean(1)  # Mean over points
        #log
        loss_dict.rgb_loss = -value_ll.mean()
        loss_dict.depth_loss = -depth_ll.mean()
        loss_dict.overlap_loss = l1_penalty.mean()
        #output
        out_dict_t = AttrDict()
        out_dict_t.surface_density = surface_reps.local_presence()
        out_dict_t.empty_density = empty_reps.local_presence()
        out_dict_t.rgb = surface_reps.local_values()
        out_dict.append(out_dict_t)
        loss_dict.sum = loss_dict.rgb_loss+loss_dict.depth_loss+loss_dict.overlap_loss
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

    def sample_query_points(self,depth,camera_proj_inv,cam_posi,rgb,ray_sample):
        #depth: B-128-128
        #camera_proj_inv: B-3-4
        #cam_posi: B-3
        #rgb: B-N-3
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
        if ray_sample:
            sample_indx = torch.randint(0,res**2,(B,4096),device=self.device)
            #sample_indx: B-4096
            pixels_surface = torch.gather(pixels_surface,1,sample_indx[:,:,None].repeat(1,1,3))
            #pixels_surface: B-4096-3
            pixels_air = torch.gather(pixels_air,1,sample_indx[:,:,None].repeat(1,1,3))
            #pixels_air: B-4096-3
            ray_target = torch.gather(ray,1,sample_indx[:,:,None].repeat(1,1,3))
            #ray_target: B-4096-3
            importance_air = torch.gather(importance_air,1,sample_indx[:,:,None])
            #importance_air: B-4096-1
            rgb_target = torch.gather(rgb,1,sample_indx[:,:,None].repeat(1,1,3))
            #rgb_target: B-4096-1
        else:
            ray_target = ray
            #ray_target: B-128x128-3
            rgb_target = rgb
            #rgb_target: B-128x128-1
        return pixels_surface,pixels_air,ray_target,importance_air,rgb_target,ray

    def get_annealing_factor(self, it, start, end):
        if end is not None:
            cur_factor = float(it - start) / (end - start)
            return max(min(1., cur_factor), 0.)
        else:
            return 1.

    def get_l1_factor(self, it):
        return self.get_annealing_factor(it, 20000, 40000) * 0.05

    def get_init(self):
        loss_dict = AttrDict()
        loss_dict.rgb_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.depth_loss = torch.zeros([1],device=self.device).sum()
        loss_dict.overlap_loss = torch.zeros([1],device=self.device).sum()
        #output
        out_dict = []
        return loss_dict,out_dict
