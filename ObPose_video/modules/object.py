import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.distributions.categorical import Categorical
from torch.distributions.kl import kl_divergence as KL
from modules.networks import MLP,ClusterEncoderKPConv
from attrdict import AttrDict
from utils.funcs import to_gaussian,get_rot_matrix,loc2extr


import os
import numpy as np
import pdb
import time
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
class ObjectModule(nn.Module):
    def __init__(self,what_encoder,decoder,args,device):
        super(ObjectModule, self).__init__()
        self.args = args
        self.device = device
        self.K = 4
        self.obj_size = 0.4
        #self.rot_matrix_grid = torch.load(CUR_PATH+'/../constants/rot_grid_high.pt',map_location=device)
        #inference
        ##where
        self.where_encoder = ClusterEncoderKPConv()
        self.q_where_rnn = nn.GRUCell(64,32)
        self.q_z_where_net = MLP(32,32,32*2)
        self.q_h_where_his = []
        self.z_where_his = []
        self.trans_his = []
        self.trans_net = MLP(32,256,3)
        #self.rot_net = MLP(32+3,256,3)

        ##what
        self.what_encoder = what_encoder
        self.q_what_rnn = nn.GRUCell(64,32)
        self.q_z_what_net = MLP(32,32,32*2)
        self.q_h_what_his = []
        self.z_what_his = []
        self.decoder = decoder
        #prior
        ##where
        self.p_h_where_his = []
        self.p_where_rnn = nn.GRUCell(32,32)
        self.p_z_where_net = MLP(32,32,32*2)
        ##what
        self.p_h_what_his = []
        self.p_what_rnn = nn.GRUCell(32,32)
        self.p_z_what_net = MLP(32,32,32*2)


    def forward(self,pcd,\
                p_eval_surface,p_eval_air,ray,\
                feature,mask,depth,bg_rob_obj_prob,\
                idle_states,t):
        #pcd: B-N2-3
        #p_eval_surface/air,ray: B-N-3
        #feature: B-K-N2-32
        #mask: B-K-N2-1
        #depth: B-128-128
        #bg_rob_obj_prob: B-N2-2
        #idle_states: B-K
        #t: 1
        B,N,_ = p_eval_surface.shape
        N2 = pcd.shape[1]
        #-------------------- ENCODE WHERE --------------------
        with torch.no_grad():
            prev_trans,rot_matrix = self.get_pose_trans_from_att(pcd,mask,bg_rob_obj_prob)

        pcd_where_inpt,feature_where_inpt = self.get_where_points(pcd,feature,prev_trans)
        #pcd_where_inpt: B-K-N_where_enc-3
        #feature_where_inpt: B-K-N_where_enc-32
        where_inpt = torch.cat([pcd_where_inpt,feature_where_inpt],dim=-1).reshape(B*self.K,-1,3+feature.shape[-1]).permute(0,2,1).contiguous()
        #where_inpt: BxK-32+3-N_where_enc
        where_enc = self.where_encoder(pcd_where_inpt.reshape(B*self.K,-1,3).contiguous(),where_inpt)
        #where_enc: BxK-128
        q_h_where = self.q_h_where_his[-1]
        #q_h_where: BxK-32
        q_h_where_n = self.q_where_rnn(where_enc,q_h_where)
        #q_h_where_n: BxK-32
        self.q_h_where_his.append(q_h_where_n)
        z_where,q_z_where = to_gaussian(self.q_z_where_net(q_h_where_n.view(B,self.K,-1)))
        #z_where: B-K-32
        self.z_where_his.append(z_where)
        #z_where: B-K-32
        if len(self.z_where_his) == 1:
            p_z_where = Normal(torch.zeros_like(z_where),torch.ones_like(z_where))
        else:
            p_h_where = self.p_h_where_his[-1]
            #p_h_where: BxK-32
            p_h_where_n = self.p_where_rnn(self.z_where_his[-2].view(-1,32),p_h_where)
            #p_h_where_n: BxK-32
            self.p_h_where_his.append(p_h_where_n)
            _,p_z_where = to_gaussian(self.p_z_where_net(p_h_where_n.view(B,self.K,-1)))
        kl_where = (KL(q_z_where,p_z_where)*(1.-idle_states)[:,:,None]).sum(dim=(1,2))
        #kl_where: B
        #trans
        delta_trans = torch.tanh(self.trans_net(z_where))*0.1
        trans = prev_trans + delta_trans
        #trans: B-K-3
        self.trans_his.append(trans)
        #-------------------- ENCODE WHAT --------------------
        p_eval_surface_out,mask_eval_surface,surface_num,\
        p_eval_air_out,mask_eval_air,air_num,\
        ray_out,\
        p_enc_what,feature_what = self.get_evaluation_points(p_eval_surface,p_eval_air,ray,\
                                                             feature,pcd,\
                                                             trans.detach(),rot_matrix.detach())
        #p_eval_surface_out/ray_out: N_surface-3
        #surface_num: BxK
        #p_enc_what/feature_what: B-K-N2-3/32
        #mask_eval_surface: B-K-N
        what_inpt = torch.cat([p_enc_what,feature_what],dim=-1).reshape(B*self.K,-1,3+feature_what.shape[-1]).permute(0,2,1).contiguous()
        #what_inpt: BxK-19-N2
        what_enc = self.what_encoder(p_enc_what.reshape(B*self.K,-1,3).contiguous(),what_inpt)
        #what_enc: BxK-128
        q_h_what = self.q_h_what_his[-1]
        #q_h_what: BxK-32
        q_h_what_n = self.q_what_rnn(what_enc,q_h_what)
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
        kl_what = (KL(q_z_what,p_z_what)*(1.-idle_states)[:,:,None]).sum(dim=(1,2))
        #kl_what: B
        #print("--- %s seconds --- obj encode time" % (time.time() - encode_time))
        #-------------------- DECODE --------------------
        decode_time = time.time()
        #object
        rgb,logits_air,logits_surface,\
        p_voxel_world,prob_occup,posi_map,mask_air = self.decoder(p_eval_surface_out,surface_num,ray_out,\
 p_eval_air_out,air_num,z_what,trans.detach(),rot_matrix.detach(),depth,torch.ones([B,self.K,3],device=self.device)*self.obj_size)
        #rgb: N_surface-3
        #logits_surface: N_surface-1
        #logits_air: N_air-1
        rot_loss_time = time.time()
        with torch.no_grad():
            trans_target = self.generate_pose_target(posi_map,mask_air,p_voxel_world)
            #trans_target: B-K-3
        #posi_mask = (posi_map.sum(dim=-1) > 10).detach()
        posi_mask = (torch.logical_and(posi_map,~mask_air).sum(dim=-1) > 10).detach() 
        #posi_mask: B-K
        trans_loss = ((trans-trans_target)**2).sum(dim=2)*(1.-idle_states)*posi_mask.float()
        trans_loss = trans_loss.sum(dim=1).mean()
        rot_loss = torch.zeros([1],device=self.device).sum()

        rgb_out = torch.zeros([B,self.K,N,3],device=self.device)
        logits_surface_out = -1000.*torch.ones([B,self.K,N,1],device=self.device)
        logits_air_out = -1000.*torch.ones([B,self.K,N,1],device=self.device)
        count_surface = 0
        count_air = 0
        for b in range(B):
            for k in range(self.K):
                surface_num_i = surface_num[b*self.K+k].long()
                surface_idx = mask_eval_surface[b,k].bool()
                rgb_out[b,k,surface_idx] = rgb[count_surface:count_surface+surface_num_i]*(1.-idle_states[b,k])
                logits_surface_out[b,k,surface_idx] = logits_surface[count_surface:count_surface+surface_num_i]*(1.-idle_states[b,k]) + idle_states[b,k]*-1e8
                count_surface += surface_num_i

                air_num_i = air_num[b*self.K+k].long()
                air_idx = mask_eval_air[b,k].bool()
                logits_air_out[b,k,air_idx] = logits_air[count_air:count_air+air_num_i]*(1.-idle_states[b,k]) + idle_states[b,k]*-1e8
                count_air += air_num_i
        #print("--- %s seconds --- obj padding time" % (time.time() - padding_time))
        #-------------------- DECODE --------------------
        #outputs
        out = AttrDict(
            rgb=rgb_out,#B-K-N-3
            logits_surface=logits_surface_out,#B-K-N-1
            logits_air=logits_air_out,#B-K-N-1
            z_what=z_what,#B-K-C
            z_where=z_where,#B-K-32
            kl_what=kl_what,#B
            kl_where=kl_where*1e-3,#B
            prev_trans=prev_trans,#B-K-3
            trans=trans,#B-K-3
            trans_target=trans_target,#B-K-3
            rot_matrix=rot_matrix,#B-K-3-3
            rot_loss=rot_loss.sum()/B,#1
            trans_loss=trans_loss.sum()/B,#1
            p_voxel_world=p_voxel_world,#B-K-cnxcnxcn-3
            prob_occup=prob_occup,#B-K-cnxcnxcn
            posi_map=posi_map,#B-K-cnxcnxcn
            mask_air=mask_air,#B-K-cnxcnxcn
            p_enc_what=p_enc_what,#B-K-N-3
            p_eval_surface_out=p_eval_surface_out,#N_surface-3
            surface_num=surface_num#B-K
        )
        return out

    def transform_points_to_box(self,p,bb_t,bb_r,is_ray=False):
        #bb_t: B-K-3, bb_R:B-K-3-3
        B,K,_ = bb_t.shape
        bb_t = bb_t.reshape(B*K,3)
        #bb_t: BxK-3
        bb_r = bb_r.reshape(B*K,3,3)
        #bb_R: BxK-3-3
        box_ex = loc2extr(bb_t,bb_r)
        #box_ex:BxK-3-4
        #p: B-N-3
        N = p.shape[1]
        if is_ray:
            p_homo = torch.cat([p,torch.zeros([B,N,1],device=self.device)],-1)
        else:
            p_homo = torch.cat([p,torch.ones([B,N,1],device=self.device)],-1)
        #p_homo: B-N-4
        p_homo = p_homo[:,None].repeat(1,K,1,1).view(B*K*N,4,1)
        #p_homo: B-N-4 -> B-K-N-4 -> BxKxN-4-1
        box_ex = box_ex[:,None].repeat(1,N,1,1).view(B*K*N,3,4)
        #box_ex: BxK-3-4 -> BxK-1-3-4 -> BxK-N-3-4 -> BxKxN-3-4
        p_box = box_ex@p_homo
        #p_box: BxKxN-3-1
        p_box = p_box[:,:,0]
        #p_box: BxKxN-3
        p_box = p_box.view(B,K,N,3)
        #p_box:B-K-N-3
        return p_box

    def get_evaluation_points(self,p_eval_surface,p_eval_air,ray,features,\
                              p_enc,bb_loc,bb_rot_matrix):
        #p_eval_surface/air: B-N-3
        #ray: B-N-3
        #p_enc: B-N2-3
        #bb_loc: B-K-3
        #bb_rot_matrix:B-K-3-3
        #features: B-K-N2-32
        obj_size = self.obj_size
        p_eval_surface_in_box = self.transform_points_to_box(p_eval_surface,bb_loc,bb_rot_matrix)
        #p_eval_surface_in_box: B-K-N-3
        p_eval_air_in_box = self.transform_points_to_box(p_eval_air,bb_loc,bb_rot_matrix)
        #p_eval_air_in_box: B-K-N-3
        ray_in_box = self.transform_points_to_box(ray,bb_loc,bb_rot_matrix,True)
        #ray: B-K-N-3
        p_enc_in_box = self.transform_points_to_box(p_enc,bb_loc,bb_rot_matrix)
        #p_eval_in_box: B-K-N2-3
        B,N,_ = ray.shape
        threshold = 1.0
        p_eval_surface_in_box = p_eval_surface_in_box/(obj_size/2.)
        #p_eval_surface_in_box: B-K-N-3
        p_eval_air_in_box = p_eval_air_in_box/(obj_size/2.)
        #p_eval_air_in_box: B-K-N-3
        p_enc_in_box = p_enc_in_box/(obj_size/2.)
        #p_eval_in_box: B-K-N-3
        #features: B-K-N2-32
        mask_eval_surface = torch.all(p_eval_surface_in_box<=threshold,dim=-1)&torch.all(p_eval_surface_in_box>=-threshold,dim=-1)
        #mask_eval_surface: B-K-N
        mask_eval_air = torch.all(p_eval_air_in_box<=threshold,dim=-1)&torch.all(p_eval_air_in_box>=-threshold,dim=-1)
        #mask_eval_air: B-K-N
        mask_enc = torch.all(p_enc_in_box<=threshold,dim=-1)&torch.all(p_enc_in_box>=-threshold,dim=-1)
        #p_enc_in_box: B-K-N2
        p_eval_surface_out = p_eval_surface_in_box.view(-1,3)[mask_eval_surface.view(-1).bool()]
        #p_eval_surface_out: N_surface-3
        p_eval_air_out = p_eval_air_in_box.view(-1,3)[mask_eval_air.view(-1).bool()]
        #p_eval_air_out: N_air-3
        ray_out = ray_in_box.view(-1,3)[mask_eval_surface.view(-1).bool()]
        #ray_out: N_surface-3
        surface_num = mask_eval_surface.sum(dim=2).view(-1)
        #surface_num: BxK
        air_num = mask_eval_air.sum(dim=2).view(-1)
        #air_num: BxK
        enc_num = mask_enc.sum(dim=2)
        #enc_num: B-K
        N_max = enc_num.max().clamp(min=256)
        p_enc_out = torch.zeros([B,self.K,N_max,3],device=self.device)
        #p_enc_out: B-K-N_max-3
        feature_out = torch.zeros([B,self.K,N_max,features.shape[-1]],device=self.device)
        #p_enc_out: B-K-N_max-D
        for b in range(B):
            for k in range(self.K):
                n_i = enc_num[b,k]
                mask_i = mask_enc[b,k]
                p_enc_out[b,k,:n_i] = p_enc_in_box[b,k,mask_i]
                feature_out[b,k,:n_i] = features[b,k,mask_i]
        #feature_out: B-K-N2-32
        return p_eval_surface_out,mask_eval_surface,surface_num,\
               p_eval_air_out,mask_eval_air,air_num,\
               ray_out,\
               p_enc_out,feature_out

    def get_where_points(self,pcd,features,trans):
        #pcd: B-N2-3
        #features: B-K-N2-32
        #trans: B-K-3
        B = pcd.shape[0]
        K = features.shape[1]
        obj_size = self.obj_size
        with torch.no_grad():
            pcd_in_box = pcd[:,None].repeat(1,K,1,1) - trans[:,:,None]
            #pcd_in_box: B-K-N2-3
            pcd_in_box = pcd_in_box/(obj_size/2.)
            #pcd_in_box: B-K-N2-3
            threshold = 1.0
            mask = torch.all(pcd_in_box<=threshold,dim=-1)&torch.all(pcd_in_box>=-threshold,dim=-1)
            #mask: B-K-N2
            enc_num = mask.sum(dim=2)
            #enc_num: B-K
            N_max = enc_num.max().clamp(min=256)
            pcd_out = torch.zeros([B,self.K,N_max,3],device=self.device)
            #pcd_out: B-K-N_max-3
            for b in range(B):
                for k in range(self.K):
                    n_i = enc_num[b,k]
                    mask_i = mask[b,k]
                    pcd_out[b,k,:n_i] = pcd_in_box[b,k,mask_i]
        feature_out = torch.zeros([B,self.K,N_max,features.shape[-1]],device=self.device)
        for b in range(B):
            for k in range(self.K):
                n_i = enc_num[b,k]
                mask_i = mask[b,k]
                feature_out[b,k,:n_i] = features[b,k,mask_i]
        #feature_out: B-K-N2-D
        return pcd_out,feature_out

    def init_his(self,B):
        #what
        q_h0_what = torch.zeros([B*self.K,32],device=self.device)
        self.q_h_what_his.append(q_h0_what)
        #q_h0_what: BxK-32
        p_h0_what = torch.zeros([B*self.K,32],device=self.device)
        self.p_h_what_his.append(p_h0_what)
        #p_h0_what: BxK-32
        #where
        q_h0_where = torch.zeros([B*self.K,32],device=self.device)
        self.q_h_where_his.append(q_h0_where)
        #q_h0_where: BxK-32
        p_h0_where = torch.zeros([B*self.K,32],device=self.device)
        self.p_h_where_his.append(p_h0_where)
        #p_h0_where: BxK-32

    def get_pose_trans_from_att(self,pcd,mask,bg_rob_obj_prob):
        #pcd: B-N2-3
        #mask: B-K-N2-1
        #bg_rob_obj_prob: B-N2-2
        B = pcd.shape[0]
        pcd_k = pcd[:,None].repeat(1,self.K,1,1)
        #pcd_k: B-K-N2-3
        pcd_centre = (pcd_k*mask).sum(dim=2)/(mask.sum(dim=2)+1e-8)
        #pcd_centre: B-K-3

        mask_full = torch.cat([mask[:,:,:,0],bg_rob_obj_prob.permute(0,2,1)[:,:1]],dim=1)
        #mask_full: B-K+1-N2
        att_mask_bin = torch.argmax(mask_full,dim=1)
        #att_mask_bin: B-N2
        att_mask_bin = att_mask_bin[:,:,None] == torch.arange(self.K,device=self.device)
        #att_mask_bin: B-N2-K
        att_rot_matrix = get_rot_matrix(torch.zeros([B,self.K,3],device=self.device))
        #att_rot_matrix: B-K-3-3
        for b in range(B):
            for k in range(self.K):
                att_points_num = att_mask_bin[b,:,k].sum()
                if att_points_num < 10 and len(self.trans_his)!=0:
                    pcd_centre[b,k] = self.trans_his[-1][b,k]
        return pcd_centre,att_rot_matrix

    def generate_pose_target(self,posi_map,mask_air,p_voxel_world):
        #posi_map: B-K-cnxcnxcn
        #p_voxel_world: B-K-cnxcnxcn-3-1
        B = posi_map.shape[0]
        trans_target = torch.zeros([B,self.K,3],device=self.device)
        #trans_target: B-K-3
        posi_map = torch.logical_and(posi_map,~mask_air)
        for b in range(B):
            for k in range(self.K):
                voxel_num = posi_map[b,k].sum()
                if voxel_num > 10:
                    posi_voxels = p_voxel_world[b,k,posi_map[b,k]]
                    #posi_voxels: N_posi-3 -> 1-N_posi-3-1
                    #rot_matrix_eval: N_grid-3-3 -> N_grid-1-3-3
                    trans_target_max = posi_voxels.max(dim=0)[0]
                    #trans_target_max: 3
                    trans_target_min = posi_voxels.min(dim=0)[0]
                    #trans_target_min: 3
                    trans_target_centre = (trans_target_max+trans_target_min)/2.
                    #trans_target_centre: 3
                    trans_target[b,k] = trans_target_centre
                    #trans_target_world: B-K-3

        return trans_target


    def free_his(self):
        #what
        self.q_h_what_his = []
        self.p_h_what_his = []
        self.z_what_his = []
        #where
        self.q_h_where_his = []
        self.p_h_where_his = []
        self.z_where_his = []
        self.trans_his = []
