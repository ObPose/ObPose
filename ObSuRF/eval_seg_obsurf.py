import json
import torch
import argparse
import numpy as np
from attrdict import AttrDict
import os

from model.obsurf_full import ObsurfFull
from utils.eval_funcs import compute_iou,compute_msc
from sklearn.metrics.cluster import adjusted_rand_score as compute_ari
from torch.utils.data import DataLoader
from dataloader.move_objs_eval import MoveObjsEval

from obsurf import config
import obsurf.model.config as model_config

parser = argparse.ArgumentParser(description='ObsurfFull')
parser.add_argument('--ex_id', default=0, type=int,
                    help='experiment id')
parser.add_argument('--it', default=0, type=int,
                    help='iteration num')
parser.add_argument('--seed', default=0, type=int,
                    help='Fixed random seed.')
args = parser.parse_args()
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

ex_id = args.ex_id
it_num = args.it
test_dataset = MoveObjsEval('./data/MoveObjs/raw/','test',500)
dataloader_test = DataLoader(test_dataset,batch_size=1,shuffle=False)
directory_output = './checkpoints/MoveObjs_obsurf/{}/'.format(args.ex_id)

with open(directory_output+'args.txt', 'r') as f:
    args.__dict__ = json.load(f)
args.ray_sample=False
device = 'cuda:0'
cfg = config.load_config('runs/3dclevr/obsurf/config.yaml','configs/default.yaml')
obsurf = model_config.get_model(cfg, device=device, dataset=test_dataset)
model = ObsurfFull(device,args,obsurf)
model.cuda()
resume_checkpoint = directory_output + f'net-{it_num}.ckpt'
checkpoint = torch.load(resume_checkpoint,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
iter_idx = checkpoint['iter_idx'] + 1
start_iter = iter_idx
print("Starting eval at iter = {}".format(iter_idx))
model.args.ray_sample=False
model.eval()
#test dataloader
count = 0.
iou_bg_sum = 0.
ari_obj_sum = 0.
msc_obj_sum = 0.
CUR_PATH = os.path.dirname(os.path.abspath(__file__))
for data in dataloader_test:
    rgb_video,pcd_video,depth_video,mask_video,obj_indx=data
    mask_video = mask_video.to(model.device).float()
    L = rgb_video.shape[1]
    B = rgb_video.shape[0]
    misc = np.load(CUR_PATH+'/constants/camera_info.npy',allow_pickle=True).item()
    cam_ex=torch.from_numpy(misc['camera_extrinsics'])
    cam_in=torch.from_numpy(misc['camera_intrinsics'])
    camera_ex = cam_ex[None].repeat(B,1,1)
    camera_in = cam_in[None].repeat(B,1,1)
    out_dict_list = []
    
    for t in range(L):
        rgb = rgb_video[:,t]
        pcd = pcd_video[:,t]
        depth = depth_video[:,t]
        with torch.no_grad():
            loss_dict,out_dict = model(pcd,rgb,depth,camera_in,camera_ex,iter_idx)
        sigma_surface = out_dict[0]['surface_density']
        sigma_surface_denom = sigma_surface.sum(dim=1,keepdim=True)
        #sigma_surface_denom: B-1-N-1
        sigma_surface = sigma_surface/sigma_surface_denom
        #sigma_surface: B-K_all-N-1
        out_dict_list.append(out_dict)
        for b in [0]:
            gt_mask = mask_video[b,t]
            #gt_mask: 128-128-5
            predicted_mask = sigma_surface[b,:,:].permute(1,0).view(128,128,5)
            #predicted_mask: 128-128-5
            predicted_mask = torch.argmax(predicted_mask,dim=2,keepdim=True)
            #predicted_mask: 128-128-1
            predicted_mask_bin = predicted_mask == torch.arange(5,device=model.device)
            #predicted_mask_bin: 128-128-5
            #############BG#################
            iou_bg = -1000.
            bg_idx = -1
            for j in range(5):
                predicted_mask_j = predicted_mask_bin[:,:,j]
                gt_mask_bg = gt_mask[:,:,0]
                iou_bg_j = compute_iou(gt_mask_bg,predicted_mask_j)
                if iou_bg_j > iou_bg:
                   iou_bg = iou_bg_j
                   bg_idx = j
            iou_bg_sum = iou_bg_sum+iou_bg
            #iou_bg_sum: 1
            #############ARI################
            gt_obj = gt_mask[:,:,1:].sum(dim=2).view(-1)
            #gt_obj: 128x128
            predicted_obj_mask = predicted_mask.view(-1)[gt_obj.bool()]
            #predicted_obj_mask: N_obj
            gt_mask_ari = torch.argmax(gt_mask,dim=2).view(-1)
            #gt_mask_ari: 128x128
            gt_obj_ari = gt_mask_ari[gt_obj.bool()]
            #gt_obj_ari: N_obj
            ari_obj = compute_ari(gt_obj_ari.cpu().data.numpy(),predicted_obj_mask.cpu().data.numpy())
            #ari_obj: 1
            ari_obj_sum = ari_obj_sum + ari_obj
            #ari_obj_sum: 1
            #############MSC################
            predicted_mask_bin = torch.cat([predicted_mask_bin[:,:,:bg_idx],predicted_mask_bin[:,:,bg_idx+1:]],dim=2)
            msc_obj = compute_msc(gt_mask[:,:,1:],predicted_mask_bin)
            msc_obj_sum = msc_obj_sum + msc_obj
            count += 1
            #count: 1

iou_bg_mean = iou_bg_sum/count
ari_obj_mean = ari_obj_sum/count
msc_obj_mean = msc_obj_sum/count
print(iou_bg_mean,ari_obj_mean,msc_obj_mean)
torch.save([iou_bg_mean,ari_obj_mean,msc_obj_mean],f'./eval_results/seg/{ex_id}-{it_num}_obsurf.pt')
