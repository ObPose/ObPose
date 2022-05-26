import json
import torch
import argparse
import numpy as np
from attrdict import AttrDict

from model.obpose import ObPose
from utils.eval_funcs import compute_iou,compute_msc
from sklearn.metrics.cluster import adjusted_rand_score as compute_ari
from torch.utils.data import DataLoader
from dataloader.move_objs_eval import MoveObjsEval

parser = argparse.ArgumentParser(description='WM3D')
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
directory_output = './checkpoints/MoveObjs/{}/'.format(args.ex_id)

with open(directory_output+'args.txt', 'r') as f:
    args.__dict__ = json.load(f)
device = 'cuda:0'
model = ObPose(device,args)
model.cuda()
resume_checkpoint = directory_output + f'net-{it_num}.ckpt'
checkpoint = torch.load(resume_checkpoint,map_location=device)
model.load_state_dict(checkpoint['model_state_dict'])
iter_idx = checkpoint['iter_idx'] + 1
start_iter = iter_idx
print("Starting eval at iter = {}".format(iter_idx))
model.eval()
#test dataloader
count = 0.
iou_bg_sum = 0.
ari_obj_sum = 0.
msc_obj_sum = 0.
for data in dataloader_test:
    rgb_video,pcd_video,depth_video,mask_video,obj_indx=data
    bs = rgb_video.shape[0]
    mask_video = mask_video.to(model.device).float()
    L = rgb_video.shape[1]
    with torch.no_grad():
        loss_dict,out_dict = model(pcd_video,rgb_video,depth_video)
        model.free_his()
    for t in range(L):
        sigma_surface,_ = out_dict[t]['sigma']
        for b in [0]:
            gt_mask = mask_video[b,t]
            #gt_mask: 128-128-5
            predicted_mask = sigma_surface[b,:,:,0].permute(1,0).view(128,128,5)
            #predicted_mask: 128-128-5
            #############ARI################
            predicted_mask = torch.argmax(predicted_mask,dim=2,keepdim=True)
            #predicted_mask: 128-128-1
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
            predicted_mask_bin = predicted_mask == torch.arange(5,device=model.device)
            #predicted_mask_bin: 128-128-5
            msc_obj = compute_msc(gt_mask[:,:,1:],predicted_mask_bin[:,:,1:])
            msc_obj_sum = msc_obj_sum + msc_obj
            #############BG#################
            predicted_bg_mask = predicted_mask_bin[:,:,0]
            #predicted_bg_mask: 128-128
            iou_bg = compute_iou(gt_mask[:,:,0],predicted_bg_mask)
            #iou_bg: 1
            iou_bg_sum = iou_bg_sum+iou_bg
            #iou_bg_sum: 1
            count += 1
            #count: 1

iou_bg_mean = iou_bg_sum/count
ari_obj_mean = ari_obj_sum/count
msc_obj_mean = msc_obj_sum/count
print(iou_bg_mean,ari_obj_mean,msc_obj_mean)
torch.save([iou_bg_mean,ari_obj_mean,msc_obj_mean],f'./eval_results/seg/{ex_id}-{it_num}.pt')
