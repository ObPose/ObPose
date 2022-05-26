import torch
import torch.optim as optim

from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from dataloader.obj_table_multiview import ObjTabelMultiview

from model.obsurf_full import ObsurfFull
import os
import time
import json
import argparse
import numpy as np
from attrdict import AttrDict

from obsurf import config
import obsurf.model.config as model_config

class LrScheduler():
    """ Implements a learning rate schedule with warum up and decay """
    def __init__(self, peak_lr=4e-4, peak_it=10000, decay_rate=0.5, decay_it=100000):
        self.peak_lr = peak_lr
        self.peak_it = peak_it
        self.decay_rate = decay_rate
        self.decay_it = decay_it

    def get_cur_lr(self, it):
        if it < self.peak_it:
            return self.peak_lr * (it / self.peak_it)
        it_since_peak = it - self.peak_it
        return self.peak_lr * (self.decay_rate ** (it_since_peak / self.decay_it))

    def update_every(self, it):
        if it <= self.peak_it:
            return 1
        return self.decay_it // 100

parser = argparse.ArgumentParser(description='WM3D')
parser.add_argument('--bs', default=4, type=int,
                    help='batch size')
parser.add_argument('--ex_id', default=0, type=int,
                    help='experiment id')
parser.add_argument('--resume', action='store_true', default=False,
                    help='whether resume')
parser.add_argument('--ray_sample', action='store_true', default=False,
                    help='whether ray_sample')
parser.add_argument('--seed', default=20, type=int,
                    help='Fixed random seed.')
args = parser.parse_args()

# Fix seeds. Always first thing to be done after parsing the config!
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)
# Make CUDA operations deterministic
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# Dataloader
#torch.autograd.set_detect_anomaly(True)
train_dataset = ObjTabelMultiview('./data/obj_tabel_multiview/raw/','train',12000)
val_dataset = ObjTabelMultiview('./data/obj_tabel_multiview/raw/','test',1500)
dataloader_train = DataLoader(train_dataset,batch_size=args.bs,shuffle=True)
dataloader_val = DataLoader(val_dataset,batch_size=args.bs,shuffle=False)
# Checkpoints
directory_output = './checkpoints/obj_tabel_multiview_obsurf/{}/'.format(args.ex_id)
os.makedirs(directory_output,exist_ok = True)
if args.resume:
    with open(directory_output+'args.txt', 'r') as f:
        args.__dict__ = json.load(f)
    args.resume = True
    all_iters = np.array([int(x[4:-5]) for x in os.listdir(directory_output) if x.endswith('ckpt')])
else:
    with open(directory_output+'args.txt', 'w') as f:
        json.dump(args.__dict__, f, indent=2)
#args.ray_sample=False
cfg = config.load_config('runs/3dclevr/obsurf/config.yaml','configs/default.yaml')
if 'lr_warmup' in cfg['training']:
    peak_it = cfg['training']['lr_warmup']
else:
    peak_it = 10000
decay_it = cfg['training']['decay_it'] if 'decay_it' in cfg['training'] else 100000
lr_scheduler = LrScheduler(peak_it=peak_it, decay_it=decay_it)
# Create model
device = 'cuda:0'
obsurf = model_config.get_model(cfg, device=device, dataset=train_dataset)
model = ObsurfFull(device,args,obsurf)
model = model.cuda()
optimiser = optim.Adam(model.parameters(), lr_scheduler.get_cur_lr(0))
start_iter = 1
writer = SummaryWriter(directory_output)
# Try to restore model and optimiser from checkpoint
if args.resume:
    resume_checkpoint = directory_output + 'net-{}.ckpt'.format(np.max(all_iters))#
    checkpoint = torch.load(resume_checkpoint)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimiser.load_state_dict(checkpoint['optimiser_state_dict'])
    iter_idx = checkpoint['iter_idx'] + 1
    start_iter = iter_idx
    print("Starting eval at iter = {}".format(iter_idx))

model.train()

def eval_model(model,dataloader_val,writer,iter_idx):
    timer_eval = time.time()
    model.eval()
    loss_dict_eval = AttrDict()
    loss_dict_eval.rgb_loss = 0.
    loss_dict_eval.depth_loss = 0.
    loss_dict_eval.overlap_loss = 0.
    loss_dict_eval.sum = 0.

    num_iter=1.
    for data_eval in dataloader_val:
        #if num_iter >5:
            #break
        rgb,pcd,depth,mask,obj_indx,camera_in,camera_ex=data_eval
        # Forward propagation
        loss_dict,out_dict = model(pcd,rgb,depth,camera_in,camera_ex,iter_idx)
        for key in list(loss_dict_eval.keys()):
            loss_dict_eval[key] += loss_dict[key].cpu().data.numpy()
        num_iter+=1.
    speed_eval = (time.time()-timer_eval)/num_iter
    for key, value in loss_dict_eval.items():
        writer.add_scalar('eval/{}'.format(key), value/num_iter, iter_idx)
    print('iter: ',iter_idx,
          " ".join(str(key)+': '+str(value/num_iter) for key, value in loss_dict_eval.items()),
          'speed: {:.3f}s/iter'.format(speed_eval))
    print('GPU usage in eval start: ', torch.cuda.max_memory_allocated()/1048576)
    model.train()

flag = 0
timer_epoch = time.time()
iter_idx = start_iter
skip_flag = False
for epoch in range(5000):
    for data in dataloader_train:
        #timer = time.time()
        if iter_idx == 5:
            timer = time.time()
        if iter_idx == 6:
            print('1 iter takes {:.3f}s.'.format(time.time()-timer))
        rgb,pcd,depth,mask,obj_indx,camera_in,camera_ex=data
        # Forward propagation
        optimiser.zero_grad()
        # Forward propagation
        loss_dict,out_dict = model(pcd,rgb,depth,camera_in,camera_ex,iter_idx)
        # Backprop and optimise
        loss_dict.sum.backward()
        optimiser.step()
        
        # Heartbeat log
        hertz = 1500
        if (iter_idx % hertz == 0):
            # Print output and write to file
            speed = (time.time()-timer_epoch)/float(hertz)
            timer_epoch = time.time()
            print('GPU usage in train: ', torch.cuda.max_memory_allocated()/1048576)
            print('iter: ',iter_idx,
                  " ".join(str(key)+': '+str(value.cpu().data.numpy()) for key, value in loss_dict.items()),
                  'speed: {:.3f}s/iter'.format(speed))
            for key, value in loss_dict.items():
                writer.add_scalar('train/{}'.format(key), value, iter_idx)
            if iter_idx % (hertz*10) == 0:
                with torch.no_grad():
                    eval_model(model,dataloader_val,writer,iter_idx)

        # Save checkpoints
        if iter_idx % hertz == 0:
            ckpt_file = directory_output + 'net-{}.ckpt'.format(iter_idx)
            print("Saving model training checkpoint to: {}".format(ckpt_file))
            model_state_dict = model.state_dict()
            ckpt_dict = {'iter_idx': iter_idx,
                         'model_state_dict': model_state_dict,
                         'optimiser_state_dict': optimiser.state_dict(),
                         'loss_dict': loss_dict}
            torch.save(ckpt_dict, ckpt_file)
        if flag == 0:
            print('Training starts, GPU usage in train start: ', torch.cuda.max_memory_allocated()/1048576)
            flag = 666
        #update iter_idx
        iter_idx+=1
writer.close()
