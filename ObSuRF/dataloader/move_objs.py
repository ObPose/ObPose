import numpy as np
import torch
from torch.utils.data import Dataset

class MoveObjs(Dataset):
    def __init__(self,img_dir,task,data_num,video_length):
        self.img_dir = img_dir+f'/{task}/'
        self.data_num = data_num
        self.L = video_length
        self.obj_table = dict()
        self.obj_table['a_cups'] = 0
        self.obj_table['apple'] = 1
        self.obj_table['bleach_cleanser'] = 2
        self.obj_table['chef_can'] = 3
        self.obj_table['cracker_box'] = 4
        self.obj_table['foam_brick'] = 5
        self.obj_table['gelatin_box'] = 6
        self.obj_table['mug'] = 7
        self.obj_table['mustard_bottle'] = 8
        self.obj_table['orange'] = 9
        self.obj_table['pear'] = 10
        self.obj_table['potted_meat_can'] = 11
        self.obj_table['pudding_box'] = 12
        self.obj_table['rubiks_cube'] = 13
        self.obj_table['softball'] = 14
        self.obj_table['sugar_box'] = 15
        self.obj_table['tomato_soup_can'] = 16
        self.obj_table['tuna_fish_can'] = 17
        self.obj_table['wood_block'] = 18

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        data_path = self.img_dir+f'{idx}.npy'
        obs=np.load(data_path,allow_pickle=True)
        max_frame_num = len(obs)
        start_idx = np.random.randint(0,max_frame_num-self.L+1)
        rgb = np.zeros([self.L,128,128,3])
        depth = np.zeros([self.L,128,128])
        mask = np.zeros([self.L,128,128,5])
        pcd = np.zeros([self.L,128,128,3])
        obj_indx = np.zeros([19])
        for obj in obs[0]['obj_list']:
            obj_indx[self.obj_table[obj['name']]] = 1
        for j,i in enumerate(range(start_idx,start_idx+self.L)):
            rgb[j]=obs[i]['rgb']
            depth[j]=obs[i]['depth']
            pcd[j]=obs[i]['pcd']
            obj_num = obs[i]['mask_bin'].shape[-1]-1
            mask[j,:,:,:(1+obj_num)]=obs[i]['mask_bin']
            
        return rgb,pcd,depth,mask,obj_indx


