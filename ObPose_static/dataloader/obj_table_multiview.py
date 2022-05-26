import numpy as np
import torch
from torch.utils.data import Dataset

class ObjTabelMultiview(Dataset):
    def __init__(self,img_dir,task,data_num):
        self.img_dir = img_dir+f'/{task}/'
        self.data_num = data_num
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
        obs=np.load(data_path,allow_pickle=True).item()

        rgb = np.zeros([128,128,3])
        depth = np.zeros([128,128])
        mask = np.zeros([128,128,5])
        pcd = np.zeros([128,128,3])
        obj_indx = np.zeros([19])
        for obj in obs['obj_list']:
            obj_indx[self.obj_table[obj['name']]] = 1
        rgb=obs['rgb']
        depth=obs['depth']
        obj_num = len(obs['obj_list'])
        mask=obs['mask_bin']
        pcd=obs['pcd']
        camera_in = obs['misc']['camera_intrinsics']
        camera_ex = obs['misc']['camera_extrinsics']
        return rgb,pcd,depth,mask,obj_indx,camera_in,camera_ex


