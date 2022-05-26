import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter
from torch.nn.init import kaiming_uniform_
from kernels.kernel_points import load_kernels
from lib.pointops.functions import pointops
import pdb
EPSILON = 1e-8
class KPConv(nn.Module):

    def __init__(self,in_dim,out_dim,KP_extent,radius,nsample=16):
        super(KPConv, self).__init__()
        # Save parameters
        self.K = 15
        self.KP_extent = KP_extent
        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K,in_dim,out_dim), dtype=torch.float32),
                                 requires_grad=True)
        kaiming_uniform_(self.weights,a=math.sqrt(5))
        # Initialize kernel points
        self.kernel_points = self.init_KP(radius)
        self.grouper = pointops.QueryAndGroupForKPConv(radius=radius,nsample=nsample,use_xyz=True)

    def init_KP(self,radius):
        K_points_numpy = load_kernels(radius,
                                      self.K,
                                      dimension=3,
                                      fixed='center')

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self,p,x,x_in,stride=None):
        #p: B-N-3
        #x: B-C-N
        #x_in: B-C_in-N
        if stride is not None:
            M = p.shape[1] // stride
            p_trans = p.transpose(1,2).contiguous()
            #p_trans: B-3-N
            p2 = pointops.gathering(p_trans,pointops.furthestsampling(p,M)).transpose(1, 2).contiguous()
            #p2: B-M-3
            n_k,_,n_idx = self.grouper(xyz=p,new_xyz=p2,features=x)
        else:
            n_k,_,n_idx = self.grouper(xyz=p,features=x)
            p2=p
        
        #n_k: B-(C+3)-M-nsample
        #n_idx: B-M-nsample
        padding_mask = torch.zeros_like(n_idx)
        #padding_mask: B-M-nsample
        padding_mask[:,:,1:] = (n_idx[:,:,1:] - n_idx[:,:,:1]) == 0.
        #padding_mask: B-M-nsample -> 1-1-M_b-nsample; 1 is padded points, 0 is not padded.
        n_k_pcd = n_k[:,:3]*(1.-padding_mask[:,None]) + 1e6*padding_mask[:,None]
        #n_k_pcd: B-3-M-nsample
        n_k_feature = n_k[:,3:]*(1.-padding_mask[:,None]) + 0.*padding_mask[:,None]
        #n_k_feature: B-C-M-nsample
        n_k = torch.cat([n_k_pcd,n_k_feature],dim=1)
        #n_k: B-(C+3)-M-nsample
        x_in_skip = pointops.grouping(x_in,n_idx)
        x_in_skip = x_in_skip*(1.-padding_mask[:,None]) + 0.*padding_mask[:,None]
        x_in_skip, _ = torch.max(x_in_skip,3)
        #x_in_skip: B-C_in-M-nsample 

        rela_posi = n_k[:,:3].permute(0,2,3,1)[:,:,:,None]
        #rela_posi: B-3-M-nsample -> B-M-nsample-3 -> B-M-nsample-1-3
        #self.kernel_points: K-3 -> 1-1-1-K-3
        differences = rela_posi - self.kernel_points[None,None,None]
        #differences: B-M-nsample-K-3
        sq_distances = torch.sum(differences**2, dim=4)
        #sq_distances: B-M-nsample-K
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        #all_weights: B-M-nsample-K
        features = n_k[:,3:].permute(0,2,3,1)
        #features: B-C-M-nsample -> B-M-nsample-C
        #all_weights: B-M-nsample-K -> B-M-K-nsample
        weighted_features = all_weights.permute(0,1,3,2)@features
        #weighted_features: B-M-K-C ->B-M-K-1-C
        #self.weights: K-C-C_out -> 1-1-K-C-C_out
        kernel_outputs = (weighted_features[:,:,:,None]@self.weights[None,None])[:,:,:,0]
        #kernel_outputs: B-M-K-1-C_out -> B-M-K-C_out
        return torch.sum(kernel_outputs,dim=2).permute(0,2,1),p2,x_in_skip,n_idx#B-C_out-M

class KPConvBlock(nn.Module):

    def __init__(self,in_dim,out_dim,radius,stride=None,nsample=16):
        super(KPConvBlock, self).__init__()
        self.stride=stride
        current_extent = radius*1.2/2.5
        self.gn = nn.GroupNorm(out_dim//16,out_dim)
        self.relu = nn.ReLU()
        self.l0 = nn.Conv1d(in_dim,out_dim//4,kernel_size=1,bias=False)
        self.gn0 = nn.GroupNorm(out_dim//16,out_dim//4)
        self.kpconv = KPConv(out_dim//4,out_dim//4,current_extent,radius,nsample)
        self.l1 = nn.Conv1d(out_dim//4,out_dim,kernel_size=1,bias=False)
        self.gn1 = nn.GroupNorm(out_dim//16,out_dim)
        self.l_shorcut = nn.Conv1d(in_dim,out_dim,kernel_size=1,bias=False)
        self.gn_shortcut = nn.GroupNorm(out_dim//16,out_dim)

    def forward(self,p,x):
        #p: B-N-3
        #x: B-C-N
        y = self.relu(self.gn0(self.l0(x)))
        #y: B-C_out//4-N
        y,p2,skip_y,n_idx = self.kpconv(p,y,x,self.stride)
        #y: B-C_out//4-M
        #p2: B-3-M
        #skip_y: B-C-M
        y = self.relu(self.gn1(self.l1(y)))
        #y: B-C_out-M
        y = y+self.gn_shortcut(self.l_shorcut(skip_y))
        return p2,y,n_idx

class KPConvTransitionUp(nn.Module):
    def __init__(self,in_dim,out_dim,skip_dim):
        super(KPConvTransitionUp, self).__init__()
        self.Linear1 = nn.Sequential(
            nn.Conv1d(in_dim,out_dim,kernel_size=1,bias=False),
            nn.GroupNorm(out_dim//16,out_dim),
            nn.ReLU()
        )
        self.Linear2 = nn.Sequential(
            nn.Conv1d(skip_dim,out_dim,kernel_size=1,bias=False),
            nn.GroupNorm(out_dim//16,out_dim),
            nn.ReLU()
        )

    def forward(self,p1,x1,p2,x2):
        # in_points, p1: (B, N, 3)
        # in_features, x1: (B, C_in, N)
        # skip_points, p2: (B, M, 3)
        # skip_features, x2: (B, C_skip, M)
        # Three nearest neighbor upsampling
        dist, idx = pointops.nearestneighbor(p2, p1)
        dist_recip = 1.0 / (dist + EPSILON)
        norm = torch.sum(dist_recip, dim=2, keepdim=True)
        weight = dist_recip / norm
        up_x1 = pointops.interpolation(self.Linear1(x1), idx, weight)
        # aggregation
        y = self.Linear2(x2) + up_x1 # (B, C_out, M)
        return [p2, y]

class InputBlock(nn.Module):
    def __init__(self,in_dim,out_dim,radius,nsample=16):
        super(InputBlock, self).__init__()
        current_extent = radius*1.2/2.5
        self.kpconv = KPConv(in_dim,out_dim,current_extent,radius,nsample)
        self.relu = nn.ReLU()
        self.gn = nn.GroupNorm(out_dim//16,out_dim)

    def forward(self,p,x):
        y,p2,skip_y,n_idx = self.kpconv(p,x,x,None)
        y = self.relu(self.gn(y))
        return p2,y,n_idx

class KPConvAggregateLayer(nn.Module):

    def __init__(self,in_dim,out_dim,radius):
        super(KPConvAggregateLayer, self).__init__()
        # Save parameters
        self.K = 15
        self.KP_extent = radius*1.2/2.5
        # Initialize weights
        self.weights = Parameter(torch.zeros((self.K,in_dim,out_dim), dtype=torch.float32),
                                 requires_grad=True)
        kaiming_uniform_(self.weights,a=math.sqrt(5))
        # Initialize kernel points
        self.kernel_points = self.init_KP(radius)

    def init_KP(self,radius):
        K_points_numpy = load_kernels(radius,
                                      self.K,
                                      dimension=3,
                                      fixed='center')

        return Parameter(torch.tensor(K_points_numpy, dtype=torch.float32),
                         requires_grad=False)

    def forward(self,p,x):
        #p: B-N-3
        #x: B-C-N
        #self.kernel_points: K-3 -> 1-1-K-3
        #p: B-N-3 -> B-N-1-3
        differences = p[:,:,None] - self.kernel_points[None,None]
        #differences: B-N-K-3
        sq_distances = torch.sum(differences**2,dim=3)
        #sq_distances: B-N-K
        all_weights = torch.clamp(1 - torch.sqrt(sq_distances) / self.KP_extent, min=0.0)
        #all_weights: B-N-K -> B-K-N
        weighted_features = all_weights.permute(0,2,1)@x.permute(0,2,1)
        #weighted_features: B-K-C ->B-K-1-C
        #self.weights: K-C-C_out -> 1-K-C-C_out
        kernel_outputs = (weighted_features[:,:,None]@self.weights[None])[:,:,0]
        #kernel_outputs: B-K-1-C_out -> B-K-C_out
        return torch.sum(kernel_outputs,dim=1)#B-C_out
