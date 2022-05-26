import torch
import torch.nn.functional as F
from torch.distributions import Normal
def loc2extr(trans,rot):
    #rot: N-3-3
    #trans: N-3
    if len(trans.shape) == 2:
        trans = trans[:,:,None]
        #trans: N-3 -> N-3-1
        rot_inv = rot.permute(0,2,1)
        #rot_inv: N-3-3
    else:
        trans = trans[:,None]
        #trans: 3 -> 3-1
        rot_inv = rot.permute(1,0)
        #rot_inv: 3-3
    rot_inv_trans = rot_inv@trans
    #rot_inv_trans: N-3-1/3-1
    extrinsic = torch.cat([rot_inv,-rot_inv_trans],dim=-1)
    #extrinsic: N-3-4/3-4s
    return extrinsic

def to_gaussian(x):
    mean,std = x.chunk(2,-1)
    std = F.softplus(std)+1e-4
    dist = Normal(mean,std)
    z = dist.rsample()
    return z,dist

def get_rot_matrix(bb_rot):
    #bb_rot:B-K-3
    rot_matrix_x = torch.stack([torch.stack([torch.ones_like(bb_rot[:,:,0]),torch.zeros_like(bb_rot[:,:,0]),torch.zeros_like(bb_rot[:,:,0])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,0]),torch.cos(bb_rot[:,:,0]),-torch.sin(bb_rot[:,:,0])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,0]),torch.sin(bb_rot[:,:,0]),torch.cos(bb_rot[:,:,0])],dim=2)],dim=2)

    rot_matrix_y = torch.stack([torch.stack([torch.cos(bb_rot[:,:,1]),torch.zeros_like(bb_rot[:,:,1]),torch.sin(bb_rot[:,:,1])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,1]),torch.ones_like(bb_rot[:,:,1]),torch.zeros_like(bb_rot[:,:,1])],dim=2),\
                                torch.stack([-torch.sin(bb_rot[:,:,1]),torch.zeros_like(bb_rot[:,:,1]),torch.cos(bb_rot[:,:,1])],dim=2)],dim=2)

    rot_matrix_z = torch.stack([torch.stack([torch.cos(bb_rot[:,:,2]),-torch.sin(bb_rot[:,:,2]),torch.zeros_like(bb_rot[:,:,2])],dim=2),\
                                torch.stack([torch.sin(bb_rot[:,:,2]),torch.cos(bb_rot[:,:,2]),torch.zeros_like(bb_rot[:,:,2])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,2]),torch.zeros_like(bb_rot[:,:,2]),torch.ones_like(bb_rot[:,:,2])],dim=2)],dim=2)
    rot_matrix = rot_matrix_x@rot_matrix_y@rot_matrix_z
    return rot_matrix#B-K-3-3

def clamp_preserve_gradients(x, lower, upper):
    return x + (x.clamp(lower, upper) - x).detach()

def paddingORfilling(x,num_points,type='zeros',task='padding'):
    #x: BxN_i-D if padding, B-N_m-D if filling
    #num_points: B
    B = num_points.shape[0]
    D = x.shape[-1]
    max_num = torch.max(num_points)
    if type == 'zeros':
        out = torch.zeros([B,max_num.long(),D],device=x.device)
    if type == 'ones':
        out = torch.ones([B,max_num.long(),D],device=x.device)
    count = 0
    for b in range(B):
        num_i = num_points[b]
        if task == 'padding':
            out[b,:num_i] = x[count:count+num_i]
        if task == 'filling':
            out[b,...,:num_i,:] = x[b,...,:num_i,:]
        count += num_i
    #out: B-N_m-D
    return out

def get_rot_matrix(bb_rot):
    #bb_rot:B-K-3
    rot_matrix_x = torch.stack([torch.stack([torch.ones_like(bb_rot[:,:,0]),torch.zeros_like(bb_rot[:,:,0]),torch.zeros_like(bb_rot[:,:,0])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,0]),torch.cos(bb_rot[:,:,0]),-torch.sin(bb_rot[:,:,0])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,0]),torch.sin(bb_rot[:,:,0]),torch.cos(bb_rot[:,:,0])],dim=2)],dim=2)

    rot_matrix_y = torch.stack([torch.stack([torch.cos(bb_rot[:,:,1]),torch.zeros_like(bb_rot[:,:,1]),torch.sin(bb_rot[:,:,1])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,1]),torch.ones_like(bb_rot[:,:,1]),torch.zeros_like(bb_rot[:,:,1])],dim=2),\
                                torch.stack([-torch.sin(bb_rot[:,:,1]),torch.zeros_like(bb_rot[:,:,1]),torch.cos(bb_rot[:,:,1])],dim=2)],dim=2)

    rot_matrix_z = torch.stack([torch.stack([torch.cos(bb_rot[:,:,2]),-torch.sin(bb_rot[:,:,2]),torch.zeros_like(bb_rot[:,:,2])],dim=2),\
                                torch.stack([torch.sin(bb_rot[:,:,2]),torch.cos(bb_rot[:,:,2]),torch.zeros_like(bb_rot[:,:,2])],dim=2),\
                                torch.stack([torch.zeros_like(bb_rot[:,:,2]),torch.zeros_like(bb_rot[:,:,2]),torch.ones_like(bb_rot[:,:,2])],dim=2)],dim=2)
    rot_matrix = rot_matrix_x@rot_matrix_y@rot_matrix_z
    return rot_matrix#B-K-3-3

def _copysign(a, b):
    signs_differ = (a < 0) != (b < 0)
    return torch.where(signs_differ, -a, a)
def random_quaternions(n,device):
    o = torch.randn((n, 4), device=device)
    s = (o * o).sum(1)
    o = o / _copysign(torch.sqrt(s), o[:, 0])[:, None]
    return o
def quaternion_to_matrix(quaternions):
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

def random_rot_matrix(n,device):
    o = random_quaternions(n,device)
    o = quaternion_to_matrix(o)
    return o

def cross_matrix(m):
    #M: B-K-3
    B,K,_ = m.size()
    M = torch.zeros([B,K,3,3],device=m.device)
    M[:,:,0,1] = -m[:,:,2]
    M[:,:,0,2] = m[:,:,1]
    M[:,:,1,0] = m[:,:,2]
    M[:,:,1,2] = -m[:,:,0]
    M[:,:,2,0] = -m[:,:,1]
    M[:,:,2,1] = m[:,:,0]
    return M

def to_group(w):
    #w: B-K-3
    w_group = torch.matrix_exp(cross_matrix(w))
    #w_group: B-K-3-3
    return w_group

def to_algebra(W):
    #W: B-K-3-3
    B,K,_,_ = W.size()
    trace = W[:,:,0,0]+W[:,:,1,1]+W[:,:,2,2]
    #trace: B-K
    w_norm = torch.arccos(((trace-1.)/2.).clamp(-1.,1.))[:,:,None]
    #w_norm: B-K-1
    w_vector = torch.zeros([B,K,3],device=W.device)
    w_vector[:,:,0] = W[:,:,2,1] - W[:,:,1,2]
    w_vector[:,:,1] = W[:,:,0,2] - W[:,:,2,0]
    w_vector[:,:,2] = W[:,:,1,0] - W[:,:,0,1]
    #w_vector: B-K-3
    w = torch.zeros([B,K,3],device=W.device)
    for b in range(B):
        for k in range(K):
            if w_norm[b,k] == 0.:
                w[b,k] = torch.zeros([3],device=W.device)
            else:
                w_normed = (1./(2.*torch.sin(w_norm[b,k])))*w_vector[b,k]
                #w_normed: B-K-3
                w[b,k] = w_normed*w_norm[b,k]
    return w,w_norm
