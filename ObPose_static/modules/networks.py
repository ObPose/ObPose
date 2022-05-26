import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from utils.funcs import to_gaussian

from modules.point_transformer_modules import (
    PointTransformerBlock,
    TransitionDown,
    TransitionUp,
    AggregateLayer
)

from modules.kpconv_modules import (
    InputBlock,
    KPConvBlock,
    KPConvTransitionUp,
    KPConvAggregateLayer
)

import pdb

class ClusterEncoder(nn.Module):
    def __init__(self):
        super(ClusterEncoder, self).__init__()
        # encoder
        self.in_mlp = nn.Sequential(
            nn.Conv1d(16+3,16, kernel_size=1,bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))#,
            #nn.Conv1d(32,32,kernel_size=1,bias=False),
            #nn.GroupNorm(2,32),
            #nn.ReLU(inplace=True)
        #)
        self.enc_layer1 = self._make_layer(16,1,nsample=8)
        self.down1to2 = TransitionDown(16,32,stride=4,num_neighbors=16)
        self.enc_layer2 = self._make_layer(32,1,nsample=16)
        self.down2to3 = TransitionDown(32,32, stride=4,num_neighbors=16)
        self.enc_layer3 = self._make_layer(32,1,nsample=16)
        self.down3to4 = TransitionDown(32,64,stride=4,num_neighbors=16)
        self.enc_layer4 = self._make_layer(64,1,nsample=16)
        self.down4to5 = TransitionDown(64,64,stride=4,num_neighbors=16)
        self.enc_layer5 = self._make_layer(64,1,nsample=16)
        self.final_layer = AggregateLayer(64,64)

    def _make_layer(self,planes,blocks,nsample):
        layers = []
        for _ in range(blocks):
            layers.append(PointTransformerBlock(planes,num_neighbors=nsample))
        return nn.Sequential(*layers)

    def forward(self,p1,x1):
        x1 = self.in_mlp(x1)
        p1x1 = self.enc_layer1([p1, x1])
        p2x2 = self.down1to2(p1x1)
        p2x2 = self.enc_layer2(p2x2)
        p3x3 = self.down2to3(p2x2)
        p3x3 = self.enc_layer3(p3x3)
        p4x4 = self.down3to4(p3x3)
        p4x4 = self.enc_layer4(p4x4)
        p5x5 = self.down4to5(p4x4)
        p5x5 = self.enc_layer5(p5x5)
        y = self.final_layer(p5x5)
        #x: BxK-C
        #x = self.gaussian_mlp(x)
        #x: BxK-C
        return y

class PointEncoder(nn.Module):
    def __init__(self):
        super(PointEncoder, self).__init__()
        # encoder
        self.in_mlp = nn.Sequential(
            nn.Conv1d(6,16, kernel_size=1,bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True))#,
            #nn.Conv1d(16,16,kernel_size=1,bias=False),
            #nn.GroupNorm(1,16),
            #nn.ReLU(inplace=True)
        #)
        self.enc_layer1 = self._make_layer(16,1,nsample=8)
        self.down1to2 = TransitionDown(16,16,stride=4,num_neighbors=16)
        self.enc_layer2 = self._make_layer(16,1,nsample=16)
        self.down2to3 = TransitionDown(16,32, stride=4,num_neighbors=16)
        self.enc_layer3 = self._make_layer(32,1,nsample=16)
        self.down3to4 = TransitionDown(32,64,stride=4,num_neighbors=16)
        self.enc_layer4 = self._make_layer(64,1,nsample=16)
        self.down4to5 = TransitionDown(64,64,stride=4,num_neighbors=16)
        self.enc_layer5 = self._make_layer(64,1,nsample=16)
        self.final_layer = AggregateLayer(64,64)
        # decoder
        self.dec_mlp = nn.Sequential(
            nn.Conv1d(64,64,kernel_size=1,bias=False),
            nn.GroupNorm(64//16,64),
            nn.ReLU(inplace=True),
            nn.Conv1d(64,64,kernel_size=1, bias=False),
            nn.GroupNorm(64//16,64),
            nn.ReLU(inplace=True)
        )
        self.dec_layer5 = self._make_layer(64,1,nsample=16)
        self.up5to4 = TransitionUp(64,64,64)
        self.dec_layer4 = self._make_layer(64,1,nsample=16)
        self.up4to3 = TransitionUp(64,32,32)
        self.dec_layer3 = self._make_layer(32,1,nsample=16)
        self.up3to2 = TransitionUp(32,16,16)
        self.dec_layer2 = self._make_layer(16,1,nsample=16)
        #self.up2to1 = TransitionUp(32,16,16)
        #self.dec_layer1 = self._make_layer(16,1,nsample=8)
        self.out_mlp = nn.Sequential(
            nn.Conv1d(16,16,kernel_size=1,bias=False),
            nn.GroupNorm(1,16),
            nn.ReLU(inplace=True),
            nn.Conv1d(16,16,kernel_size=1)
        )

    def _make_layer(self,planes,blocks,nsample):
        layers = []
        for _ in range(blocks):
            layers.append(PointTransformerBlock(planes,num_neighbors=nsample))
        return nn.Sequential(*layers)

    def forward(self,p1,x1):
        x1 = self.in_mlp(x1)
        p1x1 = self.enc_layer1([p1, x1])
        p2x2 = self.down1to2(p1x1)
        p2x2 = self.enc_layer2(p2x2)
        p3x3 = self.down2to3(p2x2)
        p3x3 = self.enc_layer3(p3x3)
        p4x4 = self.down3to4(p3x3)
        p4x4 = self.enc_layer4(p4x4)
        p5x5 = self.down4to5(p4x4)
        p5, x5 = self.enc_layer5(p5x5)
        scene_enc = self.final_layer([p5,x5])
        # decoder
        y = self.dec_mlp(x5)
        p5y = self.dec_layer5([p5, y])
        p4y = self.up5to4(p5y, p4x4)
        p4y = self.dec_layer4(p4y)
        p3y = self.up4to3(p4y, p3x3)
        p3y = self.dec_layer3(p3y)
        p2y = self.up3to2(p3y, p2x2)
        p2,y = self.dec_layer2(p2y)
        #p1y = self.up2to1(p2y, p1x1)
        #p1, y = self.dec_layer1(p1y)
        y = self.out_mlp(y)
        y = y.permute(0,2,1)
        return y,scene_enc,p2


class ClusterEncoderKPConv(nn.Module):
    def __init__(self):
        super(ClusterEncoderKPConv, self).__init__()
        # encoder
        r_1 = 0.025
        self.enc_in = InputBlock(16+3,16,r_1)
        self.enc_11 = KPConvBlock(16,16,r_1)
        r_2 = r_1*2
        self.enc_1to2 = KPConvBlock(16,32,r_2,4)
        self.enc_21 = KPConvBlock(32,32,r_2)
        self.enc_22 = KPConvBlock(32,32,r_2)
        r_3 = r_2*2
        self.enc_2to3 = KPConvBlock(32,64,r_3,4)
        self.enc_31 = KPConvBlock(64,64,r_3)
        self.enc_32 = KPConvBlock(64,64,r_3)
        r_4 = r_3*2
        self.enc_3to4 = KPConvBlock(64,64,r_4,4)
        self.enc_41 = KPConvBlock(64,64,r_4)
        self.enc_42 = KPConvBlock(64,64,r_4)
        self.enc_layer = KPConvAggregateLayer(64,64,1.0)
        self.out_mlp = nn.Sequential(
            nn.Linear(64,64,bias=False),
            nn.GroupNorm(64//16,64),
            nn.LeakyReLU(0.1),
            nn.Linear(64,64,bias=False)
        )

    def forward(self,p,x):
        #p: B-N-3
        #x: B-C-N
        p,x,idx = self.enc_in(p,x)
        p11,x11,idx11 = self.enc_11(p,x)
        p2,x2,idx2 = self.enc_1to2(p11,x11)
        p21,x21,idx21 = self.enc_21(p2,x2)
        p22,x22,idx22 = self.enc_22(p21,x21)
        p3,x3,idx3 = self.enc_2to3(p22,x22)
        p31,x31,idx31 = self.enc_31(p3,x3)
        p32,x32,idx32 = self.enc_32(p31,x31)
        p4,x4,idx4 = self.enc_3to4(p32,x32)
        p41,x41,idx41 = self.enc_41(p4,x4)
        p42,x42,idx42 = self.enc_42(p41,x41)
        enc = self.enc_layer(p42,x42)
        enc = self.out_mlp(enc)
        return enc

class PointEncoderKPConv(nn.Module):
    def __init__(self):
        super(PointEncoderKPConv, self).__init__()
        # encoder
        r_1 = 0.025
        self.enc_in = InputBlock(6,16,r_1,8)
        self.enc_11 = KPConvBlock(16,16,r_1)
        r_2 = r_1*2
        self.enc_1to2 = KPConvBlock(16,32,r_2,4)
        self.enc_21 = KPConvBlock(32,32,r_2)
        self.enc_22 = KPConvBlock(32,32,r_2)
        r_3 = r_2*2
        self.enc_2to3 = KPConvBlock(32,64,r_3,4)
        self.enc_31 = KPConvBlock(64,64,r_3)
        self.enc_32 = KPConvBlock(64,64,r_3)
        r_4 = r_3*2
        self.enc_3to4 = KPConvBlock(64,64,r_4,4)
        self.enc_41 = KPConvBlock(64,64,r_4)
        self.enc_42 = KPConvBlock(64,64,r_4)
        self.enc_layer = KPConvAggregateLayer(64,64,1.0)
        self.dec_4to3 = KPConvTransitionUp(64,64,64)
        self.dec_3to2 = KPConvTransitionUp(64,32,32)
        self.out_mlp = nn.Sequential(
            nn.Conv1d(32,32,kernel_size=1,bias=False),
            nn.GroupNorm(2,32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32,16,kernel_size=1,bias=False)
        )

    def forward(self,p,x):
        #p: B-N-3
        #x: B-C-N
        p,x,idx = self.enc_in(p,x)
        p11,x11,idx11 = self.enc_11(p,x)
        p2,x2,idx2 = self.enc_1to2(p11,x11)
        p21,x21,idx21 = self.enc_21(p2,x2)
        p22,x22,idx22 = self.enc_22(p21,x21)
        p3,x3,idx3 = self.enc_2to3(p22,x22)
        p31,x31,idx31 = self.enc_31(p3,x3)
        p32,x32,idx32 = self.enc_32(p31,x31)
        p4,x4,idx4 = self.enc_3to4(p32,x32)
        p41,x41,idx41 = self.enc_41(p4,x4)
        p42,x42,idx42 = self.enc_42(p41,x41)
        enc = self.enc_layer(p42,x42)
        p3_up,x3_up = self.dec_4to3(p42,x42,p32,x32)
        p2_up,x2_up = self.dec_3to2(p32,x32,p22,x22)
        x_out = self.out_mlp(x2_up)
        return p2_up,x_out.permute(0,2,1),enc


class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_out)

    def forward(self, inputs):
        x = F.elu(self.fc1(inputs))
        x = self.fc2(x)
        return x

def clamp_preserve_gradients(x, lower, upper):
    return x + (x.clamp(lower, upper) - x).detach()

class Block(nn.Module):
    def __init__(self,in_c,out_c,identity_downsample=None, stride=1):
        super(Block, self).__init__()

        self.conv1 = nn.Conv2d(in_c,out_c,kernel_size=3,stride=stride,padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c,out_c,kernel_size=1,stride=1,padding=0)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample

    def forward(self,x):
        identity = x
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)
        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        enc_dim = 128
        self.conv1 = nn.Conv2d(3+96+3,enc_dim,kernel_size=7,stride=2,padding=3)
        self.bn1 = nn.BatchNorm2d(enc_dim)
        self.relu = nn.ReLU()

        # ResNetLayers
        self.layer1 = self.make_layers(enc_dim,stride=1)
        self.layer2 = self.make_layers(enc_dim,stride=2)
        self.layer3 = self.make_layers(enc_dim,stride=1)
        self.layer4 = self.make_layers(enc_dim,stride=1)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

    def make_layers(self,c,stride):
        layers = []
        ids = nn.Sequential(nn.Conv2d(c,c, kernel_size=1,stride=stride),
                                      nn.BatchNorm2d(c))
        layers.append(Block(c,c,ids,stride))
        layers.append(Block(c,c)) 
        return nn.Sequential(*layers)


def build_grid(resolution,device):
    ranges = [np.linspace(0., 1., num=res) for res in resolution]
    grid = np.meshgrid(*ranges, sparse=False, indexing="ij")
    grid = np.stack(grid, axis=-1)
    grid = np.reshape(grid, [resolution[0], resolution[1], -1])
    grid = np.expand_dims(grid, axis=0)
    grid = grid.astype(np.float32)
    return torch.from_numpy(np.concatenate([grid, 1.0 - grid], axis=-1)).to(device)

class SoftPositionEmbed(nn.Module):
    def __init__(self, hidden_size, resolution,device):
        """Builds the soft position embedding layer.
        Args:
        hidden_size: Size of input feature dimension.
        resolution: Tuple of integers specifying width and height of grid.
        """
        super().__init__()
        self.embedding = nn.Linear(4, hidden_size, bias=True)
        self.grid = build_grid(resolution,device)

    def forward(self, inputs):
        grid = self.embedding(self.grid)
        return inputs + grid

