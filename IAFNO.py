# ---------------------------------------------------------------------------------------------
# Author: Yuchi Jiang
# LatestVersionDate: 08/13/2025
# ---------------------------------------------------------------------------------------------

# Many thanks to all the authors of:
# Guibas, J., Mardani, M., Li, Z., Tao, A., Anandkumar, A., Catanzaro, B.: Adaptive Fourier Neural Operators: Efficient Token Mixers for Transformers. arXiv preprint arXiv:2111.13587 (2021)

import math
import torch
import torch.fft
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
from torchvision import transforms

import matplotlib.pyplot as plt
from utilities3 import *


import operator
from functools import reduce
from functools import partial

from timeit import default_timer
import scipy.io
import os

from einops import rearrange
from timm.models.layers import DropPath, trunc_normal_

torch.manual_seed(123)
np.random.seed(123)

################################################################################################################################

class PatchEmbed(nn.Module):
    def __init__(self, length, patch_size, embed_dim, in_chans):              #####   Length & Patch_size must be 3 dims   #####
        super().__init__()
        num_patches = (length[0] // patch_size[0]) * (length[1] // patch_size[1]) * (length[2] // patch_size[2])
        self.length = length
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):

        ##### make sure an input of shape: (bs x y z c nt) #####

        x = x.flatten(4)                     ##### (bs x y z c*nt)
        x = x.permute(0, 4, 1, 2, 3)         ##### (bs c*nt x y z)
        x = self.proj(x)                     ##### (bs embed_dim x//px y//py z//pz)
        x = x.permute(0, 2, 3, 4, 1)         ##### (bs x//px y//py z//pz embed_dim)

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, act_layer=nn.GELU, drop=0.):
        super().__init__()
        self.out_features = out_features
        self.hidden_features = hidden_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):

        ##### make sure an input of shape: (bs x//px y//py z//pz embed_dim) #####

        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class Block(nn.Module):
    def __init__(
            self, nlayer, dim, patch_size, embed_dim, hidden_size_factor, num_blocks, in_chans, 
            drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, double_skip=True
        ):
        super().__init__()
        hidden_features = embed_dim * 4

        self.filter = AFNO(embed_dim, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = Mlp(embed_dim, hidden_features, embed_dim)
        self.norm1 = norm_layer(embed_dim)
        self.norm2 = norm_layer(embed_dim)
        self.double_skip = double_skip

    def forward(self, x):

        ##### must after patch_embed #####
        ##### input (bs x//px y//py z//pz embed_dim) #####

        residual = x

        x = self.norm1(x)
        x = self.filter(x)

        if self.double_skip:
            x = x + residual
            residual = x

        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x)
        x = x + residual

        ##### output (bs x//px y//py z//pz embed_dim) #####

        return x

################################################################################################################################

class AFNO(nn.Module):
    def __init__(self, hidden_size, hidden_size_factor, num_blocks, sparsity_threshold=0.01, hard_thresholding_fraction=1):
        super().__init__()

        self.hidden_size = hidden_size
        self.sparsity_threshold = sparsity_threshold
        self.num_blocks = num_blocks
        self.block_size = self.hidden_size // self.num_blocks
        self.hard_thresholding_fraction = hard_thresholding_fraction
        self.hidden_size_factor = hidden_size_factor
        self.scale = 0.02

        self.w1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size, self.block_size * self.hidden_size_factor))
        self.b1 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor))
        self.w2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size * self.hidden_size_factor, self.block_size))
        self.b2 = nn.Parameter(self.scale * torch.randn(2, self.num_blocks, self.block_size))

    def forward(self, x):
        bias = x

        dtype = x.dtype
        x = x.float()
        B, X, Y, Z, C = x.shape

        x = torch.fft.rfftn(x, dim=(1, 2, 3), norm="ortho")
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size)

        o1_real = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o1_imag = torch.zeros([B, x.shape[1], x.shape[2], x.shape[3], self.num_blocks, self.block_size * self.hidden_size_factor], device=x.device)
        o2_real = torch.zeros(x.shape, device=x.device)
        o2_imag = torch.zeros(x.shape, device=x.device)

        total_modes = Z // 2 + 1
        kept_modes = int(total_modes * self.hard_thresholding_fraction)

        o1_real[:, :, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].real, self.w1[0]) - \
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].imag, self.w1[1]) + \
            self.b1[0]
        )

        o1_imag[:, :, :, :kept_modes] = F.relu(
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].imag, self.w1[0]) + \
            torch.einsum('...bi,bio->...bo', x[:, :, :, :kept_modes].real, self.w1[1]) + \
            self.b1[1]
        )

        o2_real[:, :, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :, :kept_modes], self.w2[0]) - \
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :, :kept_modes], self.w2[1]) + \
            self.b2[0]
        )

        o2_imag[:, :, :, :kept_modes] = (
            torch.einsum('...bi,bio->...bo', o1_imag[:, :, :, :kept_modes], self.w2[0]) + \
            torch.einsum('...bi,bio->...bo', o1_real[:, :, :, :kept_modes], self.w2[1]) + \
            self.b2[1]
        )

        x = torch.stack([o2_real, o2_imag], dim=-1)
        x = F.softshrink(x, lambd=self.sparsity_threshold)
        x = torch.view_as_complex(x)
        x = x.reshape(B, x.shape[1], x.shape[2], x.shape[3], C)
        x = torch.fft.irfftn(x, s=(X, Y, Z), dim=(1, 2, 3), norm="ortho")
        x = x.type(dtype)
        return x + bias

##################################################################################################################

class IAFNONet(nn.Module):
    def __init__(
            self,
            dim,
            patch_size,
            embed_dim,
            num_blocks,
            in_chans,
            out_chans,
            ex_layer,
            nlayer,
            hidden_size_factor,
            dim_f,
            drop_rate=0.,
            drop_path_rate=0.,
            sparsity_threshold=0.01,
            hard_thresholding_fraction=1.0,
        ):
        super().__init__()
        self.dim = dim
        self.dim_f = dim_f
        self.out_chans = out_chans
        self.patch_size = patch_size
        norm_layer = partial(nn.LayerNorm, eps=1e-6)
        self.patch_embed = PatchEmbed(dim, patch_size, embed_dim, in_chans)
        self.pos_embed = nn.Parameter(torch.zeros(1, dim[0] // patch_size[0], dim[1] // patch_size[1], dim[2] // patch_size[2], embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, ex_layer)]

        self.h = self.dim[0] // self.patch_size[0]
        self.w = self.dim[1] // self.patch_size[1]
        self.z = self.dim[2] // self.patch_size[2]

        self.blocks = nn.ModuleList([
            Block(
                nlayer, dim, patch_size, embed_dim, hidden_size_factor, num_blocks, in_chans).cuda()
            for i in range(ex_layer)])

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, self.out_chans*self.patch_size[0]*self.patch_size[1]*self.patch_size[2], bias=False)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        if (ex_layer!=1 & nlayer==1):
            for j in range(ex_layer):
                x = self.blocks[j](x)
        else:
            for i in range(nlayer):
                for j in range(ex_layer):
                    coef = 1/(nlayer * ex_layer)
                    x = x + self.blocks[j](x) * coef

        return x

    def forward(self, x):

        ##### considering patch size [2,2,2] and input shape [x=64,y=65,z=32], we add a zero-information padding in y-axis to provide a smoother patching
        ##### in other words: dim_f:[64,65,32] --> dim:[64,66,32] 
        if (dim_f[0]!=dim[0]):
            pad = torch.zeros(x.shape[0], 1, x.shape[2], x.shape[3], x.shape[4], x.shape[5]).to(device)
            x = torch.cat((x, pad), 1).to(device)
        if (dim_f[1]!=dim[1]):
            pad = torch.zeros(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4], x.shape[5]).to(device)
            x = torch.cat((x, pad), 2).to(device)
        if (dim_f[2]!=dim[2]):
            pad = torch.zeros(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[4], x.shape[5]).to(device)
            x = torch.cat((x, pad), 3).to(device)
        
        x = self.forward_features(x)
        x = self.head(x)
        x = rearrange(
            x,
            "b h w z (p1 p2 p3 c_out) -> b (h p1) (w p2) (z p3) c_out",
            p1=self.patch_size[0],
            p2=self.patch_size[1],
            p3=self.patch_size[2],
            h=self.dim[0] // self.patch_size[0],
            w=self.dim[1] // self.patch_size[1],
            z=self.dim[2] // self.patch_size[2],
        )
        if (dim_f[0]!=dim[0]):
            x = x[:, :-1, :, :, :]
        if (dim_f[1]!=dim[1]):
            x = x[:, :, :-1, :, :]
        if (dim_f[2]!=dim[2]):
            x = x[:, :, :, :-1, :]
        return x

##################################################################################################################

device = torch.device("cuda")

weight_decay_value = 1e-11

learning_rate = 0.001                   ###########################################               LearningRate               ###################################################

epochs = 100                            ############################################                 EPOCHS                 ####################################################

nlayer = 20                             ############################################             IMPLICIT LAYER             ####################################################

ex_layer = 1                            ##########################################               EXPLICIT LAYER               ##################################################

hidden_size_factor = 3                  ##########################################                 HIDDENSIZE                 ##################################################

trainsets_num = 20                      ##########################################                 #TRAINSETS                 ##################################################

batch_size = 5                          ##########################################                  BATCHSIZE                 ##################################################

patch_size = (2,2,2)                    ##########################################                  PATCHSIZE                 ##################################################

embed_dim = 200                         ##########################################                  EMBEDDIM                  ##################################################

num_blocks = 1                          ##########################################                   #BLOCKS                  ##################################################

scheduler_step = 10
scheduler_gamma = 0.5

##################################################################################################################

print('Epochs:', epochs, '  LearningRate:', learning_rate, '  SchedulerStep:', scheduler_step, '  SchedulerGamma:', scheduler_gamma)
print('Batchsize:', batch_size, '  #Implicit_layer:', nlayer, '  #Explicit_layer:', ex_layer, '  #Trainsets:', trainsets_num)
print('Embed_dim:', embed_dim, '  Patch_size:', patch_size, '  Num_blocks:', num_blocks, '  HiddensizeFactor:', hidden_size_factor)

runtime = np.zeros(2, )
t1 = default_timer()

vor_data = np.load('/mnt/Simulation_Data/YuchiJiang/Data/data_re590_400d200_64_65_32.npy') 
print('Shape of your input data',vor_data.shape)
vor_data = vor_data[..., 0:3]                       # extract velocity information [u,v,w,p]-->[u,v,w]
vor_data = vor_data[0:trainsets_num,...]            # extract trainsets
vor_data = torch.from_numpy(vor_data)                  

input_list = []
output_list = []

##### input: [U_(m),U_(m+1),U_(m+2),U_(m+3),U_(m+4)]; output: [U_(m+5)-U_(m+4)]
for j in range(vor_data.shape[0]):
    for i in range(vor_data.shape[1]-5):
        input_list.append(vor_data[j,i:i+5,...])
        output_6m5 = (vor_data[j,i+5,...]-vor_data[j,i+4,...])
        output_list.append(output_6m5)            

input_set = torch.stack(input_list)
output_set = torch.stack(output_list)
input_set = input_set.permute(0,2,3,4,5,1)

in_chans = input_set.shape[-1] * input_set.shape[-2]            ############################                  INPUT WIDTHS                  ################################

out_chans = input_set.shape[-2]                                 ############################                  OUTPUT WIDTHS                  ###############################

dim_x, dim_y, dim_z = input_set.shape[1:4]
dim_f = input_set.shape[1:4]                                    ############################                  REAL DIMENSION                 ###############################

##### since patch size is 2, we must guarantee divisibility
if (dim_x%2 != 0):
    dim_x = dim_x + 1

if (dim_y%2 != 0):
    dim_y = dim_y + 1

if (dim_z%2 != 0):
    dim_z = dim_z + 1

dim = (dim_x, dim_y, dim_z)                                     ############################               DIMENSION USED IN MODEL              ############################

full_set = torch.utils.data.TensorDataset(input_set, output_set)
train_dataset, test_dataset = torch.utils.data.random_split(full_set, [int(0.8*len(full_set)), 
                                                                       len(full_set)-int(0.8*len(full_set))])

# Data loader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)


##################################################################################################################


model = IAFNONet(dim, patch_size, embed_dim, num_blocks, in_chans, out_chans, ex_layer, nlayer, hidden_size_factor, dim_f).to(device)


print('Model Total Params:', count_params(model))
PARAMS = count_params(model)

print('Input Channels:', in_chans, '  Output Channels:', out_chans, '  Dimension used in model:', dim)

##### note that to ensure fair comparison with the IUFNO model:
##### Li, Z., Peng, W., Yuan, Z., Wang, J.: Long-term predictions of turbulence by implicit U-Net enhanced Fourier neural operator. Physics of Fluids 35(7), 075145 (2023)
##### the scheduler is not actually used in model
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay_value)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)

mse_train = []
mse_test = []
timecost = []


myloss = LpLoss()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    for xx, yy in train_loader:
        
        xx = xx.to(device)
        yy = yy.to(device)
        im = model(xx).to(device)
        
        train_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    mse_train.append(train_loss.item())
        

    with torch.no_grad():
        for xx, yy in test_loader:
            xx = xx.to(device)
            yy = yy.to(device)

            im = model(xx).to(device)
            test_loss = myloss(im.reshape(im.shape[0], -1), yy.reshape(yy.shape[0], -1))
        mse_test.append(test_loss.item())

    t2 = default_timer()
    
    print(ep, "%.2f" % (t2 - t1), 'train_loss: {:.4f}'.format(train_loss.item()), 
          'test_loss: {:.4f}'.format(test_loss.item()))
    
    timecost.append(t2-t1)
    

MSE_save=np.dstack((timecost,mse_train,mse_test)).squeeze()
np.savetxt(f'/mnt/Simulation_Data/YuchiJiang/{trainsets_num}g_B{batch_size}_H{hidden_size_factor}_I{nlayer}_E{ex_layer}_ED{embed_dim}_Ep{epochs}.dat',MSE_save,fmt="%16.7f")

file_name = f'{trainsets_num}g_B{batch_size}_H{hidden_size_factor}_I{nlayer}_E{ex_layer}_ED{embed_dim}_Ep{epochs}.dat'                                                                   
FilePath = os.path.join("//mnt/Simulation_Data/YuchiJiang/", file_name)
with open(FilePath, "a") as filewrite:
    filewrite.write("\n")
    filewrite.write(f'Epochs:{epochs}, LearningRate: {learning_rate}, SchedulerStep:{scheduler_step}, SchedulerGamma:{scheduler_gamma}')
    filewrite.write("\n")
    filewrite.write(f'Batchsize:{batch_size}, #Implicit_layer:{nlayer}, #Explicit_layer:{ex_layer}, #Trainsets:{trainsets_num}')
    filewrite.write("\n")
    filewrite.write(f'Patchsize:{patch_size}, HiddensizeFactor:{hidden_size_factor}, Embed_dim:{embed_dim}, Num_blocks:{num_blocks}, Dim:{dim}')