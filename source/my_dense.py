## 
## This file is part of the Hierarchical Autoregressive Neural Network algorithm for simulations of the Potts model 
## https://github.com/piotrkorcyl/hierarchical_autoregressive_networks_for_potts_model
## Copyright (c) 2022 P. Bialas, P. Korcyl, T. Stebel
##
## Description of the algorithm can be found in:
## Hierarchical autoregressive neural networks for statistical systems, Comput.Phys.Commun. 281 (2022) 108502, arXiv:2203.10989 [cond-mat.stat-mech]
## Simulating first-order phase transition with hierarchical autoregressive networks, arXiv:2212.04955 [cond-mat.stat-mech]
##
## The implementation is based on the VAN algorithm implementation accessible in:
## https://github.com/wdphy16/stat-mech-van
## 
## This program is free software: you can redistribute it and/or modify  
## it under the terms of the GNU General Public License as published by  
## the Free Software Foundation, version 3.
##
## This program is distributed in the hope that it will be useful, but 
## WITHOUT ANY WARRANTY; without even the implied warranty of 
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
## General Public License for more details.
##
## You should have received a copy of the GNU General Public License 
## along with this program. If not, see <http://www.gnu.org/licenses/>.
## 
## File: my_dense.py
## Authors: T. Stebel, P. Korcyl
## Contact: tomasz.stebel@uj.edu.pl, piotr.korcyl@uj.edu.pl
## 
## Version: 1.0
## 
## Description:
## contains the definitions of the neural networks
## 
##


import torch
from numpy import log
from torch import nn
import numpy as np


class MaskedLinear(nn.Linear):
    def __init__(self, in_channels, out_channels, Q, n, bias, exclusive):          # n=L^2 =number of spins
        super(MaskedLinear, self).__init__(in_channels * n, out_channels * n,  # parameters for nn.Linear class - we increase n times input channels and n times output channels
                                           bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.exclusive = exclusive

        self.register_buffer('mask', torch.ones([int(self.n/Q)] * 2))
        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)                   # mask for n=3:  [[0,0,0],[1,0,0],[1,1,0]]
        else:
            self.mask = torch.tril(self.mask)                       # mask for n=3:  [[1,0,0],[1,1,0],[1,1,1]]
        self.mask = torch.cat([self.mask] * in_channels, dim=1)     # replicate mask in_channels times in dim 1
        self.mask = torch.cat([self.mask] * out_channels, dim=0)    # replicate mask out_channels times in dim 0


        mask_tmp=torch.zeros(n,n)
        for i in range(0,int(n/Q)):
 
            
            for j in range(0,int(n/Q)):
                if self.mask[i,j] == 1:
                    for x in range(0,Q):
                        for y in range(0,Q):
                            mask_tmp[Q*i+x,Q*j+y] = 1
                else:
                    for x in range(0,Q):
                        for y in range(0,Q):
                            mask_tmp[Q*i+x,Q*j+y] = 0

        self.mask = mask_tmp

        self.weight.data *= self.mask

        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

    def extra_repr(self):
        return (super(MaskedLinear, self).extra_repr() +
                ', exclusive={exclusive}'.format(**self.__dict__))



class MaskedLinear_nonsquare(nn.Linear):
    def __init__(self, in_channels, out_channels, Q, n, m, bias, exclusive):          # n=number of spins interior + borders, m= numer of spins interior
        super(MaskedLinear_nonsquare, self).__init__(in_channels * n, out_channels * m,  # parameters for nn.Linear class - we increase n times input channels and m times output channels
                                           bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.m = m
        self.exclusive = exclusive

        self.register_buffer('mask', torch.ones([int(self.m/Q), int(self.m/Q)] ))
        self.mask = torch.tril(self.mask)                       # mask for n=3:  [[1,0,0],[1,1,0],[1,1,1]]
        self.mask = torch.cat([torch.ones([int(self.m/Q), int((self.n-self.m)/Q)]), self.mask],dim=1)
        
        
        self.mask = torch.cat([self.mask] * in_channels, dim=1)     # replicate mask in_channels times in dim 1
        self.mask = torch.cat([self.mask] * out_channels, dim=0)    # replicate mask out_channels times in dim 0


        mask_tmp=torch.zeros(m,n)
        for j in range(0,int(m/Q)):
 
            for i in range(0,int(n/Q)):
                if self.mask[j,i] == 1:
                    for x in range(0,Q):
                        for y in range(0,Q):
                            mask_tmp[Q*j+x,Q*i+y] = 1
                else:
                    for x in range(0,Q):
                        for y in range(0,Q):
                            mask_tmp[Q*j+x,Q*i+y] = 0

        self.mask = mask_tmp

        self.weight.data *= self.mask

        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

class MaskedLinear_border(nn.Linear):
    def __init__(self, in_channels, out_channels, Q, n, m, bias, exclusive):          # n=number of spins interior + borders, m= numer of spins interior
        super(MaskedLinear_border, self).__init__(in_channels * n, out_channels * n,  # parameters for nn.Linear class - we increase n times input channels and m times output channels
                                           bias)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n = n
        self.m = m
        self.exclusive = exclusive


        self.register_buffer('mask', torch.ones([int(self.m/Q), int(self.m/Q)] ))
        if self.exclusive:
            self.mask = 1 - torch.triu(self.mask)                   # mask for n=3:  [[0,0,0],[1,0,0],[1,1,0]]
        else:
            self.mask = torch.tril(self.mask)                       # mask for n=3:  [[1,0,0],[1,1,0],[1,1,1]]
        self.mask = torch.cat([torch.zeros([int((self.n-self.m)/Q), int(self.m/Q)]), self.mask],dim=0)
        self.mask = torch.cat([torch.ones([int(self.n/Q), int((self.n-self.m)/Q)]), self.mask],dim=1)

        
        self.mask = torch.cat([self.mask] * in_channels, dim=1)     # replicate mask in_channels times in dim 1
        self.mask = torch.cat([self.mask] * out_channels, dim=0)    # replicate mask out_channels times in dim 0

        mask_tmp=torch.zeros(n,n)
        for i in range(0,int(n/Q)):

            for j in range(0,int(n/Q)):
                if self.mask[j,i] == 1:
                    for x in range(0,Q):
                        for y in range(0,Q):
                            mask_tmp[Q*j+x,Q*i+y] = 1
                else:
                    for x in range(0,Q):
                        for y in range(0,Q):
                            mask_tmp[Q*j+x,Q*i+y] = 0

        self.mask = mask_tmp

        self.weight.data *= self.mask

        self.weight.data *= torch.sqrt(self.mask.numel() / self.mask.sum())

    def forward(self, x):
        return nn.functional.linear(x, self.mask * self.weight, self.bias)

class MADE_b(nn.Module):
    def __init__(self, Q,L,n_block,net_depth,net_width,bias,z2,translation_y,res_block,
                    x_hat_clip,epsilon,device):
        super(MADE_b, self).__init__()
        self.L = L
        self.n_block = n_block
        self.n = Q*(2*self.L - 1)   # Number of boundary sites
        self.net_depth = net_depth
        self.net_width = net_width
        self.bias = bias
        self.z2 = z2
        self.translation_y=translation_y        
        self.res_block = res_block
        self.x_hat_clip = x_hat_clip
        self.epsilon = epsilon
        self.device = device
        
        self.default_dtype = np.float32
        self.default_dtype_torch = torch.float32

        layers = []
        layers.append(
            MaskedLinear(
                1,
                1 if self.net_depth == 1 else self.net_width,
                Q,
                self.n,
                self.bias,
                exclusive=True))
        for count in range(self.net_depth - 2):
            layers.append(
                self._build_simple_block(Q, self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(self._build_simple_block(Q, self.net_width, 1))
        self.net = nn.Sequential(*layers)

    def _build_simple_block(self, Q, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(
            MaskedLinear(
                in_channels, out_channels, Q, self.n, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block



    def forward(self, x, Q):
        x = x.view(x.shape[0], -1)
        x_hat = self.net(x)
        x_hat = x_hat.view(x_hat.shape[0], 1, int(self.n/Q), Q)
        x_hat = torch.nn.functional.softmax(x_hat,dim=3)

        return x_hat

    def sample(self, batch_size, beta, Q):
        sample = torch.zeros(
            [batch_size, 1, self.n],
            dtype=self.default_dtype_torch,
            device=self.device)
        for i in range(int(self.n/Q)):
            x_hat = self.forward(sample, Q)
            prob = x_hat[:,0,i]
            selected = torch.multinomial(prob,1)
            selected = torch.nn.functional.one_hot(selected, Q)
            selected = torch.reshape(selected,(batch_size,Q))
            for j in range(0,Q):
                sample[:,0,Q*i+j] = selected[:,j]
        sample = sample.view(batch_size,1,int(self.n/Q),Q)
        return sample

    def _log_prob(self, sample, x_hat):

        log_prob = torch.flatten(torch.sum(torch.log(torch.sum((sample*x_hat),3)),2))
        return log_prob

    def log_prob(self, sample,beta, Q):
        x_hat = self.forward(sample, Q)
        log_prob = self._log_prob(sample, x_hat)        
        return log_prob





class MADE_i(nn.Module):
    def __init__(self, Q, l ,net_depth,net_width,bias,z2,translation_y,res_block,
                    x_hat_clip,epsilon,device):
        super(MADE_i, self).__init__()
        self.l = l  ## here l size of "cross"
        self.m = Q*(2*self.l-1) ## number of spins in "cross"
        self.n = Q*(int(self.m/Q) + 4*self.l)          # Number of all spins = interior spins + spins at border
        self.net_depth = net_depth
        self.net_width = net_width
        self.bias = bias
        self.z2 = z2
        self.translation_y=translation_y        
        self.res_block = res_block
        self.x_hat_clip = x_hat_clip
        self.epsilon = epsilon
        self.device = device
        
        self.default_dtype = np.float32
        self.default_dtype_torch = torch.float32

        layers = []
        layers.append(
            MaskedLinear_border(
                1,
                1 if self.net_depth == 1 else self.net_width,
                Q,
                self.n,
                self.m,
                self.bias,
                exclusive=True))
        for count in range(self.net_depth - 2):
            layers.append(
                self._build_simple_block_border(Q, self.net_width, self.net_width))
        if self.net_depth >= 2:
            layers.append(self._build_simple_block_nonsquare(Q, self.net_width, 1))
        self.net = nn.Sequential(*layers)


    def _build_simple_block_border(self, Q, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(
            MaskedLinear_border(
                in_channels, out_channels, Q, self.n, self.m, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block

    def _build_simple_block_nonsquare(self, Q, in_channels, out_channels):
        layers = []
        layers.append(nn.PReLU(in_channels * self.n, init=0.5))
        layers.append(
            MaskedLinear_nonsquare(
                in_channels, out_channels, Q, self.n, self.m, self.bias, exclusive=False))
        block = nn.Sequential(*layers)
        return block



    def forward(self, x, border, Q):
        x = x.view(x.shape[0], -1)
        border = border.view(border.shape[0],-1)
        x = torch.cat([border,x],dim=1)
        x_hat = self.net(x)
        x_hat = x_hat.view(x_hat.shape[0], 1, int(self.m/Q), Q)
        x_hat = torch.nn.functional.softmax(x_hat,dim=3)

        return x_hat

    def sample(self, batch_size,beta, border, Q):
        sample = torch.zeros(
            [batch_size, 1, self.m],
            dtype=self.default_dtype_torch,
            device=self.device)
        for i in range(int(self.m/Q)):
            x_hat = self.forward(sample, border, Q)
            prob = x_hat[:,0,i]
            selected = torch.multinomial(prob,1)
            selected = torch.nn.functional.one_hot(selected, Q)
            selected = torch.reshape(selected,(batch_size,Q))
            for j in range(0,Q):
                sample[:,0,Q*i+j] = selected[:,j]
        sample = sample.view(batch_size,1,int(self.m/Q),Q)   

        return sample

    def _log_prob(self, sample, x_hat, Q):
        log_prob = torch.flatten(torch.sum(torch.log(torch.sum((sample*x_hat),3)),2))
        
        return log_prob

    def log_prob(self, sample,border,beta, Q):
        x_hat = self.forward(sample,border, Q)
        log_prob = self._log_prob(sample, x_hat, Q)           
        return log_prob


def make_square(batch_size, default_dtype_torch, device, L, Q, n_block, sample_b, sample_i):
    block_width=L//n_block

    configuration= torch.empty([batch_size,1,L,L,Q],dtype=default_dtype_torch, device=device)
    block= torch.empty([batch_size,1,block_width,block_width,Q],dtype=default_dtype_torch, device=device)
    sample_b = sample_b.chunk(n_block**2,dim=2)
    
    block_number=0
    for i in range(n_block):
        for j in range(n_block):
            block_boundary=sample_b[block_number]
            block[:,:,:,0,:]= block_boundary[:,:,0:block_width,:]
            block[:,:,block_width-1,1:,:]=block_boundary[:,:,block_width:,:]
            block[:,:,:-1,1:,:]=sample_i[:,:,i,j,:,:,:]
            configuration[:,:,i*block_width:(i+1)*block_width,j*block_width:(j+1)*block_width,:]=block
            block_number+= 1

    return configuration

def find_borders(square):

    assert square.shape[2] == square.shape[3] 
    Li=square.shape[2]-2
    borders = torch.cat([ square[:,:,0,1:Li+1,:], square[:,:,1:Li+1,Li+1,:], square[:,:,Li+1,1:Li+1,:], square[:,:,1:Li+1,0,:] ], dim=2)

    return borders

def divide_into_squares_0(sample_b):
   
    size=int((sample_b.shape[2]+1)/2)

    B = torch.zeros([sample_b.shape[0],1,size+1,size+1,sample_b.shape[3]],dtype=torch.float32, device='cuda:0')

    B[:,:,0,0:size,:] = sample_b[:,:,0:size:,:]
    B[:,:,size,0:size,:] = sample_b[:,:,0:size:,:]

    B[:,:,1:size,0,:] = sample_b[:,:,size:,:]
    B[:,:,1:size,size,:] = sample_b[:,:,size:,:]

    B[:,:,0,size,:] = B[:,:,0,0,:]
    B[:,:,size,size,:] = B[:,:,size,0,:]


    return [B]

def divide_into_squares(square):

    assert square.shape[2] == square.shape[3]
    Li=square.shape[2]-2


    B11= square[:,:, 0:(Li+3)//2, 0:(Li+3)//2,:]#.detach().clone()

    B12= square[:,:, 0:(Li+3)//2 , (Li+1)//2:,:]#.detach().clone()

    B21= square[:,:, (Li+1)//2:  ,0:(Li+3)//2,:]#.detach().clone()

    B22 =square[:,:, (Li+1)//2: , (Li+1)//2: ,:]#.detach().clone()


    return [B11,B12,B21,B22]


def divide_into_square(sample):
    
    size=sample.shape[2]

    B = torch.zeros([sample.shape[0],1,size+1,size+1,sample.shape[4]],dtype=torch.float32, device='cuda:0')

    B[:,:,0:size,0:size,:] = sample[:,:,:,:,:]

    B[:,:,size,0:size,:] = B[:,:,0,0:size,:]

    B[:,:,1:size,size,:] = B[:,:,1:size,0,:]

    B[:,:,0,size,:] = B[:,:,0,0,:]
    B[:,:,size,size,:] = B[:,:,size,0,:]

    return [B]


def make_a_cross(list_of_spins,Q,default_dtype_torch, device):
    Li= (list_of_spins.shape[2]+1)//2
    batch_size=list_of_spins.shape[0]
    list_of_spins=torch.split(list_of_spins, [Li,(Li-1)//2,(Li-1)//2],dim=2)
    
    interior = torch.zeros([batch_size,1,Li+2,Li+2,Q],dtype=default_dtype_torch, device=device )
    interior[:,:,1:Li+1,(Li+1)//2]= list_of_spins[0]  ### fill first vertical line of cross
    interior[:,:,(Li+1)//2, 1:(Li+1)//2]= list_of_spins[1]  ## fill left arm of cross
    interior[:,:,(Li+1)//2, (Li+1)//2+1:Li+1]= list_of_spins[2]     ## fill right arm of cross

    return interior


def breakdown_a_cross(square):
    Li= square.shape[2]-2
    batch_size=square.shape[0]
    verticalbar = square[:,:,1:Li+1,(Li+1)//2,:]
    leftbar = square[:,:,(Li+1)//2, 1:(Li+1)//2,:]
    rightbar = square[:,:,(Li+1)//2, (Li+1)//2+1:Li+1,:]
    list_of_spins= torch.cat((verticalbar, leftbar,rightbar), dim=2)

    return list_of_spins



def add_into_squares(tensor_of_squares):
    
    assert tensor_of_squares.shape[3] == tensor_of_squares.shape[4]
    n_of_sq =tensor_of_squares.shape[0]
    
    tensor_of_squares=torch.chunk(tensor_of_squares, n_of_sq//4 )
        
    list_of_tensors=[]
    for four_of_squares in tensor_of_squares:
        b11 = four_of_squares[0]
        b12 = four_of_squares[1][:,:,:,1:]
        b21 = four_of_squares[2]
        b22 = four_of_squares[3][:,:,:,1:]
        
        b11= torch.cat([b11,b12],dim=3)
        b22= torch.cat([b21,b22],dim=3)
        b22=b22[:,:,1:,:]
        
        b11= torch.cat([b11,b22],dim=2)
        b11=torch.unsqueeze(b11,0)
        list_of_tensors.append(b11)

    list_of_tensors = torch.cat(list_of_tensors,dim=0)
    return list_of_tensors



def take_boundary(batch_size, default_dtype_torch, device, L, Q, n_block, configuration):

    boundary_length = 2*L-1

    sample_b = torch.zeros([batch_size,1,boundary_length,Q],dtype=default_dtype_torch, device=device)

    sample_b[:,:,0:L,:] = configuration[:,:,0,0:L,:];

    sample_b[:,:,L:boundary_length,:] = configuration[:,:,1:L,0,:]

    return sample_b

