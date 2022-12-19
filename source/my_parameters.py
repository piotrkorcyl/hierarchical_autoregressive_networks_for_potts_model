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
## File: my_parameters.py
## Authors: T. Stebel, P. Korcyl
## Contact: tomasz.stebel@uj.edu.pl, piotr.korcyl@uj.edu.pl
## 
## Version: 1.0
## 
## Description:
## sets some of the main parameters
## 
##


import torch
n_block=2
lattice = 'sqr'
boundary = 'periodic'
ham = 'fm' 

#network parameters:
net_depth= 2
net_width =1
half_kernel_size = 6  ##also called kernell radius

bias = True
nmcmc_or_nis = True  # for nmcmc set 'True' for nis set 'False'
fitting_tau = False

res_block = False
x_hat_clip = False
final_conv = False
epsilon = 1e-8
device = torch.device('cuda:0')
#device = torch.device('cpu')
default_dtype_torch=  torch.float32

#learning parameters
max_step = 10000
start_counting = 100
colected_batches = 100
beta_anneal= 0.0; ##998

batch_size=1
print_step=1
lr = 0.0005
