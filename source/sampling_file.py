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
## File: my_sampling.py
## Authors: T. Stebel, P. Korcyl
## Contact: tomasz.stebel@uj.edu.pl, piotr.korcyl@uj.edu.pl
## 
## Version: 1.0
## 
## Description:
## contains the generation of configuration using the hierarchical set of neural network
## 
##


import torch
import numpy as np
import my_potts

from my_dense import make_square, find_borders, take_boundary, breakdown_a_cross
from my_dense import divide_into_square, divide_into_squares, divide_into_squares_0 ,make_a_cross, add_into_squares
from my_potts import energy
from my_parameters import ham, lattice, boundary


def build_sample(Q, net_b, int_nets, beta, L, batch_size, device = torch.device('cuda:0'), default_dtype_torch=  torch.float32):

    list_args_for_nets=[]


    sample_b = net_b.sample(batch_size, beta, Q)    
   
    list_args_for_nets.append(sample_b)
           
    set_squares = divide_into_squares_0(sample_b)


    del sample_b
    
    torch.cuda.empty_cache()
    
    
    list_with_cross=[]
    border_list=[]
    for square in set_squares:
        border_list.append(find_borders(square))
    batch_mult = len(border_list)
    border_list = torch.cat(border_list, dim=0)

    cross = int_nets[0].sample(batch_size*batch_mult, beta, border_list, Q ) 

    cross = cross.view(batch_mult, batch_size, 1, cross.shape[2], Q)
   
    for k, square in enumerate(set_squares):
        cross_tmp = make_a_cross(cross[k], Q, default_dtype_torch, device)
        square += cross_tmp

    size_of_square= set_squares[0].shape[2]
    square_need_cross = (size_of_square >= 5 )        

    n_i_nets=int(np.log2(L))
    blocks_widths=L//2**(np.arange(1,n_i_nets+1)-1)-1

    
    while square_need_cross:
        set_smaller_square=[]
        
        list_with_cross=[]
        border_list=[]
        for square in set_squares:
            set_square_1_4= divide_into_squares(square)
            Li = set_square_1_4[0].shape[2]-2
            number_of_net = np.where(blocks_widths==Li)[0][0]
            for smaller_square in set_square_1_4:
                border_list.append(find_borders(smaller_square))
                set_smaller_square.append(smaller_square)
        batch_mult = len(border_list)
        border_list = torch.cat(border_list, dim=0)

        cross = int_nets[number_of_net].sample(batch_size*batch_mult, beta, border_list, Q )
        cross = cross.view(batch_mult, batch_size, 1, cross.shape[2], Q)

        for k, square in enumerate(set_smaller_square):
            cross_tmp = make_a_cross(cross[k], Q, default_dtype_torch, device)
            square += cross_tmp

        set_squares = set_smaller_square
        size_of_square = set_squares[0].shape[2]
        square_need_cross = (size_of_square >= 5 )  

    set_squares=torch.stack(set_squares,dim=0)

    
    log_prob_chess=0.0
    del cross, set_smaller_square, set_square_1_4, list_with_cross
 
    torch.cuda.empty_cache()

    continue_adding = ( set_squares.shape[0] >=2 )
    
    while continue_adding:
        set_squares = add_into_squares(set_squares)
        continue_adding = (  set_squares.shape[0] >=2 )

    sample= (set_squares[0])[:,:,:-1,:-1,:]
    
    return sample




def breakdown_square(sample, beta, L, Q, batch_size, device = torch.device('cuda:0'), default_dtype_torch=  torch.float32):

    list_args_for_nets=[]
    list_args_for_nets_batch_mult=[]

    sample_b = take_boundary(batch_size, default_dtype_torch, device, L, Q, 2, sample)
    
    list_args_for_nets.append(sample_b)   
  
    set_squares = divide_into_square(sample)  ### 4 squares with boundaries

    list_with_cross=[]
    border_list=[]
    cross_list=[]
    for square in set_squares:
        border_list.append(find_borders(square))
        cross_list.append(breakdown_a_cross(square))
    
    border_list = torch.cat(border_list, dim=0)
    cross_list = torch.cat(cross_list, dim=0)

    batch_mult = len(border_list)

    list_with_cross.append([cross_list,border_list])
    list_args_for_nets.append(list_with_cross)
    list_args_for_nets_batch_mult.append(batch_mult)


    size_of_square= set_squares[0].shape[2]
    square_need_cross = (size_of_square >= 5 )        
    
    n_i_nets=int(np.log2(L))
    blocks_widths=L//2**(np.arange(1,n_i_nets+1)-1)-1

    
    while square_need_cross:
        set_smaller_square=[]
        
        list_with_cross=[]
        border_list=[]
        cross_list=[]
        for square in set_squares:
            set_square_1_4= divide_into_squares(square)
            Li = set_square_1_4[0].shape[2]-2
            number_of_net = np.where(blocks_widths==Li)[0][0]
            for smaller_square in set_square_1_4:
                border_list.append(find_borders(smaller_square))
                cross_list.append(breakdown_a_cross(smaller_square))
            set_smaller_square+=set_square_1_4
        set_squares= set_smaller_square

        border_list = torch.cat(border_list, dim=0)
        cross_list = torch.cat(cross_list, dim=0)

        list_with_cross.append([cross_list,border_list])
        list_args_for_nets.append(list_with_cross)
        list_args_for_nets_batch_mult.append(batch_mult)
  
        size_of_square= set_squares[0].shape[2]
        square_need_cross = (size_of_square >= 5 )  
      
    log_prob_chess=0.0

    return list_args_for_nets, log_prob_chess



def _calc_log_prob_net(net_b, int_nets, beta, sign, list_args_for_nets, Q):
    list_log_prob_net=[]
    for index, net_arg in enumerate(list_args_for_nets):
        if index == 0:
            batch = net_arg.shape[0]
            list_log_prob_net.append(net_b.log_prob(sign*net_arg,beta,Q))
        else:
            for arguments in net_arg:
                out = int_nets[index-1].log_prob(sign*arguments[0],sign*arguments[1], beta, Q )
                out = torch.sum(torch.reshape(out, (int(arguments[0].shape[0]/batch),batch)),dim=0)

                list_log_prob_net.append(out)

    arg_sum=torch.zeros(batch, dtype=torch.float32, device=torch.device('cuda:0'))

    for k, arg in enumerate(list_log_prob_net):
        arg_sum += list_log_prob_net[k]

    return arg_sum



def calc_log_prob(z2, translation_y, net_b, int_nets, Q, beta, sample, step, device = torch.device('cuda:0'), default_dtype_torch=  torch.float32):
    
    batch_size = sample.shape[0]
    L = sample.shape[2]
    
    log_prob_all = torch.zeros( [4,8,12,2,batch_size ], dtype=default_dtype_torch, device=device )

    count=0

    for n_y in range(0,4):
        sample_inv = torch.roll(sample, shifts=int(n_y*L/4), dims=2 ) 
        for q in range(0,12):
            sample_inv_q = torch.roll(sample_inv, shifts=q, dims=4 ) 
            for n_x in range(0,8):
                sample_inv_q_x = torch.roll(sample_inv_q, shifts=int(n_x*L/8), dims=3 ) 
                for rr in range(0,2):
                    if(rr==0):
                        sample_inv_q_x_rr = sample_inv_q_x
                    if(rr==1):
                        sample_inv_q_x_rr = sample_inv_q_x.permute(0,1,3,2,4)
            
                    list_args_for_nets_inv, log_prob_chess_inv = breakdown_square(sample_inv_q_x_rr, beta, L, Q, batch_size)

                    log_prob_inv = _calc_log_prob_net(net_b, int_nets, beta, 1 ,list_args_for_nets_inv, Q )
                    log_prob_all[n_y,n_x,q,rr,:] = log_prob_inv + log_prob_chess_inv

    log_prob = torch.logsumexp(log_prob_all, dim=(0,1,2,3))
    log_prob = log_prob - np.log(768)     

    return log_prob




