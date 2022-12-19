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
## File: my_potts.py
## Authors: T. Stebel, P. Korcyl
## Contact: tomasz.stebel@uj.edu.pl, piotr.korcyl@uj.edu.pl
## 
## Version: 1.0
## 
## Description:
## calculates the energy of a spin configuration
## 
##
import torch
import numpy as np



def energy(sample, ham, lattice, boundary):

#    print(sample)
#    print(sample.shape)

    term = sample[:, :, 1:, :, :] * sample[:, :, :-1, :, :]  # multiply all S_[i,j]*S[i-1,j] and later sum over i and j 
                                                       #(summation over dim=1 is trivial since sample is of length 1 in this direction)

#    print(term.shape)

    term = term.sum(dim=(1, 2, 3, 4))
    output = term
    term = sample[:, :, :, 1:, :] * sample[:, :, :, :-1, :] # multiply all S_[i,j]*S[i,j-1] and later sum over i and j
    term = term.sum(dim=(1, 2, 3, 4))
    output += term
    if lattice == 'tri':
        term = sample[:, :, 1:, 1:] * sample[:, :, :-1, :-1]
        term = term.sum(dim=(1, 2, 3))
        output += term

    if boundary == 'periodic':
        term = sample[:, :, 0, :, :] * sample[:, :, -1, :, :]
        term = term.sum(dim=(1, 2, 3))
        output += term
        term = sample[:, :, :, 0, :] * sample[:, :, :, -1, :]
        term = term.sum(dim=(1, 2, 3))
        output += term
        if lattice == 'tri':
            term = sample[:, :, 0, 1:] * sample[:, :, -1, :-1]
            term = term.sum(dim=(1, 2))
            output += term
            term = sample[:, :, 1:, 0] * sample[:, :, :-1, -1]
            term = term.sum(dim=(1, 2))
            output += term
            term = sample[:, :, 0, 0] * sample[:, :, -1, -1]
            term = term.sum(dim=1)
            output += term

    if ham == 'fm':
        output *= -1

#    print(output)
#    quit()

    return output

def neibours_sum(sample):

    term = torch.cat( (sample[:, :, -1, :].view(sample.shape[0],1,1,sample.shape[3]) ,sample[:, :, :-1, :]),dim=2 )                                                      
    output = term
    
    term = torch.cat( (sample[:, :, 1:, :], sample[:, :, 0, :].view(sample.shape[0],1,1,sample.shape[3]) ),dim=2 )                                                      
    output += term

    term = torch.cat( (sample[:, :, :, -1].view(sample.shape[0],1,sample.shape[2],1 ) ,sample[:, :, :, :-1]),dim=3 )                                                      
    output += term

    term = torch.cat( (sample[:, :, :, 1:],  sample[:, :, :, 0].view(sample.shape[0],1,sample.shape[2],1 ) ),dim=3 )                                                      
    output += term
        

    #if ham == 'fm':
    output *= -1

    return output


