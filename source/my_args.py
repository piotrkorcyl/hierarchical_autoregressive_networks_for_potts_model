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
## File: my_args.py
## Authors: T. Stebel, P. Korcyl
## Contact: tomasz.stebel@uj.edu.pl, piotr.korcyl@uj.edu.pl
## 
## Version: 1.0
## 
## Description:
## contains some of the main parameters which can be set from the command line
## 
##

import argparse

parser = argparse.ArgumentParser()

group = parser.add_argument_group('physics parameters')

group.add_argument('--beta0', type=float, default=1.0, help='beta = 1 / k_B T')

group.add_argument('--rn', type=int, default=1, help='run number')
group.add_argument('--lossn', type=int, default=1, help='loss choice: 1-KL, 2-LAT, 3-ESS')
group.add_argument('--Lv', type=int, default=32, help='L value')

group.add_argument('--Z2', type=int, default=0, help='Z2 symmetry on')
group.add_argument('--Ty', type=int, default=0, help='Ty symmetry on')
group.add_argument('--chess', type=int, default=0, help='chess trick on')
group.add_argument('--net_type', type=str, default='mnVAN', help='VAN or nmVAN')


my_args = parser.parse_args()

