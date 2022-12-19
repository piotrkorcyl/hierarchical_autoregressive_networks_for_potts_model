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
## File: multinets.py
## Authors: T. Stebel, P. Korcyl
## Contact: tomasz.stebel@uj.edu.pl, piotr.korcyl@uj.edu.pl
## 
## Version: 1.0
## 
## Description:
## contains the main part of the simulations: training and calculation of averages
## 
##

import numpy as np
import torch
from numpy import sqrt
from numpy import log
from torch import nn
import time
from scipy.special import logsumexp
from scipy.optimize import curve_fit
from scipy import stats
from my_args import my_args

import my_utensils as uten
import my_potts
from my_parameters import *

Q = 12
beta_final = my_args.beta0*np.log(1.0+np.sqrt(1.0*Q))
run_num = my_args.rn
loss_num = my_args.lossn
L = my_args.Lv
z2 = bool(my_args.Z2)
translation_y= bool(my_args.Ty)
chess = bool(my_args.chess)
net_type= str(my_args.net_type)



if net_type=='VAN':
    from my_dense_VAN import MADE
elif net_type=='mnVAN':
    from my_dense import MADE_b, MADE_i
    from sampling_file import build_sample, calc_log_prob, breakdown_square

else:
    print('Undefined algorithm!!!!')
    exit()





if nmcmc_or_nis:
    print('\n','NMCMC')
else:
    print('\n','training only')

if net_type=='VAN':
    print('standard VAN\n')
elif net_type=='mnVAN':
    print('multinet VAN\n')

    
print('beta_final: ', beta_final)
print('L: ', L)
print('n_block: ', n_block)
print('lattice: ', lattice) 
print('net_depth: ', net_depth)
print('net_width: ',net_width)

print('bias: ' , bias)
print('z2 symmetry:',z2)
print('T_y symmetry', translation_y)
print('chess-board trick:',chess)

print('device: ', device,'\n')
print('max_step for teaching: ', max_step)
print('start MC from epoch: ', start_counting)
print('number of batches in MC: ', colected_batches)
print('beta_anneal step: ', beta_anneal)
print('batch_size: ', batch_size)
print('learning rate: ', lr, '\n' )




n_i_nets=int(np.log2(L))
print('number of int nets: ', n_i_nets)
blocks_widths=L//2**(np.arange(1,n_i_nets+1)-1)-1
print('crosses sizes Li: ', blocks_widths)



if net_type =='mnVAN':
    net_b =     MADE_b(Q,L,n_block,net_depth,net_width,bias,z2,translation_y,res_block,x_hat_clip,epsilon,device)
    net_b.to(device)
    params_b = list(net_b.parameters())
    params_b = list(filter(lambda p: p.requires_grad, params_b))   # take only those parameters from params which has .requires_grad =True
    nparams_b = int(sum([np.prod(p.shape) for p in params_b]))
    print('Boundary net - number of trainable parameters:', nparams_b, '\n')


    params = params_b

    int_nets=[]
    for k in range(n_i_nets):
        net_i = MADE_i(Q,blocks_widths[k],net_depth,net_width,bias,z2,translation_y,res_block,x_hat_clip,epsilon,device)
        net_i.to(device)
        params_i = list(net_i.parameters())
        params_i = list(filter(lambda p: p.requires_grad, params_i))   # take only those parameters from params which has .requires_grad =True
        nparams_i = int(sum([np.prod(p.shape) for p in params_i]))
        print('Interior net - number of trainable parameters:', nparams_i, '\n')
        int_nets.append(net_i)
        params= params+ params_i
   
        
elif net_type =='VAN':
    net =     MADE(Q,L,net_depth,net_width,bias,z2,translation_y,res_block,x_hat_clip,epsilon,device)
    net.to(device)    
    params = list(net.parameters())
    params = list(filter(lambda p: p.requires_grad, params))   # take only those parameters from params which has .requires_grad =True

    nparams = int(sum([np.prod(p.shape) for p in params]))
    print('Total number of trainable parameters:', nparams, '\n')



optimizer = torch.optim.Adam(params, lr=lr, betas=(0.9, 0.999))
try:
    if net_type =='mnVAN':
    
        state = torch.load('saved_state_b_L='+str(L)+'_beta='+str(beta_final)+'_mn.out')
        net_b.load_state_dict(state['net'])
        
        first_step = state['last_step']

        for k in range(n_i_nets):
            state = torch.load('saved_state_intnet'+str(k)+'_L='+str(L)+'_beta='+str(beta_final)+'_mn.out')
            int_nets[k].load_state_dict(state['net'])
        
        print('state loaded')
        print('starting from epoch: ', first_step)
        print('')

        step_cont=[]
        F_cont=[]
        F_err_cont=[]
        time_upto_now=8000

    elif net_type =='VAN':
        state = torch.load('saved_state_VAN_L='+str(L)+'_beta='+str(beta_final)+'.out')
        net.load_state_dict(state['net'])
        
        first_step = state['last_step']
        
        print('state loaded')
        print('starting from epoch: ', first_step)
        print('')

        time_upto_now= np.loadtxt('wyn_t_L='+str(L)+'_beta='+str(beta_final)+'_VAN.out')[0]

        step_cont,F_cont,F_err_cont=  np.loadtxt('wyn_F_L='+str(L)+'_beta='+str(beta_final)+'_VAN.out') 
        
        step_cont=step_cont.tolist()
        F_cont=F_cont.tolist()
        F_err_cont=F_err_cont.tolist()
        
except FileNotFoundError:
    print('no starting point\n')
    first_step=0
    step_cont=[]
    F_cont=[]
    F_err_cont=[]
    time_upto_now=0

    Fex=0
    Uex=0



accept_ratio_cont=[]


U_cont=[]
U_err_cont=[]
absM_cont=[]
absM_err_cont=[]

tau_batch_cont=[]
Fnet_cont=[]
tau_eff_cont=[]
tau_int_cont=[]
tau_eff_err_cont=[]
ess_cont=[]



start_time = time.time()

if not first_step==start_counting:
    print('training')
    for step in range(1+first_step, first_step+max_step+1):
        last_step=step
        optimizer.zero_grad()

        beta = beta_final
        with torch.no_grad():
            if net_type =='mnVAN':
                sample = build_sample(Q, net_b, int_nets, beta, L, batch_size)
            elif net_type =='VAN':
                sample, x_hat = net.sample(batch_size, beta, Q)

        assert not sample.requires_grad

        if net_type =='mnVAN':
            log_prob = calc_log_prob(z2, translation_y, net_b, int_nets, Q, beta, sample,step)
            
        elif net_type =='VAN':
            log_prob = net.log_prob(sample,beta)     # log_prob is a 1-dim tensor of lenght = batch_size
          

        with torch.no_grad():
            energy = my_potts.energy(sample, ham, lattice, boundary)
            loss = log_prob + beta * energy


        assert not energy.requires_grad
        assert not loss.requires_grad
        
        
        loss_reinforce = torch.mean((loss - loss.mean()) * log_prob )
        
        loss_reinforce.backward()    
        optimizer.step()

        if step%print_step==0:
            free_energy_mean = loss.mean() / beta / L**2
            free_energy_error= torch.sqrt(  (loss**2).sum()*batch_size- loss.sum()**2 )/batch_size/np.sqrt(batch_size)/ beta / L**2
            free_energy_mean =free_energy_mean.data.cpu().numpy()
            free_energy_error = free_energy_error.data.cpu().numpy()

            energy_mean = energy.mean() / L**2
            energy_error= torch.sqrt(  (energy**2).sum()*batch_size- energy.sum()**2 )/batch_size/np.sqrt(batch_size)/ L**2
            energy_mean =energy_mean.data.cpu().numpy()
            energy_error = energy_error.data.cpu().numpy()    
            
        
            mag = torch.sum(sample,dim=(2,3))
            abs_mag=torch.max(mag,dim=2)[0]
            abs_mag_mean = abs_mag.mean()/L**2
            abs_mag_error=  torch.sqrt(  (abs_mag**2).sum()*batch_size- abs_mag.sum()**2 )/batch_size/np.sqrt(batch_size)/L**2        
            abs_mag_mean=abs_mag_mean.data.cpu().numpy()
            abs_mag_error=abs_mag_error.data.cpu().numpy()
 
            log_ess= 2*torch.logsumexp(-loss,0)-torch.logsumexp(-2*loss,0)
            ess = torch.exp(log_ess)/batch_size
            ess = ess.data.cpu().numpy()

            loss_m= torch.min(loss)        
            l_obs =  -torch.exp(loss_m-loss)   ## l_obs>=-1 since loss_m < loss
            eigenv_batch = 1 + l_obs
            eigenv_batch = torch.mean(eigenv_batch)
            tau_batch = -1/torch.log(eigenv_batch)
            eigenv_batch = eigenv_batch.data.cpu().numpy()
            tau_batch = tau_batch .data.cpu().numpy()


            print('beta/beta_final= ', (1 - beta_anneal**(step))  )   
                    
            print('training step: ', step, 
                    ' <|M|>: ', abs_mag_mean, '(', abs_mag_error, ')', 
                    ' F_b: ', free_energy_mean, '(', free_energy_error, ')', 
                    ' U_b: ', energy_mean, '(', energy_error, ')',                     
                    ' tau_batch: ', tau_batch,
                    ' ESS: ' , ess
                    )    
                                
            F_cont.append(free_energy_mean)
            F_err_cont.append(free_energy_error)
            U_cont.append(energy_mean)
            U_err_cont.append(energy_error)
            absM_cont.append(abs_mag_mean)
            absM_err_cont.append(abs_mag_error)
            tau_batch_cont.append(tau_batch)
            step_cont.append(step)
            ess_cont.append(ess)
            
    print("--- %s seconds ---" % (time.time() - start_time))


    if net_type =='mnVAN':

        state = {
                'net': net_b.state_dict(),
                'last_step': last_step
                }
        torch.save(state, 'saved_state_b_L='+str(L)+'_beta='+str(beta_final)+'_mn.out')

        for k in range(n_i_nets):
            state = {
                    'net': int_nets[k].state_dict(),
                    }
            torch.save(state, 'saved_state_intnet'+str(k)+'_L='+str(L)+'_beta='+str(beta_final)+'_mn.out')

    
    elif net_type =='VAN':

        state = {
                'net': net.state_dict(),
                'last_step': last_step
                }
        torch.save(state, 'saved_state_VAN_L='+str(L)+'_beta='+str(beta_final)+'.out')




start_time_NMCMC = time.time()

if nmcmc_or_nis and (first_step>=start_counting or L<=32):
## cumulation of samples:
    beta=beta_final


    print('NMCMC')
    last_step= start_counting
    
    list_mag = np.empty((0,batch_size), dtype='float32' )
    list_energy = np.empty((0,batch_size), dtype='float32' )
    list_log_prob=np.empty((0,batch_size), dtype='float32' )
    Gamma_cont= []
    accept_batch_av_cont =[]
                        
    with torch.no_grad():
        for step1 in range(1, colected_batches+1):

            
            if net_type =='mnVAN':
                sample, list_args_for_nets, log_prob_chess = build_sample(Q, net_b, int_nets, beta, L, batch_size)
                log_prob = calc_log_prob(z2, translation_y, net_b, int_nets, Q, beta, sample, list_args_for_nets , log_prob_chess,step1)


            elif net_type =='VAN':
                sample, x_hat = net.sample(batch_size, beta)
                log_prob = net.log_prob(sample,beta)     # log_prob is a 1-dim tensor of lenght = batch_size

            if step1%20==0:
                print(step1)
  

                    
            energy = my_potts.energy(sample, ham, lattice, boundary)
            loss = log_prob + beta * energy
 

            list_energy=np.append(list_energy, np.array([energy.cpu().numpy()]), axis=0 )
            list_log_prob=np.append(list_log_prob, np.array([log_prob.cpu().numpy()]), axis=0 )  


    sh=list_energy.shape                      
    N_samples = sh[0]*sh[1]

    list_energy = list_energy.reshape(N_samples)
    list_log_prob = list_log_prob.reshape(N_samples)
    list_loss = list_log_prob+beta*list_energy

    free_energy_mean = np.mean(list_loss) / beta / L**2

    ess= 2*logsumexp(-list_loss,0)-logsumexp(-2*list_loss,0)
    ess = np.exp(ess)/N_samples
    
    
    print('F all batches ', free_energy_mean)
    print('ESS all batches ', ess)
   
    
    list_energy, list_log_prob, accept_cont = uten.metropolis(beta, list_energy, list_log_prob)
    accept_ratio= np.mean(accept_cont)


    print('accept ratio= ', accept_ratio )                            

    Gamma_cont = uten.autocorr2(list_energy, 600)

    
    index= np.argmax(Gamma_cont < 0)
    
    if index >= 1:
        tau_int= 1+2*np.sum( Gamma_cont[1:index] )
    else:
        tau_int= 1+2*np.sum( Gamma_cont )
           
    print('tau_int', tau_int)
    

else:
    # ~ tau_eff_cont.append(0)
    tau_int=0
    # ~ tau_eff_err_cont.append(0)
    accept_ratio=0


print("--- %s seconds for NMCMC---" % (time.time() - start_time_NMCMC))


F_cont=np.array(F_cont)
F_err_cont=np.array(F_err_cont)

U_cont=np.array(U_cont)
U_err_cont=np.array(U_err_cont)

step_cont=np.array(step_cont)

time_upto_now+= time.time() - start_time

if net_type =='mnVAN':
    np.savetxt('wyn_t_L='+str(L)+'_beta='+str(beta_final)+'.out', np.array([time_upto_now, tau_int, accept_ratio, ess]  ) )
    np.savetxt('wyn_F_L='+str(L)+'_beta='+str(beta_final)+'_mn.out', (step_cont, F_cont, F_err_cont  )) 
elif net_type =='VAN':
    np.savetxt('wyn_t_L='+str(L)+'_beta='+str(beta_final)+'_VAN.out', np.array([time_upto_now, tau_int, accept_ratio, ess]  ) )
    np.savetxt('wyn_F_L='+str(L)+'_beta='+str(beta_final)+'_VAN.out', (step_cont, F_cont, F_err_cont  )) 

if last_step>=start_counting:
    plt.errorbar(step_cont, F_cont, yerr=F_err_cont, fmt='.r')
    plt.plot(step_cont, Fex* np.ones(step_cont.size) ,'-g')
    plt.plot(step_cont, free_energy_mean* np.ones(step_cont.size), '-b')



