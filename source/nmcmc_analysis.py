#Standalone NMCMC analysis for pre-trained HAN models
#Based on the original multinets.py implementation

import numpy as np
import torch
from scipy.special import logsumexp
import time
import argparse
import my_potts
import my_utensils as uten
from my_parameters import *
import os
from my_dense import MADE_b, MADE_i
from sampling_file import build_sample, calc_log_prob, breakdown_square

parser = argparse.ArgumentParser()
parser.add_argument('--beta0', type=float, required=True, help='beta = 1 / k_B T')
parser.add_argument('--Lv', type=int, required=True, help='L value')
parser.add_argument('--batches', type=int, default=100, help='number of batches for NMCMC')
parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
parser.add_argument('--Z2', type=int, default=0, help='Z2 symmetry')
parser.add_argument('--Ty', type=int, default=0, help='Ty symmetry')
parser.add_argument('--net_type', type=str, default='mnVAN', help='only mnVAN')
parser.add_argument('--model_dir', type=str, default='.', help='directory containing model files')
args = parser.parse_args()

Q = 2
beta_final = args.beta0 * np.log(1.0 + np.sqrt(1.0 * Q))
L = args.Lv
z2 = bool(args.Z2)
translation_y = bool(args.Ty)
net_type = args.net_type
batch_size = args.batch_size
colected_batches = args.batches
model_dir = args.model_dir

n_block = 2
lattice = 'sqr'
boundary = 'periodic'
ham = 'fm'
net_depth = 2
net_width = 1
bias = True
res_block = False
x_hat_clip = False
epsilon = 1e-8
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f'Running NMCMC analysis')
print(f'beta_final: {beta_final}')
print(f'L: {L}')
print(f'batch_size: {batch_size}')
print(f'batches to collect: {colected_batches}')
print(f'device: {device}')
print()


n_i_nets = int(np.log2(L))
blocks_widths = L // 2**(np.arange(1, n_i_nets+1) - 1) - 1
print(f'number of int nets: {n_i_nets}')
print(f'crosses sizes Li: {blocks_widths}')

net_b = MADE_b(Q, L, n_block, net_depth, net_width, bias, z2, translation_y, 
                res_block, x_hat_clip, epsilon, device)
net_b.to(device)

int_nets = []
for k in range(n_i_nets):
    net_i = MADE_i(Q, blocks_widths[k], net_depth, net_width, bias, z2, translation_y,
                    res_block, x_hat_clip, epsilon, device)
    net_i.to(device)
    int_nets.append(net_i)

model_path = os.path.join(model_dir, f'saved_state_b_L={L}_beta={beta_final}_mn.out')
state = torch.load(model_path, map_location=device)
net_b.load_state_dict(state['net'])
print(f'Boundary network loaded from {model_path}')

for k in range(n_i_nets):
    model_path = os.path.join(model_dir, f'saved_state_intnet{k}_L={L}_beta={beta_final}_mn.out')
    state = torch.load(model_path, map_location=device)
    int_nets[k].load_state_dict(state['net'])
print('Interior networks loaded')    



print('\nStarting NMCMC sampling...')
start_time = time.time()

list_energy = np.empty((0, batch_size), dtype='float32')
list_log_prob = np.empty((0, batch_size), dtype='float32')

beta = beta_final

with torch.no_grad():
    for step in range(1, colected_batches + 1):
        if step % 20 == 0:
            print(f'Batch {step}/{colected_batches}')
        
        if net_type == 'mnVAN':
            sample = build_sample(Q, net_b, int_nets, beta, L, batch_size)
            list_args_for_nets, log_prob_chess = breakdown_square(sample, beta, L, Q, batch_size)
            log_prob = calc_log_prob(z2, translation_y, net_b, int_nets, Q, beta, sample, step)
        
        energy = my_potts.energy(sample, ham, lattice, boundary)
        
        list_energy = np.append(list_energy, np.array([energy.cpu().numpy()]), axis=0)
        list_log_prob = np.append(list_log_prob, np.array([log_prob.cpu().numpy()]), axis=0)

sh = list_energy.shape
N_samples = sh[0] * sh[1]

list_energy = list_energy.reshape(N_samples)
list_log_prob = list_log_prob.reshape(N_samples)

list_energy, list_log_prob, accept_cont = uten.metropolis(beta, list_energy, list_log_prob)
accept_ratio = np.mean(accept_cont)

print(f'\nAcceptance ratio: {accept_ratio:.4f}')

Gamma_cont = uten.autocorr2(list_energy, 600)
index = np.argmax(Gamma_cont < 0)

if index >= 1:
    tau_int = 1 + 2 * np.sum(Gamma_cont[1:index])
else:
    tau_int = 1 + 2 * np.sum(Gamma_cont)

print(f'tau_int = {tau_int:.2f}')

list_loss = list_log_prob + beta * list_energy
free_energy_mean = np.mean(list_loss) / beta / L**2
free_energy_err = np.std(list_loss) / np.sqrt(N_samples / tau_int) / beta / L**2

energy_mean = np.mean(list_energy) / L**2
energy_err = np.std(list_energy) / np.sqrt(N_samples / tau_int) / L**2

ess = 2 * logsumexp(-list_loss, 0) - logsumexp(-2 * list_loss, 0)
ess = np.exp(ess) / N_samples

print(f'\nResults after Metropolis:')
print(f'F = {free_energy_mean:.6f} ± {free_energy_err:.6f}')
print(f'U = {energy_mean:.6f} ± {energy_err:.6f}')
print(f'ESS = {ess:.4f}')

elapsed_time = time.time() - start_time
print(f'\nTotal time: {elapsed_time:.2f} seconds')

output_file = f'nmcmc_results_L={L}_beta={beta_final}_{net_type}.txt'
with open(output_file, 'w') as f:
    f.write(f'# NMCMC Results\n')
    f.write(f'# L = {L}, beta = {beta_final}, batches = {colected_batches}, batch_size = {batch_size}\n')
    f.write(f'# Free energy: {free_energy_mean} ± {free_energy_err}\n')
    f.write(f'# Energy: {energy_mean} ± {energy_err}\n')
    f.write(f'# Accept ratio: {accept_ratio}\n')
    f.write(f'# tau_int: {tau_int}\n')
    f.write(f'# ESS: {ess}\n')
    f.write(f'# Time: {elapsed_time} seconds\n')

print(f'\nResults saved to {output_file}')