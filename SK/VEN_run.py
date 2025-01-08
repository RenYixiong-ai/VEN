import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
#from sklearn.manifold import TSNE
from einops import rearrange
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm 
import pickle
from multiprocessing import Pool

from VEN_core import *
from SK import SKModel
from utils import record
import datetime

import argparse

parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument('--n', type=int, default=40, help='an integer for the accumulator')

args = parser.parse_args()

kwargs = {}
kwargs['default_dtype_torch'] = torch.float
kwargs['device'] = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

kwargs['batch_size'] = 100 
kwargs['n'] = args.n
kwargs['set_beta'] = 1.0
kwargs['seed'] = np.random.randint(1, 100000)

kwargs['net_depth'] = 3
kwargs['net_width'] = 5
kwargs['half_kernel_size'] = 3
kwargs['bias'] = True
kwargs['z2'] = True
kwargs['res_block'] = True
kwargs['x_hat_clip'] = 0.01
kwargs['final_conv'] = True
kwargs['epsilon'] = float(1e-7)

kwargs['max_step'] = 50 #20

kwargs['spare_num'] = 9 + 1
kwargs['random_step'] = 3

kwargs['beta_start'] = 0.05
kwargs['beta_end'] = 2.0
kwargs['beta_num'] = 60



print(f"n={kwargs['n']}, max_step={kwargs['max_step']}")


def run(PATH, **kwargs):
    net = MADE(**kwargs).to(kwargs['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    beta_list = []
    Energy_list = []
    M_list = []
    Freeenergy_list = []
    x_hat_list = []
    S_list = []
    

    start_beta = kwargs['beta_start']
    SK = SKModel(kwargs['n'], start_beta, kwargs['device'], seed=kwargs['seed'])
    SK.save(PATH)
    start = np.random.choice([-1, 1], [kwargs['batch_size'], 1, kwargs['n']])
    s_min = np.random.choice([-1, 1], [kwargs['batch_size'], kwargs['n']])
    E_min = 100



    for step, beta in enumerate(np.linspace(kwargs['beta_start'], kwargs['beta_end'], kwargs['beta_num']*kwargs['max_step'])):
        spare = np.tile(start, (1, kwargs['spare_num'], 1))
        for spare_step in range(1, kwargs['spare_num']):
            for _ in range(kwargs['random_step']):
                i = np.random.choice(kwargs['n'])
                spare[:, spare_step, i] *= -1
        spare[:, kwargs['spare_num']-1, :] = s_min[:, :] 

        sample = rearrange(spare, "b p h1 -> (b p) 1 h1")
        sample = torch.from_numpy(sample)
        sample = sample.clone().detach().to(dtype=kwargs['default_dtype_torch']).to(kwargs['device'])
        with torch.no_grad():
            x_hat = net(sample)
            spare_energy = SK.energy(sample)

        x_hat = rearrange(x_hat, "(b p) c h1 -> b (c p) h1", p=kwargs['spare_num'])
        x_hat = x_hat.cpu().numpy()
        mask = (1+spare)//2
        x_hat = (x_hat*mask+(1-x_hat)*(1-mask))
        x_hat_mean = x_hat.mean(axis=(0, 1))
        w_hat = (x_hat/x_hat_mean[np.newaxis, np.newaxis, :]).prod(axis=(2))

        spare_energy = rearrange(spare_energy, "(b p)-> b p", p=kwargs['spare_num'])
        spare_energy = spare_energy.cpu().numpy()

        for i in range(kwargs['batch_size']):
            w_hat[i] = w_hat[i]/w_hat[i].sum()

            result = np.random.choice(np.arange(kwargs['spare_num']), p=w_hat[i])
            start[i, 0, :] = spare[i, result, :]

            if spare_energy[i].min() < E_min:
                s_min[i, :] = spare[i, spare_energy[i].argmin(), :]
                E_min = spare_energy[i].min()



        optimizer.zero_grad()
        sample = torch.from_numpy(start)
        sample = sample.clone().detach().to(dtype=kwargs['default_dtype_torch']).to(kwargs['device'])
        log_prob = net.log_prob(sample)
        with torch.no_grad():
            energy = SK.energy(sample)
            loss = log_prob + beta * energy

        sorted_loss, _ = torch.sort(loss)
        loss_reinforce = torch.mean((loss - sorted_loss[:kwargs['batch_size']*3//4].mean()) * log_prob)
        
        loss_reinforce.backward()

        optimizer.step()


        if step%kwargs['max_step'] != 0: continue
        Energy = SK.energy(sample)/kwargs['n']

        record(PATH, beta, sample[0], x_hat[0], w_hat[0], loss, energy=E_min.min())

        beta_list.append(beta)
        Energy_list.append(Energy.cpu().detach().numpy())
        Freeenergy_list.append(loss.cpu().numpy() / beta / kwargs['n'])
        S_list.append(sample.cpu().numpy())
        x_hat_list.append(x_hat)

    with open(PATH+'/beta.pickle', 'wb') as f:
        pickle.dump(beta_list, f)
    with open(PATH+'/freeenergy.pickle', 'wb') as f:
        pickle.dump(Freeenergy_list, f)
    with open(PATH+'/x_hat_list.pickle', 'wb') as f:
        pickle.dump(x_hat_list, f)
    with open(PATH+'/S_list.pickle', 'wb') as f:
        pickle.dump(S_list, f)
    with open(PATH+'/Energy.pickle', 'wb') as f:
        pickle.dump(Energy_list, f)

    

if __name__ == "__main__":
    filePath = './data'
    os.makedirs(filePath, exist_ok=True)

    with open(filePath+"/set.txt", 'w') as f:
        for key in kwargs:
            f.writelines(str(key)+"\t"+ str(kwargs[key])+"\n")


    run(filePath, **kwargs)
