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
from utils import record
import datetime



kwargs = {}
kwargs['default_dtype_torch'] = torch.float
kwargs['device'] = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

kwargs['batch_size'] = 20 
kwargs['L'] = 8
kwargs['set_beta'] = 1.0

kwargs['net_depth'] = 3
kwargs['net_width'] = 64
kwargs['half_kernel_size'] = 3
kwargs['bias'] = True
kwargs['z2'] = True
kwargs['res_block'] = True
kwargs['x_hat_clip'] = 0
kwargs['final_conv'] = True
kwargs['epsilon'] = float(1e-7)

kwargs['max_step'] = 4000 #4000

kwargs['spare_num'] = 50
kwargs['random_step'] = 3

kwargs['beta_start'] = 0.05
kwargs['beta_end'] = 1.0
kwargs['beta_num'] = 100
kwargs['seed'] = 3592769612

np.random.seed(kwargs['seed'])
torch.manual_seed(kwargs['seed'])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(kwargs['seed'])

print("max_step=%d" %kwargs['max_step'])


def run(PATH, **kwargs):
    net = MADE(**kwargs).to(kwargs['device'])
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))

    beta_list = []
    Energy_list = []
    M_list = []
    Freeenergy_list = []
    log_prob_list = []
    S_list = []
    
    start_beta = kwargs['beta_start']
    start = np.random.choice([-1, 1], [kwargs['batch_size'], 1, kwargs['L'], kwargs['L']])


    for set_beta in np.linspace(kwargs['beta_start'], kwargs['beta_end'], kwargs['beta_num']):
        for step in range(kwargs['max_step']):
            beta = (set_beta-start_beta) * (1 - 0.998**(step+1)) + start_beta

            spare = np.tile(start, (1, kwargs['spare_num'], 1, 1))

            for spare_step in range(1, kwargs['spare_num']):
                for _ in range(kwargs['random_step']):
                    i = np.random.choice(kwargs['L'])
                    j = np.random.choice(kwargs['L'])
                    spare[:, spare_step, i, j] *= -1

            sample = rearrange(spare, "b p h1 h2 -> (b p) 1 h1 h2")
            sample = torch.from_numpy(sample)
            sample = sample.clone().detach().to(dtype=kwargs['default_dtype_torch']).to(kwargs['device'])
            with torch.no_grad():
                x_hat = net(sample)
            x_hat = rearrange(x_hat, "(b p) c h1 h2 -> b (c p) h1 h2", p=kwargs['spare_num'])

            x_hat = x_hat.cpu().numpy()
            mask = (1+spare)//2
            x_hat = (x_hat*mask+(1-x_hat)*(1-mask)) * 0.8 + 0.1 # 截断，防止出现为0与1这样的概率
            x_hat_mean = x_hat.mean(axis=(0, 1))
            w_hat = (x_hat/x_hat_mean[np.newaxis, np.newaxis, :, :]).prod(axis=(2, 3))


            for i in range(kwargs['batch_size']):
                
                w_hat[i] = w_hat[i]/w_hat[i].sum()

                result = np.random.choice(np.arange(kwargs['spare_num']), p=w_hat[i])
                start[i, 0, :, :] = spare[i, result, :, :]


                
            optimizer.zero_grad()
            sample = torch.from_numpy(start)
            sample = sample.clone().detach().to(dtype=kwargs['default_dtype_torch']).to(kwargs['device'])
            log_prob = net.log_prob(sample)
            with torch.no_grad():
                energy = IsingEnergy(sample)
                loss = log_prob + set_beta * energy

            loss_reinforce = torch.mean((loss - loss.mean()) * log_prob)
            loss_reinforce.backward()

            optimizer.step()

        record(PATH, beta, sample[0], x_hat[0], w_hat[0], loss)

        start_beta = beta
        beta_list.append(beta)
        Energy_list.append(IsingEnergy(sample).mean())
        M_list.append(IsingM(sample))
        Freeenergy_list.append(loss.cpu().numpy() / beta / kwargs['L']**2)
        S_list.append(sample.cpu().numpy())
        log_prob_list.append(log_prob.detach().cpu().numpy())

    with open(PATH+'/beta.pickle', 'wb') as f:
        pickle.dump(beta_list, f)
    with open(PATH+'/freeenergy.pickle', 'wb') as f:
        pickle.dump(Freeenergy_list, f)
    with open(PATH+'/log_prob_list.pickle', 'wb') as f:
        pickle.dump(log_prob_list, f)
    with open(PATH+'/S_list.pickle', 'wb') as f:
        pickle.dump(S_list, f)

if __name__ == "__main__":
    filePath = './data/'
    os.makedirs(filePath, exist_ok=True)

    with open(filePath+"/set.txt", 'w') as f:
        for key in kwargs:
            f.writelines(str(key)+"\t"+ str(kwargs[key])+"\n")


    run(filePath, **kwargs)
