import torch
import numpy as np
import pickle
import math
import sys
import os
from sko.GA import GA


class SKModel():
    def __init__(self, n, beta=1, device="cpu", field=0, seed=None, size_pop=100, max_iter=1000, prob_mut=0.001):
        self.n = n
        self.beta = beta
        self.field = field

        self.size_pop = size_pop
        self.max_iter = max_iter
        self.prob_mut = prob_mut

        if seed is None:
            self.seed = np.random.randint(1, 100000)
        else:
            self.seed = seed
        torch.manual_seed(self.seed)

        self.J = torch.randn([self.n, self.n]) / math.sqrt(self.n)
        # Symmetric matrix, zero diagonal
        self.J = torch.triu(self.J, diagonal=1)
        self.J += self.J.t().clone()
        self.J = self.J.to(device)
        self.J.requires_grad = True

        self.C_model = []

        #print('SK model with n = {}, beta = {}, field = {}, seed = {}'.format(
        #    n, beta, field, self.seed))

    def exact(self):
        #assert self.n <= 20

        Z = 0
        n = self.n
        J = self.J.cpu().to(torch.float64)
        beta = self.beta
        E_min = 0
        n_total = int(math.pow(2, n))

        for d in range(n_total):
            s = np.binary_repr(d, width=n)
            b = np.array(list(s)).astype(np.float32)
            b[b < 0.5] = -1
            b = torch.from_numpy(b).view(n, 1).to(torch.float64)
            E = -0.5 * b.t() @ J @ b        # 计算能量
            if E < E_min:
                E_min = E
            Z += torch.exp(-beta * E)       # 计算配分函数

        self.C_model = torch.zeros([n, n]).to(torch.float64)
        for d in range(n_total):
            s = np.binary_repr(d, width=n)
            b = np.array(list(s)).astype(np.float32)
            b[b < 0.5] = -1
            b = torch.from_numpy(b).view(n, 1).to(torch.float64)
            E = -0.5 * b.t() @ J @ b
            prob = torch.exp(-beta * E) / Z
            self.C_model += b @ b.t() * prob

        Exact_FreeEnergy = -torch.log(Z).item() / beta / n

        # print(self.C_model)
        '''
        print(
            'Exact free energy = {:.8f}, paramagnetic free energy = {:.8f}, E_min = {:.8f}'
            .format(Exact_FreeEnergy, -math.log(2) / beta,
                    E_min.item() / n))
        '''
        
        return Exact_FreeEnergy, E_min.item() / n
        
    def energy(self, samples):
        """
        Compute energy of samples, samples should be of size [m, n] where n is the number of spins, m is the number of samples.
        """
        samples = samples.view(samples.shape[0], -1).to(torch.float32)
        assert samples.shape[1] == self.n
        m = samples.shape[0]
        return (-0.5 * ((samples @ self.J).view(m, 1, self.n) @ samples.view(
            m, self.n, 1)).squeeze() - self.field * torch.sum(samples, 1))

    def _ga_energy(self, samples):

        input_sample = torch.from_numpy(samples.reshape([1, self.n]))
        out =  self.energy(input_sample).detach().numpy()

        return out[0]
    
    def ga_energy(self):

        aim_func = lambda sample : self._ga_energy(sample)
        ga = GA(func=aim_func, n_dim=self.n, size_pop=self.size_pop, max_iter=self.max_iter, prob_mut=self.prob_mut,
                lb=-np.ones(self.n), ub=np.ones(self.n), precision=2*np.ones(self.n))
    
        best_sample, low_energy = ga.run()
        print('best_sample', best_sample, '\n', 'low_energy:', low_energy)
        return best_sample, low_energy/self.n


    def save(self, savepath=None):
        J = self.J.cpu()
        fsave_name = 'n{}D{}.pickle'.format(self.n, self.seed)
        if savepath is not None:
            fsave_name = os.path.join(savepath, fsave_name)

        with open(fsave_name, 'wb') as fsave:
            pickle.dump(J, fsave)
        print('SK model is saved to', fsave_name)

if __name__ == '__main__':
    n = 5
    beta = 0.4
    seed = 332

    device = torch.device('cpu')
    sk = SKModel(n, beta, device, seed=seed)
    sk.exact()
    sk.save()

    S = np.random.choice([-1, 1], [2, 5])
    print(sk.energy(S))

