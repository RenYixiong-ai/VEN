import numpy as np
import os

def record(PATH, beta, sample, x_hat, w_hat, loss, energy=None):
    path = os.path.join(PATH, 'record.txt')
    with open(path, 'a') as f:
        f.write(80*"-")
        f.write('\n')
        f.write(f'beta={beta}')
        f.write('\n')

        if energy is not None:
            f.write(f'Energy={energy}')
            f.write('\n')

        f.write(f'sample')
        f.write('\n')
        f.write(str(sample))
        f.write('\n')

        f.write(f'x_hat')
        f.write('\n')
        f.write(str(x_hat))
        f.write('\n')

        f.write(f'w_hat')
        f.write('\n')
        f.write(str(w_hat))
        f.write('\n')

        f.write(f'loss={loss}')
        f.write('\n')


        f.write(2*'\n')