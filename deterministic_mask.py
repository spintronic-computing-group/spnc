import numpy as np
import random as rnd

class fixed_seed_mask:
    def __init__(self, Nin, Nvirt, m0, seed = 1234):
        rnd.seed(a=seed)
        mask = np.zeros((Nvirt,Nin))
        for i in range(Nvirt):
            for j in range(Nin):
                mask[i,j] = 2*m0*(rnd.randint(0,1)-0.5)
        self.M = mask

    def apply(self, x):
        if x.dtype == np.object:
            J = np.copy(x)
            for i,xi in enumerate(x):
                J[i] = np.matmul(xi, self.M.T)
        else:
            J = np.matmul(x, self.M.T)
        return J