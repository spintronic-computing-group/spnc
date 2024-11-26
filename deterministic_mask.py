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
        if x.dtype == object:
            J = np.copy(x)
            for i,xi in enumerate(x):
                J[i] = np.matmul(xi, self.M.T)
        else:
            J = np.matmul(x, self.M.T)
        return J
    
class max_sequences_mask:
    def __init__(self, Nin, Nvirt, m0):
        mask = np.zeros((Nvirt,Nin))
        mask_list = mask_max_sequences(m0,Nvirt)
        for i in range(Nvirt):
            for j in range(Nin):
                mask[i,j] = mask_list[i]
        self.M = mask

    def apply(self, x):
        if x.dtype == object:
            J = np.copy(x)
            for i,xi in enumerate(x):
                J[i] = np.matmul(xi, self.M.T)
        else:
            J = np.matmul(x, self.M.T)
        return J

def mask_max_sequences(m0,Nvirt):
    p = int(np.log2(Nvirt))
    L = [1]*p
    (success,L_final) = build_optimal_list_rec(L,[],p)
    mask = [0]*Nvirt
    N_L = len(L_final)
    for i in range(N_L):
        mask[i] = m0*2*(L_final[i]-0.5)
    for i in range(Nvirt-N_L):
        mask[-1-i] = m0*2*(L_final[-1-i]-0.5)
    return(mask)

def build_optimal_list_rec(L,sequences,p):
    #print(L)
    if len(L)==2**p:
        if check_cyclic_seq(L,sequences,p):
            L_final = L
            return(True,L_final)
        else:
            return(False,[0])
    else:
        #Add a 0
        L.append(0)
        last_seq = get_last_sequence(L,p)
        if check_last_seq(last_seq,sequences):
            sequences.append(last_seq)
            (success,L_final) = build_optimal_list_rec(L,sequences,p)
            if success:
                return(True,L_final)
        L.pop()
        
        #Add a 1
        L.append(1)
        last_seq = get_last_sequence(L,p)
        if check_last_seq(last_seq,sequences):
            sequences.append(last_seq)
            (success,L_final) = build_optimal_list_rec(L,sequences,p)
            if success:
                return(True,L_final)
        L.pop()
        return(False,[0])
            
def get_last_sequence(L,p):
    key = 0
    for k in range(p):
        key += (L[-p+k])*10**(p-k-1)
    key = int(key)
    return(key)


def check_last_seq(last_seq,sequences):
    if last_seq in sequences:
        return False
    else:
        return True

def check_cyclic_seq(L,sequences,p):
    L_cyclic = L[-p+1:]+L[:p-1]
    for k in range(p-1):
        last_seq = get_last_sequence(L_cyclic[k:k+p],p)
        if check_last_seq(last_seq,sequences)==False:
            return False
    return True