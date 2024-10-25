import numpy as np
from spnc import spnc_anisotropy
from pathlib import Path

# local repos and search paths. Each repo will be searched on every path

#tuple of Path variables 
searchpaths = (
    Path(r'C:\\icloud\\iCloudDrive\\Desktop\\Code'),  
    
)
#tuple of repos
repos = ('machine_learning_library',)

from mask import binary_mask

'''
add a new parameter beta_ref 

adjust it by comparing with the single_node_res.py
'''
class single_node_heterogenous_reservoir:
    def __init__(self, Nin, Nvirt, Nout, gamma, beta_prime, beta_ref, delta_betas, h, m0=0.003, dilution = 1.0, identity = False, ravel_order = 'c',**kwargs):

        self.M = binary_mask(Nin, Nvirt, m0, dilution, identity)
        self.ravel_order = ravel_order


        # Initialize multiple spnc_anisotropy instances
        self.anisotropy_instances = [spnc_anisotropy(h, 90, 0, 45, beta_prime + delta) for delta in delta_betas]


        # for idx, instance in enumerate(self.anisotropy_instances):
        #     print(f"Anisotropy instance {idx + 1} parameters:")
        #     print(f"h: {instance.h}, beta_prime: {instance.beta_prime},theta_H: {instance.theta_H},")

    def transform(self, x, params, beta_ref, *weights, force_compute=False, nthreads=1):

        """
        Transform function supporting multiple instances and weight combination.

        """

        assert len(weights) == len(self.anisotropy_instances), "Weight count should match the number of instances"


        Nthreads = 1
        if "Nthreads" in params.keys():
            Nthreads = params["Nthreads"]

        print('Using Nthreads = ', Nthreads)

        J = self.M.apply(x)

        if J.dtype == object:
            J_1d = np.copy(J)
            if self.ravel_order is not None:
                for i,Ji in enumerate(J_1d):
                    J_1d[i] = np.expand_dims(np.ravel(Ji, order=self.ravel_order), axis = -1)
            block_sizes = np.array([ Ji.shape for Ji in J_1d])
            J_1d = np.vstack(J_1d)
        else:
            if self.ravel_order is not None:
                J_1d = np.expand_dims(np.ravel(J, order=self.ravel_order), axis = -1)
            else:
                J_1d = np.copy(J)

        if Nthreads > 1:
            if J.dtype == object:
                split_sizes = []
                for spi in np.array_split(block_sizes, Nthreads):
                    total = 0
                    for si in spi:
                        total += si[0]
                    split_sizes.append(total)

                params["thread_alloc"] = split_sizes

        S_1d_avarage = np.zeros_like(J_1d)
        for i, (instance, weight) in enumerate(zip(self.anisotropy_instances, weights)):
            mags_instance = instance.gen_signal_fast_delayed_feedback(K_s, params, beta_ref)
            S_1d_avarage += mags_instance * weight

        if J.dtype == object:
            S = np.copy(J)
            idx = 0
            for i in range(len(S)):
                size = np.product(J[i].shape) if self.ravel_order is not None else J[i].shape[0]
                S[i] = S_1d_avarage[idx:idx+size]
                idx += size
            if self.ravel_order is not None:
                for i,Si in enumerate(S):
                    S[i] = S[i].reshape(J[i].shape, order=self.ravel_order)

        else:
            S = S_1d_avarage.reshape(J.shape, order=self.ravel_order) if self.ravel_order is not None else np.copy(S_flat)

        # print("Final output S in alex:", S)

        return S, J
