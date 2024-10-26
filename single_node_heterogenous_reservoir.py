import numpy as np
from spnc import spnc_anisotropy




'''
add a new parameter beta_ref 

adjust it by comparing with the single_node_res.py
'''

# have no idea why i can't import the binary_mask from mask.py

class binary_mask:
    def __init__(self, Nin, Nvirt, m0=0.1, mask_sparse=1.0, identity=False):
        self.M = 2*m0*(np.random.randint(0,2, (Nvirt,Nin))-0.5)
        self.M *= 1.0*(np.random.random(size=(Nvirt, Nin)) <= mask_sparse)

        if identity:
            self.M =np.eye(Nin)

        # # for mask comparison
        # print('Mask shape: ', self.M)

        # # save mask matrix
        # save_path = r'C:\icloud\iCloudDrive\Desktop\Code\Uniform-Reservoir\mask_matrix.npy'
        # np.save(save_path, self.M)
        # print(f'Mask matrix saved at {save_path}')

    def apply(self, x):
        if x.dtype == np.object:
            J = np.copy(x)
            for i,xi in enumerate(x):
                J[i] = np.matmul(xi, self.M.T)
        else:
            J = np.matmul(x, self.M.T)
        return J


class single_node_heterogenous_reservoir:
    def __init__(self, Nin, Nvirt, Nout, gamma, beta_prime, beta_ref, delta_betas, h, theta, m0=0.003, dilution = 1.0, identity = False, ravel_order = 'c',**kwargs):
        print("delta_betas in single_node_heterogenous_reservoir:", delta_betas, "Type:", type(delta_betas))
        self.Nvirt = Nvirt
        self.beta_prime = beta_prime
        self.beta_ref = beta_ref   
        self.M = binary_mask(Nin, Nvirt, m0, dilution, identity)
        
        self.ravel_order = ravel_order


        # Initialize multiple spnc_anisotropy instances
        self.anisotropy_instances = [spnc_anisotropy(h, 90, 0, 45, beta_prime + delta) for delta in delta_betas]

        for idx, instance in enumerate(self.anisotropy_instances):
            print(f"Anisotropy instance {idx + 1} parameters:")
            print(f"h: {instance.h}, beta_prime: {instance.beta_prime},theta_H: {instance.theta_H},")

    def transform(self, x, params, beta_ref, *weights, force_compute=False, nthreads=1):

        print("beta_ref in single_node_heterogenous_reservoir:", beta_ref, "Type:", type(beta_ref))
        print(f"Input x shape: {x.shape}")
        print(f"Mask matrix M shape: {self.M.shape if hasattr(self.M, 'shape') else 'unknown'}")

        """
        Transform function supporting multiple instances and weight combination.

        """

        assert len(weights) == len(self.anisotropy_instances), "Weight count should match the number of instances"


        Nthreads = 1
        if "Nthreads" in params.keys():
            Nthreads = params["Nthreads"]

        print('Using Nthreads = ', Nthreads)

        J = self.M.apply(x)
        print(f"Initial J shape: {J.shape if hasattr(J, 'shape') else 'object array'}")

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
        print(f"Final J_1d shape: {J_1d.shape}")
        if Nthreads > 1:
            if J.dtype == object:
                split_sizes = []
                for spi in np.array_split(block_sizes, Nthreads):
                    total = 0
                    for si in spi:
                        total += si[0]
                    split_sizes.append(total)

                params["thread_alloc"] = split_sizes

        # S_1d_avarage = np.zeros((len(x) * self.Nvirt, 1))
        for i, (instance, weight) in enumerate(zip(self.anisotropy_instances, weights)):
             # Check shapes for debugging
            
            mags_instance = instance.gen_signal_fast_delayed_feedback_varing_temp(J_1d, params, beta_ref)
            print('shape of j_1d:', J_1d.shape)
            # weight = float(weight)
            print('weight',weight)
            print('type of weight',type(weight))
            print(f"Instance {i} - mags_instance shape: {mags_instance.shape}, weight: {weight}")
            S_1d_avarage  = mags_instance * weight
            
            print(f"S_1d_avarage shape before addition: {S_1d_avarage.shape}")
            

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
