import argparse
import glob
import numpy as np
import torch

from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset

# Definitions
DATA_PATH = "data"

# Function for the calculation of the equilibrium
def compute_Feq(Feq, rho, ux, uy, c, w, Q=9, cs2=1.0 / 3):
    uu = (ux**2 + uy**2) * (1. / cs2)

    for ip in range(Q):
        cu = (c[ip, 0] * ux[:,:]  + c[ip, 1] * uy[:,:] ) * (1. / cs2)
        Feq[:, :, ip] = w[ip] * rho * (1.0 + cu + 0.5 * (cu * cu - uu) )

    return Feq

def stencil():
    # Note: D2Q9 stencil for LBM
    Q = 9
    c = np.zeros((Q, 2), dtype=np.int32)
    w = np.zeros(Q)    

    # Note: Speed of sound squared (c_s = 1/sqrt(3))    
    cs2 = 1./3.

    c[0, 0] =  0;  c[0, 1] =  0; w[0] = 4./9.
    c[1, 0] =  1;  c[1, 1] =  0; w[1] = 1./9.
    c[2, 0] =  0;  c[2, 1] =  1; w[2] = 1./9.
    c[3, 0] = -1;  c[3, 1] =  0; w[3] = 1./9.
    c[4, 0] =  0;  c[4, 1] = -1; w[4] = 1./9.
    c[5, 0] =  1;  c[5, 1] =  1; w[5] = 1./36.
    c[6, 0] = -1;  c[6, 1] =  1; w[6] = 1./36.
    c[7, 0] = -1;  c[7, 1] = -1; w[7] = 1./36.
    c[8, 0] =  1;  c[8, 1] = -1; w[8] = 1./36.

    return c, w, cs2

def compute_random_rho_and_u(num_samples, rho_min=0.95, rho_max=1.05, u_abs_min=0.0, u_abs_max=0.01):
    
    rho   = np.random.uniform(rho_min, rho_max, size=num_samples)    
    u_abs = np.random.uniform(u_abs_min, u_abs_max, size=num_samples)
    theta = np.random.uniform(0, 2 * np.pi, size=num_samples)
    
    ux = u_abs * np.cos(theta)
    uy = u_abs * np.sin(theta)
    u  = np.array([ux,uy]).transpose()
    
    return rho, u

def compute_random_F(c, num_samples, sigma_min, sigma_max):

    Q  = 9
    K0 = 1.0 / 9
    K1 = 1.0 / 6
    
    F_random = np.zeros((num_samples, Q))
    
    if sigma_min==sigma_max:
        sigma = sigma_min*np.ones(num_samples)
    else:
        sigma = np.random.uniform(sigma_min, sigma_max, size=num_samples)         
        
    for i in range(num_samples):
        F_random[i,:] = np.random.normal(0, sigma[i], size=(1,Q))

        rho_hat = np.sum(F_random[i,:])
        ux_hat  = np.sum(F_random[i,:] * c[:,0])
        uy_hat  = np.sum(F_random[i,:] * c[:,1])

        F_random[i,:] = F_random[i,:] -K0 * rho_hat -K1 * ux_hat * c[:,0] - K1 * uy_hat * c[:,1]  

    return F_random

def compute_fpre_and_fpost(f_eq, f_neq, tau_min=1, tau_max=1):
    
    tau = np.random.uniform(tau_min, tau_max, size=f_eq.shape[0])
    f_pre = f_eq + f_neq
    f_post = f_pre + 1 / tau[:,None] * (f_eq - f_pre)

    return tau, f_pre, f_post

def delete_negative_samples(f_eq, f_pre, f_post):
    """Crucial to ensure no contributions to lower order moments (mass and momentum invariance)"""
    i_neg_f_eq = np.where(np.sum(f_eq < 0,axis=1) > 0)[0]
    i_neg_f_pre = np.where(np.sum(f_pre < 0,axis=1) > 0)[0]
    i_neg_f_post = np.where(np.sum(f_post < 0,axis=1) > 0)[0]

    i_neg_f = np.concatenate([i_neg_f_pre, i_neg_f_post, i_neg_f_eq])
    
    f_eq = np.delete(np.copy(f_eq), i_neg_f, 0)
    f_pre = np.delete(np.copy(f_pre), i_neg_f, 0)
    f_post = np.delete(np.copy(f_post), i_neg_f, 0)
    
    return f_eq, f_pre, f_post

def generate_data():
    """ Follows the outlined algorithm (Algorithm 1) in the paper """
    n_samples = 100_000
    sigma_min = 1e-15 
    sigma_max = 5e-4  

    # lattice velocities and weights
    Q = 9 
    c, w, cs2 = stencil()

    fpre_list  = np.empty((n_samples, Q))
    fpost_list = np.empty((n_samples, Q))
    feq_list   = np.empty((n_samples, Q))

    idx = 0
    # loop until we get n_samples without negative populations
    while idx < n_samples: 
        
        # get random values for macroscopic quantities
        rho, u = compute_random_rho_and_u(n_samples)

        rho = rho[:,np.newaxis]
        ux  = u[:,0][:,np.newaxis]
        uy  = u[:,1][:,np.newaxis]

        # compute the equilibrium distribution
        f_eq  = np.zeros((n_samples, 1, Q))
        f_eq  = compute_Feq(f_eq, rho, ux, uy, c, w, Q, cs2)[:,0,:]
        
        # compute a random non equilibrium part
        f_neq = compute_random_F(c, n_samples, sigma_min, sigma_max)   
        
        # apply BGK to f_pre = f_eq + f_neq
        tau , f_pre, f_post = compute_fpre_and_fpost(f_eq, f_neq)
        
        # remove negative elements
        f_eq, f_pre, f_post = delete_negative_samples(f_eq, f_pre, f_post)
        
        # accumulate 
        non_negatives = f_pre.shape[0]
        
        idx1 = min(idx + non_negatives, n_samples)
        to_be_added = min(n_samples - idx, non_negatives)
        
        fpre_list[idx:idx1] = f_pre[:to_be_added]
        fpost_list[idx:idx1] = f_post[:to_be_added]
        feq_list[idx:idx1] = f_eq[:to_be_added]
        
        idx = idx + non_negatives

    # Note: Serialize data into a training and validation sets
    # Note: Normalizations of the train/test data -> for instance, fpre_i / rho = fpre_i / SUM(fpre_i)
    feq   = feq_list / np.sum(feq_list, axis=1)[:, np.newaxis]          # [num_samples, Q=9]
    fpre  = fpre_list / np.sum(fpre_list, axis=1)[:, np.newaxis]        # [num_samples, Q=9]
    fpost = fpost_list / np.sum(fpost_list, axis=1)[:, np.newaxis]      # [num_samples, Q=9]

    # split train and test set
    fpre_train, fpre_test, fpost_train, fpost_test, feq_train, feq_test = train_test_split(fpre, fpost, feq, test_size=0.3, shuffle=True)

    np.savez('train_dataset.npz', f_pre=fpre_train, f_post=fpost_train, f_eq=feq_train)
    np.savez('test_dataset.npz', f_pre=fpre_test, f_post=fpost_test, f_eq=feq_test)

    print('Generated data')
    

class LBMDataset(Dataset):
    def __init__(self, path):
        data = np.load(path, allow_pickle=True)
        self.feq   = data['f_eq']
        self.fpre  = data['f_pre']
        self.fpost = data['f_post']

        #print(self.feq.shape)
        #print(self.fpre.shape)
        #print(self.fpost.shape)

    def __getitem__(self, idx):
        feq = self.feq[idx]
        fpre = self.fpre[idx]
        fpost = self.fpost[idx]
        #feq = (feq - feq.min()) / (feq.max() - feq.min()) # [0, 1]
        #fpre = (fpre - fpre.min()) / (fpre.max() - fpre.min()) # [0, 1]
        #fpost = (fpost - fpost.min()) / (fpost.max() - fpost.min()) # [0, 1]
        feq = torch.from_numpy(feq).float() # hwc to chw
        fpre = torch.from_numpy(fpre).float() # hwc to chw
        fpost = torch.from_numpy(fpost).float() # hwc to chw
        return fpre, fpost, feq

    def __len__(self):
        assert len(self.feq) == len(self.fpre) == len(self.fpost), "All distributions should be of the same length"
        return len(self.feq)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', action='store_true')
    parser.add_argument('--test', action='store_true')
    
    args = parser.parse_args()
    
    if args.test:
        dataset = LBMDataset("train_dataset.npz")
        print("Size of data: ", len(dataset))
    elif args.create:
        generate_data()
