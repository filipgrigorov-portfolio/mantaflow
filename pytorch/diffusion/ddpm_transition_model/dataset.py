import numpy as np
import os
import sys
sys.path.append("./tools")
import uniio

import torch
from torch.utils.data import Dataset

from utils import TOTAL_SIMULATION_TIME, GRID_SIZE

# Definitions
DEBUG = True

def load_sim_file(data_path, sim, type_name, idx):
    uniPath = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # TOTAL_SIMULATION_TIME files per sim
    print(uniPath)
    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr


class MantaFlow2DSimStatesDataset(Dataset):
    """(A) Provides dt, vt at t=t-1, t=t and t=t+1, individually"""
    def __init__(self, data_path, start_itr, end_itr, transform_ops=None):
        self.transform_ops = transform_ops

        # Note: Identical entries normalized by the TOTAL_SIMULATION_TIME
        tau_range = torch.arange(0, TOTAL_SIMULATION_TIME)
        self.tau = tau_range# / (TOTAL_SIMULATION_TIME - 1)
        print(f"Shape of tau: {self.tau.shape}")


        self.data = []
        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):

                for t in range(0, TOTAL_SIMULATION_TIME):
                    bc0 = load_sim_file(data_path, sim, 'boundary', t) # Note: Should be the same for each step of the simulation 
                    dt = load_sim_file(data_path, sim, 'density', t)
                    vt = load_sim_file(data_path, sim, 'vel', t)
                    xt = np.dstack([dt, vt, bc0])
                    self.data.append([xt, self.tau[t]])

        num_data = len(self.data)
        NUM_SIMS = 1
        assert num_data >= NUM_SIMS, f"Number of simulations should be at least {NUM_SIMS}"
        #print(f'Loaded {num_data} data triplets') # [20, 100, 64, 64, 4]
        

    def __getitem__(self, idx):
        # Note: at time t
        d0 = self.data[idx][0][..., :1]
        v0 = self.data[idx][0][..., 1:3]
        bc0 = self.data[idx][0][..., -1:]
        tau = self.data[idx][1]

        # Normalize input data
        d0 = (d0 - d0.min()) / (d0.max() - d0.min()) # [0, 1]
        v0 = (v0 - v0.min()) / (v0.max() - v0.min()) # [0, 1]

        d0_t = torch.from_numpy(d0.transpose(2, 0, 1)).float() # hwc to chw
        v0_t = torch.from_numpy(v0.transpose(2, 0, 1)).float() # hwc to chw
        bc0_t = torch.from_numpy(bc0.transpose(2, 0, 1)).float() # hwc to chw

        #print(d0_t.size())
        #print(v0_t.size())
        #print(tau_t.size())

        if self.transform_ops is not None:
            v0_t = self.transform_ops(v0_t)

        
        
        # Note: at time t - 1
        d0_prev = self.data[idx - 1][0][..., :1] if idx > 0 else self.data[idx][0][..., :1]
        v0_prev = self.data[idx - 1][0][..., 1:3] if idx > 0 else self.data[idx][0][..., 1:3]

        # Normalize input data
        d0_prev = (d0_prev - d0_prev.min()) / (d0_prev.max() - d0_prev.min()) # [0, 1]
        v0_prev = (v0_prev - v0_prev.min()) / (v0_prev.max() - v0_prev.min()) # [0, 1]

        d0_prev_t = torch.from_numpy(d0_prev.transpose(2, 0, 1)).float() # hwc to chw
        v0_prev_t = torch.from_numpy(v0_prev.transpose(2, 0, 1)).float() # hwc to chw

        if self.transform_ops is not None:
            v0_prev_t = self.transform_ops(v0_prev_t)

        
        
        # Note: at time t + 1
        d0_next = self.data[idx + 1][0][..., :1] if idx < TOTAL_SIMULATION_TIME - 1 else self.data[idx][0][..., :1]
        v0_next = self.data[idx + 1][0][..., 1:3] if idx < TOTAL_SIMULATION_TIME - 1 else self.data[idx][0][..., 1:3]

        # Normalize input data
        d0_next = (d0_next - d0_next.min()) / (d0_next.max() - d0_next.min()) # [0, 1]
        v0_next = (v0_next - v0_next.min()) / (v0_next.max() - v0_next.min()) # [0, 1]

        d0_next_t = torch.from_numpy(d0_next.transpose(2, 0, 1)).float() # hwc to chw
        v0_next_t = torch.from_numpy(v0_next.transpose(2, 0, 1)).float() # hwc to chw

        if self.transform_ops is not None:
            v0_next_t = self.transform_ops(v0_next_t)

        return d0_prev_t, v0_prev_t,   d0_t, v0_t, bc0_t,   d0_next_t, v0_next_t,   tau

    def __len__(self):
        return len(self.data)
