import numpy as np
import os
import sys
sys.path.append("./tools")
import uniio

import torch

from torch.utils.data import Dataset

def load_sim_file(data_path, sim, type_name, idx):
    uniPath = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # 100 files per sim
    print(uniPath)
    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr

class MantaFlow2DDataset(Dataset):
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.densities = []
        self.velocities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                for i in range(0, 100):
                    self.densities.append(load_sim_file(data_path, sim, 'density', i))
                    self.velocities.append(load_sim_file(data_path, sim, 'vel', i))

        num_densities = len(self.densities)
        num_velocities = len(self.velocities)
        if num_densities < 200:
            raise("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times...")

        self.densities = np.reshape( self.densities, (len(self.densities), grid_height, grid_width, 1) )
        print("Read uni files (density), total data " + format(self.densities.shape))

        self.velocities = np.reshape( self.velocities, (len(self.velocities), grid_height, grid_width, 3) )
        print("Read uni files (velocity), total data " + format(self.velocities.shape))

    def __getitem__(self, idx):
        d1 = self.densities[idx]
        v1 = self.velocities[idx]

        d0, v0 = d1.copy(), v1.copy()
        if idx - 1 >= 0:
            d0 = self.densities[idx - 1]
            v0 = self.velocities[idx - 1]

        d0_t = torch.from_numpy(d0).float()
        v0_t = torch.from_numpy(v0).float()
        d1_t = torch.from_numpy(d1).float()
        v1_t = torch.from_numpy(v1).float()

        if self.transform_ops is not None:
            d0_t = self.transform_ops(d0)
            v0_t = self.transform_ops(v0)
            d1_t = self.transform_ops(d1)
            v1_t = self.transform_ops(v1)

        return d0_t, v0_t, (idx - 1 if idx - 1 >= 0 else idx), d1_t, v1_t, idx

    def __len__(self):
        assert self.densities.shape[0] == self.velocities.shape[0]
        return self.densities.shape[0]
