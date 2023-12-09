import numpy as np
import os
import sys
sys.path.append("./tools")
import uniio

import torch
from torch.utils.data import Dataset

# Definitions
TOTAL_SIMULATION_TIME = 20 # length of sequences
DEBUG = True

def load_sim_file(data_path, sim, type_name, idx):
    uniPath = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # 100 files per sim
    print(uniPath)
    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr

class MantaFlow2DSimSequenceDataset(Dataset):
    """Provides dt, vt, tau in a sequence of length TOTAL_SIMULATION_TIME"""
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.batched_densities = []
        self.batched_velocities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                # Note: fill in each batch with a sequence of len TOTAL_SIMULATION_TIME
                seq_of_density_fields = []
                seq_of_velocity_fields = []
                for i in range(0, TOTAL_SIMULATION_TIME):
                    seq_of_density_fields.append(load_sim_file(data_path, sim, 'density', i))
                    seq_of_velocity_fields.append(load_sim_file(data_path, sim, 'vel', i))

                seq_of_density_fields = np.reshape( seq_of_density_fields, (len(seq_of_density_fields), grid_height, grid_width, 1) )
                print("Read uni files (density), total data " + format(seq_of_density_fields.shape))

                seq_of_velocity_fields = np.reshape( seq_of_velocity_fields, (len(seq_of_velocity_fields), grid_height, grid_width, 3) )
                seq_of_velocity_fields = seq_of_velocity_fields[:, :, :, :2]
                print("Read uni files (velocity), total data " + format(seq_of_velocity_fields.shape))

                # Note: Batchify the sims
                self.batched_densities.append(seq_of_density_fields)
                self.batched_velocities.append(seq_of_velocity_fields)


        num_densities = len(self.batched_densities)
        num_velocities = len(self.batched_velocities)
        NUM_SIMS = 2
        assert num_densities >= NUM_SIMS, f"Number of simulations should be at least {NUM_SIMS}"
        assert self.batched_velocities[0].shape[0] == TOTAL_SIMULATION_TIME, f"Simulation time should be {TOTAL_SIMULATION_TIME}"
        print(f'Loaded {num_densities} densities')
        print(f'Loaded {num_velocities} densities')

        self.batched_densities = np.array(self.batched_densities)
        self.batched_velocities = np.array(self.batched_velocities)
        print(f"Shape of batchified densities sims: {self.batched_densities.shape}")
        print(f"Shape of batchified velocities sims: {self.batched_velocities.shape}") # [20, 100, 64, 64, 2]

        if DEBUG:
            from utils_seq import show_compound_images_seq_single_ch
            print('Debug printing data')
            vel_vals = self.batched_velocities[3]
            show_compound_images_seq_single_ch(vel_vals, title="dataset_debug_")
            raise('debug')

        # Note: Identical entries normalized by the TOTAL_SIMULATION_TIME
        tau_range = torch.arange(0, TOTAL_SIMULATION_TIME)
        XX, _ = torch.meshgrid(tau_range, tau_range, indexing="ij")
        self.tau_t = torch.stack([ torch.tile(XX[t], dims=(TOTAL_SIMULATION_TIME, 1))[: grid_height, : grid_width].unsqueeze(-1).permute(2, 0, 1) for t in range(TOTAL_SIMULATION_TIME) ], dim=0).float()
        self.tau_t /= (TOTAL_SIMULATION_TIME - 1) # [100, 1, 64, 64]
        print(f"Shape of tau: {self.tau_t.shape}")

        # Note: Transform at construction to save compute (in-memory)

        self.batched_densities_t = []
        self.batched_velocities_t = []
        for idx in range(self.__len__()):
            d0_t = torch.from_numpy(self.batched_densities[idx].transpose(0, 3, 1, 2)).float()
            v0_t = torch.from_numpy(self.batched_velocities[idx].transpose(0, 3, 1, 2)).float()
            d0_t -= d0_t.min()
            d0_t /= d0_t.max()
            v0_t -= v0_t.min()
            v0_t /= v0_t.max()

            if self.transform_ops is not None:
                d0_t = self.transform_ops(d0_t)
                v0_t = self.transform_ops(v0_t)
            
            self.batched_densities_t.append(d0_t)
            self.batched_velocities_t.append(v0_t)
        
        self.batched_densities_t = torch.stack(self.batched_densities_t, dim=0).float()
        self.batched_velocities_t = torch.stack(self.batched_velocities_t, dim=0).float()
        print(f"Shape of torch batchified densities sims: {self.batched_densities_t.size()}")
        print(f"Shape of torch batchified velocities sims: {self.batched_velocities_t.size()}") # [20, 100, 64, 64, 2]


    def __getitem__(self, idx):
        d0_seq_t = self.batched_densities_t[idx]
        v0_seq_t = self.batched_velocities_t[idx]

        return d0_seq_t, v0_seq_t, self.tau_t

    def __len__(self):
        assert self.batched_densities.shape[0] == self.batched_velocities.shape[0]
        return self.batched_velocities.shape[0]
    
class MantaFlow2DSimTupleDataset(Dataset):
    """Provides dt, vt, tau as a triplet, individually"""
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.data = []

        # Note: Identical entries normalized by the TOTAL_SIMULATION_TIME
        tau_range = np.arange(0, grid_height)
        XX, _ = np.meshgrid(tau_range, tau_range, indexing="ij")
        self.tau = np.stack([ np.tile(XX[t], reps=(grid_height, 1))[: grid_height, : grid_width, np.newaxis] for t in range(TOTAL_SIMULATION_TIME) ], axis=0)
        print(self.tau[0].shape)
        self.tau = self.tau / (TOTAL_SIMULATION_TIME - 1) # [TOTAL_SIMULATION_TIME, 64, 64, 1]
        print(f"Shape of tau: {self.tau.shape}")

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):

                # ICs/BCs
                d0 = load_sim_file(data_path, sim, 'density', 0)
                bc0 = load_sim_file(data_path, sim, 'boundary', 0)

                # Input data
                for t in range(0, TOTAL_SIMULATION_TIME):
                    dt = load_sim_file(data_path, sim, 'density', t)
                    vt = load_sim_file(data_path, sim, 'vel', t)[..., :2]
                    xt = np.dstack([dt, vt, d0, bc0, self.tau[t]])
                    self.data.append(xt)

                    # debug
                    #import matplotlib.pyplot as plt
                    #plt.imshow(self.tau[t], vmin=0, vmax=1)
                    #plt.savefig(f"tau_train_{t + 1}.jpg")

                #raise('debug')

        num_data = len(self.data)
        print(f"Number data points: {num_data}")
        NUM_SIMS = 1
        assert num_data >= NUM_SIMS, f"Number of simulations should be at least {NUM_SIMS}"
        print(f'Loaded {num_data} data triplets') # [20, 100, 64, 64, 4]

    def __getitem__(self, idx):
        dt = self.data[idx][..., 0][..., np.newaxis]
        vt = self.data[idx][..., 1:3]

        # ICs and BCs
        d0 = self.data[idx][..., 3][..., np.newaxis]
        bc0 = self.data[idx][..., 4][..., np.newaxis]
        tau = self.data[idx][..., 5][..., np.newaxis]

        vt = (vt - vt.min()) / (vt.max() - vt.min()) # [0, 1]

        dt_t = torch.from_numpy(dt.transpose(2, 0, 1)).float() # hwc to chw
        vt_t = torch.from_numpy(vt.transpose(2, 0, 1)).float() # hwc to chw

        # ICs and BCs
        d0_t = torch.from_numpy(d0.transpose(2, 0, 1)).float() # hwc to chw
        bc0_t = torch.from_numpy(bc0.transpose(2, 0, 1)).float() # hwc to chw
        tau_t = torch.from_numpy(tau.transpose(2, 0, 1)).float() # hwc to chw

        #print(d0_t.size())
        #print(bc0_t.size())
        #print(v0_t.size())
        #print(tau_t.size())

        if self.transform_ops is not None:
            dt_t = self.transform_ops(dt_t)
            vt_t = self.transform_ops(vt_t)

        return dt_t, vt_t, d0_t, bc0_t, tau_t # 6 channels

    def __len__(self):
        return len(self.data)

class MantaFlow2DDataset(Dataset):
    """Provides dt, vt, t"""
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.densities = []
        self.velocities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                for i in range(0, TOTAL_SIMULATION_TIME):
                    self.densities.append(load_sim_file(data_path, sim, 'density', i))
                    self.velocities.append(load_sim_file(data_path, sim, 'vel', i))

        num_densities = len(self.densities)
        num_velocities = len(self.velocities)
        if num_densities < 2 * TOTAL_SIMULATION_TIME:
            raise("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times...")
        
        print(f'Loaded {num_densities} densities')
        print(f'Loaded {num_velocities} densities')

        self.densities = np.reshape( self.densities, (len(self.densities), grid_height, grid_width, 1) )
        print("Read uni files (density), total data " + format(self.densities.shape))

        self.velocities = np.reshape( self.velocities, (len(self.velocities), grid_height, grid_width, 3) )
        print("Read uni files (velocity), total data " + format(self.velocities.shape))

    def __getitem__(self, idx):
        d0 = self.densities[idx]
        v0 = self.velocities[idx]

        d0_t = torch.from_numpy(d0).float()
        v0_t = torch.from_numpy(v0).float()

        if self.transform_ops is not None:
            d0_t = self.transform_ops(d0)
            v0_t = self.transform_ops(v0)

        return d0_t, v0_t, idx

    def __len__(self):
        assert self.densities.shape[0] == self.velocities.shape[0]
        return self.densities.shape[0]


class MantaFlow2DTemporalDataset(Dataset):
    """Provides dt-1, vt-1, t-1, dt, vt, t"""
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.densities = []
        self.velocities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                for i in range(0, TOTAL_SIMULATION_TIME):
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
    

    
class MantaFlow2DSimXYStatesDataset(Dataset):
    """Provides dt, vt, tau as a triplet at t-1, t and t+1, individually"""
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.data = []

        # Note: Identical entries normalized by the TOTAL_SIMULATION_TIME
        tau_range = np.arange(0, TOTAL_SIMULATION_TIME)
        XX, _ = np.meshgrid(tau_range, tau_range, indexing="ij")
        self.tau = np.stack([ np.tile(XX[t], reps=(TOTAL_SIMULATION_TIME, 1))[: grid_height, : grid_width, np.newaxis] for t in range(TOTAL_SIMULATION_TIME) ], axis=0)
        self.tau = self.tau / (TOTAL_SIMULATION_TIME - 1) # [TOTAL_SIMULATION_TIME, 64, 64, 1]
        print(f"Shape of tau: {self.tau.shape}")

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):

                for t in range(0, TOTAL_SIMULATION_TIME):
                    dt = load_sim_file(data_path, sim, 'density', t)
                    vt = load_sim_file(data_path, sim, 'vel', t)
                    xt = np.dstack([dt, vt, self.tau[t]])
                    self.data.append(xt)

        num_data = len(self.data)
        NUM_SIMS = 1
        assert num_data >= NUM_SIMS, f"Number of simulations should be at least {NUM_SIMS}"
        #print(f'Loaded {num_data} data triplets') # [20, 100, 64, 64, 4]

    def __getitem__(self, idx):
        # Note: at time t
        d0 = self.data[idx][..., :1]
        v0 = self.data[idx][..., 1:3]
        tau = self.data[idx][..., -1:]

        v0 = (v0 - v0.min()) / (v0.max() - v0.min()) # [0, 1]

        d0_t = torch.from_numpy(d0.transpose(2, 0, 1)).float() # hwc to chw
        v0_t = torch.from_numpy(v0.transpose(2, 0, 1)).float() # hwc to chw
        tau_t = torch.from_numpy(tau.transpose(2, 0, 1)).float() # hwc to chw

        #print(d0_t.size())
        #print(v0_t.size())
        #print(tau_t.size())

        if self.transform_ops is not None:
            v0_t = self.transform_ops(v0_t)

        # Note: at time t - 1
        d0_prev = self.data[idx - 1][..., :1] if idx > 0 else self.data[idx][..., :1]
        v0_prev = self.data[idx - 1][..., 1:3] if idx > 0 else self.data[idx][..., 1:3]
        tau_prev = self.data[idx - 1][..., -1:] if idx > 0 else self.data[idx][..., -1:]

        v0_prev = (v0_prev - v0_prev.min()) / (v0_prev.max() - v0_prev.min()) # [0, 1]

        d0_prev_t = torch.from_numpy(d0_prev.transpose(2, 0, 1)).float() # hwc to chw
        v0_prev_t = torch.from_numpy(v0_prev.transpose(2, 0, 1)).float() # hwc to chw
        tau_prev_t = torch.from_numpy(tau_prev.transpose(2, 0, 1)).float() # hwc to chw

        if self.transform_ops is not None:
            v0_prev_t = self.transform_ops(v0_prev_t)

        # Note: at time t + 1
        d0_next = self.data[idx + 1][..., :1] if idx < TOTAL_SIMULATION_TIME - 1 else self.data[idx][..., :1]
        v0_next = self.data[idx + 1][..., 1:3] if idx < TOTAL_SIMULATION_TIME - 1 else self.data[idx][..., 1:3]
        tau_next = self.data[idx + 1][..., -1:] if idx < TOTAL_SIMULATION_TIME - 1 else self.data[idx][..., -1:]

        v0_next = (v0_next - v0_next.min()) / (v0_next.max() - v0_next.min()) # [0, 1]

        d0_next_t = torch.from_numpy(d0_next.transpose(2, 0, 1)).float() # hwc to chw
        v0_next_t = torch.from_numpy(v0_next.transpose(2, 0, 1)).float() # hwc to chw
        tau_next_t = torch.from_numpy(tau_next.transpose(2, 0, 1)).float() # hwc to chw

        if self.transform_ops is not None:
            v0_next_t = self.transform_ops(v0_next_t)

        return d0_prev_t, v0_prev_t, tau_prev_t,   d0_t, v0_t, tau_t,   d0_next_t, v0_next_t, tau_next_t

    def __len__(self):
        return len(self.data)
