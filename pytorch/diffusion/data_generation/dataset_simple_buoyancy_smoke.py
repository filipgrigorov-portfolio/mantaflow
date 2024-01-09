import glob
import numpy as np
import os
import re
import torch

from torchvision import transforms as T
from torch.utils.data import Dataset

CHANNELS = 4
DEBUG = True

def default_transform_ops():
    return T.Compose([
        # Note: [-1, 1] as DDPM generates a normally distributed data (assumes data in [0, 1])
        T.Lambda(lambda t: (t * 2) - 1)],
    )

def inverse_default_transform_ops():
    return T.Compose([
        T.Lambda(lambda t: (t + 1) / 2)],
    )

def load_sim_file(data_path, sim, type_name, idx):
    data_full_path = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # TOTAL_SIMULATION_TIME files per sim
    print(f"Loading \"{data_full_path}\"")
    arr_np = np.load(data_full_path).astype(np.float32)
    return arr_np

# TODO: load and reproduce the visualization
class PhiflowDataset(Dataset):
    """Dataset class providing previous data, current data, next data and a time matrix for a given index"""
    def __init__(self, root, transform_ops, sim_time):
        self.transform_ops = transform_ops
        self.sim_time = sim_time

        folder_names = glob.glob(f"{root}/simSimple_*/", recursive = True)

        if DEBUG:
            print(folder_names)

        self.time_mat_range = np.arange(0, sim_time)

        self.data = []
        for folder_name in folder_names:
            density_file_names = sorted(glob.glob(f"{folder_name}/density_*.npy"), key=os.path.getmtime)
            velocity_x_file_names = sorted(glob.glob(f"{folder_name}/velocity_x_*.npy"), key=os.path.getmtime)
            velocity_y_file_names = sorted(glob.glob(f"{folder_name}/velocity_y_*.npy"), key=os.path.getmtime)
            solid_file_names = sorted(glob.glob(f"{folder_name}/solid_*.npy"), key=os.path.getmtime)

            if DEBUG:
                #print(density_file_names)
                print(velocity_x_file_names)
                #print(velocity_y_file_names)
                #print(solid_file_names)

            assert len(density_file_names) == sim_time, "len(density_file_names) == sim_time"
            assert len(velocity_x_file_names) == sim_time, "len(velocity_x_file_names) == sim_time"
            assert len(velocity_y_file_names) == sim_time, "len(velocity_y_file_names) == sim_time"
            assert len(solid_file_names) == sim_time, "len(solid_file_names) == sim_time"

            density_np = np.array([ np.load(filename).astype(np.float32) for filename in density_file_names ])
            vel_x_np = np.array([ np.load(filename).astype(np.float32) for filename in velocity_x_file_names ])
            vel_y_np = np.array([ np.load(filename).astype(np.float32) for filename in velocity_y_file_names ])
            solid_np = np.array([ np.load(filename).astype(np.float32) for filename in solid_file_names ])

            # Note: In-memory loading since small
            for t in range(sim_time):
                # Note: Implicit time encoding
                time_mat = np.ones_like(density_np[t]) * self.time_mat_range[t]
                stacked_data_t_np = np.concatenate([ density_np[t], vel_x_np[t], vel_y_np[t], solid_np[t], time_mat ], axis=-1)
                self.data.append(stacked_data_t_np)

        self.data = np.array(self.data)
            
    def __getitem__(self, index):
        batch = self.data[index]
        batch_prev = self.data[max(0, index - 1)]
        batch_next = self.data[min(index + 1, self.sim_time - 1)]
        tensors = []
        tensors_prev = []
        tensors_next = []
        for c in range(CHANNELS):
                batch[..., c] = (batch[..., c] - batch[..., c].min()) / (batch[..., c].max() - batch[..., c].min()) # range: [0, 1]
                batch_prev[..., c] = (batch_prev[..., c] - batch_prev[..., c].min()) / (batch_prev[..., c].max() - batch_prev[..., c].min())    # range: [0, 1]
                batch_next[..., c] = (batch_next[..., c] - batch_next[..., c].min()) / (batch_next[..., c].max() - batch_next[..., c].min())    # range: [0, 1]

                if self.transform_ops:
                    tensors.append( self.transform_ops(torch.from_numpy(batch[..., c].transpose(0, 3, 1, 2)).float()) )    # hwc to chw
                    tensors_prev.append( self.transform_ops(torch.from_numpy(batch_prev[..., c].transpose(0, 3, 1, 2)).float()) )  # hwc to chw -> [-1, 1]
                    tensors_next.append( self.transform_ops(torch.from_numpy(batch_next[..., c].transpose(0, 3, 1, 2)).float()) )  # hwc to chw -> [-1, 1]

        # debug
        print(tensors[0].size())
        # debug

        return batch

if __name__ == "__main__":
    ROOT = "data"
    dataset = PhiflowDataset(ROOT, sim_time=16, transform_ops=default_transform_ops())
    dataset[0]
