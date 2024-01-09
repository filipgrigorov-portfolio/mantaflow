import numpy as np
import os

def generate_next_available_idx(dir_name):
    file_names = list(os.walk(dir_name))
    return len(file_names) - 1 if len(file_names) > 1 else 0

def save_np_tensor(dir_name, x, type_str, time_step):
    assert os.path.exists(dir_name), f"{dir_name} should exist"
    assert isinstance(x, np.ndarray), "tesnor should be of type numpy.ndarray"

    full_file_path = os.path.join(dir_name, f"{type_str}_{time_step}.npy")
    print(f"Saving {full_file_path}", end='\r')
    np.save(full_file_path, x)
