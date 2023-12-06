import einops
import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import MantaFlow2DDataset, MantaFlow2DSimSequenceDataset, MantaFlow2DSimXYDataset
from torchvision import transforms as T

#Definitions
BATCH_SIZE = 8
GRID_SIZE = 64
CHANNELS = 4 # d0 (ch = 1), tau (1), v (2)

OUTPUT_DATA_PATH = "results/"

from dataset import TOTAL_SIMULATION_TIME

# Display utils
def show_compound_images_seq_single_ch(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays:
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(16, 16))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                img = (images[idx][0]).astype(np.float32)
                #print(img.shape) # debug
                plt.imshow(img, vmin=0, vmax=1)
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_test.jpg"))
    

def show_compound_images_seq(images, title=""):
    """Shows the provided images as sub-pictures in a square (v0 (x and y))"""

    # Converting images to CPU numpy arrays: images -> (16, 2, 64, 64)
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(20, 20))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # Velocities
    print(f"Compositing x velocities images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                img = images[idx][0]
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_vel_x.jpg"))

    print(f"Compositing y velocities images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                img = images[idx][1]
                img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_vel_y.jpg"))

@torch.no_grad()
def generate_new_images_seq(ddpm, d_init, tau, sim_time=TOTAL_SIMULATION_TIME, device=None, frames_per_gif=100, gif_name="sampling.gif", channels=CHANNELS, h=GRID_SIZE, w=GRID_SIZE):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    #frame_idxs = np.linspace(0, ddpm.diffusion_steps, frames_per_gif).astype(np.uint)
    #frames = []

    if device is None:
        device = ddpm.device

    # Starting from random noise
    x = torch.randn(sim_time, channels, h, w).to(device)

    # From T to 0, denoise
    for _, t in enumerate(list(range(ddpm.diffusion_steps))[::-1]):
        # Estimating noise to be removed
        time_tensor = (torch.ones(sim_time, 1) * t).to(device).long()
        x_stacked = torch.concat([x, d_init, tau], dim=1)
        eta_theta = ddpm.backward(x_stacked, time_tensor)[:, :2, :, :]

        alpha_t = ddpm.alphas[t]
        alpha_t_bar = ddpm.alpha_bars[t]

        # Partially denoising the image
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

        if t > 0:
            # Step 3 in https://arxiv.org/pdf/2006.11239.pdf (Algorithm 2)
            z = torch.randn(sim_time, channels, h, w).to(device)

            # Option 1: sigma_t squared = beta_t
            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()

            # Adding some more noise like in Langevin Dynamics fashion
            x = x + sigma_t * z
            
        #TODO; go back from -1 to 1 to 0 to 1 in nfloat32

    return x

'''
        # Adding frames to the GIF
        if idx in frame_idxs or t == 0:
            # Putting digits in range [0, 255]
            normalized = x.clone()[:, 0, :, :][:, np.newaxis, :, :]
            for i in range(len(normalized)):
                normalized[i] -= torch.min(normalized[i])
                normalized[i] *= 255 / torch.max(normalized[i])

            # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
            frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(sim_time**0.5), c=1)
            frame = frame.cpu().numpy().astype(np.uint8)

            # Rendering frame
            frames.append(frame)

    # Storing the gif
    with io.get_writer(gif_name, mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)

            # Showing the last frame for a longer time
            if idx == len(frames) - 1:
                last_rgb_frame = np.repeat(frames[-1], 3, axis=2)
                for _ in range(frames_per_gif // 3):
                    writer.append_data(last_rgb_frame)
'''
    #return x
