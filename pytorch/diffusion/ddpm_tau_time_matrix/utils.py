import einops
import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torch.utils.data import Dataset, DataLoader

from dataset import MantaFlow2DDataset, MantaFlow2DSimSequenceDataset, MantaFlow2DSimTupleDataset
from torchvision import transforms as T

#Definitions
BATCH_SIZE = 8
GRID_SIZE = 64
INPUT_CHANNELS = 5

OUTPUT_DATA_PATH = "results/"

from dataset import TOTAL_SIMULATION_TIME

# Display utils
def show_images(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(16, 16))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # Populating figure with sub-plots
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0])
                idx += 1
    fig.suptitle(title, fontsize=30)

    # Showing the figure
    #plt.show()
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + ".jpg"))

def show_compound_images(images, title=""):
    """Shows the provided images as sub-pictures in a square (d0, v0 (x and y))"""

    # Converting images to CPU numpy arrays: images -> (16, 3, 64, 64)
    if type(images) is torch.Tensor:
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(8, 8))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # Densities
    print(f"Compositing densities images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][0], cmap="gray")
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_densities.jpg"))

    # Velocities
    print(f"Compositing x velocities images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][1])
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_vel_x.jpg"))

    print(f"Compositing y velocities images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                plt.imshow(images[idx][2])
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_vel_y.jpg"))


def show_first_batch(loader):
    """Shows the images in the first batch of a DataLoader object"""
    for batch in loader:
        show_images(batch[1], "Images in the first batch")
        break

def show_forward(ddpm, loader, device):
    """Showing the forward process"""
    for batch in loader:
        imgs = batch[0]

        show_images(imgs, 0, f"Original images at {0}")
        show_images(imgs, 1, f"Original images at {1}")

        for percent_noise in [0.25, 0.5, 0.75, 1]:
            print(f"Generating noisy images with {percent_noise} % noise")
            imgs = ddpm(imgs.to(device), [int(percent_noise * ddpm.n_steps) - 1 for _ in range(len(imgs))])
            show_images(imgs, 0, f"DDPM Noisy images {int(percent_noise * 100)}% at {0}")
            show_images(imgs, 1, f"DDPM Noisy images {int(percent_noise * 100)}% at {1}")

        break

def generate_new_images(ddpm, n_samples=16, device=None, frames_per_gif=100, gif_name="sampling.gif", c=INPUT_CHANNELS, h=GRID_SIZE, w=GRID_SIZE):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""

    frame_idxs = np.linspace(0, ddpm.diffusion_steps, frames_per_gif).astype(np.uint)
    frames = []

    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_samples, c, h, w).to(device)

        # From T to 0, denoise
        for idx, t in enumerate(list(range(ddpm.diffusion_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_samples, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                # Step 3 in https://arxiv.org/pdf/2006.11239.pdf (Algorithm 2)
                z = torch.randn(n_samples, c, h, w).to(device)

                # Option 1: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Option 2: sigma_t squared = beta_tilda_t
                # prev_alpha_t_bar = ddpm.alpha_bars[t-1] if t > 0 else ddpm.alphas[0]
                # beta_tilda_t = ((1 - prev_alpha_t_bar)/(1 - alpha_t_bar)) * beta_t
                # sigma_t = beta_tilda_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

            # Adding frames to the GIF
            if idx in frame_idxs or t == 0:
                # Putting digits in range [0, 255]
                normalized = x.clone()[:, 2, :, :][:, np.newaxis, :, :]
                for i in range(len(normalized)):
                    normalized[i] -= torch.min(normalized[i])
                    normalized[i] *= 255 / torch.max(normalized[i])

                # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
                frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(n_samples**0.5), c=1)
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
    return x


# Network utils
def default_transform_ops():
    return T.Compose([
        # Note: [-1, 1] as DDPM generates a normally distributed data (assumes data in [0, 1])
        T.Lambda(lambda t: (t * 2) - 1)],
    )

def inverse_default_transform_ops():
    return T.Compose([
        T.Lambda(lambda t: (t + 1) / 2)],
    )

def generate_dataset(datapath, grid_h=GRID_SIZE, grid_w=GRID_SIZE, start=1000, end=2000):
    return MantaFlow2DSimTupleDataset(data_path=datapath, start_itr=start, end_itr=end, grid_height=grid_h, grid_width=grid_w, transform_ops=default_transform_ops())

def generate_dataloader(datapath, eval=False):
    if eval:
        return DataLoader(generate_dataset(datapath), batch_size=1, shuffle=False, pin_memory=True)
    return DataLoader(generate_dataset(datapath), batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
