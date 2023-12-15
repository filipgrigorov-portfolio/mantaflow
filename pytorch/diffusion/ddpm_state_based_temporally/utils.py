import einops
import imageio as io
import matplotlib.pyplot as plt
import numpy as np
import os

import torch
from torchvision import transforms as T

#Definitions
BATCH_SIZE = 8
GRID_SIZE = 64
INPUT_CHANNELS = 7
TOTAL_SIMULATION_TIME = 20 # length of sequences

OUTPUT_DATA_PATH = "results/"

# Display utils
def show_images(images, img_idx, title="", is_binary=False):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays
    if type(images) is torch.Tensor:
        images = inverse_default_transform_ops()(images)
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(16, 16))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # Populating figure with sub-plots
    idx = 0
    for _ in range(rows):
        for _ in range(cols):
            _ = fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                img = images[idx][img_idx]
                if not is_binary:
                    plt.imshow(img, vmin=0, vmax=1)
                else:
                    plt.imshow(img, cmap='gray')
                idx += 1

    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + ".jpg"))

def show_compound_images(images, title=""):
    """Shows the provided images as sub-pictures in a square (d0, v0 (x and y))"""

    # Converting images to CPU numpy arrays: images -> (16, 3, 64, 64)
    if type(images) is torch.Tensor:
        # EXP
        images = inverse_default_transform_ops()(images)
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
                plt.title(idx)
                plt.imshow(images[idx][2])
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_vel_y.jpg"))


def show_first_batch(loader):
    """Shows the images in the first batch of a DataLoader object"""
    for idx, batch in enumerate(loader):
        if idx == 0:
            dt = batch[0]
            vt = batch[1]
            d0 = batch[2]
            bc0 = batch[3]
            tau = batch[4]
            print(f"Printing batch {idx}")
            show_images(d0, 0, f"Images in batch {idx} for initial density")
            show_images(bc0, 0, f"Images in batch {idx} for boundary", is_binary=True)
            show_images(dt, 0, f"Images in batch {idx} for density")
            show_images(vt, 0, f"Images in batch {idx} for vel x")
            show_images(vt, 1, f"Images in batch {idx} for vel y")
            show_images(tau, 0, f"Images in batch {idx} for tau")
            break

def show_forward(ddpm, loader, device):
    """Showing the forward process"""
    stop_at = 3
    for batch in loader:
        v0 = batch[1]
        tau = batch[4]

        show_images(v0, 0, f"Original images at {0}")
        show_images(v0, 1, f"Original images at {1}")
        show_images(tau, 0, f"Original images for tau")

        for percent_noise in [0.25, 0.5, 0.75, 1]:
            print(f"Generating noisy images with {percent_noise} % noise")
            v0 = ddpm(v0.to(device), [int(percent_noise * ddpm.diffusion_steps) - 1 for _ in range(len(v0))])
            show_images(v0, 0, f"DDPM Noisy images {int(percent_noise * 100)}% at {0}")
            show_images(v0, 1, f"DDPM Noisy images {int(percent_noise * 100)}% at {1}")

        if stop_at == 3:
            break
        stop_at += 1

def show_compound_images_seq_single_ch(images, title=""):
    """Shows the provided images as sub-pictures in a square"""

    # Converting images to CPU numpy arrays:
    if type(images) is torch.Tensor:
        # EXP
        images = inverse_default_transform_ops()(images)
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
    

def show_compound_images(images, title=""):
    """Shows the provided images as sub-pictures in a square (v0 (x and y))"""

    # Converting images to CPU numpy arrays: images -> (N, OUTPUT_CHANNELS, 64, 64)
    if type(images) is torch.Tensor:
        # Note: convert from [-1, 1] to [0, 1]
        images = inverse_default_transform_ops()(images) # Note: This works on numpy as well (funny observation)
        images = images.detach().cpu().numpy()

    # Defining number of rows and columns
    fig = plt.figure(figsize=(20, 20))
    num_images = len(images)
    rows = int(num_images ** (1 / 2))
    cols = round(num_images / rows)

    # Density
    print(f"Compositing density images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                img = images[idx][0]
                #img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img, cmap='gray')
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_density.jpg"))

    # Velocities
    print(f"Compositing x velocities images")
    idx = 0
    for r in range(rows):
        for c in range(cols):
            fig.add_subplot(rows, cols, idx + 1)

            if idx < len(images):
                img = images[idx][1]
                #img = (img - img.min()) / (img.max() - img.min())
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
                img = images[idx][2]
                #img = (img - img.min()) / (img.max() - img.min())
                plt.imshow(img)
                idx += 1
    fig.suptitle(title, fontsize=30)
    plt.savefig(os.path.join(OUTPUT_DATA_PATH, title + "_vel_y.jpg"))

@torch.no_grad()
def sample(ddpm, dt_prev, vt_prev, bc_init, output_channels, sim_time=TOTAL_SIMULATION_TIME, device=None, grid_size=GRID_SIZE):
    """Given a DDPM model, a number of samples to be generated and a device, returns some newly generated samples"""
    if device is None:
        device = ddpm.device

    # Starting from random noise
    x = torch.randn(sim_time, output_channels, grid_size, grid_size).to(device)

    rho = torch.zeros_like(x[:, :1, :, :])
    vel = torch.zeros_like(x[:, 1:, :, :])

    # From T to 0, denoise
    for _, t in enumerate(list(range(ddpm.diffusion_steps))[::-1]):
        # Estimating noise to be removed
        time_tensor = (torch.ones(sim_time, 1) * t).to(device).long()
        x_stacked = torch.concat([x, dt_prev, vt_prev, bc_init], dim=1)
        eta_theta = ddpm.backward(x_stacked, time_tensor)

        alpha_t = ddpm.alphas[t]
        alpha_t_bar = ddpm.alpha_bars[t]

        # Partially denoising the image
        x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

        if t > 0:
            # Step 3 in https://arxiv.org/pdf/2006.11239.pdf (Algorithm 2)
            z = torch.randn(sim_time, output_channels, grid_size, grid_size).to(device)

            beta_t = ddpm.betas[t]
            sigma_t = beta_t.sqrt()

            # Adding some more noise like in Langevin Dynamics fashion
            x = x + sigma_t * z

    # Note: explicit Euler (forward Euler)
    drho = x[:, :1, :, :]
    dvel = x[:, 1:, :, :]
    rho = dt_prev + drho
    vel = vt_prev + dvel

    return torch.concat([rho, vel], dim=1)

def default_transform_ops():
    return T.Compose([
        # Note: [-1, 1] as DDPM generates a normally distributed data (assumes data in [0, 1])
        T.Lambda(lambda t: (t * 2) - 1)],
    )

def inverse_default_transform_ops():
    return T.Compose([
        T.Lambda(lambda t: (t + 1) / 2)],
    )
