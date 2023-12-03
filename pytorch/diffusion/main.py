# Python
import random
import imageio
import numpy as np
np.set_printoptions(suppress=True)
import math

# From
from utils import *

from argparse import ArgumentParser
from tqdm.auto import tqdm

# Torch
import einops
import torch
import torch.nn as nn

import torch.optim as optim

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Definitions
INPUT_DATA_PATH = "data/"
STORE_PATH_WEIGHTS = f"weights/ddpm_model_smoke.pt"
TOTAL_SIMULATION_TIME = 100 # length of sequences

@torch.no_grad()
def sample(ddpm, n_sample=1, channels=CHANNELS, height=GRID_SIZE, width=GRID_SIZE, device=None):
    """Sample next image from noise"""
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        # Starting from random noise
        x = torch.randn(n_sample, channels, height, width).to(device)

        # From T to 0, denoise
        for idx, t in enumerate(list(range(ddpm.diffusion_steps))[::-1]):
            # Estimating noise to be removed
            time_tensor = (torch.ones(n_sample, 1) * t).to(device).long()
            eta_theta = ddpm.backward(x, time_tensor)

            alpha_t = ddpm.alphas[t]
            alpha_t_bar = ddpm.alpha_bars[t]

            # Partially denoising the image
            x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

            if t > 0:
                # Step 3 in https://arxiv.org/pdf/2006.11239.pdf (Algorithm 2)
                z = torch.randn(n_sample, channels, height, width).to(device)

                # Note: sigma_t squared = beta_t
                beta_t = ddpm.betas[t]
                sigma_t = beta_t.sqrt()

                # Adding some more noise like in Langevin Dynamics fashion
                x = x + sigma_t * z

    return x[:, :1, :, :], x[:, 1:3, :, :] # d, v

#TODO: Validate if this works (WIP)
@torch.no_grad()
def sample_sequence(ddpm, d0, v0, seq_len=9, channels=CHANNELS, height=GRID_SIZE, width=GRID_SIZE, device=None):
    """Sample next image from noise"""
    BATCH_SIZE = 1
    with torch.no_grad():
        if device is None:
            device = ddpm.device

        d_prev = d0.to(device)
        v_prev = v0.to(device)

        frames = []
        for i in range(seq_len):
            # Starting from random noise
            print(f'Processing frame {i}')
            x = torch.randn(BATCH_SIZE, channels, height, width).to(device)
            x[:, 3:4, :, :] = d_prev
            x[:, 4:, :, :] = v_prev

            # From T to 0, denoise
            for idx, t in enumerate(list(range(ddpm.diffusion_steps))[::-1]):
                # Estimating noise to be removed
                time_tensor = (torch.ones(BATCH_SIZE, 1) * t).to(device).long()
                eta_theta = ddpm.backward(x, time_tensor)

                alpha_t = ddpm.alphas[t]
                alpha_t_bar = ddpm.alpha_bars[t]

                # Partially denoising the image
                x = (1 / alpha_t.sqrt()) * (x - (1 - alpha_t) / (1 - alpha_t_bar).sqrt() * eta_theta)

                if t > 0:
                    # Step 3 in https://arxiv.org/pdf/2006.11239.pdf (Algorithm 2)
                    z = torch.randn(BATCH_SIZE, channels, height, width).to(device)

                    # Note: sigma_t squared = beta_t
                    beta_t = ddpm.betas[t]
                    sigma_t = beta_t.sqrt()

                    # Adding some more noise like in Langevin Dynamics fashion
                    x = x + sigma_t * z

            d_prev = x[:, 3:4, :, :].clone()
            v_prev = x[:, 4:, :, :].clone()

            plt.imshow(x[0, 0, :, :].cpu().numpy())
            plt.savefig(f"SEQ_TEST/test_seq_{i}.png")
            frames.append(x)
    raise('debug')
    return frames

@torch.no_grad()
def run_sequence_sampling():
    from model import DDPM, UNet, DEVICE, AttentionUNet

    # Originally used by the authors
    diffusion_steps = 400
    min_beta = 1e-4
    max_beta = 2e-2
    ddpm = DDPM(AttentionUNet(output_channels=CHANNELS, diffusion_steps=diffusion_steps), diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

    sum([p.numel() for p in ddpm.parameters()])

    loader = generate_dataloader(datapath="data/", eval=True)

    print('Continuing from checkpoint')
    ddpm.load_state_dict(torch.load(STORE_PATH_WEIGHTS, map_location=DEVICE))

    d0, v0, _ = next(iter(loader))
    d0 = d0.to(DEVICE)
    v0 = v0[:, :2, ...].to(DEVICE)
    frames = sample_sequence(ddpm, d0, v0, seq_len=20)

    # Adding frames to the GIF
    for idx in range(len(frames)):
        d = frames[idx]
        # Putting digits in range [0, 255]
        normalized = d.clone()[:, 1, :, :][:, np.newaxis, :, :]
        # 1, 1, 64, 64
        normalized -= torch.min(normalized)
        normalized *= 255 / torch.max(normalized)

        # Reshaping batch (n, c, h, w) to be a (as much as it gets) square frame
        frame = einops.rearrange(normalized, "(b1 b2) c h w -> (b1 h) (b2 w) c", b1=int(1**0.5), c=1)
        # 64, 64, 1
        frame = frame.cpu().numpy().astype(np.uint8)

        # Rendering frame
        frames[idx] = frame

    # gif generation
    # Storing the gif
    with io.get_writer("sequence_test.gif", mode="I") as writer:
        for idx, frame in enumerate(frames):
            rgb_frame = np.repeat(frame, 3, axis=2)
            writer.append_data(rgb_frame)
    
    print('End')



def train(display=True, continue_from_checkpoint=False, show_forward_process=False, show_backward_process=False, with_attention=False):
    from model import DDPM, UNet, DEVICE, AttentionUNet
    EPOCHS = 500
    LR = 1e-4

    # Originally used by the authors
    diffusion_steps = 400
    min_beta = 1e-4
    max_beta = 2e-2
    ddpm = DDPM(AttentionUNet(output_channels=CHANNELS, diffusion_steps=diffusion_steps), diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE) if with_attention else \
        DDPM(UNet(diffusion_steps=diffusion_steps), diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

    sum([p.numel() for p in ddpm.parameters()])

    # Note: data loading
    loader = DataLoader(generate_dataset(INPUT_DATA_PATH), batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

    # Display at start of training (Optional)
    if continue_from_checkpoint:
        print('Continuing from checkpoint')
        ddpm.load_state_dict(torch.load(STORE_PATH_WEIGHTS, map_location=DEVICE))

    if show_forward_process:
        show_forward(ddpm, loader, DEVICE)

    if show_backward_process:
        generated = generate_new_images(ddpm, gif_name="before_training.gif")
        show_images(generated, "Images generated before training")

    mse = nn.MSELoss()
    optimizer = optim.Adam(ddpm.parameters(), LR)

    best_loss = float("inf")
    diffusion_steps = ddpm.diffusion_steps
    for epoch in tqdm(range(EPOCHS), desc=f"Training progress", colour="#00ff00"):
        
        
        # Note: INPUT DATA
        # Note: Initial d1, v1 are d0 and v0 (conditioning on previous state)
        d_prev, v_prev, _ = next(iter(loader))
        #d_prev = d_prev.to(DEVICE)
        d_init = d_prev.to(DEVICE)
        v_prev = v_prev[:, :2, ...].to(DEVICE)
        epoch_loss = 0.0
        tau = torch.ones_like(d_init) # temporal information matrix
        tau = # Note: identical entries that have been normalized with the total simulation time


        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{EPOCHS}", colour="#005500")):
            # Loading data
            #d0 = batch[0].to(DEVICE)
            v0 = batch[1][:, :2, ...].to(DEVICE)
            # Note: x0 (original image) is the vx and vy and y is the d0 or any other ICs or BCs (grid-like inputs)
            x0 = v0#torch.concat([d_init, tau, v0], dim=1)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(DEVICE)                           # ~N(0, 1)
            t = torch.randint(0, diffusion_steps, (n,)).to(DEVICE)          # t ~N(0, 1) -> random steps during the noisifying forward process from the batch

            # Computing the noisy image based on x0 and the time-step (FORWARD)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step (BACKWARD)
            x0 = torch.concat([d_init, tau, noisy_imgs], dim=1) # Note: noise is added only on the velocity fields (vx and vy)
            eta_pred = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            #d_prev, v_prev = sample(ddpm, n_sample=BATCH_SIZE, device=DEVICE)
            #d_prev = d0
            #v_prev = v0

            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_pred, eta)

            optimizer.zero_grad()
            
            loss.backward()
            
            optimizer.step()

            epoch_loss += loss.item() * len(x0) / len(loader.dataset)

        # Display images generated at this epoch
        if display and epoch % 10 == 9:
            show_compound_images(generate_new_images(ddpm, device=DEVICE), f"Images generated at epoch {epoch + 1}")

        log_string = f"\tLoss at epoch {epoch + 1}: {epoch_loss:.3f}"

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), STORE_PATH_WEIGHTS)
            log_string += " --> Best model ever (stored)"

        print(log_string)


    # Display at end of training (Optional)
    if show_forward_process:
        show_forward(ddpm, loader, DEVICE)

    if show_backward_process:
        generated = generate_new_images(ddpm, gif_name="after_training.gif")
        show_images(generated, "Images generated after training")

    print('End of training')

def evaluate(with_attention=False):
    from model import DDPM, UNet, DEVICE, AttentionUNet

    # Loading the trained model
    diffusion_steps = 1000
    best_model = DDPM(AttentionUNet(output_channels=CHANNELS, diffusion_steps=diffusion_steps), diffusion_steps=diffusion_steps, device=DEVICE) if with_attention else \
        DDPM(UNet(diffusion_steps=diffusion_steps), diffusion_steps=diffusion_steps, device=DEVICE)
    best_model.load_state_dict(torch.load(STORE_PATH_WEIGHTS, map_location=DEVICE))
    best_model.eval()
    print("Model loaded")

    print("Generating new images")
    generated = generate_new_images(best_model, n_samples=100, device=DEVICE, gif_name="smoke_eval.gif")
    show_images(generated, "Final result")


def show_data():
    data_path = "data/"
    show_first_batch(generate_dataloader(data_path))



if __name__ == "__main__":
    parser  = ArgumentParser()
    parser.add_argument('--show_data', action='store_true')
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--infer', action='store_true')
    parser.add_argument('--from_checkpoint', action='store_true')
    parser.add_argument('--sample_sequence', action='store_true')
    parser.add_argument('--with_attention', action='store_true')

    args = parser.parse_args()

    if args.show_data:
        print('Showing first batch of training data')
        show_data()
    elif args.train:
        """Learn the distribution"""
        print('Training mode')
        train(continue_from_checkpoint=args.from_checkpoint, show_backward_process=True, with_attention=args.with_attention)
    elif args.eval:
        """Generate block images from distribution"""
        print('Evaluation mode')
        evaluate(with_attention=args.with_attention)
    elif args.infer:
        """Infer from input setup (ICs, BCs etc.)"""
        pass
    elif args.sample_sequence:
        """Samples a sequence from ICs and BCs"""
        run_sequence_sampling()
    else:
        print('No action has been selected!')