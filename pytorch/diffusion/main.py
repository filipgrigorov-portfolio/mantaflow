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
STORE_PATH_WEIGHTS = f"weights/ddpm_model_smoke.pt"


def train(display=True, continue_from_checkpoint=False, show_forward_process=False, show_backward_process=False):
    from model import DDPM, UNet, DEVICE
    EPOCHS = 300
    LR = 1e-3

    # Originally used by the authors
    diffusion_steps = 400
    min_beta = 1e-4
    max_beta = 2e-2
    ddpm = DDPM(UNet(diffusion_steps), diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

    sum([p.numel() for p in ddpm.parameters()])

    loader = generate_dataloader(datapath="data/")

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

        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{EPOCHS}", colour="#005500")):
            # Loading data
            d0 = batch[0].to(DEVICE)
            v0 = batch[1][:, :2, ...].to(DEVICE)
            # Note: x0 (original image) is the vx and vy and y is the d0 or any other ICs or BCs (grid-like inputs)
            x0 = torch.concat([v0, d0], dim=1)
            n = len(x0)

            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            eta = torch.randn_like(x0).to(DEVICE)                           # ~N(0, 1)
            t = torch.randint(0, diffusion_steps, (n,)).to(DEVICE)          # t ~N(0, 1) -> random steps during the noisifying forward process from the batch

            # Computing the noisy image based on x0 and the time-step (FORWARD)
            noisy_imgs = ddpm(x0, t, eta)

            # Getting model estimation of noise based on the images and the time-step (BACKWARD)
            eta_pred = ddpm.backward(noisy_imgs, t.reshape(n, -1))

            # Optimizing the MSE between the noise plugged and the predicted noise
            #v0_pred = get preidction here
            #d0_pred = get prediction here
            loss = mse(eta_pred, eta)# + mse(v0, )

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
        generated = generate_new_images(ddpm, gif_name="before_training.gif")
        show_images(generated, "Images generated before training")

    print('End of training')

def evaluate():
    from model import DDPM, UNet, DEVICE

    # Loading the trained model
    diffusion_steps = 1000
    best_model = DDPM(UNet(), diffusion_steps=diffusion_steps, device=DEVICE)
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

    args = parser.parse_args()

    if args.show_data:
        print('Showing first batch of training data')
        show_data()
    elif args.train:
        """Learn the distribution"""
        print('Training mode')
        train(continue_from_checkpoint=args.from_checkpoint)
    elif args.eval:
        """Generate block images from distribution"""
        print('Evaluation mode')
        evaluate()
    elif args.infer:
        """Infer from input setup (ICs, BCs etc.)"""
        pass
    else:
        print('No action has been selected!')