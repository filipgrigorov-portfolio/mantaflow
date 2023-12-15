# Python
import random
import imageio as io
import numpy as np
np.set_printoptions(suppress=True)

from argparse import ArgumentParser
from tqdm.auto import tqdm

# Torch
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from model import DDPM, UNet, DEVICE, SelfAttentionUNet

# Setting reproducibility
SEED = 0
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

OUTPUT_CHANNELS = 3
AT_EVERY = 10

# Definitions
INPUT_DATA_PATH = "data_20s/"
STORE_PATH_WEIGHTS = f"weights/ddpm_model_smoke.pt"

from utils import TOTAL_SIMULATION_TIME, GRID_SIZE, OUTPUT_DATA_PATH
from utils import default_transform_ops, sample, inverse_default_transform_ops, show_forward, show_compound_images, show_first_batch
from dataset import MantaFlow2DSimStatesDataset


#TODO: Explore further
@torch.no_grad()
def sample_sequence(with_attention=False):
    # Originally used by the authors
    diffusion_steps = 400
    min_beta = 1e-4
    max_beta = 2e-2
    ddpm = DDPM(SelfAttentionUNet(output_channels=OUTPUT_CHANNELS, diffusion_steps=diffusion_steps, time_emb_dim=TOTAL_SIMULATION_TIME), 
                diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE) if with_attention \
        else DDPM(UNet(output_channels=OUTPUT_CHANNELS, diffusion_steps=diffusion_steps, time_emb_dim=TOTAL_SIMULATION_TIME), 
                diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

    print("Parameters: ", sum([p.numel() for p in ddpm.parameters()]))

    print('Continuing from checkpoint')
    ddpm.load_state_dict(torch.load(STORE_PATH_WEIGHTS, map_location=DEVICE))


    dataset = MantaFlow2DSimStatesDataset(data_path=INPUT_DATA_PATH, start_itr=1003, end_itr=1004, transform_ops=default_transform_ops())
    loader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=True)

    dt_prev, vt_prev, d0, v0, bc0, dt_next, vt_next = next(iter(loader))
    d0 = d0.to(DEVICE)
    v0 = v0.to(DEVICE)
    bc0 = bc0.to(DEVICE)

    for time_step in range(1, TOTAL_SIMULATION_TIME):
        # Starting from random noise
        time_idx = time_step + 1
        print(f'Processing frame {time_idx}')
        
        x = sample(ddpm, dt_prev=d0, vt_prev=v0, bc_init=bc0, output_channels=OUTPUT_CHANNELS, sim_time=1)
        d0 = x[:, :1, ...].clone()
        v0 = x[:, 1:, ...].clone()

        x_display = torch.clamp(x, -1, 1)
        x_display = inverse_default_transform_ops()(x)

        sampled_rho = x_display[0, 0, :, :].cpu().numpy()
        io.imwrite(f"SEQ_TEST/density_{time_idx}.png", sampled_rho)

    print('End')



def train(display=True, continue_from_checkpoint=False, show_forward_process=False, with_attention=False):
    EPOCHS = 1000
    LR = 1e-4
    BATCH_SIZE = 8

    # Originally used by the authors
    diffusion_steps = 400
    min_beta = 1e-4
    max_beta = 2e-2
    ddpm = DDPM(SelfAttentionUNet(output_channels=OUTPUT_CHANNELS, diffusion_steps=diffusion_steps, time_emb_dim=TOTAL_SIMULATION_TIME), 
                diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE) if with_attention else \
        DDPM(UNet(output_channels=OUTPUT_CHANNELS, diffusion_steps=diffusion_steps, time_emb_dim=TOTAL_SIMULATION_TIME), 
                diffusion_steps=diffusion_steps, min_beta=min_beta, max_beta=max_beta, device=DEVICE)

    sum([p.numel() for p in ddpm.parameters()])

    # Note: data loading
    dataset = MantaFlow2DSimStatesDataset(data_path=INPUT_DATA_PATH, start_itr=1000, end_itr=2100, transform_ops=default_transform_ops())
    loader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    # Display at start of training (Optional)
    if continue_from_checkpoint:
        print('Continuing from checkpoint')
        ddpm.load_state_dict(torch.load(STORE_PATH_WEIGHTS, map_location=DEVICE))

    if show_forward_process:
        show_forward(ddpm, loader, DEVICE)
        raise('show_forward')

    mse = nn.MSELoss()
    optimizer = optim.Adam(ddpm.parameters(), LR)

    best_loss = float("inf")
    diffusion_steps = ddpm.diffusion_steps
    for epoch in tqdm(range(EPOCHS), desc=f"Training progress", colour="#00ff00"):

        epoch_loss = 0.0
        for _, batch in enumerate(tqdm(loader, leave=False, desc=f"Epoch {epoch + 1}/{EPOCHS}", colour="#005500")):
            # Input data
            dt_prev = batch[0].to(DEVICE)
            vt_prev = batch[1].to(DEVICE)
            dt = batch[2].to(DEVICE)
            vt = batch[3].to(DEVICE)
            dt_next = batch[5].to(DEVICE)
            vt_next = batch[6].to(DEVICE)
            
            # ICs/BCs conditioning
            bc0 = batch[4].to(DEVICE)


            # Picking some noise for each of the images in the batch, a timestep and the respective alpha_bars
            # Assuming time step (h) is 1
            density_flow = (dt_next - dt_prev) / 2 # <=> (dt - dt_prev)
            vel_flow = (vt_next - vt_prev) / 2 # <=> (vt - vt_prev)
            flow_input_data = torch.concat([density_flow, vel_flow], dim=1)
            n = len(flow_input_data)

            eta = torch.randn_like(flow_input_data).to(DEVICE) # eta ~N(0, 1)
            t = torch.randint(0, diffusion_steps, (n,)).to(DEVICE) # t ~N(0, T)



            # (FORWARD)
            noisy_imgs = ddpm(flow_input_data, t, eta)



            # (BACKWARD)
            # Note: [batch, OUTPUT_CHANNELS, 64, 64]
            # print(noisy_imgs.size())
            # print(dt_prev.size())
            # print(vt_prev.size())
            # print(bc0.size())
            noisy_x0 = torch.concat([noisy_imgs, dt_prev, vt_prev, bc0], dim=1)
            eta_pred = ddpm.backward(noisy_x0, t.reshape(n, -1))



            # Optimizing the MSE between the noise plugged and the predicted noise
            loss = mse(eta_pred, eta) #+ KL(P||Q) -> nn.KLDivLoss(pred, gt)


            optimizer.zero_grad()
            loss.backward()            
            optimizer.step()

            epoch_loss += loss.item() * len(noisy_x0) / len(loader.dataset)




        # Display images generated at this epoch
        if display and epoch % AT_EVERY == AT_EVERY -1 :
            print(f"\nSaving results at {OUTPUT_DATA_PATH}")
            show_compound_images(sample(ddpm, dt_prev, vt_prev, bc0, 
                output_channels=OUTPUT_CHANNELS, sim_time=BATCH_SIZE, device=DEVICE), f"Images generated at epoch {epoch + 1}")

        log_string = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.3f}"


        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(ddpm.state_dict(), STORE_PATH_WEIGHTS)
            log_string += " --> Best model ever (stored)"
        print(log_string)

    print('End of training')


def show_data():
    #show_first_batch(generate_dataloader(data_path))
    dataset = MantaFlow2DSimStatesDataset(
        data_path=INPUT_DATA_PATH, 
        grid_height=GRID_SIZE, grid_width=GRID_SIZE, 
        start_itr=1000, end_itr=2100, transform_ops=default_transform_ops())
    loader = DataLoader(dataset=dataset, batch_size=8, shuffle=False, pin_memory=True)
    show_first_batch(loader=loader)



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
        train(continue_from_checkpoint=args.from_checkpoint, with_attention=args.with_attention, show_forward_process=False)
    elif args.infer:
        """Infer from input setup (ICs, BCs etc.)"""
        raise NotImplementedError
    elif args.sample_sequence:
        """Samples a sequence from ICs and BCs"""
        sample_sequence(with_attention=args.with_attention)
    else:
        print('No action has been selected!')