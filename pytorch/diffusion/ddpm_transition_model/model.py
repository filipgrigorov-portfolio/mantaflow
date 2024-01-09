import torch
import torch.nn as nn

from networks import SelfAttentionUNet
from utils import GRID_SIZE, INPUT_CHANNELS # rho, vx and vy

# Definitions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {DEVICE}\t" + (f"{torch.cuda.get_device_name(0)}" if torch.cuda.is_available() else "CPU"))

DIFFUSION_STEPS = 400
MIN_BETA = 1e-4
MAX_BETA = 2e-2

INPUT_SHAPE = (INPUT_CHANNELS, GRID_SIZE, GRID_SIZE)

# Summary:
# Note: forward process adds noise relative to x0 (original image) instead of adding noise gradually
# Note: backward process of denoising is taken care of by the network

class DDPM(nn.Module):
    def __init__(self, network, diffusion_steps=DIFFUSION_STEPS, min_beta=MIN_BETA, max_beta=MAX_BETA, device=DEVICE, image_shape=INPUT_SHAPE):
        """General framework for forward and backward processes (network could be anything)"""
        super(DDPM, self).__init__()

        self.diffusion_steps = diffusion_steps
        self.device = device
        self.image_shape = image_shape
        self.network = network.to(device)

        self.betas = torch.linspace(min_beta, max_beta, diffusion_steps).to(device)

        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0).to(device)

    def forward(self, x0, t, eta=None):
        """Forward process to introduce noise at different t out of T (diffusion steps)"""
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]
        
        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy_x = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy_x
    
    def backward(self, x, t, tau):
        """Run each image through the network for each timestep t in the vector t. The network returns its estimation of the noise that was added."""
        # p(xt-1|xt) ~ N(xt-1; mu_theta(xt, t), sigma_t**2.I)
        return self.network(x, t, tau)
    

