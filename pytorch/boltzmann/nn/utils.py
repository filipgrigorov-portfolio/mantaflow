import numpy as np
import torch
import torch.nn as nn

def boltzmann_distribution(quantity, temperature): #f(x, v, t)
    # Boltzmann constant
    k = 1.380649e-23  # J/K
    
    # Calculate the Boltzmann distribution
    exponent = -0.5 * quantity**2 / (k * temperature)
    prob_density = torch.exp(exponent)
    
    return prob_density

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10_000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings