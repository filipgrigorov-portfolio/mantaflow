import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms as T 
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam

import matplotlib.pyplot as plt
import numpy as np
import imageio as io
import os
import numpy as np
import random
import sys

random.seed(9)
np.random.seed(9)
torch.manual_seed(9)

sys.path.append("../../tensorflow/tools")
import uniio
import math

np.set_printoptions(suppress=True)

DATA_PATH = '../../tensorflow/data/'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
GRID_SIZE = 64
BATCH_SIZE = 50
TIME = 50
# Note: densities have channel of 1
CHANNELS = 1

SAVE_DATA_PATH = 'diffusion_data_comparisons/'
cwd = os.getcwd()
save_path = os.path.join(cwd, SAVE_DATA_PATH)
if not os.path.exists(save_path):
    print(f'Creating {save_path}\n')
    os.mkdir(save_path)


def load_sim_file(data_path, sim, type_name, idx):
    uniPath = "%s/simSimple_%04d/%s_%04d.uni" % (data_path, sim, type_name, idx)  # 100 files per sim
    print(uniPath)
    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
    h = header['dimX']
    w  = header['dimY']
    arr = content[:, ::-1, :, :] # reverse order of Y axis
    arr = np.reshape(arr, [w, h, arr.shape[-1]]) # discard Z
    return arr

class MantaFlow2DDataset(Dataset):
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.densities = []
        self.velocities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                for i in range(0, 100):
                    self.densities.append(load_sim_file(data_path, sim, 'density', i))
                    self.velocities.append(load_sim_file(data_path, sim, 'vel', i))

        num_densities = len(self.densities)
        num_velocities = len(self.velocities)
        if num_densities < 200:
            raise("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times...")

        self.densities = np.reshape( self.densities, (len(self.densities), grid_height, grid_width, 1) )
        print("Read uni files (density), total data " + format(self.densities.shape))

        self.velocities = np.reshape( self.velocities, (len(self.velocities), grid_height, grid_width, 3) )
        print("Read uni files (velocity), total data " + format(self.velocities.shape))

    def __getitem__(self, idx):
        d0 = self.densities[idx]
        v0 = self.velocities[idx]

        d0_t = torch.from_numpy(d0).float()
        v0_t = torch.from_numpy(v0).float()

        if self.transform_ops is not None:
            d0_t = self.transform_ops(d0)
            v0_t = self.transform_ops(v0)

        return d0_t, v0_t, idx

    def __len__(self):
        assert self.densities.shape[0] == self.velocities.shape[0]
        return self.densities.shape[0]



def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """ 
    Returns a specific index t of a passed list of values vals
    while considering the batch dimension.
    """
    batch_size = t.shape[0]
    out = vals.gather(-1, t.cpu())
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    """ 
    Takes an image and a timestep as input and 
    returns the noisy version of it
    """
    noise = torch.randn_like(x_0)
    sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x_0.shape)

    # mean + variance
    return sqrt_alphas_cumprod_t.to(DEVICE) * x_0.to(DEVICE) + sqrt_one_minus_alphas_cumprod_t.to(DEVICE) * noise.to(DEVICE), noise.to(DEVICE)


class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        
        if up:
            self.conv1 = nn.Conv2d(2 * in_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)

        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)

        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t, ):
        # First Conv
        h = self.bnorm1(self.relu(self.conv1(x)))

        # Time embedding
        time_emb = self.relu(self.time_mlp(t))
        
        # Extend last 2 dimensions
        time_emb = time_emb[(...,) + (None, ) * 2]
        
        # Add time channel
        h = h + time_emb
        
        # Second Conv
        h = self.bnorm2(self.relu(self.conv2(h)))
        
        # Down or Upsample
        return self.transform(h)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings


class SimpleUnet(nn.Module):
    """
    A simplified variant of the Unet architecture.
    """
    def __init__(self):
        super().__init__()
        image_channels = CHANNELS
        down_channels = (64, 128, 256, 512, 1024)
        up_channels = (1024, 512, 256, 128, 64)
        out_dim = CHANNELS
        time_emb_dim = 32

        # Time embedding
        self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(time_emb_dim),
                nn.Linear(time_emb_dim, time_emb_dim),
                nn.ReLU()
            )
        
        # Initial projection
        self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

        # Downsample
        self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels) - 1)])

        # Upsample
        self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])
        
        self.output = nn.Conv2d(up_channels[-1], out_dim, 1)

    def forward(self, x, timestep):
        # Embedd time
        t = self.time_mlp(timestep)

        # Initial conv
        x = self.conv0(x)
        
        # Unet
        residual_inputs = []
        
        for down in self.downs:
            x = down(x, t)
            residual_inputs.append(x)
        
        for up in self.ups:
            residual_x = residual_inputs.pop()
            # Add residual x as additional channels
            x = torch.cat((x, residual_x), dim=1)           
            x = up(x, t)
        
        return self.output(x)
    

def get_loss(model, x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod):
    x_noisy, noise = forward_diffusion_sample(x_0, t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
    noise_pred = model(x_noisy, t).to(DEVICE)
    return F.l1_loss(noise, noise_pred)



# data: [50, 1, 64, 64]
def show_tensor_image(image, epoch_idx=None):
    reverse_transforms = T.Compose([
        T.Lambda(lambda t: (t + 1) / 2),
        #T.Lambda(lambda t: t.permute(1, 2, 0)), # CHW to HWC
        #T.Lambda(lambda t: t * 255.),
        #T.Lambda(lambda t: t.numpy().astype(np.uint8)),
        T.ToPILImage(),
    ])

    # Take first image of batch
    if len(image.shape) == 4:
        image = image[0, :, :, :] 

    img_transformed = reverse_transforms(image)
    print(img_transformed.size)
    plt.imshow(img_transformed)
    if epoch_idx is not None:
        plt.savefig(f"saved_data/train_progress_image{epoch_idx}.png")


@torch.no_grad()
def sample_timestep(model, x, t, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    """
    Calls the model to predict the noise in the image and returns 
    the denoised image. 
    Applies noise to this image, if we are not in the last step yet.
    """

    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, t, x.shape)
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)
    
    # Call model (current image - noise prediction)
    model_mean = sqrt_recip_alphas_t * (x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)
    
    if t == 0:
        # As pointed out by Luis Pereira (see YouTube comment)
        # The t's are offset from the t's in the paper
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise


@torch.no_grad()
def sample_plot_image(model, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, epoch_idx):
    # Sample noise
    img = torch.randn((1, CHANNELS, GRID_SIZE, GRID_SIZE), device=DEVICE)
    plt.figure(figsize=(15, 15))
    plt.axis('off')
    num_images = 10
    stepsize = int(TIME / num_images)

    # Note: Denoise backwards
    for i in range(0, TIME)[::-1]:
        t = torch.full((1,), i, device=DEVICE, dtype=torch.long)
        img = sample_timestep(model, img, t, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance)

        # Edit: This is to maintain the natural range of the distribution
        img = torch.clamp(img, -1.0, 1.0)
        if i % stepsize == 0:
            plt.subplot(1, num_images, int(i/stepsize)+1)
            show_tensor_image(img.detach().cpu(), epoch_idx)


# TODO: Sample for the right time
# This is taking in noisy image and time embedding to reproduce an image of the distribution,
# so it has the capacity to do it.
@torch.no_grad()
def sample(model, dataloader, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance):
    # Sample noise
    noisy_img = torch.randn(size=(1, CHANNELS, GRID_SIZE< GRID_SIZE), device=DEVICE)

    # Denoise
    for batch_idx, (d0, v0, time) in enumerate(dataloader):
        d0 = d0.to(DEVICE)
        v0 = v0.to(DEVICE)
        time = torch.full((1,), time, device=DEVICE, dtype=torch.long)
        
        betas_t = get_index_from_list(betas, time, noisy_img.shape)
        sqrt_one_minus_alphas_cumprod_t = get_index_from_list(sqrt_one_minus_alphas_cumprod, time, noisy_img.shape)
        sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, time, noisy_img.shape)
        
        # Call model (current image - noise prediction)
        model_mean = sqrt_recip_alphas_t * (noisy_img - betas_t * model(noisy_img, time) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = get_index_from_list(posterior_variance, time, noisy_img.shape)
        
        # Get the image for t + 1
        d0_pred = model_mean if time == 0 else model_mean + torch.sqrt(posterior_variance_t) * torch.randn_like(noisy_img)

        # Save images (pred, gt)
        io.imwrite("%s/in_%d.png" % (SAVE_DATA_PATH, time), d0.cpu().numpy().squeeze(0).squeeze(0))
        io.imwrite("%s/out_%d.png" % (SAVE_DATA_PATH, time), d0_pred.cpu().squeeze(0).squeeze(0))


def simulate_forward_diffusion(dataloader, sqrt_one_minus_alphas_cumprod, sqrt_alphas_cumprod):
    # d0: [batch, 1, h, w]
    # v0: [batch, 3, h, w]
    d0, v0, dt = next(iter(dataloader))

    plt.figure(figsize=(15,15))
    plt.axis('off')
    num_images = 10
    stepsize = int(TIME / num_images)

    for idx in range(0, TIME, stepsize):
        time = torch.Tensor([idx]).type(torch.int64)
        plt.subplot(1, num_images + 1, int(idx / stepsize) + 1)
        # noise: [50, 1, 64, 64]
        d0_noisy, noise = forward_diffusion_sample(d0, time, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)
        show_tensor_image(d0_noisy)

    plt.show()

def main(data_path, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance):
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])

    model = SimpleUnet()
    model.to(DEVICE)
    optimizer = Adam(model.parameters(), lr=0.001)

    # Data
    densities = MantaFlow2DDataset(data_path=data_path, transform_ops=transforms)

    train_len = len(densities)
    print(f'Length of dataset: {train_len}')
    validation_len = max(100, int(train_len * 0.1))
    train_dataset, validation_dataset = torch.utils.data.random_split(densities, [train_len - validation_len, validation_len])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, pin_memory=True)

    EPOCHS = 300
    for epoch in range(EPOCHS):
        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()

            t = torch.randint(0, TIME, (BATCH_SIZE,), device=DEVICE).long()
            
            loss = get_loss(model, batch[0], t, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod)

            loss.backward()
            
            optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

            if epoch % 15 == 0 and step == 0:
                print('Printing progress')
                sample_plot_image(model, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, epoch)
    
    print('Serialize final evaluation')
    sample(model, validation_dataloader, betas, sqrt_recip_alphas, sqrt_one_minus_alphas_cumprod, posterior_variance, epoch)


def test(data_path, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance):
    transforms = T.Compose([
        T.ToTensor(),
        T.Lambda(lambda t: (t * 2) - 1)
    ])

    dataset = MantaFlow2DDataset(data_path=data_path, transform_ops=transforms)
    data_len = len(dataset)
    print(f'Length of dataset: {data_len}')
    dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    simulate_forward_diffusion(dataloader, sqrt_one_minus_alphas_cumprod, sqrt_alphas_cumprod)


if __name__ == '__main__':
    # Define beta schedule
    betas = linear_beta_schedule(timesteps=TIME)

    # Pre-calculate different terms for closed form
    alphas = 1.0 - betas

    # alpha hats
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
    sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

    main(DATA_PATH, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance)
    #test(DATA_PATH, betas, alphas, alphas_cumprod, alphas_cumprod_prev, sqrt_recip_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod, posterior_variance)