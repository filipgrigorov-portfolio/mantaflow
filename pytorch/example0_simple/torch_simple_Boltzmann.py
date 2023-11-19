import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T # from torchvision.transforms import v2 (in newer versions)

import imageio as io
import os
import matplotlib.pyplot as plt
import numpy as np
import random
import sys

random.seed(9)
np.random.seed(9)
torch.manual_seed(9)

sys.path.append("../../tensorflow/tools")
import uniio

from torch.utils.data import Dataset, DataLoader

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
        d1 = self.densities[idx]
        v1 = self.velocities[idx]

        d0, v0 = d1.copy(), v1.copy()
        if idx - 1 >= 0:
            d0 = self.densities[idx - 1]
            v0 = self.velocities[idx - 1]

        d0_t = torch.from_numpy(d0).float()
        v0_t = torch.from_numpy(v0).float()
        d1_t = torch.from_numpy(d1).float()
        v1_t = torch.from_numpy(v1).float()

        if self.transform_ops is not None:
            d0_t = self.transform_ops(d0)
            v0_t = self.transform_ops(v0)
            d1_t = self.transform_ops(d1)
            v1_t = self.transform_ops(v1)

        return d0_t, v0_t, (idx - 1 if idx - 1 >= 0 else idx), d1_t, v1_t, idx

    def __len__(self):
        assert self.densities.shape[0] == self.velocities.shape[0]
        return self.densities.shape[0]
    

def boltzmann_distribution(velocity, temperature): #f(x, v, t)
    # Boltzmann constant
    k = 1.380649e-23  # J/K
    
    # Calculate the Boltzmann distribution
    exponent = -0.5 * velocity**2 / (k * temperature)
    prob_density = torch.exp(exponent)
    
    return prob_density

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = np.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # TODO: Double check the ordering here
        return embeddings

class SimpleBoltzmann(nn.Module):
    def __init__(self, grid_h, grid_w, chs):
        super(SimpleBoltzmann, self).__init__()

        time_emb_dim = 32

        self.boltzmann_const = 1.380649e-23  # J/K
        #self.temperature = torch.FloatTensor(1)
        #self.temperature[0] = 300 # K (trainable)
        self.T = 300

        self.w = grid_w
        self.h = grid_h
        self.k = chs
        self.input_size = grid_h * grid_w * chs

        self.emb = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        self.model = nn.Sequential(
            nn.Linear(in_features=self.input_size + time_emb_dim, out_features=32),
            nn.Tanh(),
            nn.BatchNorm1d(num_features=32),
        )

        self.df_head = nn.Sequential(
            nn.Linear(in_features=32, out_features=self.input_size, bias=True)
        )
        torch.nn.init.xavier_uniform_(self.df_head[0].weight)

        #self.temp_param = nn.Parameter(data=self.temperature, requires_grad=True)
        #torch.nn.init.constant(self.temp_param, 300)

    def forward(self, d0, v0, dt):
        emb_out = self.emb(dt)
        in_for_emb = torch.concat([v0.view(-1, self.input_size), emb_out], dim=1)
        out = self.model(in_for_emb)
        df = self.df_head(out)
        df = df.view(-1, self.k, self.h, self.w) # [batch x h x w x 3]
        exponent = (-0.5 * v0**2 / (self.boltzmann_const * self.T))
        f = torch.exp(exponent)
        # Note: densities are weighted according to their probabilities -> rho * P(rho| r, v, t)
        # Here, there is no advection/streaming per se, as the model is ran on the whole fluid stream
        densities_next = d0 * (f + df)
        d1_pred = torch.zeros((f.size(0), 1, f.size(2), f.size(3)))
        d1_pred[:, 0, ...] = densities_next[:, 0, ...] + densities_next[:, 1, ...]
        return d1_pred


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0.0

    for batch_idx, (d0, v0, dt0, d1, v1, dt1) in enumerate(dataloader):
        d0, v0 = d0.to(device), v0.to(device)
        d1, v1 = d1.to(device), v1.to(device)

        d1_pred = model(d0, v0, dt0)

        loss = loss_fn(d1_pred, d1)
        train_loss += loss.item()

        if batch_idx % 200 == 199:
            print(f'\t{batch_idx + 1}/{len(dataloader)}: {train_loss / (batch_idx + 1)}')
    
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return train_loss

def validation_step(model, dataloader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        for batch_idx, (d0, v0, dt0, d1, v1, dt1) in enumerate(dataloader):
            d0, v0 = d0.to(device), v0.to(device)
            d1, v1 = d1.to(device), v1.to(device)

            d1_pred = model(d0, v0, dt0)

            loss = loss_fn(d1_pred, d1)
            valid_loss += loss.item()

            if batch_idx % 10 == 9:
                print(f'\t{batch_idx + 1}/{len(dataloader)}: {valid_loss / (batch_idx + 1)}')

    return valid_loss

def images_step(data_path, model, dataloader, device):
    out_dir = "%s/test_simple_pytorch" % data_path
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    print('start writing')
    model.eval()
    with torch.no_grad():        
        for idx, (d0, v0, dt0, d1, v1, dt1) in enumerate(dataloader):
            d0, v0 = d0.to(device), v0.to(device)
            d1, v1 = d1.to(device), v1.to(device)

            assert d1.size(0) == 1

            d1_pred = model(d0, v0, dt0)

            io.imwrite("%s/in_%d.png" % (out_dir, dt1), d1.cpu().numpy().squeeze(0).squeeze(0))
            io.imwrite("%s/out_%d.png" % (out_dir, dt1), d1_pred.cpu().squeeze(0).squeeze(0))

    print('end')


def main(data_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 500
    BATCH_SIZE = 10

    transforms = T.Compose([
        T.ToTensor()
    ])

    dataset = MantaFlow2DDataset(data_path=data_path, transform_ops=transforms)

    train_len = len(dataset)
    print(f'Length of dataset: {train_len}')
    validation_len = max(100, int(train_len * 0.1))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_len - validation_len, validation_len])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = SimpleBoltzmann(grid_h=64, grid_w=64, chs=3).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_step(model, train_dataloader, criterion, optimizer, DEVICE)
        num_batches = len(train_dataloader)
        print(f'{epoch + 1}/{EPOCHS}: {train_loss / num_batches}')

        if epoch == EPOCHS - 1:
            valid_loss = validation_step(model, validation_dataloader, criterion, DEVICE)
            num_batches = len(validation_dataloader)
            print(f'Validation -> {epoch + 1}/{EPOCHS}: {valid_loss / num_batches}')

            # Write out the produced images
            images_step(DATA_PATH, model, validation_dataloader, DEVICE)

def test_dataset(data_path):
    dataset = MantaFlow2DDataset(data_path)
    size = len(dataset)

    dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, pin_memory=True)
    
    '''
    print(f'Size of loaded data: {size}')
    densities, velocities, _ = dataset[0]

    print(f'Size of density and velocity data: {densities.shape}, {velocities.shape}')
    print(f'Max value of densities: {densities.max()}')
    print(f'Max value of velocities: {velocities.max()}')

    assert size > 0, "data should be loaded"
    assert densities.shape[0] == 64, "grid widht should be 64"
    assert densities.max() > 0, "max density value > 0"

    loaded_densities, loaded_velocities, _ = zip(next(iter(dataloader)))

    print(f'Loaded densities size: {loaded_densities[0].size()}')
    print(f'Loaded densities np shape: {loaded_densities[0].numpy().shape}')
    print(f'Max value of loaded densities: {loaded_densities[0].numpy().max()}')

    print(f'Loaded velocities size: {loaded_velocities[0].size()}')
    print(f'Loaded velocities np shape: {loaded_velocities[0].numpy().shape}')
    print(f'Max value of loaded velocities: {loaded_velocities[0].numpy().max()}')

    assert (loaded_velocities[0].numpy() == velocities.numpy()).all(), "arrays should be equal"
    '''

    count = 0
    for idx, (d0, v0, dt0, d1, v1, dt1) in enumerate(dataloader, start=1):   
        io.imwrite(f'img_densities_{dt1}.png', d1.squeeze(0))
        plt.quiver(v1[..., 0].squeeze(0), v1[..., 1].squeeze(0))
        plt.savefig(f'img_velocities_{dt1}.png')

        if count == 20:
            raise('stop')
        count += 1

if __name__ == '__main__':
    DATA_PATH = '../../tensorflow/data/'
    main(DATA_PATH)
    #test_dataset(DATA_PATH)
