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

from dataset import MantaFlow2DDataset
from torch.utils.data import Dataset, DataLoader
from utils import SinusoidalPositionEmbeddings, boltzmann_distribution

# Definitions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
OUTPUT_DATA_PATH = "results/"
WEIGHTS_OUTPUT_PATH = "weights/"
DATA_PATH = 'data/'

class SimpleBoltzmann(nn.Module):
    def __init__(self, grid_h, grid_w, chs):
        super(SimpleBoltzmann, self).__init__()
        self.temperature = 300

        time_emb_dim = 32

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
            nn.LeakyReLU(),
            nn.BatchNorm1d(num_features=32),
            nn.Linear(in_features=32, out_features=32),
            nn.LeakyReLU(),
        )

        self.df_head = nn.Sequential(
            nn.Linear(in_features=32, out_features=self.input_size, bias=True)
        )
        torch.nn.init.xavier_uniform_(self.df_head[0].weight)

    def forward(self, d0, v0, dt):
        emb_out = self.emb(dt)
        in_for_emb = torch.concat([d0.view(-1, self.input_size), emb_out], dim=1)
        out = self.model(in_for_emb)
        
        df = self.df_head(out)
        # Note: [batch x 3 x h x w]
        df = df.view(-1, self.k, self.h, self.w)
        
        f = boltzmann_distribution(v0, self.temperature)
        d = f + df
        v1_pred = v0 * d

        # Note: [10, 1, 64, 64]
        d1_pred = torch.zeros((f.size(0), 1, f.size(2), f.size(3))).to(DEVICE).float()
        d1_pred[:, 0, ...] = (d[:, 0, ...] + d[:, 1, ...]) / 2

        return d1_pred, v1_pred


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0.0

    for batch_idx, (d0, v0, dt0, d1, v1, dt1) in enumerate(dataloader):
        d0, v0 = d0.to(device), v0.to(device)
        d1, v1 = d1.to(device), v1.to(device)
        dt0 = dt0.to(device)
        dt1 = dt1.to(device)

        d1_pred = model(d0, v0, dt0)

        loss = loss_fn(d1_pred, d1)
        train_loss += loss.item()

        if batch_idx % 200 == 199:
            print(f'\t{batch_idx + 1}/{len(dataloader)}: {train_loss / (batch_idx + 1)}')
    
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

    return train_loss

# Note: Uses predictions instead of gt data, after the process has started
def train_stepV2(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0.0

    #d0, v0, dt0, d1, v1, dt1 = next(iter(dataloader))
    for batch_idx, (d0, v0, dt0, d1, v1, dt1) in enumerate(dataloader):
        d0, v0 = d0.to(device), v0.to(device)
        d1, v1 = d1.to(device), v1.to(device)
        dt0 = dt0.to(device)
        dt1 = dt1.to(device)

        d1_pred, v1_pred = model(d0, v0, dt0)
        if torch.any(torch.isnan(d1_pred)):
            raise('d1_pred has NaN values')
        
        if torch.any(torch.isnan(v1_pred)):
            raise('v1_pred has NaN values')

        loss = 0.5 * loss_fn(d1_pred, d1) + 0.5 * loss_fn(v1_pred, v1)
        train_loss += loss.item()

        if batch_idx % 200 == 199:
            print(f'\t{batch_idx + 1}/{len(dataloader)}: {train_loss / (batch_idx + 1)}')
    
        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        # dt+1 = d`t+1
        d0 = d1_pred.detach().clone()
        v0 = v1_pred.detach().clone()

    return train_loss

@torch.no_grad()
def validation_step(model, dataloader, loss_fn, device):
    model.eval()

    with torch.no_grad():
        valid_loss = 0.0
        d0, v0, dt0, d1, v1, dt1 = next(iter(dataloader))
        for batch_idx, (_, _, dt0, d1, v1, dt1) in enumerate(dataloader):
            d0, v0 = d0.to(device), v0.to(device)
            d1, v1 = d1.to(device), v1.to(device)
            dt0 = dt0.to(device)
            dt1 = dt1.to(device)

            d1_pred, v1_pred = model(d0, v0, dt0)

            # Note: Updated with velocities update
            loss = loss_fn(d1_pred, d1) + loss_fn(v1_pred, v1)
            valid_loss += loss.item()

            if batch_idx % 10 == 9:
                print(f'\t{batch_idx + 1}/{len(dataloader)}: {valid_loss / (batch_idx + 1)}')

            # dt+1 = d`t+1
            d0 = d1_pred.detach().clone()
            v0 = v1_pred.detach().clone()

    return valid_loss

@torch.no_grad()
def images_step(data_path, model, dataloader, device, out_subdir=''):
    print('start writing')
    model.eval()

    d0, v0, dt0, _, _, _ = next(iter(dataloader))
    for _, (_, _, dt0, d1, v1, dt1) in enumerate(dataloader):
        d0, v0 = d0.to(device), v0.to(device)
        d1, v1 = d1.to(device), v1.to(device)
        dt0 = dt0.to(device)
        dt1 = dt1.to(device)

        assert d1.size(0) == 1

        # Note: Updated with velocities update
        d1_pred, v1_pred = model(d0, v0, dt0)

        io.imwrite("%s/in_%d.png" % (OUTPUT_DATA_PATH, dt1), d1.cpu().numpy().squeeze(0).squeeze(0))
        io.imwrite("%s/out_%d.png" % (OUTPUT_DATA_PATH, dt1), d1_pred.cpu().squeeze(0).squeeze(0))

        # dt+1 = d`t+1
        d0 = d1_pred.detach().clone()
        v0 = v1_pred.detach().clone()

    print('end')


def main(data_path):
    EPOCHS = 100
    BATCH_SIZE = 30

    transforms = T.Compose([
        T.ToTensor(),
    ])

    dataset = MantaFlow2DDataset(data_path=data_path, transform_ops=transforms)

    train_len = len(dataset)
    print(f'Length of dataset: {train_len}')
    validation_len = max(100, int(train_len * 0.1))
    train_dataset, validation_dataset = torch.utils.data.random_split(dataset, [train_len - validation_len, validation_len])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = SimpleBoltzmann(grid_h=64, grid_w=64, chs=1).to(DEVICE)

    criterion = nn.MSELoss().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-2)

    print("Starting training...")
    for epoch in range(EPOCHS):
        train_loss = train_stepV2(model, train_dataloader, criterion, optimizer, DEVICE)
        num_batches = len(train_dataloader)
        print(f'{epoch + 1}/{EPOCHS}: {train_loss / num_batches}')

        if epoch == 0 or epoch == EPOCHS - 1:
            abs_inter_data_path = os.path.join(os.getcwd(), f'{OUTPUT_DATA_PATH}/')
            images_step(abs_inter_data_path, model, validation_dataloader, DEVICE, f"epoch{epoch}")

        if epoch == EPOCHS - 1:
            valid_loss = validation_step(model, validation_dataloader, criterion, DEVICE)
            num_batches = len(validation_dataloader)
            print(f'Validation -> {epoch + 1}/{EPOCHS}: {valid_loss / num_batches}')

            # Write out the produced images
            #images_step(DATA_PATH, model, validation_dataloader, DEVICE)

    # Save the model
    torch.save(model.state_dict(), f"{WEIGHTS_OUTPUT_PATH}/checkpoint.pt")

    # Run iteratively on produced images
    model.load_state_dict(torch.load(f"{WEIGHTS_OUTPUT_PATH}/checkpoint.pt"))
    model.eval()


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
    main(DATA_PATH)
    #test_dataset(DATA_PATH)
