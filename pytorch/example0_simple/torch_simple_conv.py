import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T # from torchvision.transforms import v2 (in newer versions)

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

from torch.utils.data import Dataset, DataLoader

class MantaFlow2DDataset(Dataset):
    def __init__(self, data_path, start_itr=1000, end_itr=2000, grid_width=64, grid_height=64, transform_ops=None):
        self.transform_ops = transform_ops

        self.densities = []

        for sim in range(start_itr, end_itr): 
            if os.path.exists( "%s/simSimple_%04d" % (data_path, sim) ):
                for i in range(0,100): 
                    filename = "%s/simSimple_%04d/density_%04d.uni" 
                    uniPath = filename % (data_path, sim, i)  # 100 files per sim
                    header, content = uniio.readUni(uniPath) # returns [Z,Y,X,C] np array
                    h = header['dimX']
                    w  = header['dimY']
                    arr = content[:, ::-1, :, :] # reverse order of Y axis
                    arr = np.reshape(arr, [w, h, 1]) # discard Z
                    self.densities.append( arr )

        loadNum = len(self.densities)
        if loadNum < 200:
            raise("Error - use at least two full sims, generate data by running 'manta ./manta_genSimSimple.py' a few times...")

        self.densities = np.reshape( self.densities, (len(self.densities), grid_height, grid_width, 1) )

        print("Read uni files, total data " + format(self.densities.shape))

    def __getitem__(self, idx):
        sampled_densities = self.densities[idx]
        densities_tensor = torch.from_numpy(sampled_densities).float()
        if self.transform_ops is not None:
            densities_tensor = self.transform_ops(sampled_densities)
        return densities_tensor, densities_tensor

    def __len__(self):
        return self.densities.shape[0]
    

class SimpleAEv3(nn.Module):
    def __init__(self, grid_h, grid_w, chs):
        super(SimpleAEv3, self).__init__()

        self.pipe = nn.Sequential(
            # 64x64x1
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            # 56x56x1
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.MaxPool2d(2, 2),
            #28x28x1
            nn.Upsample(scale_factor=2),
            #56x56x1
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=0),
            nn.ConvTranspose2d(in_channels=64, out_channels=1, kernel_size=5, stride=1, padding=0)
        )

    def forward(self, X, train=True):
        out = self.pipe(X)
        return out


def train_step(model, dataloader, loss_fn, optimizer, device):
    model.train()

    train_loss = 0.0

    for batch_idx, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        y_pred = model(X)

        loss = loss_fn(y_pred, y)
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
        for batch_idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            y_pred = model(X, train=False)

            loss = loss_fn(y_pred, y)
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
        for idx, (X, y) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)

            assert X.size(0) == 1

            y_pred = model(X, train=False)

            io.imwrite("%s/in_%d.png" % (out_dir, idx), y.cpu().numpy().squeeze(0).squeeze(0))
            io.imwrite("%s/out_%d.png" % (out_dir, idx), y_pred.cpu().numpy().squeeze(0).squeeze(0))
    print('end')


def main(data_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 10
    BATCH_SIZE = 10

    transforms = T.Compose([
        T.ToTensor()
    ])

    densities = MantaFlow2DDataset(data_path=data_path, transform_ops=transforms)

    train_len = len(densities)
    print(f'Length of dataset: {train_len}')
    validation_len = max(100, int(train_len * 0.1))
    train_dataset, validation_dataset = torch.utils.data.random_split(densities, [train_len - validation_len, validation_len])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = SimpleAEv3(grid_h=64, grid_w=64, chs=1).to(DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

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

if __name__ == '__main__':
    DATA_PATH = '../../tensorflow/data/'
    main(DATA_PATH)
