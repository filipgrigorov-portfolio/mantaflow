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

class SimpleAE(nn.Module):
    def __init__(self, grid_h, grid_w, chs):
        super(SimpleAE, self).__init__()
        
        self.w = grid_w
        self.h = grid_h
        self.k = chs
        self.input_size = grid_h * grid_w * chs

        self.w1 = nn.parameter.Parameter(torch.rand(size=(self.input_size, 50)), requires_grad=True)
        self.b1 = nn.parameter.Parameter(torch.rand(size=(1, 50)), requires_grad=True)

        self.w2 = nn.parameter.Parameter(torch.rand(size=(50, self.input_size)), requires_grad=True)
        self.b2 = nn.parameter.Parameter(torch.rand(size=(1, self.input_size)), requires_grad=True)

    def forward(self, X, train=True):
        X = X.view(-1, self.input_size)
        fc1_out = torch.tanh(torch.matmul(X, self.w1) + self.b1)
        out = torch.dropout(fc1_out, p=0.3, train=train)
        fc2_out = torch.matmul(out, self.w2) + self.b2
        return fc2_out.view(-1, self.h, self.w, self.k)
    
class SimpleAEv2(nn.Module):
    def __init__(self, grid_h, grid_w, chs):
        super(SimpleAEv2, self).__init__()
        
        self.w = grid_w
        self.h = grid_h
        self.k = chs
        self.input_size = grid_h * grid_w * chs

        self.fc = nn.Sequential(
            nn.Linear(in_features=self.input_size, out_features=50, bias=True),
            nn.Tanh(),
            nn.Dropout(),
            nn.Linear(in_features=50, out_features=self.input_size, bias=True)
        )

    def forward(self, X, train=True):
        X = X.view(-1, self.input_size)
        out = self.fc(X)
        return out.view(-1, self.h, self.w, self.k)


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

            io.imwrite("%s/in_%d.png" % (out_dir, idx), y.cpu().numpy().squeeze(0).squeeze(-1))
            io.imwrite("%s/out_%d.png" % (out_dir, idx), y_pred.cpu().numpy().squeeze(0).squeeze(-1))
    print('end')


def test_dataset(data_path):
    densities = MantaFlow2DDataset(data_path)
    size = len(densities)
    print(f'Size of loaded data: {size}')
    X, y = densities[0]
    print(f'Size of first data point: {X.shape}, {y.shape}')
    print(f'Max value of X: {X.max()}')

    assert size > 0, "data should be loaded"
    assert X.shape[0] == 64, "grid widht should be 64"
    assert X.max() > 0, "max density value > 0"

    dataloader = DataLoader(dataset=densities, batch_size=1, shuffle=False, pin_memory=True)
    loaded_X, loaded_y = zip(next(iter(dataloader)))
    print(f'Loaded X size: {loaded_X[0].size()}')
    print(f'Loaded X np shape: {loaded_X[0].numpy().shape}')
    print(f'Max value of loaded X: {loaded_X[0].numpy().max()}')

    assert (loaded_X[0].numpy() == X.numpy()).all(), "arrays should be equal"

    count = 0
    for idx, (X, y) in enumerate(dataloader):
        io.imwrite(f'img_{idx}.png', X.numpy().squeeze(0).squeeze(-1))
        if count == 5:
            raise('stop')
        count += 1


def main(data_path):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    EPOCHS = 500
    BATCH_SIZE = 10

    densities = MantaFlow2DDataset(data_path)
    train_len = len(densities)
    print(f'Length of dataset: {train_len}')
    validation_len = max(100, int(train_len * 0.1))
    train_dataset, validation_dataset = torch.utils.data.random_split(densities, [train_len - validation_len, validation_len])

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=1, shuffle=False, pin_memory=True)

    model = SimpleAEv2(grid_h=64, grid_w=64, chs=1).to(DEVICE)

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
    #test_dataset(DATA_PATH)
