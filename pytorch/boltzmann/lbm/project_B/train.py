import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm.auto import tqdm
from model import CollisionNN
from torch.utils.data import DataLoader
from torchmetrics import MeanAbsolutePercentageError
from dataset import LBMDataset

# Definitions
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TRAIN_DATA_PATH = "train_dataset.npz"
TEST_DATA_PATH = "test_dataset.npz"
PRETRAINED = False

AT_EVERY = 20

EPOCHS = 300
BATCH_SIZE = 32
LR = 1e-4
IS_MAPE = True

def train():
    traindataset = LBMDataset(TRAIN_DATA_PATH)
    trainloader = DataLoader(dataset=traindataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    valdataset = LBMDataset(TEST_DATA_PATH)
    valloader = DataLoader(dataset=valdataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, drop_last=True)

    model = CollisionNN(in_chs=9).to(DEVICE)

    if PRETRAINED:
        model.load_state_dict(torch.load("checkpoint.pt"))

    mape = MeanAbsolutePercentageError().to(DEVICE)
    mse = nn.MSELoss().to(DEVICE)
    criterion = mape if IS_MAPE else mse
    optimizer = optim.Adam(model.parameters(), LR)

    training_losses = []
    val_losses = []

    best_loss = float("inf")
    for epoch in tqdm(range(EPOCHS), desc=f"Training progress", colour="#00ff00"):

        epoch_loss = 0.0
        for _, batch in enumerate(tqdm(trainloader, leave=False, desc=f"Epoch {epoch + 1}/{EPOCHS}", colour="#005500")):
            fpre = batch[0].to(DEVICE)
            fpost = batch[1].to(DEVICE)
            
            fpost_pred = model(fpre)
            loss = criterion(fpost_pred, fpost)
            optimizer.zero_grad()
            loss.backward()       
            optimizer.step()

            epoch_loss += loss.item() * BATCH_SIZE / len(trainloader.dataset)

        training_losses.append(epoch_loss)
        logs = f"\nLoss at epoch {epoch + 1}: {epoch_loss:.6f}"

        if epoch % AT_EVERY == AT_EVERY - 1:
            model.eval()
            with torch.no_grad():
                tot_val_loss = 0.0
                count = 0
                for _, batch in enumerate(valloader):
                    fpre_val = batch[0].to(DEVICE)
                    fpost_val = batch[1].to(DEVICE)
                    fpost_val_pred = model(fpre_val)
                    val_loss = criterion(fpost_val_pred, fpost_val)
                    tot_val_loss += val_loss.item()
                    count += 1
                val_losses.append(tot_val_loss / count)
                print(f"\nValidation loss: {val_losses[-1]}")
            model.train()

        # Storing the model
        if best_loss > epoch_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), os.path.join(os.getcwd(), "checkpoint_mape.pt"))
            logs += " --> Best model ever (stored)"
        print(logs)

    plt.semilogy(training_losses, lw=3, label='Training')
    plt.semilogy(val_losses, lw=3, label='Validation')

    plt.legend(loc='best', frameon=False)

    plt.savefig("loss_plot.png")

if __name__ == "__main__":
    print("Training has begun")
    train()
    print("Training has ended")