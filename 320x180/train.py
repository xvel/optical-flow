import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from autoflowdataset import AutoFlowAug2Dataset  # Zakładam, że dataset jest w tym samym folderze
from InceptionNext import InceptionNext
import matplotlib.pyplot as plt

# Hyperparametry
EPOCHS = 10
BATCH_SIZE = 64
LEARNING_RATE = 1e-4
FINAL_LR = LEARNING_RATE * 0.02
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train():
    # Dataset i DataLoader
    dataset = AutoFlowAug2Dataset(root_dir="/mnt/d/datasety/autoflowaug2")
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

    # Model, loss, optimizer, scheduler
    model = InceptionNext().to(DEVICE)
    print(f"Model parameters: {count_parameters(model):,}")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=LEARNING_RATE,
        total_steps=len(dataloader) * EPOCHS,
        pct_start=0.1,
        anneal_strategy='linear',
        final_div_factor=1 / 0.03
    )

    # Mixed precision
    scaler = GradScaler()

    loss_history = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for step, ((im0, im1), target) in enumerate(dataloader):
            im0, im1, target = im0.to(DEVICE), im1.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass z mixed precision
            with autocast():
                output = model(im0, im1)
                loss = criterion(output, target)

            # Backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            loss_history.append(loss.item())

            if step % 500 == 0:
                print(f"Epoch {epoch+1}/{EPOCHS}, Step {step}, Loss: {loss.item():.6f}")

        print(f"Epoch {epoch+1}/{EPOCHS}, Average Loss: {epoch_loss/len(dataloader):.6f}")

    # Zapis modelu
    torch.save(model.state_dict(), "inceptionnext_autoflowaug2.pth")

    # Wykres loss
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("loss_plot.png")
    plt.show()

if __name__ == "__main__":
    train()
