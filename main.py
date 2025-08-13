"""Main training script for MNIST diffusion model."""

import torch
import torch.optim as optim
import os
from tqdm import tqdm
import argparse

from generate import generate_images
from config import Config
from models import DiffusionModel
from utils import get_mnist_dataloader, save_images, plot_loss


def train_model():
    """
    Train the diffusion model on MNIST.
    
    Returns:
        tuple: (trained_model, loss_history)
    """
    # TODO: 
    # 1. Setup device, create directories
    # 2. Load MNIST data
    # 3. Initialize model and optimizer
    # 4. Training loop with loss computation
    # 5. Periodic checkpointing and sample generation
    # 6. Return trained model and losses
    dataloader = get_mnist_dataloader()
    device = Config.DEVICE
    model = DiffusionModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    loss_history = []

    for epoch in tqdm(range(Config.EPOCHS)):
        avg_loss = 0
        loss_cnt = 0
        for batch, _ in tqdm(dataloader):
            # print('batch: ', batch.shape)
            optimizer.zero_grad()
            x0 = batch.to(device)
            t = torch.randint(0, Config.TIMESTEPS, (x0.size(0),), device=device).long()
            pred_noise, actual_noise = model(x0, t)
            loss = model.compute_loss(pred_noise, actual_noise)
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())
            avg_loss += loss.item()
            loss_cnt += 1
        print(f'Epoch {epoch+1}: avg_loss {avg_loss / loss_cnt if loss_cnt > 0 else 0}')
        if epoch % Config.SAVE_EACH_EPOCHS == 0:
            checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, f"model_epoch_{epoch}.pth")
            os.makedirs(Config.CHECKPOINT_DIR, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
            generate_images(model, Config.NUM_SAMPLES, Config.DEVICE, save_path=Config.OUTPUT_DIR+f'/pic{epoch}.jpg', show_images=False)

    return model, loss_history


def main() -> None:
    """
    Main function with argument parsing.
    
    Returns:
        None
    """
    # TODO: Parse command line arguments, update config, call train_model
    train_model()


if __name__ == "__main__":
    main()
