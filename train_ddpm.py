import os

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from consts import config
from linear_noise_scheduler import LinearNoiseScheduler
from mnist_dataset import FashionMnistDataset
from unet_base import Unet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train():
    diffusion_config = config.diffusion_params
    model_config = config.model_params
    train_config = config.train_params

    if config.wandb.enable:
        import wandb
        wandb.login(key="5fda0926085bc8963be5e43c4e501d992e35abe8")
        wandb.init(project=config.wandb.project_name, config=config)

    # Create the noise scheduler
    scheduler = LinearNoiseScheduler(
        diffusion_config
    )

    # Create the dataset
    fashion_mnist = FashionMnistDataset("train")
    fashion_mnist_loader = DataLoader(
        fashion_mnist, batch_size=train_config.batch_size, shuffle=True, num_workers=4
    )

    # Instantiate the model
    model = Unet(model_config).to(device)
    model.train()

    # Create output directories
    if not os.path.exists(train_config.task_name):
        os.mkdir(train_config.task_name)

    # Load checkpoint if found
    if os.path.exists(os.path.join(train_config.task_name, train_config.ckpt_name)):
        print("Loading checkpoint as found one")
        model.load_state_dict(
            torch.load(
                os.path.join(train_config.task_name, train_config.ckpt_name),
                map_location=device,
            )
        )

    # Specify training parameters
    num_epochs = train_config.num_epochs
    optimizer = Adam(model.parameters(), lr=train_config.lr)
    criterion = torch.nn.MSELoss()

    # Run training
    for epoch_idx in range(num_epochs):
        losses = []
        for im in tqdm(fashion_mnist_loader):
            optimizer.zero_grad()
            im = im.float().to(device)

            # Sample random noise
            noise = torch.randn_like(im).to(device)

            # Sample timestep
            t = torch.randint(0, diffusion_config.num_timesteps, (im.shape[0],)).to(device)

            # Add noise to images according to timestep
            noisy_im = scheduler.add_noise(im, noise, t)
            noise_pred = model(noisy_im, t)

            loss = criterion(noise_pred, noise)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

        avg_loss = np.mean(losses)
        print(f"Finished epoch: {epoch_idx + 1} | Loss : {np.mean(losses):.4f}")
        torch.save(
            model.state_dict(), os.path.join(train_config.task_name, train_config.ckpt_name)
        )

        if config.wandb.enable:
            wandb.log({"loss": avg_loss, "epoch": epoch_idx + 1})

    print("Done Training ...")
    if config.wandb.enable:
        wandb.finish()


if __name__ == "__main__":
    train()
