import argparse
import os

import torch
from tqdm import tqdm
import yaml
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from consts import num_timesteps_list

from mnist_dataset import FashionMnistDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_lambda(x):
    return x.repeat(3, 1, 1)


class FIDCalculator:
    def __init__(self, root_dir, batch_size=64):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.fid = FrechetInceptionDistance(normalize=True).to(device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Lambda(transform_lambda),  # Convert grayscale to 3 channels
            ]
        )

    def calculate_fid(self, sampling_dir, fashion_mnist):
        fashion_mnist_loader = DataLoader(fashion_mnist, batch_size=64, shuffle=True, num_workers=4)

        # Load generated images and update FID
        self.load_generated_images(sampling_dir, self.fid)
        # Update FID with FashionMNIST images
        for batch in tqdm(fashion_mnist_loader, desc="Real images"):
            images = batch
            self.fid.update(images.to(device), real=True)

        # Compute FID
        fid_score = self.fid.compute()
        return fid_score.item()

    def load_generated_images(self, sampling_dir, fid, batch_size=100):
        batch = []
        for img_file in tqdm(sorted(os.listdir(sampling_dir)), desc="Generated images"):
            img_path = os.path.join(sampling_dir, img_file)
            img = Image.open(img_path)
            img = self.transform(img)
            img = img.unsqueeze(0).to(device)
            batch.append(img)

            # Update FID when the batch is full
            if len(batch) == batch_size:
                fid.update(torch.cat(batch), real=False)
                batch = []

        # Update FID with any remaining images in the last batch
        if batch:
            fid.update(torch.cat(batch), real=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Arguments for ddpm image generation")
    parser.add_argument("--config", dest="config_path", default="config.yaml", type=str)
    args = parser.parse_args()

    root_dir = "./data/fashion_mnist_data"
    output_dir = "./default"

    # Read the config file #
    with open(args.config_path, "r") as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)
    ########################

    diffusion_config = config["diffusion_params"]
    sampling_config = config["sampling_params"]

    fid_calculator = FIDCalculator(root_dir=root_dir)

    # Load FashionMNIST training dataset
    fashion_mnist = FashionMnistDataset(
        "train", transform=fid_calculator.transform
    )  # Updated to use FashionMnistDataset

    # Calculate FID for different sampling configurations
    for num_timesteps in num_timesteps_list:
        sampling_dir = os.path.join(
            output_dir, f"{sampling_config['sampling_algorithm']}_sampling_{num_timesteps}"
        )
        print(f"Calculating FID for {sampling_dir}")
        fid_score = fid_calculator.calculate_fid(sampling_dir, fashion_mnist)
        print(f"FID score for vanilla_sampling_{num_timesteps}: {fid_score:.4f}")
