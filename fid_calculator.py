import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import transforms
from tqdm import tqdm

from consts import config
from mnist_dataset import FashionMnistDataset

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def transform_lambda(x):
    return x.repeat(3, 1, 1)


class FIDCalculator:
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.fid = FrechetInceptionDistance(normalize=True).to(device)

        self.transform = transforms.Compose(
            [
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
                transforms.Lambda(transform_lambda),
            ]
        )

    def calculate_fid(self, sampling_dir, fashion_mnist):
        fashion_mnist_loader = DataLoader(
            fashion_mnist, batch_size=config.train_params.batch_size, shuffle=True, num_workers=4
        )

        self.load_generated_images(sampling_dir)
        for batch in tqdm(fashion_mnist_loader, desc="Real images"):
            images = batch
            self.fid.update(images.to(device), real=True)

        fid_score = self.fid.compute()
        return fid_score.item()

    def load_generated_images(self, sampling_dir):
        batch = []
        for img_file in tqdm(sorted(os.listdir(sampling_dir)), desc="Generated images"):
            img_path = os.path.join(sampling_dir, img_file)
            img = Image.open(img_path)
            img = self.transform(img)
            img = img.unsqueeze(0).to(device)
            batch.append(img)

            if len(batch) == config.sampling_params.sampling_batch_size:
                self.fid.update(torch.cat(batch), real=False)
                batch = []

        if batch:
            self.fid.update(torch.cat(batch), real=False)


if __name__ == "__main__":
    root_dir = "./data/fashion_mnist_data"
    output_dir = "./default"

    fid_calculator = FIDCalculator(root_dir=root_dir)

    fashion_mnist = FashionMnistDataset("train", transform=fid_calculator.transform)

    for num_timesteps in config.diffusion_params.num_timesteps_list:
        sampling_dir = os.path.join(
            output_dir, f"{config.sampling_params.sampling_algorithm}_sampling_{num_timesteps}"
        )
        print(f"Calculating FID for {sampling_dir}")
        fid_score = fid_calculator.calculate_fid(sampling_dir, fashion_mnist)
        print(
            f"FID score {config.sampling_params.sampling_algorithm}_sampling_{num_timesteps}:"
            f" {fid_score:.4f}"
        )
