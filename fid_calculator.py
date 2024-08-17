import os

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision import datasets, transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FIDCalculator:
    def __init__(self, root_dir, batch_size=64):
        self.root_dir = root_dir
        self.batch_size = batch_size
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.transform = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),  # Convert grayscale to 3 channels
        ])

    def calculate_fid(self, sampling_dir, split="train"):
        # Load FashionMNIST training dataset
        fashion_mnist = datasets.FashionMNIST(root=self.root_dir, train=(split == "train"),
                                              download=True, transform=self.transform)
        fashion_mnist_loader = DataLoader(fashion_mnist, batch_size=self.batch_size, shuffle=False)

        # Update FID with FashionMNIST images
        for batch in fashion_mnist_loader:
            images, _ = batch
            self.fid.update(images.to(device), real=True)

        # Load generated images and update FID
        generated_images = self.load_generated_images(sampling_dir)
        self.fid.update(generated_images.to(device), real=False)

        # Compute FID
        fid_score = self.fid.compute()
        return fid_score.item()

    def load_generated_images(self, sampling_dir):
        img_list = []
        for img_file in sorted(os.listdir(sampling_dir)):
            img_path = os.path.join(sampling_dir, img_file)
            img = Image.open(img_path)
            img = self.transform(img)
            img_list.append(img.unsqueeze(0))
        return torch.cat(img_list)


if __name__ == "__main__":
    root_dir = "./data/fashion_mnist_data"
    output_dir = "./default"

    fid_calculator = FIDCalculator(root_dir=root_dir)

    # Calculate FID for different sampling configurations
    for num_timesteps in [5, 10, 50, 200, 1000]:
        sampling_dir = os.path.join(output_dir, f"vanilla_sampling_{num_timesteps}")
        fid_score = fid_calculator.calculate_fid(sampling_dir)
        print(f"FID score for vanilla_sampling_{num_timesteps}: {fid_score:.4f}")
