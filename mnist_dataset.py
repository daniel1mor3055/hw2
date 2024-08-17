import os
import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


def scale_to_range(x):
    return (x * 2) - 1  # Scale to range [-1, 1]


class FashionMnistDataset(Dataset):
    r"""
    Dataset class for Fashion-MNIST images using torchvision's FashionMNIST dataset.
    This class allows flexibility to switch to another dataset if needed.
    """

    def __init__(self, split="train", root_dir="."):
        r"""
        Init method for initializing the dataset properties
        :param split: 'train' or 'test' to specify the dataset split.
        :param root_dir: Root directory for saving the Fashion-MNIST data.
        """
        self.split = split
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(scale_to_range),  # Use the regular function instead of lambda
            ]
        )

        # Ensure the directory structure is correct
        data_dir = os.path.join(root_dir, "data", "fashion_mnist_data")

        self.dataset = torchvision.datasets.FashionMNIST(
            root=data_dir,
            train=True if split == "train" else False,
            download=True,
            transform=self.transform,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, label = self.dataset[index]
        return im
