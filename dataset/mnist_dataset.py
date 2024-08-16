import os

import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


def scale_to_range(x):
    return (x * 2) - 1  # Scale to range [-1, 1]


class MnistDataset(Dataset):
    r"""
    Dataset class for MNIST images using torchvision's MNIST dataset.
    This class allows flexibility to switch to another dataset if needed.
    """

    def __init__(self, split='train', root_dir='.'):
        r"""
        Init method for initializing the dataset properties
        :param split: 'train' or 'test' to specify the dataset split.
        :param root_dir: Root directory for saving the MNIST data.
        """
        self.split = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(scale_to_range)  # Use the regular function instead of lambda
        ])

        # Ensure the directory structure is correct
        data_dir = os.path.join(root_dir, 'data', 'mnist_data')

        self.dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True if split == 'train' else False,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, label = self.dataset[index]
        return im
