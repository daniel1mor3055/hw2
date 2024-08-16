import torchvision
from torch.utils.data import Dataset
from torchvision import transforms


class MnistDataset(Dataset):
    r"""
    Dataset class for MNIST images using torchvision's MNIST dataset.
    This class allows flexibility to switch to another dataset if needed.
    """

    def __init__(self, split='train'):
        r"""
        Init method for initializing the dataset properties
        :param split: 'train' or 'test' to specify the dataset split.
        """
        self.split = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x * 2) - 1)  # Scale to range [-1, 1]
        ])

        self.dataset = torchvision.datasets.MNIST(
            root='mnist_data',
            train=True if split == 'train' else False,
            download=True,
            transform=self.transform
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        im, label = self.dataset[index]
        return im
