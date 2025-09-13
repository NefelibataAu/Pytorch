from torch.utils import data
import torchvision
import torchvision.transforms as transforms
import torch
from config import BATCH_SIZE, DATA_ROOT

def get_dataloaders():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = torchvision.datasets.MNIST(
        root = DATA_ROOT,
        train = True,
        transform = transform,
        download = True
    )
    test_dataset = torchvision.datasets.MNIST(
        root = DATA_ROOT,
        train = False,
        transform = transform,
        download = True
    )
    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False
    )
    return train_loader, test_loader

