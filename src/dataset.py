from torch.utils.data import Dataset, DataLoader
from PIL import Image 
from pathlib import Path

import torchvision.transforms as transforms
import os
import torch


class ImageDataset(Dataset):
    """
    """
    def __init__(self, 
            data_path: str="data/train",
            transform: transforms=None
        ):
        self.classes = {i: v for i, v in enumerate(os.listdir(data_path))}
        self.label_id = {v: i for i, v in enumerate(os.listdir(data_path))}
        self.data_path = Path(data_path)
        self.transform = transform
        self._read_data()

    def _read_data(self):
        """
        """
        self.image_names = []
        self.labels = []
        for _data in os.listdir(self.data_path):
            sub_data = [f"{self.data_path}/{_data}/{file_name}" for file_name in os.listdir(self.data_path / _data)]
            self.image_names = self.image_names + sub_data
            self.labels = self.labels + [self.label_id[_data]] * len(sub_data)

    def __getitem__(self, index: int):
        image_path = self.image_names[index]
        label = self.labels[index]
        img = Image.open(image_path)
        if self.transform is not None:
            img = self.transform(img)
            label = torch.tensor(label)
        return img, label

    def __len__(self):
        return len(self.labels)


def get_transforms():
    """
    Define the data augmentation and normalization transformations to be applied on the images.
    """
    train_transforms = transforms.Compose([
        # transforms.RandomResizedCrop(size=224),
        transforms.Resize(size=32),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_test_transforms = transforms.Compose([
        transforms.Resize(size=32),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    return train_transforms, val_test_transforms


def get_data_loader(data_dir, batch_size, mode='train'):
    """
    Returns a PyTorch DataLoader object that loads and preprocesses the data.
    """
    train_transforms, val_test_transforms = get_transforms()

    if mode == 'train':
        transform = train_transforms
    else:
        transform = val_test_transforms

    dataset = ImageDataset(os.path.join(data_dir, mode), transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=(mode=='train'))

    return loader
