import os
import random
import pickle
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

def preprocess_train(tensor):
    tensor -= tensor.mean().item()
    tensor = F.pad(tensor, pad=(4, 4, 4, 4), mode='constant', value=0)
    t = random.randrange(8)
    l = random.randrange(8)
    tensor = tensor[:, t:t+32, l:l+32]
    if random.random() < 0.5:
        tensor = transforms.functional.hflip(tensor)

    return tensor


def preprocess_test(tensor):
    tensor -= tensor.mean().item()
    return tensor



class CIFAR10NoLabels(Dataset):
    def __init__(self, file_path, transform=None):
        """
        Args:
            file_path (string): Path to the folder with the CIFAR test data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.transform = transform
        self.data = []
                    
        # Load cifar_test_nolabels
        test_file = os.path.join(file_path, 'cifar_test_nolabels.pkl')
        with open(test_file, 'rb') as fo:
            test_data = pickle.load(fo, encoding='bytes')
        self.data.append(test_data[b'data'])
        
        self.data = np.concatenate(self.data, axis=0)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Args:
            idx (int): Index
            
        Returns:
            sample (Tensor): Image in CxHxW format
        """
        image = self.data[idx]
        image = np.transpose(np.reshape(image, (3, 32, 32)), (1, 2, 0))  # Convert to HxWxC format
        if self.transform:
            image = self.transform(image)
            
        return image
    
def get_dataloader(is_train, batch_size, path):
    if is_train:
        return DataLoader(
            datasets.CIFAR10(path,
                            train=True,
                            download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Lambda(preprocess_train)
                            ])),
            batch_size=batch_size,
            shuffle=True
        )
    else:
        return DataLoader(
            datasets.CIFAR10(path,
                train=False,
                download=True,
                transform=transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Lambda(preprocess_test)
                ])),
            batch_size=batch_size,
            shuffle=False
        )

def get_kaggle_dataloader(batch_size, path):
    # Define your preprocessing and augmentations here
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(preprocess_test)
    ])
    
    # Instantiate your custom dataset
    dataset = CIFAR10NoLabels(file_path=path, transform=transform)
    
    # Create DataLoader
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return data_loader