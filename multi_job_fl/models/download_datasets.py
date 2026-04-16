"""
Download CIFAR-10 and MNIST datasets
"""
import torch
from torchvision import datasets, transforms
import os

def download_datasets(data_dir='./data'):
    """Download all required datasets"""
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading CIFAR-10...")
    transform_cifar = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    
    datasets.CIFAR10(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_cifar
    )
    datasets.CIFAR10(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_cifar
    )
    
    print("Downloading MNIST...")
    transform_mnist = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    datasets.MNIST(
        root=data_dir, 
        train=True, 
        download=True, 
        transform=transform_mnist
    )
    datasets.MNIST(
        root=data_dir, 
        train=False, 
        download=True, 
        transform=transform_mnist
    )
    
    print("All datasets downloaded successfully!")

if __name__ == "__main__":
    download_datasets()