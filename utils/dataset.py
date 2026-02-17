import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_dataloaders(data_dir, batch_size=16, image_size=128):

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5],
                             [0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(
        root=f"{data_dir}/train",
        transform=transform
    )

    val_dataset = datasets.ImageFolder(
        root=f"{data_dir}/valid",
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        root=f"{data_dir}/test",
        transform=transform
    )

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True)

    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False)

    return train_loader, val_loader, test_loader
