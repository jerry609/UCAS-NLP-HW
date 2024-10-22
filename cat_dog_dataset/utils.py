from torch.utils.data import DataLoader
from torchvision import transforms
from typing import Tuple, Optional
import torch

def get_data_loaders(data_dir: str,
                     batch_size: int = 32,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     num_workers: int = 4,
                     train_transform: Optional[transforms.Compose] = None,
                     val_test_transform: Optional[transforms.Compose] = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """创建训练集、验证集和测试集的DataLoader"""
    from .dataset import CatDogDataset

    dataset = CatDogDataset(data_dir, transform=None)

    total_size = len(dataset)
    train_size = int(train_ratio * total_size)
    val_size = int(val_ratio * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )

    # 为每个数据集指定不同的transform
    train_dataset.dataset.transform = train_transform
    val_dataset.dataset.transform = val_test_transform
    test_dataset.dataset.transform = val_test_transform

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    return train_loader, val_loader, test_loader
