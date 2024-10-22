import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple, Optional


class CatDogDataset(Dataset):
    """
    用于加载猫狗图片数据集的自定义Dataset类
    """

    def __init__(self,
                 root_dir: str,
                 transform: Optional[transforms.Compose] = None,
                 train: bool = True) -> None:
        """
        初始化数据集

        Args:
            root_dir: 数据集根目录路径
            transform: 图像预处理转换
            train: 是否为训练集
        """
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        # 如果没有指定transform，使用默认的预处理
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),  # 调整图片大小
                transforms.ToTensor(),  # 转换为tensor
                transforms.Normalize(  # 标准化
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])

        # 获取所有图片路径和标签
        self.images = []
        self.labels = []

        # 加载猫的图片
        cat_files = [f for f in os.listdir(root_dir) if f.startswith('cat')]
        for cat_file in cat_files:
            self.images.append(os.path.join(root_dir, cat_file))
            self.labels.append(0)  # 猫的标签为0

        # 加载狗的图片
        dog_files = [f for f in os.listdir(root_dir) if f.startswith('dog')]
        for dog_file in dog_files:
            self.images.append(os.path.join(root_dir, dog_file))
            self.labels.append(1)  # 狗的标签为1

    def __len__(self) -> int:
        """返回数据集大小"""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """获取指定索引的图片和标签"""
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label