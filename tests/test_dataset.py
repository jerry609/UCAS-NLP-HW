import unittest
import os
import torch
from torchvision import transforms
from cat_dog_dataset import CatDogDataset, get_data_loaders


class TestCatDogDataset(unittest.TestCase):
    def setUp(self):
        """初始化测试环境"""
        self.train_dir = r"D:\nlp\hw1\data\train"
        self.test_dir = r"D:\nlp\hw1\data\test1"

        self.assertTrue(os.path.exists(self.train_dir), "训练数据目录不存在")
        self.assertTrue(os.path.exists(self.test_dir), "测试数据目录不存在")

    def test_train_dataset_loading(self):
        """测试训练数据集加载"""
        dataset = CatDogDataset(self.train_dir)
        self.assertTrue(len(dataset) > 0, "数据集为空")

        image, label = dataset[0]
        self.assertIsInstance(image, torch.Tensor, "图像应该是tensor类型")
        self.assertEqual(image.shape, (3, 224, 224), "图像维度应该是[3, 224, 224]")
        self.assertIn(label, [0, 1], "标签值应该是0或1")

    def test_train_data_loaders(self):
        """测试训练数据加载器"""
        train_loader, val_loader = get_data_loaders(self.train_dir)

        self.assertIsInstance(train_loader, torch.utils.data.DataLoader,
                              "train_loader应该是DataLoader类型")
        self.assertIsInstance(val_loader, torch.utils.data.DataLoader,
                              "val_loader应该是DataLoader类型")

        images, labels = next(iter(train_loader))
        self.assertEqual(len(images.shape), 4,
                         "batch图像应该是4维的[batch_size, channels, height, width]")
        self.assertEqual(images.shape[1], 3, "图像应该有3个通道")
        self.assertEqual(images.shape[2], 224, "图像高度应该是224")
        self.assertEqual(images.shape[3], 224, "图像宽度应该是224")

    def test_custom_transform(self):
        """测试自定义转换"""
        custom_transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])

        dataset = CatDogDataset(self.train_dir, transform=custom_transform)
        image, _ = dataset[0]
        self.assertEqual(image.shape, (3, 128, 128),
                         "使用自定义转换后图像维度应该是[3, 128, 128]")

    def test_data_split_ratio(self):
        """测试数据集分割比例"""
        train_ratio = 0.8
        train_loader, val_loader = get_data_loaders(
            self.train_dir,
            batch_size=32,
            train_ratio=train_ratio
        )

        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        expected_train_size = int(total_samples * train_ratio)

        self.assertEqual(len(train_loader.dataset), expected_train_size,
                         "训练集大小不符合预期")
        self.assertEqual(len(val_loader.dataset), total_samples - expected_train_size,
                         "验证集大小不符合预期")

    def test_batch_size(self):
        """测试批次大小设置"""
        batch_size = 16
        train_loader, val_loader = get_data_loaders(
            self.train_dir,
            batch_size=batch_size
        )

        images, labels = next(iter(train_loader))
        self.assertEqual(images.shape[0], batch_size,
                         f"batch大小应该是{batch_size}")

    def test_dataset_split_indices(self):
        """测试数据集分割的索引"""
        train_ratio = 0.8
        train_loader, val_loader = get_data_loaders(
            self.train_dir,
            batch_size=32,
            train_ratio=train_ratio
        )

        # 获取训练集和验证集的索引
        train_indices = train_loader.dataset.indices
        val_indices = val_loader.dataset.indices

        # 检查索引是否有重叠
        train_indices_set = set(train_indices)
        val_indices_set = set(val_indices)
        intersection = train_indices_set.intersection(val_indices_set)

        self.assertEqual(len(intersection), 0, "训练集和验证集不应该有重叠的索引")

        # 检查索引总数是否正确
        total_indices = len(train_indices) + len(val_indices)
        original_dataset_size = len(train_loader.dataset.dataset)
        self.assertEqual(total_indices, original_dataset_size,
                         "索引总数应该等于原始数据集大小")

        # 检查索引范围是否正确
        self.assertTrue(all(0 <= idx < original_dataset_size for idx in train_indices),
                        "训练集索引超出范围")
        self.assertTrue(all(0 <= idx < original_dataset_size for idx in val_indices),
                        "验证集索引超出范围")


if __name__ == '__main__':
    unittest.main()