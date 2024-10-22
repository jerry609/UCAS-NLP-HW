import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

from model.DNN import CatDogDNN


class TestDataset(Dataset):
    def __init__(self, test_dir, transform=None):
        self.test_dir = test_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(test_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]

        # 根据文件名排序，确保评估的一致性
        self.image_files.sort()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        img_path = os.path.join(self.test_dir, img_name)

        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)

            # 从文件名中提取标签（如果文件名包含标签信息）
            # 这里假设文件名格式为：cat.1.jpg 或 dog.1.jpg
            label = 0 if 'cat' in img_name.lower() else 1

            return image, label, img_name

        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            # 返回一个空白图像作为替代
            blank_image = torch.zeros((3, 224, 224))
            return blank_image, -1, img_name


class TestEvaluator:
    def __init__(self, model, device, test_loader):
        self.model = model
        self.device = device
        self.test_loader = test_loader

    def evaluate(self):
        self.model.eval()
        predictions = []
        true_labels = []
        image_names = []
        incorrect_predictions = []

        with torch.no_grad():
            for data, target, img_names in tqdm(self.test_loader, desc='Testing'):
                data, target = data.to(self.device), target.to(self.device)
                output = self.model(data)

                pred = output.argmax(dim=1)

                # 记录预测结果
                predictions.extend(pred.cpu().numpy())
                true_labels.extend(target.cpu().numpy())
                image_names.extend(img_names)

                # 记录错误预测
                mask = pred.cpu().numpy() != target.cpu().numpy()
                if any(mask):
                    incorrect_predictions.extend([
                        (img_names[i], true_labels[-len(mask):][i], predictions[-len(mask):][i])
                        for i in range(len(mask)) if mask[i]
                    ])

        # 计算准确率
        correct = sum(1 for p, t in zip(predictions, true_labels) if p == t)
        total = len(predictions)
        accuracy = 100. * correct / total if total > 0 else 0

        return accuracy, predictions, true_labels, image_names, incorrect_predictions

    def save_results(self, predictions, image_names):
        # 保存预测结果到文本文件
        with open('test_predictions.txt', 'w') as f:
            f.write("Image_Name,Prediction\n")
            for img_name, pred in zip(image_names, predictions):
                label = 'cat' if pred == 0 else 'dog'
                f.write(f"{img_name},{label}\n")

    def plot_confusion_matrix(self, true_labels, predictions):
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(true_labels, predictions)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Cat', 'Dog'],
                    yticklabels=['Cat', 'Dog'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()

    def visualize_incorrect_predictions(self, incorrect_predictions, test_dir, transform):
        n = min(len(incorrect_predictions), 16)  # 最多显示16张图片
        if n == 0:
            return

        rows = int(np.ceil(n / 4))
        fig, axes = plt.subplots(rows, 4, figsize=(15, rows * 4))
        axes = axes.ravel()

        for idx, (img_name, true_label, pred_label) in enumerate(incorrect_predictions[:n]):
            img_path = os.path.join(test_dir, img_name)
            img = Image.open(img_path).convert('RGB')

            # 显示原始图像，不需要做测试时的变换
            axes[idx].imshow(img)
            axes[idx].set_title(f'True: {"Cat" if true_label == 0 else "Dog"}\n'
                                f'Pred: {"Cat" if pred_label == 0 else "Dog"}',
                                color='red')
            axes[idx].axis('off')

        # 隐藏未使用的子图
        for idx in range(n, len(axes)):
            axes[idx].axis('off')

        plt.tight_layout()
        plt.savefig('incorrect_predictions.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # 设置测试数据变换
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # 加载测试数据
    test_dir = r"D:\nlp\hw1\data\test1"
    test_dataset = TestDataset(test_dir, transform=test_transform)
    test_loader = DataLoader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # 加载模型
    try:
        checkpoint = torch.load('best_model.pth')
        model = CatDogDNN().to(device)  # 确保这里使用了正确的模型类
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return

    # 创建评估器
    evaluator = TestEvaluator(model, device, test_loader)

    # 进行测试评估
    accuracy, predictions, true_labels, image_names, incorrect_predictions = evaluator.evaluate()
    print(f'\nTest Accuracy: {accuracy:.2f}%')

    # 保存预测结果
    evaluator.save_results(predictions, image_names)
    print('Predictions saved to test_predictions.txt')

    # 绘制混淆矩阵
    evaluator.plot_confusion_matrix(true_labels, predictions)
    print('Confusion matrix saved as confusion_matrix.png')

    # 打印分类报告
    print("\nClassification Report:")
    print(classification_report(true_labels, predictions,
                                target_names=['Cat', 'Dog'],
                                digits=4))

    # 可视化错误预测
    evaluator.visualize_incorrect_predictions(incorrect_predictions, test_dir, test_transform)
    print('Incorrect predictions visualization saved as incorrect_predictions.png')


if __name__ == '__main__':
    main()