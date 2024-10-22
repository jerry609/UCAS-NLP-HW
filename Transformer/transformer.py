import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from torchvision.models import vit_b_16
from cat_dog_dataset import get_data_loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler


class CatDogTransformer(nn.Module):
    def __init__(self, num_classes=2):
        super(CatDogTransformer, self).__init__()
        # 使用预训练的 ViT 模型
        self.model = vit_b_16(pretrained=True)
        # 获取隐藏维度
        hidden_dim = self.model.hidden_dim
        # 替换分类头
        self.model.heads = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)


class Trainer:
    def __init__(self, model, device, train_loader, val_loader, test_loader,
                 criterion, optimizer, scheduler, mixed_precision=True, max_grad_norm=5):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        self.max_grad_norm = max_grad_norm  # 梯度剪裁的阈值

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []
        self.test_losses = []
        self.test_accs = []

        self.best_val_acc = 0.0
        self.best_model_state = None
        self.early_stopping_counter = 0
        self.early_stopping_patience = 5
        self.min_lr = 1e-6

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc='Training')
        for data, target in pbar:
            data, target = data.to(self.device), target.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            if self.mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.scaler.scale(loss).backward()
                # 梯度剪裁
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                # 梯度剪裁
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

            pbar.set_postfix({
                'loss': f'{total_loss / total:.4f}',
                'acc': f'{100. * correct / total:.2f}%'
            })

        return total_loss / len(self.train_loader), 100. * correct / total

    def validate(self, loader, mode='Val'):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(loader, desc=f'{mode}'):
                data, target = data.to(self.device), target.to(self.device)

                if self.mixed_precision:
                    with autocast():
                        output = self.model(data)
                        loss = self.criterion(output, target)
                else:
                    output = self.model(data)
                    loss = self.criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        avg_loss = total_loss / len(loader)
        avg_acc = 100. * correct / total

        if mode == 'Val':
            # 学习率调度
            if isinstance(self.scheduler, ReduceLROnPlateau):
                self.scheduler.step(avg_loss)
            else:
                self.scheduler.step()

            # 早停和模型保存
            if avg_acc > self.best_val_acc:
                self.best_val_acc = avg_acc
                self.best_model_state = self.model.state_dict()
                self.early_stopping_counter = 0
                torch.save({
                    'model_state_dict': self.best_model_state,
                    'val_acc': self.best_val_acc,
                    'epoch': len(self.train_losses) + 1,
                    'optimizer_state_dict': self.optimizer.state_dict(),
                }, 'best_model.pth')
            else:
                self.early_stopping_counter += 1

        return avg_loss, avg_acc

    def should_stop_training(self):
        if self.early_stopping_counter >= self.early_stopping_patience:
            return True
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr < self.min_lr:
            return True
        return False

    def train(self, epochs):
        for epoch in range(epochs):
            print(f'\nEpoch {epoch + 1}/{epochs}')

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate(self.val_loader, mode='Val')
            test_loss, test_acc = self.validate(self.test_loader, mode='Test')

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            self.test_losses.append(test_loss)
            self.test_accs.append(test_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

            if self.should_stop_training():
                print("早停触发！")
                break

        self.model.load_state_dict(self.best_model_state)
        return self.model

    def plot_training_history(self):
        plt.style.use('seaborn')
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.plot(self.test_losses, label='Test Loss', linewidth=2)
        plt.title('训练、验证和测试损失', pad=15)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc', linewidth=2)
        plt.plot(self.val_accs, label='Val Acc', linewidth=2)
        plt.plot(self.test_accs, label='Test Acc', linewidth=2)
        plt.title('训练、验证和测试准确率', pad=15)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    np.random.seed(42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 数据增强和归一化
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5],
                             std=[0.5, 0.5, 0.5])
    ])

    train_loader, val_loader, test_loader = get_data_loaders(
        data_dir=r"D:\nlp\hw1\data\train",
        batch_size=16,  # 由于 ViT 模型较大，建议减小批量大小以防止显存不足
        train_ratio=0.7,
        val_ratio=0.15,
        num_workers=4,
        train_transform=train_transform,
        val_test_transform=val_test_transform
    )

    model = CatDogTransformer(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        mixed_precision=True,
        max_grad_norm=5  # 梯度剪裁阈值
    )

    model = trainer.train(epochs=30)
    trainer.plot_training_history()

    print(f'\n最佳验证准确率: {trainer.best_val_acc:.2f}%')
    print('训练完成！模型已保存为 best_model.pth')
    print('训练过程图已保存为 training_history.png')


if __name__ == '__main__':
    main()
