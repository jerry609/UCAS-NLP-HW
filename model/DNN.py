import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from torchvision import transforms
from cat_dog_dataset import get_data_loaders
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from torch.cuda.amp import autocast, GradScaler


class CatDogDNN(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(CatDogDNN, self).__init__()

        # Using batch normalization and smaller layer sizes for better generalization
        self.features = nn.Sequential(
            nn.Linear(224 * 224 * 3, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout_rate),

            nn.Linear(128, 2)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Using Xavier initialization for better convergence
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return x


class Trainer:
    def __init__(self, model, device, train_loader, val_loader,
                 criterion, optimizer, scheduler, mixed_precision=True):
        self.model = model
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None

        self.train_losses = []
        self.train_accs = []
        self.val_losses = []
        self.val_accs = []

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

            self.optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()

            if self.mixed_precision:
                with autocast():
                    output = self.model(data)
                    loss = self.criterion(output, target)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
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

    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in tqdm(self.val_loader, desc='Validating'):
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

        val_loss = total_loss / len(self.val_loader)
        val_acc = 100. * correct / total

        # Learning rate scheduling
        if isinstance(self.scheduler, ReduceLROnPlateau):
            self.scheduler.step(val_loss)
        else:
            self.scheduler.step()

        # Early stopping and model saving
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
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

        return val_loss, val_acc

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
            val_loss, val_acc = self.validate()

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

            if self.should_stop_training():
                print("Early stopping triggered!")
                break

        self.model.load_state_dict(self.best_model_state)
        return self.model

    def plot_training_history(self):
        plt.style.use('seaborn')
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.train_losses, label='Train Loss', linewidth=2)
        plt.plot(self.val_losses, label='Val Loss', linewidth=2)
        plt.title('Training and Validation Loss', pad=15)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.plot(self.train_accs, label='Train Acc', linewidth=2)
        plt.plot(self.val_accs, label='Val Acc', linewidth=2)
        plt.title('Training and Validation Accuracy', pad=15)
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
    print(f'Using device: {device}')

    # Enhanced data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.RandomAffine(0, shear=10, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    train_loader, val_loader = get_data_loaders(
        data_dir=r"D:\nlp\hw1\data\train",
        batch_size=128,  # Increased batch size
        train_ratio=0.8,
        num_workers=4,
        custom_transform=train_transform
    )

    model = CatDogDNN(dropout_rate=0.4).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    scheduler = CosineAnnealingLR(optimizer, T_max=10, eta_min=1e-6)

    trainer = Trainer(
        model=model,
        device=device,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        mixed_precision=True
    )

    model = trainer.train(epochs=30)
    trainer.plot_training_history()

    print(f'\nBest validation accuracy: {trainer.best_val_acc:.2f}%')
    print('Training completed! Model saved as best_model.pth')
    print('Training history plot saved as training_history.png')


if __name__ == '__main__':
    main()