import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class LightMNIST(nn.Module):
    def __init__(self):
        super(LightMNIST, self).__init__()
        # Increased initial channels and added one more conv layer
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.3)  # Increased dropout
        self.dropout2 = nn.Dropout(0.3)
        self.fc = nn.Linear(64 * 7 * 7, 10)

    def forward(self, x):
        # Enhanced forward pass with additional conv layer
        identity = x = self.conv1(x)
        x = F.relu(self.bn1(x))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x + identity
        x = self.pool(x)
        x = self.dropout1(x)
        
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout2(x)
        
        x = x.view(-1, 64 * 7 * 7)
        x = self.fc(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LightMNIST().to(device)
    
    # Modified training parameters
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.001,
        weight_decay=0.005,  # Reduced weight decay
        amsgrad=True
    )
    
    # Enhanced data augmentation
    transform = transforms.Compose([
        transforms.RandomRotation(10),  # Reduced rotation
        transforms.RandomAffine(
            degrees=0,
            translate=(0.08, 0.08),  # Reduced translation
            scale=(0.9, 1.1),  # Reduced scale variation
            shear=5  # Reduced shear
        ),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=32,  # Smaller batch size
        shuffle=True,
        pin_memory=True,
        num_workers=4
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Calculate total steps for one epoch
    total_steps = len(train_loader)
    
    # Modified scheduler parameters
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=0.005,  # Reduced max learning rate
        steps_per_epoch=total_steps,
        epochs=1,
        pct_start=0.2,  # Increased warm-up period
        div_factor=15,
        final_div_factor=1000,
        anneal_strategy='cos'
    )
    
    model.train()
    correct = 0
    total = 0
    running_loss = 0.0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        
        # Label smoothing loss
        if batch_idx < len(train_loader) // 3:
            loss = criterion(output, target)
        else:
            # Confidence based loss weighting
            prob = F.softmax(output, dim=1)
            confidence, predicted = prob.max(1)
            correct_mask = predicted.eq(target)
            
            # Higher weight for misclassified samples
            loss_weights = 1.0 - (confidence * correct_mask.float())
            loss = (criterion(output, target) * loss_weights).mean()
        
        loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if batch_idx < total_steps - 1:
            scheduler.step()
        
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()
        
        if batch_idx % 100 == 0:
            print(f'Batch: {batch_idx}, Accuracy: {100.*correct/total:.2f}%, Loss: {loss.item():.4f}')

    epoch_accuracy = 100.*correct/total
    return model, epoch_accuracy

if __name__ == "__main__":
    model = LightMNIST()
    print(f"Total parameters: {count_parameters(model)}")