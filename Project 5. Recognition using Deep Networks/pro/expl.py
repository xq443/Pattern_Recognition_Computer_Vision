#   Xujia Qin 
#   29th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# 1. MPS Configuration
device = torch.device("mps")
torch.mps.set_per_process_memory_fraction(0.8)

# 2. Data Preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.2860,), (0.3530))
])
train_set = datasets.FashionMNIST('./data', train=True, download=True, transform=transform)
test_set = datasets.FashionMNIST('./data', train=False, transform=transform)

# 3. Conv-Optimized Model (Fixed FC layer)
class ConvArchSearch(nn.Module):
    def __init__(self, conv_layers, filter_sizes, filter_counts):
        super().__init__()
        layers = []
        in_channels = 1
        
        for i in range(conv_layers):
            layers += [
                nn.Conv2d(in_channels, filter_counts[i], filter_sizes[i], 
                         padding=filter_sizes[i]//2),
                nn.BatchNorm2d(filter_counts[i]),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_channels = filter_counts[i]
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self._get_flattened_size(), 256),  # Fixed FC size
            nn.ReLU(),
            nn.Linear(256, 10)
        )
    
    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.rand(1, 1, 28, 28)
            return self.features(x).view(1, -1).shape[1]
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

# 4. Experiment Runner
class ConvExperiment:
    def __init__(self):
        self.device = device
        self.train_loader = DataLoader(
            train_set, batch_size=512, shuffle=True, num_workers=2)
        self.val_loader = DataLoader(
            test_set, batch_size=1024, num_workers=2)
    
    def evaluate(self, conv_layers, filter_sizes, filter_counts, epochs=8):
        model = ConvArchSearch(conv_layers, filter_sizes, filter_counts).to(device)
        optimizer = optim.AdamW(model.parameters())
        
        for _ in range(epochs):
            model.train()
            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                loss = nn.CrossEntropyLoss()(model(images), labels)
                loss.backward()
                optimizer.step()
                torch.mps.empty_cache()
        
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in self.val_loader:
                outputs = model(images.to(device))
                correct += (outputs.argmax(1).cpu() == labels).sum().item()
        
        return correct / len(test_set)

# 5. Configuration Generator
# def generate_conv_configs():
#     return [
#         # (conv_layers, filter_sizes, filter_counts)
#         (2, [3, 3], [16, 32]),    # Small
#         (2, [5, 5], [32, 64]),    # Medium filters
#         (3, [3, 3, 3], [16, 32, 64]),  # Deep narrow
#         (3, [3, 5, 3], [32, 64, 128]), # Hybrid
#         (3, [5, 5, 5], [64, 128, 256]) # Large
#     ]


def generate_conv_configs():
    base_configs = [
        # (layers, filter_sizes, filter_counts)
        (2, [3,3], [16,32]),        # Baseline
        (2, [5,5], [32,64]),        # Larger filters
        (3, [3,3,3], [16,32,64]),   # Standard 3-layer
        (3, [5,5,5], [32,64,128]),  # Deep large filters
        (3, [3,5,3], [32,64,128]),  # Sandwich
        (4, [3,3,3,3], [16,32,64,128]), # Deep narrow
        (4, [5,5,5,5], [32,64,128,256]) # Deep wide
    ]
    
    # Bonus variations
    bonus_configs = [
        (3, [3,5,7], [32,64,128]),      # Progressive
        (3, [5,3,5], [64,32,64]),       # Bottleneck
        (4, [3,5,3,5], [32,64,128,256]) # Alternate
    ]
    return base_configs + bonus_configs

# Main Execution
if __name__ == "__main__":
    experiment = ConvExperiment()
    results = []
    
    for config in tqdm(generate_conv_configs()):
        acc = experiment.evaluate(*config)
        results.append({
            'config': config,
            'accuracy': acc,
            'params': sum(p.numel() for p in 
                         ConvArchSearch(*config).parameters())
        })
    
    # Save and display
    with open('conv_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    best = max(results, key=lambda x: x['accuracy'])
    print(f"\nBest Architecture: {best['config']}")
    print(f"Accuracy: {best['accuracy']:.2%}")
    print(f"Parameters: {best['params']:,}")