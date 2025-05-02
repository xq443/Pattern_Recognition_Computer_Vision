#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import numpy as np
from tqdm import tqdm
import json
from pathlib import Path

# 1. MPS Configuration
device = torch.device("mps")
torch.mps.set_per_process_memory_fraction(0.8)  # Prevent memory overflow

# 2. Data Preparation
def get_datasets():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530))  # FashionMNIST stats
    ])
    train_set = datasets.FashionMNIST(
        './data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_set = datasets.FashionMNIST(
        './data', 
        train=False, 
        transform=transform
    )
    return train_set, test_set

# 3. MPS-Optimized Model
class FashionMNIST_CNN(nn.Module):
    def __init__(self, conv_params, dense_nodes=256):
        super().__init__()
        layers = []
        in_channels = 1
        
        # Dynamic convolutional layers
        for out_c, k_size in zip(conv_params['filters'], conv_params['kernel_sizes']):
            layers += [
                nn.Conv2d(in_channels, out_c, k_size, padding=k_size//2),
                nn.BatchNorm2d(out_c),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ]
            in_channels = out_c
        
        self.features = nn.Sequential(*layers)
        
        # Calculate flattened size
        with torch.no_grad():
            x = torch.rand(1, 1, 28, 28)
            x = self.features(x)
            self.flattened_size = x.numel()
        
        # Dynamic classifier
        self.classifier = nn.Sequential(
            nn.Linear(self.flattened_size, dense_nodes),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(dense_nodes, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(-1, self.flattened_size)
        return self.classifier(x)

# 4. Experiment Runner
class ExperimentRunner:
    def __init__(self):
        self.device = device
        self.train_set, self.test_set = get_datasets()
        
        # Create validation subset (20% of test)
        val_indices = np.random.choice(
            len(self.test_set), 
            size=int(0.2*len(self.test_set)), 
            replace=False
        )
        self.val_set = Subset(self.test_set, val_indices)
    
    def run_experiment(self, params, epochs=10):
        model = FashionMNIST_CNN(
            conv_params={
                'filters': params['filters'],
                'kernel_sizes': [params['kernel_size']] * len(params['filters'])
            },
            dense_nodes=params['dense_nodes']
        ).to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        
        # MPS-optimized dataloaders
        train_loader = DataLoader(
            self.train_set, 
            batch_size=512, 
            shuffle=True,
            num_workers=2,
            persistent_workers=True
        )
        val_loader = DataLoader(
            self.val_set,
            batch_size=1024,
            num_workers=2
        )
        
        # Training loop
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images = images.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                torch.mps.empty_cache()  # Critical for MPS
        
        # Validation
        model.eval()
        correct = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(self.device, non_blocking=True)
                outputs = model(images)
                correct += (outputs.argmax(1).cpu() == labels).sum().item()
        
        return {
            'params': params,
            'accuracy': correct / len(self.val_set),
            'params_count': sum(p.numel() for p in model.parameters())
        }

# 5. Configuration Generator
def generate_configs():
    configs = []
    filter_configs = [
        {'name': 'small', 'filters': [16, 32], 'kernel_size': 3},
        {'name': 'medium', 'filters': [32, 64, 128], 'kernel_size': 3},
        {'name': 'large', 'filters': [64, 128, 256], 'kernel_size': 5}
    ]
    dense_nodes = [128, 256, 512]
    
    for fc in dense_nodes:
        for conv in filter_configs:
            configs.append({
                'filters': conv['filters'],
                'kernel_size': conv['kernel_size'],
                'dense_nodes': fc,
                'config_name': f"{conv['name']}_fc{fc}"
            })
    return configs

# Main Execution
if __name__ == "__main__":
    runner = ExperimentRunner()
    results = []
    
    for config in tqdm(generate_configs()):
        try:
            result = runner.run_experiment(config, epochs=10)
            results.append(result)
            
            # Save intermediate results
            with open('mps_results.json', 'w') as f:
                json.dump(results, f, indent=2)
                
        except Exception as e:
            print(f"Failed {config['config_name']}: {str(e)}")
            torch.mps.empty_cache()
            continue
    
    # Analysis
    if results:
        best = max(results, key=lambda x: x['accuracy'])
        print(f"\nBest Config: {best['params']['config_name']}")
        print(f"Accuracy: {best['accuracy']:.2%}")
        print(f"Params: {best['params_count']:,}")
    else:
        print("No successful runs!")