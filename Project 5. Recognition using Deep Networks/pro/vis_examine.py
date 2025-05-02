#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import cv2
from torchvision import datasets, transforms

class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

def apply_filters():
    # Load the trained network
    network = MyNetwork()
    network.load_state_dict(torch.load('./results/model.pth'))
    network.eval()

    # Access the weights of the first convolutional layer
    conv1_weights = network.conv1.weight.data

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    mnist_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
    image, label = mnist_data[0]

    # Convert image to numpy
    image_np = image.squeeze(0).numpy()

    # Apply filters
    filtered_images = []
    with torch.no_grad():
        for i in range(10):
            filter_weights = conv1_weights[i, 0].cpu().numpy()
            filtered_image = cv2.filter2D(image_np, -1, filter_weights)
            filtered_images.append(filtered_image)

    # Create figure with 5 rows and 4 columns
    fig, axes = plt.subplots(5, 4, figsize=(15, 18))
    
    # Display all 10 filters (20 items total)
    for i in range(10):
        row = i * 2 // 4  # Calculate row position (0-4)
        col = (i * 2) % 4  # Calculate column position (0-3)
        
        # First column pair for each filter
        axes[row, col].imshow(conv1_weights[i, 0].cpu().numpy(), cmap='viridis')
        axes[row, col].axis('off')
        axes[row, col].set_title(f'Filter {i+1} weights')
        
        axes[row, col+1].imshow(filtered_images[i], cmap='gray')
        axes[row, col+1].axis('off')
        axes[row, col+1].set_title(f'Filter {i+1} result')

    plt.tight_layout()
    plt.savefig('conv1_filter_results.png')
    plt.show()

def main():
    apply_filters()

if __name__ == "__main__":
    main()