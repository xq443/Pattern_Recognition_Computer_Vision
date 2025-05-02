import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch.nn.functional as F

# Define the network class
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        
        # Convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # Max pooling layer with 2x2 window and ReLU activation
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # Dropout layer with 30% dropout rate
        self.dropout = nn.Dropout(0.3)
        
        # Max pooling layer with 2x2 window and ReLU activation
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Flattening layer
        self.flatten = nn.Flatten()
        
        # Fully connected layer with 50 nodes + ReLU
        self.fc1 = nn.Linear(20 * 4 * 4, 50)  # Output size after convolution + pooling
        self.relu = nn.ReLU()
        
        # Fully connected layer with 10 nodes + log_softmax
        self.fc2 = nn.Linear(50, 10)
        
    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))      # Conv1 -> Pool1 -> ReLU
        x = F.relu(self.pool2(self.dropout(self.conv2(x))))  # Conv2 -> Dropout -> Pool2 -> ReLU
        x = self.flatten(x)                        # Flatten
        x = self.relu(self.fc1(x))                 # Fully Connected 1 -> ReLU
        x = F.log_softmax(self.fc2(x), dim=1)      # Fully Connected 2 -> Log Softmax
        return x

# Print the network summary
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNet().to(device)
print(model)
