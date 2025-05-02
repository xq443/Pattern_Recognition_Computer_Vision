#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# Define the LeNet model
class LeNet(nn.Module):
    """
    Creates a LeNet based architecture to classify digits in the MNIST dataset.
    """
    def __init__(self):
        super().__init__()

        # Block 1: Convolutional layer, Max pooling, and ReLU activation
        self.block_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=10,
                kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        # Block 2: Convolutional layer, Dropout, Max pooling, and ReLU activation
        self.block_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=10,
                out_channels=20,
                kernel_size=5,
            ),
            nn.Dropout2d(0.5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        # Classifier: Flatten and two fully connected layers
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20 * 4 * 4, out_features=50),
            nn.Linear(50, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)
        return nn.functional.log_softmax(x, dim=1)

# Initialize the model
model = LeNet()

# Create a dummy input tensor (for MNIST, it's 1x28x28 image)
dummy_input = torch.randn(1, 1, 28, 28)

# Get the output of the model
output = model(dummy_input)

# Visualize the computational graph
make_dot(output, params=dict(model.named_parameters())).render("LeNet_architecture", format="png")

