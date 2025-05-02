# MNIST CNN Model with Visualization
# Author: [Your Name]

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchviz import make_dot

# CNN Model Definition
class MyNetwork(nn.Module):
    """Custom CNN for MNIST digit recognition"""
    
    def __init__(self, dropout_rate=0.5):
        super(MyNetwork, self).__init__()
        
        # Convolution Layer 1: 10 filters of size 5x5 -> Output: 10x24x24
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        
        # Dropout Layer: 50% dropout rate
        self.dropout1 = nn.Dropout(p=dropout_rate)
        
        # Max Pooling Layer with 2x2 window and ReLU
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Convolution Layer 2: 20 filters of size 5x5 -> Output: 20x8x8
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        
        # Max Pooling Layer with 2x2 window and ReLU
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully Connected Layer: 320 inputs -> 50 outputs
        self.fc1 = nn.Linear(20 * 4 * 4, 50)
        
        # Fully Connected Layer: 50 inputs -> 10 outputs
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        """Forward pass through the network"""
        
        # Layer 1: Conv1 -> Dropout -> Max Pool -> ReLU
        x = F.relu(self.pool1(self.dropout1(self.conv1(x))))
        
        # Layer 2: Conv2 -> Max Pool -> ReLU
        x = F.relu(self.pool2(self.conv2(x)))
        
        # Flattening the tensor: 20x4x4 -> 320
        x = x.view(-1, 20 * 4 * 4)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc2(x), dim=1)
        
        return x

# Function to visualize the model with torchviz
def visualize_model(model, file_name="model_visualization.png"):
    """Visualizes the CNN model architecture using torchviz"""
    sample_input = torch.randn(1, 1, 28, 28)
    
    # Forward pass to create the computational graph
    output = model(sample_input)
    
    # Create the visualization graph
    dot = make_dot(output, params=dict(model.named_parameters()))
    
    # Save as a .png file
    dot.format = 'png'
    dot.render(file_name)
    print(f"Model visualization saved as {file_name}")

# Main execution
if __name__ == "__main__":
    model = MyNetwork(dropout_rate=0.5)
    
    # Visualize the model
    visualize_model(model, "mnist_cnn_model.png")
