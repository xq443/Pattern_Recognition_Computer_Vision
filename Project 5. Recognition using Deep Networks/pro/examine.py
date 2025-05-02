#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the network architecture (same as Task 1)
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

def analyze_network():
    # Load the trained network
    network = MyNetwork()
    network.load_state_dict(torch.load('./results/model.pth'))
    network.eval()  # Set the model to evaluation mode

    # Print the structure of the model
    print("Model Architecture:\n")
    print(network)

    # Access the weights of the first convolutional layer (conv1)
    conv1_weights = network.conv1.weight.data

    # Print the shape of the weights
    print(f"\nShape of conv1 weights: {conv1_weights.shape}")  # [10, 1, 5, 5]

    # Print the individual filters (weights) for conv1
    for i in range(conv1_weights.shape[0]):
        print(f"\nFilter {i} weights:\n{conv1_weights[i, 0]}")

    # Visualize the 10 filters using a color colormap (viridis)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))

    # Loop through the 10 filters and plot them
    for i in range(10):
        ax = axes[i // 4, i % 4]
        ax.imshow(conv1_weights[i, 0].cpu().numpy(), cmap='viridis')  # Use 'viridis' colormap for color
        ax.axis('off')
        ax.set_title(f'Filter {i+1}')

    # Remove the empty subplot in the 3x4 grid
    axes[2, 2].axis('off')
    axes[2, 3].axis('off')

    plt.show()

def main():
    # Call the analyze_network function
    analyze_network()

if __name__ == "__main__":
    main()
