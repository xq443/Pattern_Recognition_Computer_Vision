#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Define the network class (same as before)
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

# Prepare the MNIST test data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

test_loader = DataLoader(test_data, batch_size=10, shuffle=False)

# Load the pre-trained model and optimizer
network = MyNetwork()
network.load_state_dict(torch.load('./results/model.pth'))
network.eval()  # Set the model to evaluation mode

# Function to print results and plot predictions
def run_model_on_samples():
    # Get the first 10 samples from the test set
    data_iter = iter(test_loader)
    images, labels = next(data_iter)

    # Run the model on each sample
    for i in range(10):
        image = images[i].unsqueeze(0)  # Add batch dimension
        label = labels[i]
        
        # Get model prediction
        output = network(image)
        predicted_label = output.argmax(dim=1, keepdim=True).item()
        predicted_values = output[0].detach().numpy()

        # Print the 10 output values, the max index, and the correct label
        print(f"Example {i+1}:")
        print(f"Network Output Values: {', '.join([f'{x:.2f}' for x in predicted_values])}")
        print(f"Predicted Index: {predicted_label}, Correct Label: {label.item()}")
        print("")

        # Create a plot for the first 9 examples in a 3x3 grid
        if i < 9:
            ax = plt.subplot(3, 3, i+1)
            ax.imshow(images[i].squeeze(), cmap='gray')
            ax.set_title(f"Pred: {predicted_label}, True: {label.item()}")
            ax.axis('off')

    # Show the 3x3 grid plot of the first 9 images
    plt.tight_layout()
    plt.show()

# Run the model on the first 10 examples and display the results
run_model_on_samples()
