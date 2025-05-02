import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# Define the network architecture (same as the training script)
class LeNet(nn.Module):
    """
    Creates a LenNet based architecture to
    classify digits in Mnist dataset.
    """
    def __init__(self):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=10,
            kernel_size=5),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

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

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=20 * 4 * 4, out_features=50),
            nn.Linear(50, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return nn.functional.log_softmax(x)
    
# Load the saved model
model = LeNet()
model.load_state_dict(torch.load('./results/model.pth'))
model.eval()  # Set model to evaluation mode

# Load the MNIST test dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # Match the training normalization!
])

test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=10, shuffle=False)

# Get the first 10 images and labels
images, labels = next(iter(test_loader))

# Run the model on the first 10 examples
with torch.no_grad():
    outputs = model(images)

# Print results for each image
print("\nModel predictions for the first 10 test images:")
print("=" * 60)

for i in range(10):
    output_values = outputs[i].numpy()
    output_rounded = [f"{x:.2f}" for x in output_values]  # Round to 2 decimal places
    pred_index = output_values.argmax()
    correct_label = labels[i].item()

    print(f"Image {i+1}:")
    print(f"  Output values: {output_rounded}")
    print(f"  Predicted: {pred_index} | Correct: {correct_label}")
    print("-" * 60)

# Plot the first 9 digits in a 3x3 grid with predictions above
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
fig.suptitle("First 9 MNIST Test Samples with Predictions", fontsize=16)

for i, ax in enumerate(axes.flat):
    if i >= 9:
        break
    
    image = images[i][0].numpy()
    pred_label = outputs[i].argmax().item()
    
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Pred: {pred_label}')
    ax.axis('off')

plt.tight_layout()
plt.show()
