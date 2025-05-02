#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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

# Load the trained network (make sure the model and optimizer are loaded as before)
network = MyNetwork()
network.load_state_dict(torch.load('./results/model.pth'))
network.eval()  # Set the model to evaluation mode

# Preprocess the image for input to the network
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.Resize((28, 28)),  # Resize to 28x28
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Match MNIST intensity scaling
])

def predict_digit(image_path):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Add batch dimension
    output = network(image)
    prediction = output.argmax(dim=1, keepdim=True)
    return prediction.item()

# Test the handwritten digits
image_paths = ['0.png','1.png','2.png','3.png', '4.png','5.png', '6.png', '7.png', '8.png', '9.png']
predictions = []

# Plot the images and predictions
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, image_path in enumerate(image_paths):
    ax = axes[i // 5, i % 5]
    image = Image.open(image_path)
    ax.imshow(image, cmap='gray')
    ax.axis('off')
    pred = predict_digit(image_path)
    predictions.append(pred)
    ax.set_title(f'Pred: {pred}')

plt.show()
# Evaluate performance and display results
correct_predictions = 0
incorrect_predictions = []

for i, image_path in enumerate(image_paths):
    pred = predictions[i]
    # Here you could add the true label to compare against for more detailed analysis
    true_label = int(image_path.split('.')[0])  # Assumes the filename is the digit (e.g., '0.png')
    if pred == true_label:
        correct_predictions += 1
    else:
        incorrect_predictions.append((image_path, true_label, pred))

# Report the results
accuracy = correct_predictions / len(image_paths) * 100
print(f"Accuracy: {accuracy}%")

# Display the incorrectly classified images
if incorrect_predictions:
    print("\nIncorrectly classified images:")
    for image_path, true_label, pred in incorrect_predictions:
        print(f"Image: {image_path} | True: {true_label} | Pred: {pred}")
else:
    print("All images classified correctly!")