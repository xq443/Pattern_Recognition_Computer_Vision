#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import torch.nn.functional as F

class MNISTNet(nn.Module):
    def __init__(self):
        super(MNISTNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(p=0.5)
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training, p=0.5)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def visualize_predictions(model, test_loader, class_names):
    model.eval()
    data_iter = iter(test_loader)
    images, labels = next(data_iter)
    
    plt.figure(figsize=(10, 8))
    for i in range(min(9, len(images))):  # Show first 9 samples
        image = images[i].unsqueeze(0)
        with torch.no_grad():
            output = model(image)
            probs = torch.exp(output)  # Convert log-softmax to probabilities
            predicted_idx = output.argmax().item()
        
        # Create subplot
        ax = plt.subplot(3, 3, i+1)
        ax.imshow(images[i].squeeze(), cmap='gray')
        
        # Display prediction info
        true_label = class_names[labels[i].item()]
        pred_label = class_names[predicted_idx]
        confidence = probs[0][predicted_idx].item() * 100
        
        ax.set_title(f"True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)", fontsize=9)
        ax.axis('off')
        
        # Print detailed output for each sample
        print(f"Sample {i+1}:")
        print(f"True class: {true_label}")
        print("Prediction probabilities:")
        for j, prob in enumerate(probs[0]):
            print(f"{class_names[j]}: {prob.item()*100:.1f}%")
        print("-"*40)
    
    plt.tight_layout()
    plt.savefig('greek_predictions.png')
    plt.show()

def main():
    # Initialize model
    model = MNISTNet()
    model.load_state_dict(torch.load('./results/model.pth'))
    
    # Modify for Greek letters
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, 3)  # alpha, beta, gamma
    
    # Data pipeline
    transform = transforms.Compose([
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.ImageFolder('./greek_train', transform=transform)
    test_dataset = datasets.ImageFolder('./greek_test', transform=transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=3, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=3, shuffle=False, num_workers=2)
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.001, momentum=0.9)
    
    # Train model
    best_accuracy = 0
    for epoch in range(1, 21):
        model.train()
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # Evaluate
        current_acc = evaluate(model, test_loader)
        if current_acc > best_accuracy:
            best_accuracy = current_acc
            torch.save(model.state_dict(), 'best_greek_model.pth')
            print(f"New best model saved at epoch {epoch} with accuracy {best_accuracy:.2f}%")
    
    # Load best model and visualize predictions
    model.load_state_dict(torch.load('best_greek_model.pth'))
    visualize_predictions(model, test_loader, train_dataset.classes)

def evaluate(model, test_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total

if __name__ == '__main__':
    Path('./results').mkdir(exist_ok=True)
    main()