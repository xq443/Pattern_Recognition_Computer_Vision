#   Xujia Qin 
#   30th Mar, 2025
#   S21
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np

# Load pretrained MNIST model (from Task 1)
class MyNetwork(nn.Module):
    def __init__(self):
        super(MyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 6)  # Changed to 6 classes

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# Greek letter transform (updated for 6 classes)
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)

def plot_test_predictions(model, test_loader, class_names):
    model.eval()
    samples_per_class = 2  # 2 samples per class for 6 classes
    collected_images = []
    collected_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                if len(collected_images) >= len(class_names) * samples_per_class:
                    break
                collected_images.append(img)
                collected_labels.append(lbl)
    
    plt.figure(figsize=(12, 8))
    for i in range(len(collected_images)):
        ax = plt.subplot(3, 4, i+1)
        image = collected_images[i].squeeze()
        ax.imshow(image, cmap='gray')
        
        output = model(collected_images[i].unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        true_label = class_names[collected_labels[i].item()]
        pred_label = class_names[predicted.item()]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')
    plt.tight_layout()
    plt.savefig('greek_predictions_extended.png')
    plt.show()

def main():
    model = MyNetwork()
    model.load_state_dict(torch.load('./results/model.pth'))
    
    # Freeze all layers except last
    for param in model.parameters():
        param.requires_grad = False
    model.fc2 = nn.Linear(50, 6)  # 6 classes now
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./greek_train_extended', transform=transform),
        batch_size=6, shuffle=True  # Increased batch size
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./greek_test_extended', transform=transform),
        batch_size=6, shuffle=False
    )

    # Training setup
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    criterion = nn.CrossEntropyLoss()
    
    best_accuracy = 0
    for epoch in range(20):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        for images, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            total_loss += loss.item()
        
        train_accuracy = 100 * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={train_accuracy:.2f}%')
        
        # Early stopping if perfect accuracy
        if train_accuracy >= 99.5:
            break
        scheduler.step()
    
    # Evaluation
    model.eval()
    test_correct = 0
    test_total = 0
    class_correct = [0] * 6
    class_total = [0] * 6
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            c = (predicted == labels).squeeze()
            for i in range(len(labels)):
                label = labels[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1
    
    print('\nTest Accuracy per Class:')
    for i in range(6):
        print(f'{test_loader.dataset.classes[i]:5s}: {100*class_correct[i]/class_total[i]:.2f}%')
    
    total_accuracy = 100 * sum(class_correct) / sum(class_total)
    print(f'\nOverall Test Accuracy: {total_accuracy:.2f}%')
    
    # Visualize predictions
    plot_test_predictions(model, test_loader, test_loader.dataset.classes)

if __name__ == '__main__':
    main()