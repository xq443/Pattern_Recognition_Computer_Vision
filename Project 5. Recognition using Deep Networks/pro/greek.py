#   Xujia Qin 
#   28th Mar, 2025
#   S21

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Load pretrained MNIST model (from Task 1)
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
    
# Greek letter transform
class GreekTransform:
    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale(x)
        x = torchvision.transforms.functional.affine(x, 0, (0,0), 36/128, 0)
        x = torchvision.transforms.functional.center_crop(x, (28, 28))
        return torchvision.transforms.functional.invert(x)
    
def plot_test_predictions(model, test_loader, class_names):
    model.eval()
    
    # Collect samples from each class
    samples_per_class = 3  # We want 3 samples per class for 9 total (3 classes)
    collected_images = []
    collected_labels = []
    
    with torch.no_grad():
        # Iterate through test data until we get enough samples
        for images, labels in test_loader:
            for img, lbl in zip(images, labels):
                if len(collected_images) >= len(class_names) * samples_per_class:
                    break
                collected_images.append(img)
                collected_labels.append(lbl)
    
    # Create a 3x3 grid
    plt.figure(figsize=(10, 10))
    for i in range(9):
        # Get prediction
        output = model(collected_images[i].unsqueeze(0))
        _, predicted = torch.max(output.data, 1)
        
        # Create subplot
        ax = plt.subplot(3, 3, i+1)
        ax.imshow(collected_images[i].squeeze(), cmap='gray')
        
        # Set title with colored text
        true_label = class_names[collected_labels[i].item()]
        pred_label = class_names[predicted.item()]
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f"True: {true_label}\nPred: {pred_label}", color=color)
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('test_predictions_balanced.png')
    plt.show()



def main():
    # Load pretrained MNIST model
    model = MyNetwork()
    model.load_state_dict(torch.load('./results/model.pth'))
    
    # Freeze all layers
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace last layer for Greek letters (alpha, beta, gamma)
    model.fc2 = nn.Linear(50, 3)
    
    # Data loading
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./greek_train', transform=transform),
        batch_size=3, shuffle=True
    )
    
    test_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder('./greek_test', transform=transform),
        batch_size=3, shuffle=False
    )

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.fc2.parameters(), lr=0.01, momentum=0.5)
    
    # Train until perfect accuracy
    epochs = 0
    max_epochs = 20  # Safety limit
    train_accuracies = []
    
    while True:
        epochs += 1
        model.train()
        running_loss = 0.0
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
            running_loss += loss.item()
        
        train_accuracy = 100 * correct / total
        train_accuracies.append(train_accuracy)
        
        print(f'Epoch {epochs}: Loss = {running_loss/len(train_loader):.4f}, '
              f'Train Accuracy = {train_accuracy:.2f}%')
        
        # Stop if perfect accuracy or max epochs reached
        if train_accuracy >= 99 or epochs >= max_epochs:
            break
    
    # Plot training progress
    plt.plot(range(1, epochs+1), train_accuracies, 'b-o')
    plt.xlabel('Epoch')
    plt.ylabel('Training Accuracy (%)')
    plt.title('Convergence to Perfect Accuracy')
    plt.grid(True)
    plt.savefig('greek_training_progress.png')
    plt.show()
    
    # Evaluate on test set
    model.eval()
    test_correct = 0
    test_total = 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += (predicted == labels).sum().item()
    
    print(f'\nFinal Test Accuracy: {100 * test_correct / test_total:.2f}%')
    print(f'Total Epochs to Perfect Training Accuracy: {epochs}')
    
    # Get class names from the dataset
    class_names = test_loader.dataset.classes
    
    # Plot predictions on first 9 test samples
    plot_test_predictions(model, test_loader, class_names)

if __name__ == '__main__':
    main()