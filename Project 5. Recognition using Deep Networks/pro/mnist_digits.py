#   Xujia Qin 
#   28th Mar, 2025
#   S21
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Load the MNIST test dataset
def load_mnist_test():
    """Load the MNIST test dataset without shuffling"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    test_set = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transform
    )
    
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=6, shuffle=False)
    return test_loader

# Display and save the first six digits
def display_and_save_digits(output_path="mnist_test_digits.jpg"):
    """Display and save the first six digits from the MNIST test set"""
    test_loader = load_mnist_test()
    
    images, labels = next(iter(test_loader))
    
    fig, axes = plt.subplots(1, 6, figsize=(12, 4))
    
    for i in range(6):
        axes[i].imshow(images[i].squeeze(), cmap='gray')
        axes[i].set_title(f'Label: {labels[i].item()}')
        axes[i].axis('off')
    
    plt.tight_layout()

    # Save the figure as a .jpg file
    plt.savefig(output_path, format='jpg')
    print(f"Plot saved as {output_path}")

    plt.close()  # Close the plot to end the program

# Main execution
if __name__ == "__main__":
    display_and_save_digits("mnist_test_digits.jpg")
