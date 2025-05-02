#   Xujia Qin 
#   29th Mar, 2025
#   S21
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.models import resnet18, ResNet18_Weights
from torchviz import make_dot
import matplotlib.pyplot as plt

def analyze_pretrained_network():
    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()  # Set to evaluation mode

    # Print model structure
    print("ResNet-18 Architecture:\n")
    print(model)

    # Access the first convolutional layer (conv1)
    conv1 = model.conv1
    conv1_weights = conv1.weight.data

    # Print weights shape [out_channels, in_channels, kernel_size, kernel_size]
    print(f"\nShape of conv1 weights: {conv1_weights.shape}")  # [64, 3, 7, 7]

    # Visualize first 12 filters (4 per input channel - R,G,B)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Display first 4 filters for each RGB channel
    for i in range(12):
        ax = axes[i // 4, i % 4]
        channel = i % 3  # 0=R, 1=G, 2=B
        filter_idx = i // 3
        
        # Get single 7x7 filter for one channel
        ax.imshow(conv1_weights[filter_idx, channel].cpu().numpy(), cmap='viridis')
        ax.axis('off')
        ax.set_title(f'Filter {filter_idx+1} ({"RGB"[channel]})')

    plt.tight_layout()
    plt.show()

    # Analyze first block's conv layers
    layer1 = model.layer1[0]
    print("\nFirst block conv layers:")
    print(f"conv1: {layer1.conv1.weight.shape}")  # [64, 64, 3, 3]
    print(f"conv2: {layer1.conv2.weight.shape}")  # [64, 64, 3, 3]

def visualize_pretrained_resnet():
    # Load pre-trained ResNet-18
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    
    # Generate a dummy input and visualize computation graph
    dummy_input = torch.randn(1, 3, 224, 224)  # ImageNet input size
    output = model(dummy_input)
    
    # Use torchviz to visualize the architecture
    dot = make_dot(output, params=dict(model.named_parameters()))
    dot.render("resnet18_architecture", format="png")  # Saves as .png file
    
    # Visualize first-layer filters (same as before)
    conv1_weights = model.conv1.weight.data
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    for i in range(12):
        ax = axes[i // 4, i % 4]
        channel = i % 3  # R/G/B
        filter_idx = i // 3
        ax.imshow(conv1_weights[filter_idx, channel].cpu().numpy(), cmap='viridis')
        ax.set_title(f'Filter {filter_idx+1} ({"RGB"[channel]})')
        ax.axis('off')
    plt.tight_layout()
    plt.savefig("resnet18_filters.png")
    plt.show()

def main():
    analyze_pretrained_network()
    visualize_pretrained_resnet()

if __name__ == "__main__":
    main()