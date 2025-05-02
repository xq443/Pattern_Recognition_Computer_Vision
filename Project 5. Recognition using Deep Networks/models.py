import torch
from torch import nn, optim

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


class fashionModel(nn.Module):
    def __init__(self, out_channels: int, kernel_size: int, droput_rate: float, hidden_neurons: int): 
        """
        Model for training Fashion_MNIST model

        Args:
            out_channels: Number of channels to use in each filter.
            kernel_size: Kernel_size in each filter.
            dropout_rate: dropout_rate to use.
            hidden_nuerons: number of hidden_neurons to use in each hidden layer.
        """
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(
            in_channels=1,
            out_channels=out_channels,
            kernel_size=kernel_size),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.block_2 = nn.Sequential(
            nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            ),
            nn.Dropout2d(droput_rate),
            nn.MaxPool2d(kernel_size=2),
            nn.ReLU()
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=out_channels * 4 * 4, out_features=hidden_neurons),
            nn.Linear(hidden_neurons, 10)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.classifier(x)

        return nn.functional.log_softmax(x)

class tinyVgg(nn.Module):
    """
    Creates a tinyVgg type architecture to
    classify images in FashionMnist dataset.
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=hidden_units * 7 * 7, out_features=output_shape)
        )

    def forward(self, x:torch.Tensor):
        x = self.block1(x)
        x = self.block2(x)
        x = self.classifier(x)
        return x


        