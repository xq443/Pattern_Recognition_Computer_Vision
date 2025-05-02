
from dataloader import create_dataloaders, get_greekdata, get_FashionMnist
from models import LeNet, tinyVgg, fashionModel
import torch
from torch import nn
from tqdm.auto import tqdm
from train_prep import train_step, test_step
from helpers import save_model, save_results, plot_loss_curves, load_model, plot_six_images
import pandas as pd
import matplotlib.pyplot as plt
import sys
from sklearn.model_selection import GridSearchCV

# A function that trains the network


def train_network(model: torch.nn.Module,
                  train_dataloader: torch.utils.data.DataLoader,
                  test_dataloader: torch.utils.data.DataLoader,
                  optimizer: torch.optim.Optimizer,
                  loss_fn: torch.nn.Module,
                  epochs: int,
                  device: torch.device = "mps"):
    """
    trains the model and saves the model into local directory.

    Args:
        model: Model to use.
        train_dataloader: train_dataloader object with training data.
        test_dataloader: test_dataloader object with testing data.
        optiimizer: Type of optimization to use.
        loss: loss function to use for training the model.
        epochs: Number of epochs to train the model.
        device: device to use for computing.

    Returns: 
        A dictionary of training and testing losses and accuracies.
    """

    # Create a results dictionary.
    results = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_dataloader,
                                           loss_fn=loss_fn,
                                           optimizer=optimizer)

        test_loss, test_acc = test_step(model=model,
                                        dataloader=test_dataloader,
                                        loss_fn=loss_fn)
        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f} | "
            f"test_loss: {test_loss:.4f} | "
            f"test_acc: {test_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)

    return results


def fine_tune(train_data: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              output_layers: int,
              device: torch.device = "mps"):
    """
    A function that takes the train_data, an existing model
    and fine_tunes the model to train with new data.

    Args:
        train_data: Train data to use for fine-tuning.
        model: model to finetune.
        device: target device to use.
    """

    # Dictionary to store results.
    results = {
        "train_acc": [],
        "train_loss": []
    }
    # Freeze all the layers.
    for name, param in model.named_parameters():
        param.requires_grad = False

    # Fine-tune the last layer.
    model.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=20 * 4 * 4, out_features=50),
        nn.Linear(50, output_layers)
    ).to(device=device)

    # change to model to target device.
    model.to(device=device)
    loss = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
    for epoch in range(15):
        train_loss, train_acc = train_step(model=model,
                                           dataloader=train_data,
                                           loss_fn=loss,
                                           optimizer=optim)

        # Print out what's happening
        print(
            f"Epoch: {epoch+1} | "
            f"train_loss: {train_loss:.4f} | "
            f"train_acc: {train_acc:.4f}"
        )

        # Update results dictionary
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)

    return results


# A function that reads cmd arguements and trains the model.
def main(argv):
    batch_size = int(argv[1])
    epochs = int(argv[2])
    digit_train_mode = int(argv[3])
    fashion_train_mode = int(argv[4])
    fashion_train_vgg = int(argv[5])
    test_mode = int(argv[6])
    greek_train = int(argv[7])
    output_layers = int(argv[8])
    gabor_train = int(argv[9])

    digit_train_dataloader, digit_test_dataloader, digit_class_names = create_dataloaders(
        batch_size=batch_size)

    plot_six_images(digit_train_dataloader)

    fashion_train_dataloader, fashion_test_dataloader, fashion_class_names = get_FashionMnist(
        batch_size=batch_size)

    if digit_train_mode == 1:
        # Initialize the model
        model = LeNet().to("mps")
        loss = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(params=model.parameters(), lr=0.1)
        torch.manual_seed(42)
        results = train_network(model=model,
                                train_dataloader=digit_train_dataloader,
                                test_dataloader=digit_test_dataloader,
                                optimizer=optim,
                                loss_fn=loss,
                                epochs=epochs)

        print(results)
        # Save the model to disk.
        save_model(model, "Models", "base_model.pth")
        # Save the results of model in a text file.
        save_results(results, "Models")

    if fashion_train_mode == 1:
        if fashion_train_vgg == 1:
            torch.manual_seed(42)
            model2 = tinyVgg(input_shape=1, hidden_units=10,
                             output_shape=len(fashion_class_names)).to("mps")
            loss = nn.CrossEntropyLoss()
            optim = torch.optim.SGD(params=model2.parameters(), lr=0.1)
            results = train_network(model=model2,
                                    train_dataloader=fashion_train_dataloader,
                                    test_dataloader=fashion_test_dataloader,
                                    optimizer=optim,
                                    loss_fn=loss,
                                    epochs=epochs)
            print(results)
            # Save the model to disk.
            save_model(model2, "Models", "fashion_model.pth")
            # Save the results of model in a text file.
            save_results(results, "Models")

        else:
            # Train with different kernels.
            kernels = [5]
            for kernel in kernels:
                model2 = fashionModel(out_channels=10, kernel_size=kernel,
                                      droput_rate=0.5, hidden_neurons=50).to("mps")
                loss = nn.CrossEntropyLoss()
                optim = torch.optim.SGD(params=model2.parameters(), lr=0.1)
                results = train_network(model=model2,
                                        train_dataloader=fashion_train_dataloader,
                                        test_dataloader=fashion_test_dataloader,
                                        optimizer=optim,
                                        loss_fn=loss,
                                        epochs=epochs)
                print(f"Results with {kernel} kernels is {results}")

            channels = [5, 7, 9, 11, 13, 15, 17,
                        19, 21, 24, 25, 25, 28, 31, 33, 40]
            for channel in channels:
                model2 = fashionModel(out_channels=channel, kernel_size=5,
                                      droput_rate=0.5, hidden_neurons=50).to("mps")
                loss = nn.CrossEntropyLoss()
                optim = torch.optim.SGD(params=model2.parameters(), lr=0.1)
                results = train_network(model=model2,
                                        train_dataloader=fashion_train_dataloader,
                                        test_dataloader=fashion_test_dataloader,
                                        optimizer=optim,
                                        loss_fn=loss,
                                        epochs=epochs)
                print(f"Results with {channel} channels is {results}")

            # Train with different dropouts.
            dropouts = [0.2, 0.25, 0.3, 0.35, 0.4,
                        0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75]
            for dropout in dropouts:
                model2 = fashionModel(out_channels=10, kernel_size=5,
                                      droput_rate=dropout, hidden_neurons=50).to("mps")
                loss = nn.CrossEntropyLoss()
                optim = torch.optim.SGD(params=model2.parameters(), lr=0.1)
                results = train_network(model=model2,
                                        train_dataloader=fashion_train_dataloader,
                                        test_dataloader=fashion_test_dataloader,
                                        optimizer=optim,
                                        loss_fn=loss,
                                        epochs=epochs)
                print(f"Results with {dropout} dropout is {results}")

            # Train with different hidden_neurons.
            hidden_neurons = [20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130,
                              140, 150, 160, 170, 180, 190, 200, 210, 220]
            for hidden_neuron in hidden_neurons:
                model2 = fashionModel(out_channels=10, kernel_size=5,
                                      droput_rate=0.5, hidden_neurons=hidden_neuron).to("mps")
                loss = nn.CrossEntropyLoss()
                optim = torch.optim.SGD(params=model2.parameters(), lr=0.1)
                results = train_network(model=model2,
                                        train_dataloader=fashion_train_dataloader,
                                        test_dataloader=fashion_test_dataloader,
                                        optimizer=optim,
                                        loss_fn=loss,
                                        epochs=epochs)
                print(
                    f"Results with {hidden_neuron} hidden neurons is {results}")

    if test_mode == 1:
        # Analyze and plot the results above training mode.
        df1 = pd.read_csv("Models/results2.csv")
        df1.drop("Unnamed: 0", axis=1, inplace=True)
        my_dict = df1.to_dict('list')
        plot_loss_curves(my_dict)
        plt.show()

    if greek_train == 1:
        model_path = "Models/base_model.pth"
        model = LeNet()
        trained_model = load_model(target_dir=model_path,
                                   model=model,
                                   device=torch.device("mps"))

        greek_train, class_names = get_greekdata(5)
        results = fine_tune(greek_train, trained_model,
                            output_layers=output_layers)

        # Save the model to disk.
        save_model(model, "Models", "greek_model.pth")
        # Save the results of model into a text file.
        save_results(results, "Models")

    if gabor_train == 1:
        model_path = "Models/base_model.pth"
        model = LeNet()
        trained_model = load_model(target_dir=model_path,
                                   model=model,
                                   device=torch.device("mps")).to("mps")

        trained_model.conv1 = GaborConv2d(
            in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=2).to("mps")
        for name, param in trained_model.named_parameters():
            if name == "conv1.weight":
                param.requires_grad = False
        loss = nn.CrossEntropyLoss()
        optim = torch.optim.SGD(params=trained_model.parameters(), lr=0.1)
        results = train_network(model=trained_model,
                                train_dataloader=fashion_train_dataloader,
                                test_dataloader=fashion_test_dataloader,
                                optimizer=optim,
                                loss_fn=loss,
                                epochs=epochs)

        # Save the model to disk.
        save_model(trained_model, "Models", "gabor_model.pth")
        # Save the results of model into a text file.
        save_results(results, "Models")

# A class that creates a gabor filter.


class GaborConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(GaborConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # Define Gabor filter bank parameters
        self.sigma = nn.Parameter(torch.randn(out_channels, in_channels))
        self.theta = nn.Parameter(torch.randn(out_channels, in_channels))
        self.lambda_ = nn.Parameter(torch.randn(out_channels, in_channels))
        self.psi = nn.Parameter(torch.randn(out_channels, in_channels))
        self.gamma = nn.Parameter(torch.randn(out_channels, in_channels))

        # Initialize filter bank weights
        self.weight = nn.Parameter(torch.randn(
            out_channels, in_channels, kernel_size, kernel_size))

    def forward(self, x):
        # Apply Gabor filter bank to input image
        filters = []
        for i in range(self.out_channels):
            real, imag = F.gabor_filter(x, self.kernel_size, self.sigma[i], self.theta[i],
                                        self.lambda_[i], self.psi[i], self.gamma[i])
            filters.append(torch.stack([real, imag], dim=1))
        filters = torch.cat(filters, dim=0)

        # Apply convolution operation to filtered image
        output = F.conv2d(filters, self.weight,
                          stride=self.stride, padding=self.padding)
        return output


if __name__ == "__main__":
    main(sys.argv)


if __name__ == "__main__":
    main(sys.argv)