from collections import OrderedDict
import warnings

import flwr as fl
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import CIFAR10

import utils

warnings.filterwarnings("ignore", category=UserWarning)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# #############################################################################
# 1. PyTorch pipeline: model/train/test/dataloader
# #############################################################################

# Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ConvNet(nn.Module):
    def __init__(self) -> None:
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), stride=1
        )
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=1
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(5 * 5 * 64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = self.dropout(x)
        x = F.softmax(self.fc(x))
        return x


def train(net, trainloader, epochs):
    """Train the network on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for images, labels in trainloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(net(images), labels)
            loss.backward()
            optimizer.step()


def test(net, testloader):
    """Validate the network on the entire test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, total, loss = 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for images, labels in testloader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    loss /= len(testloader.dataset)
    accuracy = correct / total
    return loss, accuracy


def load_data():
    """Load CIFAR-10 (training and test set)."""
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )
    trainset = CIFAR10("./dataset", train=True, download=True, transform=transform)
    testset = CIFAR10("./dataset", train=False, download=True, transform=transform)
    trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
    testloader = DataLoader(testset, batch_size=32)
    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainloader, testloader, num_examples


class MyDataset(Dataset):
    """Create a Torch Dataset
    Taken from:
    stackoverflow.com/questions/44429199/how-to-load-a-list-of-numpy-arrays-to-pytorch-dataset-loader
    """

    def __init__(self, data: np.ndarray, targets: np.ndarray, transform=None):
        self.data = data
        self.targets = torch.LongTensor(targets)
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.targets[index]

        if self.transform:
            # insert your transforms here
            pass
        return x, y

    def __len__(self):
        return len(self.data)


# #############################################################################
# 2. Federation of the pipeline with Flower
# #############################################################################


def _main():
    """Create model, load data, define Flower client, start Flower client."""

    # Load model
    net = ConvNet().to(DEVICE)

    # Load data (MNIST)
    trainloader, testloader, num_examples = load_data()
    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (X_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]

    # Flower client
    class CifarClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}

    # Start client
    fl.client.start_numpy_client("[::]:8080", client=CifarClient())


def main():
    # Load MNIST dataset from https://www.openml.org/d/554
    (X_train, y_train), (X_test, y_test) = utils.load_mnist()

    # Split train set into 10 partitions and randomly use one for training.
    partition_id = np.random.choice(10)
    (x_train, y_train) = utils.partition(X_train, y_train, 10)[partition_id]
    (x_test, y_test) = utils.partition(X_test, y_test, 10)[partition_id]

    # reshape for convolution
    x_train = x_train.reshape(-1, 1, 28, 28)
    x_test = x_test.reshape(-1, 1, 28, 28)

    trainDataset = MyDataset(x_train, y_train)
    testDataset = MyDataset(x_train, y_train)

    trainloader = DataLoader(trainDataset, batch_size=32)
    testloader = DataLoader(testDataset, batch_size=32)

    net = ConvNet().to(DEVICE)

    train(net, trainloader, epochs=1)
    loss, accuracy = test(net, testloader)

    # Flower client
    class TorchClient(fl.client.NumPyClient):
        def get_parameters(self):
            return [val.cpu().numpy() for _, val in net.state_dict().items()]

        def set_parameters(self, parameters):
            params_dict = zip(net.state_dict().keys(), parameters)
            state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
            net.load_state_dict(state_dict, strict=True)

        def fit(self, parameters, config):
            self.set_parameters(parameters)
            train(net, trainloader, epochs=1)
            return self.get_parameters(), num_examples["trainset"], {}

        def evaluate(self, parameters, config):
            self.set_parameters(parameters)
            loss, accuracy = test(net, testloader)
            return float(loss), num_examples["testset"], {"accuracy": float(accuracy)}


if __name__ == "__main__":
    main()
