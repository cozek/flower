from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from torch.utils.data import DataLoader


class FlowerClient(fl.client.NumPyClient):
    def __init__(
            self, 
            private_data: Tuple, 
            public_data: Tuple, 
            net: nn.Module
        ) -> None:
        super().__init__()
        self.private_train, self.private_test = private_data
        self.public_train, self.public_test = public_data
        self.net = net

        self.private_trainloader = DataLoader(
            self.private_train, batch_size=32, shuffle=False
        )
        self.private_testloader = DataLoader(
            self.private_test, batch_size=32, shuffle=False
        )

        self.public_trainloader = DataLoader(
            self.public_train, batch_size=32, shuffle=True
        )
        self.public_testloader = DataLoader(
            self.public_test, batch_size=32, shuffle=True
        )
        self.intial_training(10)


    def intial_training(self, epochs: int):
        # TransferLearning: Train to convergence on public data and private data
        res_public = utils.train(self.net, self.public_trainloader, epochs)
        res_private = utils.train(self.net, self.private_trainloader, epochs)
        print(f'Result Public: {res_public}')
        print(f'Result Private: {res_private}')

    def get_class_scores(self, config, device:str ='cpu'):
        # calculate class scores on the public dataset
        scores = []
        with torch.no_grad():
            for images, labels in self.public_trainloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                scores.append(outputs.numpy())
        return scores

    def digest(self, consensus, epochs, device='cpu'):
        # Train private model to approach concensus f on public dataset
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            net.parameters(), lr=0.1, momentum=0.9, weight_decay = 1e-4
        )
        net.train()
        for _ in range(epochs):
            for images, _ in zip(self.public_trainloader, consensus):
                images = images.to(device)
                optimizer.zero_grad()
                loss = criterion(net(images), consensus)
                loss.backward()
                optimizer.step()
        net.to('cpu')

    def revisit(self, epochs:int):
        results = utils.train(self.net, self.public_trainloader, epochs)
        return results

    def get_parameters(self, config: Dict[str, Scalar]) -> NDArrays:
        # Communicate: return class scores on the public dataset
        # no parameters are returned 
        return self.get_class_scores(config)

    def fit(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[NDArrays, int, Dict[str, Scalar]]:
        self.digest(parameters, 10, device='cpu')
        results = self.revisit(2)
        scores = self.get_class_scores(config)
        parameters = scores
        num_examples_train = len(self.private_train)
        return parameters, num_examples_train, results

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        loss, accuracy = utils.test(net, self.private_testloader, None)
        return float(loss), len(self.private_test), {"accuracy": float(accuracy)}


if __name__ == "__main__":
    public_data: Tuple = utils.load_partition(0) # let partition 0 be public dataset
    private_data: Tuple = utils.load_partition(1)
    net = utils.Net()

    private_trainloader = DataLoader(
            private_data[0], batch_size=32, shuffle=False
        )
    for imgs,labels in private_trainloader:
        output = net(imgs)

    print(labels.shape)
    print(labels)
    print(output)
    print(output.shape)

    # client = FlowerClient(public_data, private_data, net)

    

