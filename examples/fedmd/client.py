from typing import Dict, Tuple
from flwr.common import NDArrays, Scalar
import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import flwr as fl
from torch.utils.data import DataLoader


class FedMDClient(fl.client.NumPyClient):
    def __init__(
            self, 
            private_data: Tuple, 
            public_data: Tuple, 
            net: nn.Module,
            initial_train_epochs: int = 10
        ) -> None:
        '''
            private_data: Tuple[trainset, testset] 
            public_data: Tuple[trainset, testset] 
            net: nn.Module
            initial_train_epochs: int = 10
        '''
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

        self.intial_training_epochs = initial_train_epochs
        # self.intial_training(initial_train_epochs)


    def intial_training(self, epochs: int):
        # TransferLearning: Train to convergence on public data and private data
        res_public = utils.train(self.net, self.public_trainloader, epochs)
        res_private = utils.train(self.net, self.private_trainloader, epochs)
        print(f'Training Result on Public Data: {res_public}')
        print(f'Training Result on Private Data: {res_private}')

        return res_public

    def get_class_scores(self, config, device:str ='cpu'):
        # calculate class scores on the public dataset
        scores = []
        with torch.no_grad():
            for images, labels in self.public_trainloader:
                images, labels = images.to(device), labels.to(device)
                outputs = self.net(images)
                scores.append(outputs.numpy())
        return scores

    def digest(self, consensus, epochs, device='cpu'):
        # Train private model to approach concensus f on public dataset
        criterion = torch.nn.CrossEntropyLoss().to(device)
        optimizer = torch.optim.SGD(
            self.net.parameters(), lr=0.1, momentum=0.9, weight_decay = 1e-4
        )
        self.net.train()
        for _ in range(epochs):
            for sample, con in zip(self.public_trainloader, consensus):
                images,lables = sample
                con = torch.Tensor(con).to(device)
                images = images.to(device)
                optimizer.zero_grad()
                loss = criterion(self.net(images), con)
                loss.backward()
                optimizer.step()
        self.net.to('cpu')

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
        if config['server_round'] == 1:
            results = self.intial_training(self.intial_training_epochs)
        else:
            self.digest(parameters, 10, device='cpu')
            results = self.revisit(2)
        scores = self.get_class_scores(config)
        parameters = scores
        num_examples_train = len(self.private_train)
        return parameters, num_examples_train, results

    def evaluate(
        self, parameters: NDArrays, config: Dict[str, Scalar]
    ) -> Tuple[float, int, Dict[str, Scalar]]:
        loss, accuracy = utils.test(self.net, self.private_testloader, None)
        return float(loss), len(self.private_test), {"accuracy": float(accuracy)}


def test_net():
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

def test_client():
    public_data: Tuple = utils.load_partition(0) # let partition 0 be public dataset
    private_data: Tuple = utils.load_partition(1)
    net = utils.Net()
    initial_train_epochs = 1

    C = FedMDClient(
        private_data=  private_data, 
        public_data= public_data, 
        net = net,
        initial_train_epochs = initial_train_epochs
    )

    scores = C.get_class_scores({})
    print(len(scores))
    print(scores[0].shape)

    res = C.fit(scores,{})
    print(res)

def main():
    public_data: Tuple = utils.load_partition(0) # let partition 0 be public dataset
    private_data: Tuple = utils.load_partition(1)
    net = utils.Net()
    initial_train_epochs = 1

    C = FedMDClient(
        private_data=  private_data, 
        public_data= public_data, 
        net = net,
        initial_train_epochs = initial_train_epochs
    )

    fl.client.start_numpy_client(server_address="0.0.0.0:8080", client=C)

if __name__ == "__main__":
    # test_net()
    # test_client()
    main()


    

    # client = FlowerClient(public_data, private_data, net)

    

