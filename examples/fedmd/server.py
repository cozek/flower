import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl
import numpy as np


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    """
    config = {
        "server_round":server_round
    }
    return config



if __name__ == "__main__":
    # trainset, testset = utils.load_partition(0)

    # trainLoader = DataLoader(trainset, batch_size=32, shuffle=True)

      # Create strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        min_fit_clients=2,
        min_evaluate_clients=2,
        min_available_clients=3,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(np.array([0])),
    )

    
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
    )

    # for imgs, labels in trainLoader:
    #     print(imgs.shape)
    #     print(labels.shape)
