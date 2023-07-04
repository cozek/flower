import utils
from torch.utils.data import DataLoader
import torchvision.datasets
import torch
import flwr as fl


if __name__ == "__main__":
    trainset, testset = utils.load_partition(0)

    trainLoader = DataLoader(trainset, batch_size=32, shuffle=True)

    for imgs, labels in trainLoader:
        print(imgs.shape)
        print(labels.shape)
