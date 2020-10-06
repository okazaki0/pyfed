from torchvision import datasets, transforms
import torch
import random
import numpy as np
import torch.utils.data as Data
import sys

sys.path.append("./")
from datasets.mnist.preprocess import *
from datasets.cifar10.preprocess import *
from datasets.fashionmnist.preprocess import *
from datasets.sent140.preprocess import *
from datasets.shakespeare.preprocess import *


# load data
def load_data(args):
    """The load function of the dataset

    Args:
       args: to get the global dataset argument.

    Returns:
        train_loader, test_loader, global_loader: the data loaders.
    """
    if args.dataset == "cifar10":
        train_loader, test_loader, global_loader = load_cifar10(args)

    elif args.dataset == "mnist":
        train_loader, test_loader, global_loader = load_mnist(args)

    elif args.dataset == "fashionmnist":
        train_loader, test_loader, global_loader = load_fashionmnist(args)

    elif args.dataset == "sent140":
        train_loader, test_loader, global_loader = load_sent140(args)

    elif args.dataset == "shakespeare":
        train_loader, test_loader, global_loader = load_shakespeare(args)

    # eval_dataset = eval_data(test_loader)
    return train_loader, test_loader, global_loader


# Split function
def splitDataset(args, train_loader, global_loader=None):
    """The split function

    Args:
       args: the arguments.
       train_loader,global_loader: the data loaders
    Returns:
        sub_datasets: the subset of data of each worker.
    """
    sub_datasets = [[] for i in range(args.clients)]

    if args.dataset == "cifar10":
        sub_datasets = split_cifar10(args, train_loader, global_loader)

    elif args.dataset == "mnist":
        sub_datasets = split_mnist(args, train_loader, global_loader)

    elif args.dataset == "fashionmnist":
        sub_datasets = split_fashionmnist(args, train_loader, global_loader)

    elif args.dataset == "sent140":
        sub_datasets = split_sent140(args, train_loader, global_loader)

    elif args.dataset == "shakespeare":
        sub_datasets = split_shakespeare(args, train_loader, global_loader)

    return sub_datasets
