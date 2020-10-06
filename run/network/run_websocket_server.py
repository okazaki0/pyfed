import warnings

warnings.filterwarnings("ignore")
import logging
import argparse
import numpy as np
import torch
from torchvision import datasets
from torchvision import transforms
import syft as sy

# from syft.workers import websocket_server
from syft.workers.websocket_server import WebsocketServerWorker
from random import sample
import pickle
import sys

sys.path.append("./")
import utils.arguments as arguments
from utils.func import *


def start_websocket_server_worker(
    id, host, port, hook, verbose, dataset, training=True
):
    """Helper function for spinning up a websocket server and setting up the local datasets."""

    server = WebsocketServerWorker(
        id=id, host=host, port=port, hook=hook, verbose=verbose
    )
    dataset_key = dataset
    # if we are in the traning loop
    if training:
        with open("./data/split/%d" % int(id), "rb") as fp:  # Unpickling
            data = pickle.load(fp)
        dataset_data, dataset_target = readnpy(data)
        print(type(dataset_data.long()))
        logger.info("Number of samples for client %s is %s : ", id, len(dataset_data))
        dataset = sy.BaseDataset(data=dataset_data, targets=dataset_target)
        key = dataset_key

    nb_labels = len(torch.unique(dataset_target))
    server.add_dataset(dataset, key=key)
    count = [0] * nb_labels
    logger.info("Dataset(train set) ,available numbers on %s: ", id)
    for i in range(nb_labels):
        count[i] = (dataset.targets == i).sum().item()
        logger.info("      %s: %s", i, count[i])
    logger.info("datasets: %s", server.datasets)
    if training:
        logger.info("len(datasets): %s", len(server.datasets[key]))

    server.start()
    return server


if __name__ == "__main__":
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_server")
    logger.setLevel(level=logging.DEBUG)
    parser = argparse.ArgumentParser(description="parameters.")

    parser.add_argument(
        "--port",
        "-p",
        type=int,
        help="port number of the websocket server worker, e.g. --port 8777",
    )
    parser.add_argument(
        "--host", type=str, default="localhost", help="host for the connection"
    )
    parser.add_argument(
        "--id",
        type=str,
        help="name (id) of the websocket server worker, e.g. --id alice",
    )
    parser.add_argument(
        "--testing",
        action="store_true",
        help="if set, websocket server worker will load the test dataset instead of the training dataset",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="if set, websocket server worker will be started in verbose mode",
    )
    parser.add_argument(
        "--dataset",
        help="dataset key",
    )

    args = parser.parse_args()
    # Hook and start server
    hook = sy.TorchHook(torch)
    server = start_websocket_server_worker(
        id=args.id,
        host=args.host,
        port=args.port,
        hook=hook,
        verbose=args.verbose,
        dataset=args.dataset,
        training=not args.testing,
    )
