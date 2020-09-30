import logging
import argparse
import sys
import asyncio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import syft as sy
from syft.workers import websocket_client
from syft.frameworks.torch.fl import utils
from syft.workers.websocket_client import WebsocketClientWorker
import logging
import importlib
from run.training.serializable_loss import *
LOG_INTERVAL = 25
logger = logging.getLogger("run_websocket_client")



#Instance the clients
def instance(args,kwargs_websocket):
    """ instance the workers  
    args:
        args: the arguments
        kwargs_websocket: the dict of host and hook 
    Return:
        worker_instances: the list of the workers instances 
    """
    clients = []
    for i in range(args.clients):
        clients.append(WebsocketClientWorker(id=str(i), port=args.client_port+i, **kwargs_websocket))
    worker_instances = [client for client in clients]
    for worker in worker_instances:
        print("Client: " ,worker)
    return worker_instances


async def fit_model_on_worker(
    args,
    worker: websocket_client.WebsocketClientWorker,
    traced_model: torch.jit.ScriptModule,
    curr_round: int,
    lr: float,


):
    train_config = sy.TrainConfig(model=traced_model,
                              loss_fn=get_serializable_loss(args.loss),
                              optimizer=args.optimizer,
                              batch_size=args.batch_size,
                              optimizer_args={"lr": lr},
                              epochs=args.federate_after_n_batches,
                              shuffle=True,
                              max_nr_batches=args.max_nr_batches
                              )
    
    train_config.send(worker)
    loss = await worker.async_fit(dataset_key=args.dataset, return_ids=[0])
    model = train_config.model_ptr.get().obj

    
    return worker.id, model,loss



