
import warnings
warnings.filterwarnings('ignore')
# Dependencies
import sys
sys.path.append('./')
import asyncio
import syft as sy
from syft.frameworks.torch.fl import utils
import torch
from torchvision import datasets, transforms
import torch.nn.functional as f

import numpy as np
import os
import datetime
import logging
import random
import importlib

import run.training.fit_on_worker as rwc
from utils.early_stopping import EarlyStopping
from utils.func import *
from utils.arguments import trainArgs
from datasets.dataloader  import load_data
from run.training.model import *
from models.metrics.save_results import *
import argparse




# Arguments
args = trainArgs()
_ , test_loader, _ = load_data(args)
es = EarlyStopping(patience=0)
hook = sy.TorchHook(torch)
use_cuda = args.cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
device = torch.device("cuda" if use_cuda else "cpu")
dataset_key = args.dataset

# Configure logging
logger = logging.getLogger("run_websocket_client")
if not len(logger.handlers):
    FORMAT = "%(asctime)s - %(message)s"
    DATE_FMT = "%H:%M:%S"
    formatter = logging.Formatter(FORMAT, DATE_FMT)
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
LOG_LEVEL = logging.DEBUG
logger.setLevel(LOG_LEVEL)


#Instance the clients
kwargs_websocket = {"host": "127.0.0.1", "hook": hook, "verbose": args.verbose}
worker_instances = rwc.instance(args,kwargs_websocket)

#Model
mdl = importlib.import_module("models."+args.dataset+"."+args.model)
model = getattr(mdl, args.model)
model = model().to(device)
print("Model: ",model)
# Making the model serializable
# In order to send the model to the workers we need the model to be serializable, for this we use jit.
model.eval()
traced_model = torch.jit.trace(model, mdl.get_example_input())
#print(args)

#Server data
#X,Y = server_data()

#Starting learning loop
print("Start Training")
loss_scores,accuracy_scores,training_loss = asyncio.run(fit(args,traced_model,worker_instances,device,test_loader,logger,es))
#Save resulta and show the plots
saveres(args,accuracy_scores,loss_scores,training_loss)

#Save the trained model
if args.save_model:
    model_name = "./" + args.model + ".pt"
    torch.save(model.state_dict(), model_name)
