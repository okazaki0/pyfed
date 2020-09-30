import torch
import numpy as np
import argparse
import random
import subprocess
import signal
import sys
sys.path.append('./')
import pickle
from pathlib import Path
import os
from utils.func import *
from datasets.dataloader import *
import utils.arguments as arguments


FILE_PATH = "./run/network/run_websocket_server.py"


def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    for p in process_clients:
        p.terminate()
    sys.exit(0)

args = arguments.defineArgs()
print(args.dataset)
if args.server_data == True:
    args.clients = args.clients +1
train_loader, _ ,global_loader = load_data(args)
sub_dataset = splitDataset(args,train_loader,global_loader)

cleanFolder("./data/split/*")
for i in range(args.clients):
    if (i == args.clients - 1) and (args.server_data == True):
        with open("./data/split/s", "wb") as fp:   #Pickling
            pickle.dump(sub_dataset[i], fp)
    else:
        with open("./data/split/%d" % i, "wb") as fp:   #Pickling
            pickle.dump(sub_dataset[i], fp)
   
python = Path(sys.executable).name


process_clients = []
if args.server_data == True:
    args.clients = args.clients - 1
for i in range(args.clients):
    print("Starting server for client ", i)
    process_clients.append(subprocess.Popen(["python", FILE_PATH,"--port", str(args.client_port+i), "--id", str(i), "--host", "127.0.0.1", "--dataset", args.dataset]))


#kill the process
signal.signal(signal.SIGINT, signal_handler)
signal.pause()
