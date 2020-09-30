import numpy as np
import torch
import matplotlib.pyplot as plt
import sys
import os
import glob
import pickle
import datetime


def cleanFolder(path):
    """ the clean function 
    args:
        path: the path to clean
    """
    files = glob.glob(path)
    for f in files:
        os.remove(f)

def readnpy(dataset):
    """ divise the dataset into data and target  
    args:
        dataset:[[imgs, label], [imgs, label]...., [imgs, label]]
    Return:
        dataset_data,dataset_target: the data and targets 
    """

    np_array = dataset
    imgs = []
    label = []
    for index in range(len(np_array)):
        imgs.append(np_array[index][0])
        label.append(np_array[index][1])
    dataset_data = torch.from_numpy(np.array(imgs))
    dataset_target =  torch.from_numpy(np.array(label))

    return dataset_data,dataset_target

#Get the data from the server data
def server_data():
    """ read the data of the server 
    Return:
        dataset_data,dataset_target: the dataset of the server 
    """
    #create instance of the server data
    with open("./split/s", "rb") as fp:   # Unpickling
        data = pickle.load(fp)  
    dataset_data,dataset_target = readnpy(data)
    return dataset_data,dataset_target