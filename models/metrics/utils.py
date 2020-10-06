from sklearn.metrics import f1_score
import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.ticker import MaxNLocator


def prepare_data(data):
    """this function is used for the training loss
    args:
        data: training loss data
        Return:
                res: the data as dicts
    """
    res = []
    for _, y in data.values:
        res.append(yaml.load(y))
    return res


def micro_loss(data):
    """the micro loss function
    args:
        data: training loss data after the preparation
        Return:
                avg_loss: the average micro loss
    """
    avg_loss = []
    for dicts in data:
        total = 0
        for key, value in dicts.items():
            total += value
        avg_total = total / len(dicts)
        avg_loss.append(avg_total)
    return avg_loss


def macro_loss(data, data_size):
    """the macro loss function
    args:
        data: training loss data after the preparation
        data_size: data_size of each workers
        Return:
                avg_loss: the average macro loss
    """
    avg_loss = []
    for dicts in data:
        total = 0
        for key, value in dicts.items():
            total += value * (data_size[int(key)] * 100) / sum(data_size)
        avg_loss.append(total)
    return avg_loss
