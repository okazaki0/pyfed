import torch
from typing import List
import logging
from aggregation.utils.func import *


def federated_avg(models: List[torch.nn.Module]) -> torch.nn.Module:
    """Calculate the federated average of a list of models.

    Args:
        models (List[torch.nn.Module]): the models of which the federated average is calculated.

    Returns:
        torch.nn.Module: the module with averaged parameters.
    """
    nr_models = len(models)
    model_list = list(models.values())
    model = model_list[0]
    for i in range(1, nr_models):
        model = add_model(model, model_list[i])
    model = scale_model(model, 1.0 / nr_models)
    return model
