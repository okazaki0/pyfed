import torch
from typing import List
import logging
from aggregation.utils.func import *
logger = logging.getLogger(__name__)

def weighted_avg(models: List[torch.nn.Module], args) -> torch.nn.Module:
    """Calculate the federated weighted average of a list of models.

    Args:
        models (List[torch.nn.Module]): the models of which the federated average is calculated.
        args : to get the data size of each worker

    Returns:
        torch.nn.Module: the module with averaged parameters.
    """
    nr_models = len(models)
    model_list = list(models.values())
    #model = model_list[0]*data_size[0]
    size = args.data_size
    sum_data_size = sum(size)
    if args.split_mode == 'niid':
        model = scale_model(model_list[0], size[0])
        for i in range(1, nr_models):
            model = add_model_coef(model, model_list[i], size[i])
        model = scale_model(model, 1.0 / sum_data_size)
    elif args.split_mode == 'iid':
        model = model_list[0]
        for i in range(1, nr_models):
            model = add_model(model, model_list[i])
        model = scale_model(model, 1.0 / nr_models)


    return model