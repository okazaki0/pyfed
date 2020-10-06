import torch
from torchvision import datasets, transforms
import torch.nn.functional as f
import os
import datetime
import logging
import random
import asyncio
import torch.nn as nn

import sys

sys.path.append("./")

import fit_on_worker as rwc
from utils.func import *
import importlib


# fit function
async def fit(args, traced_model, worker_instances, device, test_loader, logger, es):
    """the fit method
    args:
        ...
    Return:
        loss_scores,accuracy_scores,training_loss: the training results
    """
    learning_rate = args.lr
    accuracy_scores = []
    loss_scores = []
    training_loss = []
    for curr_round in range(1, args.training_rounds + 1):
        logger.info("Training round %s/%s", curr_round, args.training_rounds)
        # fraction of client that will particepated

        worker_instance = random.sample(
            worker_instances, k=int(args.clients * args.fraction_client)
        )
        try:
            results = await asyncio.gather(
                *[
                    rwc.fit_model_on_worker(
                        args=args,
                        worker=worker,
                        traced_model=traced_model,
                        curr_round=curr_round,
                        lr=learning_rate,
                    )
                    for worker in worker_instance
                ]
            )
        except Exception as e:
            print("Error in the round :", e)
            continue
        models = {}
        loss_values = {}

        test_models = (
            curr_round % args.eval_every == 1 or curr_round == args.training_rounds
        )
        if test_models:
            logger.info("Evaluating models")
            np.set_printoptions(formatter={"float": "{: .0f}".format})
            for worker_id, worker_model, _ in results:
                res = test(
                    "Worker " + worker_id,
                    worker_model,
                    args.loss,
                    device,
                    test_loader,
                    logger,
                )

        # Federate models (note that this will also change the model in models[0]
        for worker_id, worker_model, worker_loss in results:
            if worker_model is not None:
                models[worker_id] = worker_model
                loss_values[worker_id] = worker_loss.item()
                logger.info(
                    "Worker {} , training loss : {:.4f}".format(
                        worker_id, loss_values[worker_id]
                    )
                )
        # aggregation
        aggregation = importlib.import_module("aggregation." + args.aggregation)
        agg = getattr(aggregation, args.aggregation)
        traced_model = agg(models)
        if test_models:
            res = test(
                "Federated model", traced_model, args.loss, device, test_loader, logger
            )

            if es.step(res):
                print("Early Stopping...")
                break  # early stop criterion is met, we can stop now

            print("-------------------", res["loss"])
            training_loss.append([curr_round, loss_values])
            # print(training_loss)
            loss_scores.append(res["loss"])
            correct = res["correct"]
            accuracy_scores.append(res["accuracy"])

            # decay learning rate
            learning_rate = max(0.98 * learning_rate, args.lr * 0.01)

    return loss_scores, accuracy_scores, training_loss


def test(model_identifier, model, criterion, device, test_loader, logger):
    """the test method
    args:
        ...
    Return:
        res: dict contains the results
    """
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = getattr(f, criterion)(output, target)  # sum up batch loss
            test_loss += loss.item()
            pred = output.argmax(
                1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    accuracy = 100.0 * correct / len(test_loader.dataset)
    logger.info("%s :", model_identifier)
    logger.info(
        "Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)".format(
            test_loss, correct, len(test_loader.dataset), accuracy
        )
    )
    res = {"loss": test_loss, "correct": correct, "accuracy": accuracy}
    return res
