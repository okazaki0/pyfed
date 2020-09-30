import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import logging


def get_example_input():
    """ The function to get the input example for the jit.trace
    Returns:
        example_input: the example input
    """
    example_input = torch.zeros([1, 1, 28, 28], dtype=torch.float)
    return example_input

class cnn_batch(nn.Module):
    def __init__(self):
        super(cnn_batch, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.conv2_bn = nn.BatchNorm2d(50)
        self.fc1 = nn.Linear(4 * 4 * 50, 500)
        self.dense1_bn = nn.BatchNorm1d(500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(F.max_pool2d(self.conv2_bn(self.conv2(x)), 2))
        #x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 4 * 4 * 50)
        x = F.relu(self.dense1_bn(self.fc1(x)))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
