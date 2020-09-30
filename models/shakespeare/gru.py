import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as model
from torch.autograd import Variable


def get_example_input():
    """ The function to get the input example for the jit.trace
    Returns:
        example_input: the example input
    """
    example_input = torch.zeros([1, 1, 8], dtype=torch.int64)
    return example_input


class gru(nn.Module):
    def __init__(self, input_size=8, hidden_size=256, output_size=79, num_layer=2):
        super(gru, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.embedding = nn.Embedding(79, input_size)
        self.lstm = nn.GRU(input_size, hidden_size, num_layer)
        #nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        decoder = self.embedding(input)
        output, hidden = self.lstm(decoder.squeeze(1))
        output = self.linear(output[:, -1, :])
        print(output)
        return output



