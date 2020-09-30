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
    example_input = torch.zeros([1, 1, 300], dtype=torch.float)
    return example_input

class lstm(nn.Module):
    def __init__(self, input_size=300, hidden_size=120, output_size=2, num_layer=2):
        super(lstm, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layer = num_layer
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layer, batch_first=True)
        self.linear = nn.Linear(hidden_size, output_size)

    def forward(self, input):
        output, hidden = self.lstm(input)
        #output, out_len = torch.nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        output = self.linear(output[:, -1, :])
        # output = self.linear(output.contiguous().view(output.shape[0], -1))
        return output
