import torch
import torch.nn as nn


class ConvDecoder(nn.Module):
    def __init__(self):
        super(ConvDecoder, self).__init__()

    def forward(self, previous_decoded_input, encoder_outputs):
        pass