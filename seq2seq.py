import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, input):
        encoder_output, encoder_attention = self._encoder(input)

        decoder_output = Variable(torch.from_numpy(numpy.array([[1]]*input.shape[0])))
        while decoder_output.data[0][-1] != 2:
            decoder_output = self._decoder(decoder_output, encoder_output, encoder_attention)

        return decoder_output