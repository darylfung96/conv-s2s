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

        decoder_whole_output = Variable(torch.from_numpy(numpy.array([[0, 0, 0, 1]]*input.shape[0])))
        iteration = 0
        while any(decoder_whole_output.data[:][-1]) != 2 and iteration < 15:
            decoder_output = self._decoder(decoder_whole_output[:, -4:], encoder_output, encoder_attention)
            next_decoder_output = torch.max(decoder_output, 2)[1] # get the highest probability for the next word
            decoder_whole_output = torch.cat((decoder_whole_output, next_decoder_output), 1)
            iteration += 1

        return decoder_whole_output