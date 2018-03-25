import torch
import torch.nn as nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, input):
        encoder_output = self._encoder(input)
        decoder_output = self._decoder(encoder_output)
        return decoder_output