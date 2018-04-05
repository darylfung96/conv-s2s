import torch
import torch.nn as nn

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder

    def forward(self, input):
        encoder_output, encoder_attention = self._encoder(input)

        decoder_output = 'start'
        while decoder_output != 'end':
            decoder_output = self._decoder(decoder_output, encoder_output, encoder_attention)

        return decoder_output