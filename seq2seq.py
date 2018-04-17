import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self.criterion = nn.NLLLoss()

    def forward(self, inputs, target=None, is_training=True):
        inputs = Variable(torch.from_numpy(inputs))

        if is_training:
            if target is None:
                raise ValueError("target parameter has to be passed with value.")
            target = target[:, 1:]
            target = Variable(torch.from_numpy(target))
            return self.start_train(inputs, target)
        else:
            return self.start_eval(inputs)

    def start_train(self, inputs, target):
        encoder_output, encoder_attention = self._encoder(inputs)
        decoder_input = target
        decoder_output = self._decoder(decoder_input, encoder_output, encoder_attention)
        #TODO fix loss
        self.criterion(decoder_output, target)


    def start_eval(self, input):
        encoder_output, encoder_attention = self._encoder(input)

        decoder_inputs = Variable(torch.from_numpy(numpy.array([[1]] * encoder_output.size(0))))
        next_decoder_output = None

        while next_decoder_output is None or next_decoder_output.data[0][0] is not 2:
            decoder_output = self._decoder(decoder_inputs, encoder_output, encoder_attention)
            next_decoder_output = torch.max(decoder_output, 3)[1][:, :, -1]
            decoder_inputs = torch.cat([decoder_inputs, next_decoder_output], dim=1)

        return decoder_inputs
