import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super(Seq2seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vocab_size = vocab_size
        self.criterion = nn.NLLLoss()
        self.optim = optim.Adam(self.parameters(), lr=0.001)

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

        for i in range(1000):
            encoder_output, encoder_attention = self._encoder(inputs)
            decoder_input = target
            decoder_output = self._decoder(decoder_input, encoder_output, encoder_attention)
            decoder_output = decoder_output.squeeze(1)

            loss = None
            self.optim.zero_grad()
            for index in range(decoder_output.size(0)):
                loss = self.criterion(decoder_output[index], target[index]).backward(retain_graph=True)
            self.optim.step()

        return torch.max(decoder_output, 2)[1]


    def start_eval(self, input):
        encoder_output, encoder_attention = self._encoder(input)

        decoder_inputs = Variable(torch.from_numpy(numpy.array([[1]] * encoder_output.size(0))))
        next_decoder_output = None

        while (next_decoder_output is None or next_decoder_output.data[0][0] is not 1) and len(decoder_inputs.data[0]) < self._decoder._max_length:
            decoder_output = self._decoder(decoder_inputs, encoder_output, encoder_attention)
            next_decoder_output = torch.max(decoder_output, 3)[1][:, :, -1]
            decoder_inputs = torch.cat([decoder_inputs, next_decoder_output], dim=1)

        return decoder_inputs
