import numpy
import torch
import torch.nn as nn
from torch.autograd import Variable

class Seq2seq(nn.Module):
    def __init__(self, encoder, decoder, vocab_size):
        super(Seq2seq, self).__init__()
        self._encoder = encoder
        self._decoder = decoder
        self._vocab_size = vocab_size
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
        decoder_output = decoder_output.squeeze(1)

        one_hot_target = torch.LongTensor(target.size(0), target.size(1), self._vocab_size).zero_()
        target_reshaped = target.contiguous().view(target.size(0), target.size(1), 1)
        one_hot_target.scatter_(2, target_reshaped.data, 1)
        one_hot_target = Variable(one_hot_target)

        loss = None
        for index in range(decoder_output.size(0)):
            loss = loss + self.criterion(decoder_output[index], target[index]) if loss is not None else self.criterion(decoder_output[index], target[index])

        loss.backward()


    def start_eval(self, input):
        encoder_output, encoder_attention = self._encoder(input)

        decoder_inputs = Variable(torch.from_numpy(numpy.array([[1]] * encoder_output.size(0))))
        next_decoder_output = None

        while next_decoder_output is None or next_decoder_output.data[0][0] is not 2:
            decoder_output = self._decoder(decoder_inputs, encoder_output, encoder_attention)
            next_decoder_output = torch.max(decoder_output, 3)[1][:, :, -1]
            decoder_inputs = torch.cat([decoder_inputs, next_decoder_output], dim=1)

        return decoder_inputs
