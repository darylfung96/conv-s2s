import torch.nn as nn
import torch.nn.functional as F

import math

from embedding_position import EmbeddingPosition


class ConvEncoder(nn.Module):

    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, num_layers, dropout, is_training):
        super(ConvEncoder, self).__init__()

        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._conv_out_channels = 2 * self._hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self.is_training = is_training

        self.embedding = nn.Embedding(vocab_size, self._embedding_size)
        self.embedding_position = EmbeddingPosition(max_length, self._embedding_size) # max_length + 1 to include the padding placeholders
        self.fc1 = nn.Linear(self._embedding_size, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, self._embedding_size)
        self.kernel_size = (3, self._hidden_size)
        self.conv = nn.Conv2d(1, self._conv_out_channels, self.kernel_size, padding=((self.kernel_size[0]-1)//2, 0))

    def forward(self, inputs):

        # embedding
        embedded_input = self.embedding(inputs) + self.embedding_position(inputs)
        embedded_input = F.dropout(embedded_input, p=self._dropout, training=self.is_training)
        embedded_input = embedded_input.unsqueeze(1)

        fc1_output = self.fc1(embedded_input)

        layer_output = fc1_output

        for _ in range(self._num_layers):
            residual_output = layer_output

            fc1_output = F.dropout(layer_output, p=self._dropout)
            conv_output = self.conv(fc1_output).transpose(1, 3)

            glu_output = F.glu(conv_output, 3)

            layer_output = (glu_output + residual_output) * math.sqrt(0.5) # scale value


        # convert back to embedding dimension
        encoder_output = self.fc2(layer_output)
        encoder_attention = (encoder_output + embedded_input) * math.sqrt(0.5)

        return encoder_output, encoder_attention


