import torch.nn as nn
import torch.nn.functional as F

import math

from embedding_position import EmbeddingPosition


class ConvEncoder(nn.Module):

    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, kernel_size, num_layers, dropout, is_training):
        super(ConvEncoder, self).__init__()

        self._embedding_size = embedding_size
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._conv_out_channels = 2 * self._hidden_size
        self._num_layers = num_layers
        self._dropout = dropout
        self.is_training = is_training

        self.embedding = nn.Embedding(vocab_size, self._embedding_size)
        self.embedding_position = EmbeddingPosition(max_length, self._embedding_size)
        self.fc1 = nn.Linear(self._embedding_size, self._hidden_size)
        self.fc2 = nn.Linear(self._hidden_size, self._embedding_size)
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(hidden_size, self._conv_out_channels, kernel_size)

    def forward(self, input):

        # embedding
        embedded_input = self.embedding(input) + self.embedding_position(input)
        embedded_input = F.dropout(embedded_input, p=self._dropout, training=self.is_training)

        fc1_output = self.fc1(embedded_input)

        layer_output = embedded_input

        for _ in range(self._num_layers):
            residual_output = layer_output

            fc1_output = F.dropout(fc1_output, p=self._dropout)

            conv_output = self.conv(embedded_input)
            glu_output = F.glu(conv_output, 2)

            layer_output = (glu_output + residual_output) * math.sqrt(0.5) # scale value


        # convert back to embedding dimension
        encoder_output = self.fc2(layer_output)
        encoder_attention = (encoder_output + embedded_input) * math.sqrt(0.5)

        return encoder_output, encoder_attention


