import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_position import EmbeddingPosition


class ConvDecoder(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, kernel_size, num_layers, dropout, is_training):
        super(ConvDecoder, self).__init__()
        self._vocab_size = vocab_size
        self._max_length = max_length
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._kernel_size = kernel_size
        self._num_layers = num_layers
        self._dropout = dropout
        self._is_training = is_training

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.conv = nn.Conv1d(hidden_size, 2 * hidden_size, kernel_size)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, vocab_size)

    def forward(self, previous_decoded_input, encoder_outputs, encoder_attention):
        embedded_output = nn.Embedding(previous_decoded_input, self._embedding_size) + EmbeddingPosition(self._max_length, self._embedding_size)
        embedded_output = F.dropout(embedded_output, p=self._dropout, training=self._is_training)

        layer_output = self.fc1(embedded_output)

        for _ in self._num_layers:
            residual = layer_output

            conv_output = self.conv(layer_output)
            glu_output = F.glu(conv_output, axis=2)

            encoder_attention_logits = torch.bmm(glu_output, encoder_attention)
            encoder_attention = F.softmax(encoder_attention_logits)

            encoder_attention_output = torch.bmm(encoder_attention, encoder_outputs)
            attention_output = (encoder_attention_output + glu_output) * math.sqrt(0.5)

            layer_output = (attention_output + residual) * math.sqrt(0.5)

        # back to vocab size
        fc2_output = self.fc2(layer_output)
        fc2_output = F.dropout(fc2_output, p=self._dropout, training=self._is_training)
        fc3_output = self.fc3(fc2_output)
        prob_output = F.log_softmax(fc3_output)

        return prob_output
