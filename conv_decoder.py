import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from embedding_position import EmbeddingPosition


class ConvDecoder(nn.Module):
    def __init__(self, vocab_size, max_length, hidden_size, embedding_size, num_layers, dropout, is_training):
        super(ConvDecoder, self).__init__()
        self._vocab_size = vocab_size
        self._max_length = max_length
        self._hidden_size = hidden_size
        self._embedding_size = embedding_size
        self._kernel_size = (3, self._hidden_size)
        self._num_layers = num_layers
        self._dropout = dropout
        self._is_training = is_training

        self.embedding = nn.Embedding(vocab_size+3, self._embedding_size)
        self.embedding_position = EmbeddingPosition(self._vocab_size+3, self._embedding_size) # + 1 to include padding which act as none

        self.fc1 = nn.Linear(embedding_size, hidden_size)
        self.conv = nn.Conv2d(1, 2 * hidden_size, self._kernel_size, padding=(self._kernel_size[0]-1, 0))
        self.fc_conv_embedding = nn.Linear(hidden_size, embedding_size)
        self.fc_embedding_conv = nn.Linear(embedding_size, hidden_size)
        self.fc_next_single_char = nn.Linear(4, 1)
        self.fc2 = nn.Linear(hidden_size, embedding_size)
        self.fc3 = nn.Linear(embedding_size, vocab_size)

    def forward(self, previous_decoded_input, encoder_outputs, encoder_attention):
        embedded_output = self.embedding(previous_decoded_input) + self.embedding_position(previous_decoded_input)
        embedded_output = F.dropout(embedded_output, p=self._dropout, training=self._is_training)

        layer_output = self.fc1(embedded_output).unsqueeze(1)

        residual = layer_output
        for _ in range(self._num_layers):

            fc1_output = F.dropout(layer_output, p=self._dropout)

            conv_output = self.conv(fc1_output)
            conv_output = conv_output.narrow(2, 0, conv_output.size(2)-self._kernel_size[0]+1)
            conv_output = conv_output.transpose(1, 3)

            glu_output = F.glu(conv_output, 3)
            post_glu_output = self.fc_conv_embedding(glu_output)

            encoder_attention_logits = torch.bmm(post_glu_output.squeeze(1), encoder_attention.transpose(2, 3).squeeze(1))
            encoder_attention_output = F.softmax(encoder_attention_logits, 2)

            attention_output = torch.bmm(encoder_attention_output, encoder_outputs.squeeze(1))
            # scale attention output
            attention_output = attention_output * (encoder_outputs.size(3) * math.sqrt(2.0 / encoder_outputs.size(3)))
            layer_output = (self.fc_embedding_conv(attention_output).unsqueeze(1) + glu_output) * math.sqrt(0.5)

        layer_output = (layer_output + residual) * math.sqrt(0.5)

        # back to vocab size
        fc2_output = self.fc2(layer_output)
        fc2_output = F.dropout(fc2_output, p=self._dropout, training=self._is_training)
        fc3_output = self.fc3(fc2_output)
        prob_output = F.log_softmax(fc3_output, 2)

        return prob_output
