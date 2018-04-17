import torch
import torch.nn as nn
from torch.autograd import Variable


class EmbeddingPosition(nn.Embedding):
    def __init__(self, max_length: int, embedding_dim: int):
        super(EmbeddingPosition, self).__init__(max_length, embedding_dim)

    def forward(self, inputs: Variable):

        inputs_position = torch.zeros_like(inputs)

        for row_index in range(len(inputs.data)):
            for col_index in range(len(inputs.data[row_index])):
                # change the value to the absolute position
                if inputs.data[row_index][col_index] != 0:
                    inputs_position.data[row_index][col_index] = col_index+1
                else:
                    break

        return super(EmbeddingPosition, self).forward(inputs_position)