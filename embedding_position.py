import torch
import torch.nn as nn

class EmbeddingPosition(nn.Embedding):
    def __init__(self, max_length, embedding_dim):
        super(EmbeddingPosition, self).__init__(max_length, embedding_dim)

    def forward(self, input):
        for row in input:
            for index in range(len(row)):
                if input[row][index] == 0:
                    break
                input[row][index] = index+1

        super(EmbeddingPosition, self).forward(input)