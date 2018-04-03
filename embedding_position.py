import torch
import torch.nn as nn
from torch.autograd import Variable


class EmbeddingPosition(nn.Embedding):
    def __init__(self, max_length: int, embedding_dim: int):
        super(EmbeddingPosition, self).__init__(max_length, embedding_dim)

    def forward(self, inputs: Variable):
        inputs = inputs.nonzero()
        super(EmbeddingPosition, self).forward(inputs)