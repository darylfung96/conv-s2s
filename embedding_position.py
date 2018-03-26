import torch
import torch.nn as nn

class EmbeddingPosition(nn.Embedding):
    def __init__(self, num_embeddings, embedding_dim):
        super(EmbeddingPosition, self).__init__(num_embeddings, embedding_dim)
        pass