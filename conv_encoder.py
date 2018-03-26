import torch
import torch.nn as nn
import torch.functional as F

class ConvEncoder(nn.Module):

    def __init__(self, vocab_size, hidden_size, kernel_size, dropout):
        super(ConvEncoder, self).__init__()

        self._embedding_size = 512
        self._vocab_size = vocab_size
        self._hidden_size = hidden_size
        self._dropout = dropout

        self.embedding = nn.Embedding(vocab_size, self._embedding_size)

        #TODO create embedding position
        # self.embedding_pos = nn.Embedding()
        self.fc1 = nn.Linear(self._embedding_size, self._hidden_size)
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size)

    def forward(self, input):

        embedded_input = self.embedding(input)
        pass
