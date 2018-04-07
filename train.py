from collections import OrderedDict
import string
import numpy as np

from seq2seq import Seq2seq
from conv_encoder import ConvEncoder
from conv_decoder import ConvDecoder



examples = [
    "hello how are you doing?",
    "Do you have a dog?",
    "I have the best phone ever"
]

examples_target = [
    "I am fine, thank you. What about yourself?",
    "I have no dogs. Do you like dogs?",
    "What kind of phone do you have?"
]


word_to_index = OrderedDict({'<start>': 1, '<end>': 2})
index_to_word = OrderedDict({1: '<start>', 2: '<end>'})
max_length = 0


for index in range(len(examples)):
    removed_punc = examples[index].translate(examples[index].maketrans("", "", string.punctuation))
    splitted = removed_punc.split()

    if len(splitted) > max_length:
        max_length = len(splitted)

    for text_index in range(len(splitted)):
        if not word_to_index.get(splitted[text_index]):
            word_to_index[splitted[text_index]] = len(word_to_index)+1
            index_to_word[len(index_to_word)+1] = splitted[text_index]
            splitted[text_index] = len(word_to_index)+1
        else:
            splitted[text_index] = word_to_index.get(splitted[text_index])

    examples[index] = splitted

for index in range(len(examples_target)):
    removed_punc = examples_target[index].translate(examples_target[index].maketrans("", "", string.punctuation))
    splitted = removed_punc.split()

    if len(splitted) > max_length:
        max_length = len(splitted)

    for text_index in range(len(splitted)):
        if not word_to_index.get(splitted[text_index]):
            word_to_index[splitted[text_index]] = len(word_to_index)+1
            index_to_word[len(index_to_word)+1] = splitted[text_index]
            splitted[text_index] = len(word_to_index)+1
        else:
            splitted[text_index] = word_to_index.get(splitted[text_index])

        examples_target[index] = splitted


examples = [np.pad(example, [0, max_length-len(example)], mode='constant') for example in examples]

conv_encoder = ConvEncoder(len(word_to_index), max_length, hidden_size=128, embedding_size=512, kernel_size=2, num_layers=3, dropout=0.5, is_training=True)
conv_decoder = ConvDecoder(len(word_to_index), max_length, hidden_size=128, embedding_size=512, kernel_size=2, num_layers=3, dropout=0.5, is_training=True)


import torch
from torch.autograd import Variable
seq2seq = Seq2seq(conv_encoder, conv_decoder)
examples = Variable(torch.from_numpy(np.array(examples)))
seq2seq(examples)
