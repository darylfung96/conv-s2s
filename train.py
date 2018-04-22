from collections import OrderedDict
import string
import numpy as np
from typing import Union

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


word_to_index = OrderedDict({'<unk>': 0, '<end>': 1})
index_to_word = OrderedDict({0: '<unk>', 1: '<end>'})
max_input_length = 0
max_target_length = 0


def index_to_word_sentence(word_indexes: Union[list, np.ndarray]):
    return [index_to_word[word_index] for word_index in word_indexes]



for index in range(len(examples)):
    removed_punc = examples[index].translate(examples[index].maketrans("", "", string.punctuation))
    splitted = [word.lower() for word in removed_punc.split()]

    if len(splitted) > max_input_length:
        max_input_length = len(splitted)

    for text_index in range(len(splitted)):
        if not word_to_index.get(splitted[text_index]):
            word_to_index[splitted[text_index]] = len(word_to_index)
            index_to_word[len(index_to_word)] = splitted[text_index]
            splitted[text_index] = word_to_index[splitted[text_index]]
        else:
            splitted[text_index] = word_to_index.get(splitted[text_index])

    examples[index] = splitted

    # append end and start token
    examples[index].append(1)


for index in range(len(examples_target)):
    removed_punc = examples_target[index].translate(examples_target[index].maketrans("", "", string.punctuation))
    splitted = [word.lower() for word in removed_punc.split()]

    if len(splitted) > max_target_length:
        max_target_length = len(splitted)

    for text_index in range(len(splitted)):
        if not word_to_index.get(splitted[text_index]):
            word_to_index[splitted[text_index]] = len(word_to_index)
            index_to_word[len(index_to_word)] = splitted[text_index]
            splitted[text_index] = word_to_index[splitted[text_index]]
        else:
            splitted[text_index] = word_to_index.get(splitted[text_index])

    examples_target[index] = splitted

    # append start and end token
    examples_target[index].append(1)
    examples_target[index].insert(0, 1)

examples = [np.pad(example, [0, max_input_length+2-len(example)], mode='constant') for example in examples] # + 2 to add the end token and the padding token
examples_target =[np.pad(example, [0, max_target_length+2-len(example)], mode='constant') for example in examples_target] # + 1 to add the end token and the padding token

conv_encoder = ConvEncoder(len(word_to_index), max_input_length+2, hidden_size=128, embedding_size=512, num_layers=1, dropout=0, is_training=True)
conv_decoder = ConvDecoder(len(word_to_index), max_target_length+2, hidden_size=128, embedding_size=512, num_layers=1, dropout=0, is_training=True)

examples = np.array(examples)
examples_target = np.array(examples_target)

seq2seq = Seq2seq(conv_encoder, conv_decoder, len(word_to_index))

seq_output = seq2seq(examples, examples_target)
seq_output = seq_output.data.numpy()
sentences = [index_to_word_sentence(seq) for seq in seq_output]
print(sentences)


while True:
    new_text = input('type in text to predict:')
    new_text_token = np.array([[word_to_index[token] for token in new_text.lower().split()]])
    new_text_token = np.concatenate([new_text_token, [[1]]], axis=1)

    outputs = seq2seq(new_text_token, is_training=False)
    outputs = outputs.data.numpy()
    sentences = [index_to_word_sentence(seq) for seq in outputs]
    print(sentences)