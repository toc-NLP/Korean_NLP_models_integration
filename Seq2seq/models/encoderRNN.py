import os
import sys

import torch
import torch.nn as nn

from models.baseRNN import BaseRNN

class EncoderRNN(BaseRNN):

    def __init__(self, vocab_size, max_length, hidden_size,
                 word_embedding_size, input_dropout=0, rnn_dropout=0,
                 number_of_layers=1, bidirectional=False, rnn_cell='lstm', variable_lengths=False,
                 pretrained_embedding=None, update_embedding=True):
        super(EncoderRNN, self).__init__(vocab_size, max_length, hidden_size,
                input_dropout, rnn_dropout, number_of_layers, rnn_cell)

        self.variable_lengths = variable_lengths
        self.word_embedding = nn.Embedding(vocab_size, embedding_size)
        if pretrained_embedding is not None:
            self.embedding.weight = nn.Parameter(pretrained_embedding)
        self.embedding.weight.requires_grad = update_embedding
        self.rnn = self.rnn_cell(embedding_size, hidden_size, number_of_layers,
                                 batch_first=True, bidirectional=bidirectional, dropout=rnn_dropout)

    def forward(self, input_variable, input_lengths=None):
        word_embedded = self.word_embedding(input_variable)

        if self.variable_lengths:
            word_embedded = nn.utils.rnn.pack_padded_sequence(word_embedded, input_lengths, batch_first=True)
        output, hidden = self.rnn(word_embedded)
        if self.variable_lengths:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        return output, hidden
