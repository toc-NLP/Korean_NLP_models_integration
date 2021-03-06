import os
import sys
import random

import numpy as np

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F

from models.attention import Attention, Attention_Bahdanau
from models.baseRNN import BaseRNN

if torch.cuda.is_available():
    import torch.cuda as device
else:
    import torch as device

class DecoderRNN(BaseRNN):
    KEY_ATTN_SCORE = 'attention_score'
    KEY_LENGTH = 'length'
    KEY_SEQUENCE = 'sequence'
    KEY_ENCODER_OUTPUTS = 'encoder_outputs'
    KEY_ENCODER_CONTEXT = 'encoder_context'
    KEY_ENCODER_HIDDEN = 'encoder_hidden'

    def __init__(self, vocab_size, max_length, hidden_size, word_embedding_size,
            sos_id, eos_id, input_dropout=0, rnn_dropout=0,
            number_of_layers=1, bidirectional=False, rnn_cell='lstm',use_attention=True):
        super(DecoderRNN, self).__init__(vocab_size, max_length, hidden_size,
                input_dropout, rnn_dropout,
                number_of_layers, rnn_cell)

        self.bidirectional_encoder = bidirectional

        self.output_size = vocab_size
        self.max_length = max_length
        self.use_attention = use_attention
        self.eos_id = eos_id
        self.sos_id = sos_id

        self.init_input = None

        self.embedding = nn.Embedding(self.output_size, word_embedding_size)

        if use_attention is "Luong":
            self.rnn = self.rnn_cell(word_embedding_size, hidden_size, number_of_layers,
                                 batch_first=True, bidirectional=bidirectional, rnn_dropout=rnndropout)
            self.attention = Attention(self.hidden_size)
        elif use_attention is "Bahdanau":
            self.rnn = self.rnn_cell(hidden_size + word_embedding_size, hidden_size, number_of_layers,
                                 batch_first=True, bidirectional=bidirectional, rnn_dropout=rnndropout)
            self.attention = Attention_Bahdanau(self.hidden_size)
        else:
            self.rnn = self.rnn_cell(word_embedding_size, hidden_size, number_of_layers,
                                 batch_first=True, bidirectional=bidirectional, rnn_dropout=rnndropout)

        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward_step(self, input_var, hidden, encoder_outputs, function):
        batch_size = input_var.size(0)
        output_size = input_var.size(1)
        
        embedded = self.embedding(input_var)

        if self.use_attention == "Luong":
            output, hidden = self.rnn(embedded, hidden)
            output, attn = self.attention(output, encoder_outputs)
        elif use_attention is "Bahdanau":
            attn = self.attention(hidden[-1], encoder_outputs)
            input_v = input_var.unsqueeze(2)
            input_v = input_v.float()
            # input_v = batch_size * out_len * 1
            # attn = batch_size * 1 * in_len
            # attn_v = batch_size * out_len * in_len
            attn_v = torch.bmm(input_v, attn)
            #attn = attn.view(batch_size, output_size, -1)
            # ontext = batch * out_len * (hidden_size*2)
            context = attn_v.bmm(encoder_outputs)  # (B,s,V)
            rnn_input = torch.cat((embedded, context), 2)
            attn = attn_v
            output, hidden = self.rnn(rnn_input, hidden)
        else:
            output, hidden = self.rnn(embedded, hidden)
            attn = None

        predicted_softmax = function(self.out(output.contiguous().view(-1, self.hidden_size)), dim=1).view(batch_size, output_size, -1)
        return predicted_softmax, hidden, attn

    def forward(self, inputs=None, encoder_hidden=None, encoder_outputs=None,
                    function=F.log_softmax, teacher_forcing_ratio=0):
        ret_dict = dict()
        ret_dict[DecoderRNN.KEY_ENCODER_OUTPUTS] = encoder_outputs.squeeze(0)
        ret_dict[DecoderRNN.KEY_ENCODER_HIDDEN] = encoder_hidden[0].squeeze(0)

        if self.use_attention:
            ret_dict[DecoderRNN.KEY_ATTN_SCORE] = list()

        # input.shape = batch_size x sequence_length
        # encoder_outputs.shape = batch_size x sequence_length (50) x hidden_size (50 x 2)
        # encoder_hidden = tuple of the last hidden state and the last cell state.
        # Last cell state = number of layers * batch_size * hidden_size
        # Last hidden state = the same as above

        inputs, batch_size, max_length = self._validate_args(inputs, encoder_hidden, encoder_outputs,
                                                             function, teacher_forcing_ratio)
        decoder_hidden = self._init_state(encoder_hidden)

        use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

        decoder_outputs = []
        sequence_symbols = []
        lengths = np.array([max_length] * batch_size)

        def decode(step, step_output, step_attn):
            decoder_outputs.append(step_output)
            if self.use_attention:
                ret_dict[DecoderRNN.KEY_ATTN_SCORE].append(step_attn)
            symbols = decoder_outputs[-1].topk(1)[1]
            sequence_symbols.append(symbols)

            eos_batches = symbols.data.eq(self.eos_id)
            if eos_batches.dim() > 0:
                eos_batches = eos_batches.cpu().view(-1).numpy()
                update_idx = ((lengths > step) & eos_batches) != 0
                lengths[update_idx] = len(sequence_symbols)
            return symbols

        if use_teacher_forcing:
            decoder_input = inputs[:, :-1]
            decoder_output, decoder_hidden, attn = self.forward_step(decoder_input, 
                    decoder_hidden, encoder_outputs, function=function)

            for di in range(decoder_output.size(1)):
                step_output = decoder_output[:, di, :]
                if attn is not None:
                    step_attn = attn[:, di, :]
                else:
                    step_attn = None
                decode(di, step_output, step_attn)
        else:
            decoder_input = inputs[:, 0].unsqueeze(1)
            for di in range(max_length):
                decoder_output, decoder_hidden, step_attn = self.forward_step(decoder_input, 
                        decoder_hidden, encoder_outputs, function=function)
                step_output = decoder_output.squeeze(1)
                symbols = decode(di, step_output, step_attn)
                decoder_input = symbols

        ret_dict[DecoderRNN.KEY_SEQUENCE] = sequence_symbols
        ret_dict[DecoderRNN.KEY_LENGTH] = lengths.tolist()

        return decoder_outputs, decoder_hidden, ret_dict

    def _init_state(self, encoder_hidden):
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple([self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

    def _validate_args(self, inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio):
        if self.use_attention:
            if encoder_outputs is None:
                raise ValueError("Argument encoder_outputs cannot be None when attention is used.")

        # inference batch size
        if inputs is None and encoder_hidden is None:
            batch_size = 1
        else:
            if inputs is not None:
                batch_size = inputs.size(0)
            else:
                if self.rnn_cell is nn.LSTM:
                    batch_size = encoder_hidden[0].size(1)
                elif self.rnn_cell is nn.GRU:
                    batch_size = encoder_hidden.size(1)

        # set default input and max decoding length
        if inputs is None:
            if teacher_forcing_ratio > 0:
                raise ValueError("Teacher forcing has to be disabled (set 0) when no inputs is provided.")
            inputs = torch.LongTensor([self.sos_id] * batch_size).view(batch_size, 1)
            if torch.cuda.is_available():
                inputs = inputs.cuda()
            max_length = self.max_length
        else:
            max_length = inputs.size(1) - 1 # minus the start of sequence symbol

        return inputs, batch_size, max_length
