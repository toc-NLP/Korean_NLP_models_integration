import torch.nn as nn
import torch.nn.functional as F
import sys
import os

from models.encoderRNN import EncoderRNN
from models.decoderRNN import DecoderRNN

class Seq2seq(nn.Module):

    def __init__(self, config, src_vocab_size, tgt_vocab_size, sos_id, eos_id):
        super(Seq2seq, self).__init__()

        self.encoder = EncoderRNN(vocab_size=src_vocab_size,
                                  max_len=config["max_len"],
                                  hidden_size=config["hidden_size"],
                                  embedding_size=config["embedding_size"],
                                  input_dropout_p=config["input_dropout_p"],
                                  dropout_p=config["dropout_p"],
                                  n_layers=config["n_layers"],
                                  bidirectional=config["bidirectional"],
                                  rnn_cell=config["rnn_cell"],
                                  variable_lengths=config["variable_lengths"],
                                  embedding=config["embedding"],
                                  update_embedding=config["update_embedding"])
        self.decoder = DecoderRNN(vocab_size=tgt_vocab_size,
                                  max_len=config["max_len"],
                                  hidden_size=config["hidden_size"]*2 if config["bidirectional"] else config["hidden_size"],
                                  embedding_size=config["embedding_size"],
                                  sos_id=sos_id,
                                  eos_id=eos_id,
                                  input_dropout_p=config["input_dropout_p"],
                                  dropout_p=config["dropout_p"],
                                  n_layers=config["n_layers"],
                                  bidirectional=config["bidirectional"],
                                  rnn_cell=config["rnn_cell"],
                                  use_attention=config["use_attention"])

        self.decode_function = F.log_softmax

    def flatten_parameters(self):
        self.encoder.rnn.flatten_parameters()
        self.decoder.rnn.flatten_parameters()

    def forward(self, input_variable, input_lengths=None, target_variable=None,
                teacher_forcing_ratio=0):
        encoder_outputs, encoder_hidden = self.encoder(input_variable, input_lengths)
        result = self.decoder(inputs=target_variable,
                              encoder_hidden=encoder_hidden,
                              encoder_outputs=encoder_outputs,
                              function=self.decode_function,
                              teacher_forcing_ratio=teacher_forcing_ratio)
        return result
