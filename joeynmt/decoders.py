# coding: utf-8

"""
Various decoders
"""
from typing import Optional, Tuple, Union, List
import time
from copy import deepcopy

import random # remove me TODO FIXME
import numpy as np # TODO remove!

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, cat
from joeynmt.attention import BahdanauAttention, LuongAttention, KeyValRetAtt
from joeynmt.encoders import Encoder
from joeynmt.helpers import freeze_params, ConfigurationError, subsequent_mask, tile, product, Timer
from joeynmt.transformer_layers import PositionalEncoding, \
    TransformerDecoderLayer, MultiHeadedKbAttention


class VariableCellLSTM(nn.Module):
    def __init__(self, size, num_layers, dropout=.1, forget_bias=1., **kwargs):
        super(VariableCellLSTM, self).__init__()
        self.size = size
        self.num_layers = num_layers
        self.dropout = dropout
        self.forget_bias = float(forget_bias)

        num_inbetween_layers = max(0, self.num_layers-2)

        # Modules as in LSTM; as per Chris Olah Blog Graphic, FLTR
        self.forget = nn.Sequential(
            nn.Linear(2*size,size),nn.ReLU(), nn.Dropout(self.dropout),
            *(nn.Linear(size,size),nn.ReLU(), nn.Dropout(self.dropout))*(num_inbetween_layers), 
            nn.Linear(size,size),nn.Sigmoid(), nn.Dropout(self.dropout)
        )
        with torch.no_grad():
            # bias last linear layer of forget gate
            self.forget[-3].weight += self.forget_bias

        self.adding_residual_normalizer = nn.Sequential(
            nn.Linear(2*size,size),nn.ReLU(), nn.Dropout(self.dropout),
            *(nn.Linear(size,size),nn.ReLU(), nn.Dropout(self.dropout))*(num_inbetween_layers), 
            nn.Linear(size,size),nn.Sigmoid(), nn.Dropout(self.dropout)
        )
        self.add_net = nn.Sequential(
            nn.Linear(2*size,size),nn.ReLU(), nn.Dropout(self.dropout),
            *(nn.Linear(size,size), nn.ReLU(), nn.Dropout(self.dropout))*(num_inbetween_layers), 
            nn.Linear(size,size), nn.Tanh(), nn.Dropout(self.dropout)
        )
        self.add_energy_net = nn.Linear(size, 1, bias=False) # learns scalar of how to add to main

        self.tanh = nn.Tanh()

        self.weight_cell_input_to_hidden = nn.Sequential(
            nn.Linear(2*size,size),nn.ReLU(), nn.Dropout(self.dropout),
            *(nn.Linear(size,size), nn.ReLU(), nn.Dropout(self.dropout))*(num_inbetween_layers+1), 
            nn.Linear(size,size), nn.Sigmoid(), nn.Dropout(self.dropout)
        )

    def forward(self, x, hidden, cell):
        """
        update cell using x like in LSTM 
        transformer hidden is cell state: this is what we want to update
        """
        assert x.shape == cell.shape, [t.shape for t in [x,cell]]

        info = torch.cat([x, hidden], dim=-1)
        
        cleared_cell = self.forget(info) * cell

        new_info = self.adding_residual_normalizer(info) * self.add_net(info)
        new_info_weighted = new_info * self.add_energy_net(cell)

        # updated transformer hidden
        updated_cell = cell + new_info_weighted
        # finally updated this module's hidden for next call
        updated_hidden = self.weight_cell_input_to_hidden(info) * self.tanh(updated_cell)

        return updated_hidden, updated_cell



# pylint: disable=abstract-method
class Gen(nn.Module):
    """
    Base generator class
    """

    @property
    def output_size(self):
        """
        Return the output size (size of the target vocabulary)

        :return:
        """
        return self._output_size

# pylint: disable=abstract-method
class Decoder(nn.Module):
    """
    Base decoder class
    """

    # decoder output signature is:
    # return hidden, att_probs, att_vectors, kb_probs

    @property
    def output_size(self):
        """
        Return the output size (hidden, NOT vocab size)

        :return:
        """
        return self._output_size

    def stats(self):
        """
        Return self.timer statistics
        """
        assert hasattr(self, "timer"), f"In this version of JoeyNMT, all Decoders must have self.timer Timer objects for logging"
        return self.timer.logAllParams()


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class RecurrentDecoder(Decoder):
    """A conditional RNN decoder with attention."""

    def __init__(self,
                 rnn_type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder: Encoder = None,
                 attention: str = "bahdanau",
                 num_layers: int = 1,
                 vocab_size: int = 0,
                 dropout: float = 0.,
                 emb_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 init_hidden: str = "bridge",
                 input_feeding: bool = True,
                 freeze: bool = False,
                 **kwargs) -> None:
        """
        Create a recurrent decoder with attention.

        :param rnn_type: rnn type, valid options: "lstm", "gru"
        :param emb_size: target embedding size
        :param hidden_size: size of the RNN
        :param encoder: encoder connected to this decoder
        :param attention: type of attention, valid options: "bahdanau", "luong"
        :param num_layers: number of recurrent layers
        :param vocab_size: target vocabulary size
        :param hidden_dropout: Is applied to the input to the attentional layer.
        :param dropout: Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param init_hidden: If "bridge" (default), the decoder hidden states are
            initialized from a projection of the last encoder state,
            if "zeros" they are initialized with zeros,
            if "last" they are identical to the last encoder state
            (only if they have the same size)
        :param input_feeding: Use Luong's input feeding.
        :param freeze: Freeze the parameters of the decoder during training.
        :param kwargs:
        """

        super(RecurrentDecoder, self).__init__()

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        if self.input_feeding: # Luong-style
            # combine embedded prev word +attention vector before feeding to rnn
            self.rnn_input_size = emb_size + hidden_size
        else:
            # just feed prev word embedding
            self.rnn_input_size = emb_size

        # the decoder RNN
        self.rnn = rnn(self.rnn_input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True)

        self._output_size = vocab_size

        if attention == "bahdanau":
            self.attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=encoder.output_size,
                                               query_size=hidden_size)
        elif attention == "luong":
            self.attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=encoder.output_size)
        else:
            raise ConfigurationError("Unknown attention mechanism: %s. "
                                     "Valid options: 'bahdanau', 'luong'."
                                     % attention)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # to initialize from the final encoder state of last layer
        self.init_hidden_option = init_hidden
        if self.init_hidden_option == "bridge":
            self.bridge_layer = nn.Linear(
                encoder.output_size, hidden_size, bias=True)
        elif self.init_hidden_option == "last":
            if encoder.output_size != self.hidden_size:
                if encoder.output_size != 2*self.hidden_size:  # bidirectional
                    raise ConfigurationError(
                        "For initializing the decoder state with the "
                        "last encoder state, their sizes have to match "
                        "(encoder: {} vs. decoder:  {})".format(
                            encoder.output_size, self.hidden_size))
        if freeze:
            freeze_params(self)

    def _check_shapes_input_forward_step(self,
                                         prev_embed: Tensor,
                                         prev_att_vector: Tensor,
                                         encoder_output: Tensor,
                                         src_mask: Tensor,
                                         hidden: Tensor) -> None:
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        """
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size(
            [1, self.hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert src_mask.shape[0] == prev_embed.shape[0]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[2] == encoder_output.shape[1]
        if isinstance(hidden, tuple):  # for lstm
            hidden = hidden[0]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == self.hidden_size

    def _check_shapes_input_forward(self,
                                    trg_embed: Tensor,
                                    encoder_output: Tensor,
                                    encoder_hidden: Tensor,
                                    src_mask: Tensor,
                                    hidden: Tensor = None,
                                    prev_att_vector: Tensor = None) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param hidden:
        :param prev_att_vector:
        """
        assert len(encoder_output.shape) == 3
        assert len(encoder_hidden.shape) == 2
        assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[0] == encoder_output.shape[0]
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert trg_embed.shape[0] == encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):  # for lstm
                hidden = hidden[0]
            assert hidden.shape[1] == encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(self,
                      prev_embed: Tensor,
                      prev_att_vector: Tensor,  # context or att vector
                      encoder_output: Tensor,
                      src_mask: Tensor,
                      hidden: Tensor) -> (Tensor, Tensor, Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
            shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
            shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
            shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
            shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
            shape (num_layers, batch_size, hidden_size)
        :return:
            - att_vector: new attention vector (batch_size, 1, hidden_size),
            - hidden: new hidden state with shape (batch_size, 1, hidden_size),
            - att_probs: attention probabilities (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(prev_embed=prev_embed,
                                              prev_att_vector=prev_att_vector,
                                              encoder_output=encoder_output,
                                              src_mask=src_mask,
                                              hidden=hidden)

        if self.input_feeding:
            # concatenate the input with the previous attention vector
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        # rnn_input: batch x 1 x emb + 2 * enc_size
        _, hidden = self.rnn(rnn_input, hidden)

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple):
            query = hidden[0][-1].unsqueeze(1)
        else:
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]

        # compute context vector using attention mechanism
        # only use last layer for attention mechanism
        # key projections are pre-computed
        context, att_probs = self.attention(
            query=query, values=encoder_output, mask=src_mask)

        # return attention vector (Luong)
        # combine context with decoder hidden state before prediction
        att_vector_input = torch.cat([query, context], dim=2)
        # batch x 1 x 2*enc_size+hidden_size
        att_vector_input = self.hidden_dropout(att_vector_input)

        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs

    def forward(self,
                trg_embed: Tensor,
                encoder_output: Tensor,
                encoder_hidden: Tensor,
                src_mask: Tensor,
                unroll_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None,
                **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
         Unroll the decoder one step at a time for `unroll_steps` steps.
         For every step, the `_forward_step` function is called internally.

         During training, the target inputs (`trg_embed') are already known for
         the full sequence, so the full unrol is done.
         In this case, `hidden` and `prev_att_vector` are None.

         For inference, this function is called with one step at a time since
         embedded targets are the predictions from the previous time step.
         In this case, `hidden` and `prev_att_vector` are fed from the output
         of the previous call of this function (from the 2nd step on).

         `src_mask` is needed to mask out the areas of the encoder states that
         should not receive any attention,
         which is everything after the first <eos>.

         The `encoder_output` are the hidden states from the encoder and are
         used as context for the attention.

         The `encoder_hidden` is the last encoder hidden state that is used to
         initialize the first hidden decoder state
         (when `self.init_hidden_option` is "bridge" or "last").

        :param trg_embed: emdedded target inputs,
            shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
            shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder,
            shape (batch_size x encoder.output_size)
        :param src_mask: mask for src states: 0s for padded areas,
            1s for the rest, shape (batch_size, 1, src_length)
        :param unroll_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
            if not given it's initialized as in `self.init_hidden`,
            shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
            if not given it's initialized with zeros,
            shape (batch_size, 1, hidden_size)
        :return:
            - outputs: shape (batch_size, unroll_steps, vocab_size),
            - hidden: last hidden state (num_layers, batch_size, hidden_size),
            - att_probs: attention probabilities
                with shape (batch_size, unroll_steps, src_length),
            - att_vectors: attentional vectors
                with shape (batch_size, unroll_steps, hidden_size)
        """

        # shape checks
        self._check_shapes_input_forward(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            hidden=hidden,
            prev_att_vector=prev_att_vector)

        # initialize decoder hidden state from final encoder hidden state
        if hidden is None:
            hidden = self._init_hidden(encoder_hidden)

        # pre-compute projected encoder outputs
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        if hasattr(self.attention, "compute_proj_keys"):
            self.attention.compute_proj_keys(keys=encoder_output)
        
        # here we store all intermediate attention vectors (used for prediction)
        att_vectors = []
        att_probs = []

        batch_size = encoder_output.size(0)

        if prev_att_vector is None:
            with torch.no_grad():
                prev_att_vector = encoder_output.new_zeros(
                    [batch_size, 1, self.hidden_size])

        # unroll the decoder RNN for `unroll_steps` steps
        for i in range(unroll_steps):
            prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb
            prev_att_vector, hidden, att_prob = self._forward_step(
                prev_embed=prev_embed,
                prev_att_vector=prev_att_vector,
                encoder_output=encoder_output,
                src_mask=src_mask,
                hidden=hidden)
            att_vectors.append(prev_att_vector)
            att_probs.append(att_prob)

        att_vectors = torch.cat(att_vectors, dim=1)
        # att_vectors: batch, unroll_steps, hidden_size
        att_probs = torch.cat(att_probs, dim=1)
        # att_probs: batch, unroll_steps, src_length

        # decoder output signature is:
        # return hidden, att_probs, att_vectors, kb_probs
        return hidden, att_probs, att_vectors, None, None, None

    def _init_hidden(self, encoder_final: Tensor = None) \
            -> (Tensor, Optional[Tensor]):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        In case of `self.init_hidden_option == "bridge"`
        and a given `encoder_final`, this is a projection of the encoder state.

        In case of `self.init_hidden_option == "last"`
        and a size-matching `encoder_final`, this is set to the encoder state.
        If the encoder is twice as large as the decoder state (e.g. when
        bi-directional), just use the forward hidden state.

        In case of `self.init_hidden_option == "zero"`, it is initialized with
        zeros.

        For LSTMs we initialize both the hidden state and the memory cell
        with the same projection/copy of the encoder hidden state.

        All decoder layers are initialized with the same initial values.

        :param encoder_final: final state from the last layer of the encoder,
            shape (batch_size, encoder_hidden_size)
        :return: hidden state if GRU, (hidden state, memory cell) if LSTM,
            shape (batch_size, hidden_size)
        """
        batch_size = encoder_final.size(0)

        # for multiple layers: is the same for all layers
        if self.init_hidden_option == "bridge" and encoder_final is not None:
            # num_layers x batch_size x hidden_size
            hidden = torch.tanh(
                    self.bridge_layer(encoder_final)).unsqueeze(0).repeat(
                    self.num_layers, 1, 1)
        elif self.init_hidden_option == "last" and encoder_final is not None:
            # special case: encoder is bidirectional: use only forward state
            if encoder_final.shape[1] == 2*self.hidden_size:  # bidirectional
                encoder_final = encoder_final[:, :self.hidden_size]
            hidden = encoder_final.unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:  # initialize with zeros
            with torch.no_grad():
                hidden = encoder_final.new_zeros(
                    self.num_layers, batch_size, self.hidden_size)

        return (hidden, hidden) if isinstance(self.rnn, nn.LSTM) else hidden

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)

class KeyValRetRNNDecoder(RecurrentDecoder):
    """A conditional RNN decoder with attention AND bahdanau attention over a knowledgebase"""

    def __init__(self,
                 rnn_type: str = "gru",
                 emb_size: int = 0,
                 hidden_size: int = 0,
                 encoder: Encoder = None,
                 attention: str = "bahdanau",
                 num_layers: int = 1,
                 vocab_size: int = 0,
                 dropout: float = 0.,
                 emb_dropout: float = 0.,
                 hidden_dropout: float = 0.,
                 init_hidden: str = "bridge",
                 input_feeding: bool = True,
                 freeze: bool = False,
                 kb_key_emb_size: int=0,
                 k_hops: int = 1,
                 kb_max: Tuple[int]= (256,),
                 kb_input_feeding: bool = True,
                 kb_feed_rnn: bool = True,
                 same_module_for_all_hops: bool = False,
                 kb_multihead_feed: bool = False,
                 **kwargs) -> None:
        """
        Create a recurrent decoder with attention and key value attention over a knowledgebase.

        :param rnn_type: rnn type, valid options: "lstm", "gru"
        :param emb_size: target embedding size
        :param hidden_size: size of the RNN
        :param encoder: encoder connected to this decoder
        :param attention: type of attention, valid options: "bahdanau", "luong"
        :param num_layers: number of recurrent layers
        :param vocab_size: target vocabulary size
        :param hidden_dropout: Is applied to the input to the attentional layer.
        :param dropout: Is applied between RNN layers.
        :param emb_dropout: Is applied to the RNN input (word embeddings).
        :param init_hidden: If "bridge" (default), the decoder hidden states are
            initialized from a projection of the last encoder state,
            if "zeros" they are initialized with zeros,
            if "last" they are identical to the last encoder state
            (only if they have the same size)
        :param input_feeding: Use Luong's input feeding.
        :param freeze: Freeze the parameters of the decoder during training.
        :param k_hops: how many kvr attention layers?
        :param kb_max: tuple with maximum size of knowledgebase for each of its dimensions
        :param kb_input_feeding: bool whether to use kb scores of prev timestep as input to attention's feeding layer
        :param kwargs:
        """

        super(KeyValRetRNNDecoder, self).__init__(rnn_type=rnn_type,\
            emb_size=emb_size, hidden_size=hidden_size,encoder=encoder,\
                attention=attention, num_layers=num_layers, vocab_size=vocab_size,
                hidden_dropout=hidden_dropout, dropout=dropout, emb_dropout=emb_dropout,\
                    init_hidden=init_hidden, input_feeding=input_feeding, freeze=freeze,\
                        kwargs=kwargs)

        self.timer = Timer(printout=False)

        self.emb_dropout = torch.nn.Dropout(p=emb_dropout, inplace=False)
        self.type = rnn_type
        self.hidden_dropout = torch.nn.Dropout(p=hidden_dropout, inplace=False)
        self.hidden_size = hidden_size
        assert self.hidden_size, self.hidden_size
        self.emb_size = emb_size

        rnn = nn.GRU if rnn_type == "gru" else nn.LSTM

        self.input_feeding = input_feeding
        if self.input_feeding: # Luong-style
            # combine embedded prev word +attention vector before feeding to rnn
            self.rnn_input_size = emb_size + hidden_size
        else:
            # just feed prev word embedding
            self.rnn_input_size = emb_size

        # the decoder RNN
        self.rnn = rnn(self.rnn_input_size, hidden_size, num_layers,
                       batch_first=True,
                       dropout=dropout if num_layers > 1 else 0.)

        # combine output with context vector before output layer (Luong-style)
        self.att_vector_layer = nn.Linear(
            hidden_size + encoder.output_size, hidden_size, bias=True)

        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)
        self._output_size = vocab_size

        if attention == "bahdanau":
            self.attention = BahdanauAttention(hidden_size=hidden_size,
                                               key_size=encoder.output_size,
                                               query_size=hidden_size)
        elif attention == "luong":
            self.attention = LuongAttention(hidden_size=hidden_size,
                                            key_size=encoder.output_size)
        else:
            raise ConfigurationError("Unknown attention mechanism: %s. "
                                     "Valid options: 'bahdanau', 'luong'."
                                     % attention)

        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # to initialize from the final encoder state of last layer
        self.init_hidden_option = init_hidden
        if self.init_hidden_option == "bridge":
            self.bridge_layer = nn.Linear(
                encoder.output_size, hidden_size, bias=True)
        elif self.init_hidden_option == "last":
            if encoder.output_size != self.hidden_size:
                if encoder.output_size != 2*self.hidden_size:  # bidirectional
                    raise ConfigurationError(
                        "For initializing the decoder state with the "
                        "last encoder state, their sizes have to match "
                        "(encoder: {} vs. decoder: {})".format(
                            encoder.output_size, self.hidden_size))
        if freeze:
            freeze_params(self)

        # kvr attention after bahdanau attention:
        # multihops:
        self.k_hops = k_hops # how many kvr attention modules?

        # make sure kb_max is iterable, e.g. tuple of ints, instead of a single int
        if not hasattr(kb_max, "__iter__"): kb_max = (kb_max,)
        self.kb_max = kb_max 
        # tuple of maximum size for each knowledgebase dimension
        # for each dimension, 
        # e.g. subj and relation or city x weekday x weather_attribute
        # stores maximum allowable size for this dataset
        # e.g. (7,3) for 7 weekdays and 3 weather attributes

        self.kb_dims = len(self.kb_max)
        allowableDims = [1,2]
        if self.kb_dims not in allowableDims:
            raise NotImplementedError(f"Multidimensional KB attentions only implemented for n={allowableDims}; n={self.kb_dims}; kb_max:{self.kb_max}")

        # as in joeynmt.search.beam_search, repeat and tile KB dimension along each dimension:
        # dims before: repeat value along flat kb_total dimension for each dimension
        #           (product of all previous dimensions)
        # dims after: tile value along flat kb_total dimension for each dimension
        #           (product of all successive dimensions)

        # precalculate these products; use them for indexing in attention hop
        self.dims_before = [product(self.kb_max[:dim]) for dim in range(self.kb_dims)]
        self.dims_after = [product(self.kb_max[:dim:-1]) for dim in range(self.kb_dims)]
        self.kb_total = product(self.kb_max)

        self.kb_feed_rnn = kb_feed_rnn # bool of whether to use LSTM (True) or feed forward network (False) for input feeding (LSTM remembers everything)

        # if yes, use only one att module (per dim) for all hops 
        # and feed its output into itself 
        self.same_module_for_all_hops = same_module_for_all_hops
        if self.same_module_for_all_hops:
            _k_hops = 1
            _k_hops_same_module = self.k_hops 
        else:
            _k_hops = self.k_hops
            _k_hops_same_module = 1 
            
        self.kb_multihead_feed = bool(kb_multihead_feed)

        # list of [kvr for dim 0, kvr for dim 1] * k_hops
        self.kvr_attention = nn.ModuleList([
                                KeyValRetAtt(hidden_size=hidden_size, # use same hidden size as decoder
                                            key_size=kb_key_emb_size, 
                                            query_size=hidden_size, # queried with decoder hidden
                                            kb_max=self.kb_max, # maximum key size for the attention module for this KB dimension,e.g. subj = 10, rel = 5
                                            dim=i%self.kb_dims,
                                            feed_rnn=self.kb_feed_rnn,
                                            num_layers=num_layers,
                                            dropout=dropout,
                                            pad_keys=True,
                                            multihead_feed=self.kb_multihead_feed,
                                            )
                                for i in range(_k_hops * self.kb_dims)] * _k_hops_same_module
                                )
        self.kb_input_feeding = kb_input_feeding

    def _kvr_att_step(self, kb_utils_dims_cache, kb_feed_hidden_cache, query, kb_mask=None):
        return kvr_att_step(
                kb_utils_dims_cache, kb_feed_hidden_cache, query,
                self.kvr_attention, 
                self.k_hops, self.kb_dims, self.curr_dims_before, self.curr_dims_after,
                self.curr_kb_total, self.curr_kb_sizes, self.kb_multihead_feed, kb_mask=kb_mask
        )

    def _init_proj_keys(self, kb_keys):
        return init_proj_keys(self.kvr_attention, self.kb_dims, self.k_hops, kb_keys)

       
    def _check_shapes_input_forward_step(self,
                                         prev_embed: Tensor,
                                         prev_att_vector: Tensor,
                                         encoder_output: Tensor,
                                         src_mask: Tensor,
                                         hidden: Tensor) -> None:
        """
        Make sure the input shapes to `self._forward_step` are correct.
        Same inputs as `self._forward_step`.

        :param prev_embed:
        :param prev_att_vector:
        :param encoder_output:
        :param src_mask:
        :param hidden:
        """
        assert prev_embed.shape[1:] == torch.Size([1, self.emb_size])
        assert prev_att_vector.shape[1:] == torch.Size(
            [1, self.hidden_size])
        assert prev_att_vector.shape[0] == prev_embed.shape[0]
        assert encoder_output.shape[0] == prev_embed.shape[0]
        assert len(encoder_output.shape) == 3
        assert src_mask.shape[0] == prev_embed.shape[0]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[2] == encoder_output.shape[1]
        if isinstance(hidden, tuple):  # for lstm
            hidden = hidden[0]
        assert hidden.shape[0] == self.num_layers
        assert hidden.shape[1] == prev_embed.shape[0]
        assert hidden.shape[2] == self.hidden_size

    def _check_shapes_input_forward(self,
                                    trg_embed: Tensor,
                                    encoder_output: Tensor,
                                    encoder_hidden: Tensor,
                                    src_mask: Tensor,
                                    hidden: Tensor = None,
                                    prev_att_vector: Tensor = None) -> None:
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param trg_embed:
        :param encoder_output:
        :param encoder_hidden:
        :param src_mask:
        :param hidden:
        :param prev_att_vector:
        """

        assert len(encoder_output.shape) == 3
        assert len(encoder_hidden.shape) == 2
        assert encoder_hidden.shape[-1] == encoder_output.shape[-1]
        assert src_mask.shape[1] == 1
        assert src_mask.shape[0] == encoder_output.shape[0]
        assert src_mask.shape[2] == encoder_output.shape[1]
        assert trg_embed.shape[0] == encoder_output.shape[0]
        assert trg_embed.shape[2] == self.emb_size
        if hidden is not None:
            if isinstance(hidden, tuple):  # for lstm
                hidden = hidden[0]
            assert hidden.shape[1] == encoder_output.shape[0]
            assert hidden.shape[2] == self.hidden_size
        if prev_att_vector is not None:
            assert prev_att_vector.shape[0] == encoder_output.shape[0]
            assert prev_att_vector.shape[2] == self.hidden_size
            assert prev_att_vector.shape[1] == 1

    def _forward_step(self,
                      prev_embed: Tensor,
                      prev_att_vector: Tensor,  # context or att vector
                      encoder_output: Tensor,
                      src_mask: Tensor,
                      hidden: Tensor,
                      kb_mask: Tensor = None,
                      kb_utils_dims_cache: List[Union[Tensor, None]] = None,
                      kb_feed_hidden_cache: List[Union[Tensor, None]] = None) -> (Tensor, Tensor, Tensor):
        """
        Perform a single decoder step (1 token).

        1. `rnn_input`: concat(prev_embed, prev_att_vector [possibly empty])
        2. update RNN with `rnn_input`
        3. calculate attention and context/attention vector

        :param prev_embed: embedded previous token,
            shape (batch_size, 1, embed_size)
        :param prev_att_vector: previous attention vector,
            shape (batch_size, 1, hidden_size)
        :param encoder_output: encoder hidden states for attention context,
            shape (batch_size, src_length, encoder.output_size)
        :param src_mask: src mask, 1s for area before <eos>, 0s elsewhere
            shape (batch_size, 1, src_length)
        :param hidden: previous hidden state,
            shape (num_layers, batch_size, hidden_size)
        :param kb_mask: mask to prevent unassigned KB entries from being given score (happens in scheduling)
        :parma kb_utils_dims_cache: provided if kb input feeding is true, else None
        :parma kb_feed_hidden_cache: provided if kb input feeding is true, else None
        :return:
            - att_vector: new attention vector (batch_size, 1, hidden_size),
            - hidden: new hidden state with shape (batch_size, 1, hidden_size),
            - att_probs: attention probabilities (batch_size, 1, src_len)
        """

        # shape checks
        self._check_shapes_input_forward_step(prev_embed=prev_embed,
                                              prev_att_vector=prev_att_vector,
                                              encoder_output=encoder_output,
                                              src_mask=src_mask,
                                              hidden=hidden)

        if self.input_feeding:
            # concatenate the input with the previous attention vector
            rnn_input = torch.cat([prev_embed, prev_att_vector], dim=2)
        else:
            rnn_input = prev_embed

        rnn_input = self.emb_dropout(rnn_input)

        # rnn_input: batch x 1 x emb+2*enc_size
        _, hidden = self.rnn(rnn_input, hidden)
        # hidden: num_layers x batch x hidden
        # provided exactly as is on next step

        # use new (top) decoder layer as attention query
        if isinstance(hidden, tuple): # for lstms, states are tuples of tensors
            query = hidden[0][-1].unsqueeze(1)
        else: # for grus, states are just a tensor
            query = hidden[-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]


        # only use last layer for attention mechanism
        # key projections are pre-computed
        context, att_probs = self.attention(
            query=query, values=encoder_output, mask=src_mask)
        
        ### -------------------- start KVR attention -------------------- ###
        
        # terminology:
        # t: timestep, k: multihop, dim: KB dimension, u: utilities

        with self.timer("kbatt step", logname=f"kbatt step;k={self.k_hops},n={len(self.kb_max)},mhead={self.kb_multihead_feed},feedRNN={self.kb_feed_rnn}"):

            # initialize caches
            batch_size = att_probs.size(0)
            if kb_utils_dims_cache == None:
                # cache for attention heads for each dim to access previous utilities of corresponding head of previous hop
                with torch.no_grad():
                    kb_utils_dims_cache = [
                        query.new_zeros((query.size(0), 1, dim)).to(device=query.device)
                        for dim in self.kb_max 
                    ] 

            if kb_feed_hidden_cache == None:
                # cache of initial hidden states for feeding GRU
                if self.kb_feed_rnn:
                    init_kb_hidden = hidden[0] if isinstance(hidden, tuple) else hidden # GRU vs LSTM output types
                    kb_feed_hidden_cache = [init_kb_hidden] * (self.kb_dims * self.k_hops)
                else:
                    kb_feed_hidden_cache = [None] * (self.kb_dims * self.k_hops)

            # multiple attention hops

            u_t, kb_utils_dims_cache, kb_feed_hidden_cache = self._kvr_att_step(
                kb_utils_dims_cache, kb_feed_hidden_cache, query, kb_mask=kb_mask
            )

        # u_t = batch x 1 x kb_total

        ### -------------------- end KVR attention -------------------- ###

        # return attention vector (Luong)
        # combine context with decoder hidden state before prediction
        att_vector_input = torch.cat([query, context], dim=2)
        # batch x 1 x (2*)enc_size+hidden_size
        
        att_vector_input = self.hidden_dropout(att_vector_input)

        att_vector = torch.tanh(self.att_vector_layer(att_vector_input))

        # output: batch x 1 x hidden_size
        return att_vector, hidden, att_probs, u_t, kb_utils_dims_cache, kb_feed_hidden_cache

    def forward(self,
                trg_embed: Tensor,
                encoder_output: Tensor,
                encoder_hidden: Tensor,
                src_mask: Tensor,
                unroll_steps: int,
                hidden: Tensor = None,
                prev_att_vector: Tensor = None,
                kb_keys: Union[Tuple, Tensor] = None,
                k_hops: int = 1,
                kb_mask: Tensor = None,
                **kwargs) \
            -> (Tensor, Tensor, Tensor, Tensor):
        """
         Unroll the decoder one step at a time for `unroll_steps` steps.
         For every step, the `_forward_step` function is called internally.

         During training, the target inputs (`trg_embed') are already known for
         the full sequence, so the full unrol is done.
         In this case, `hidden` and `prev_att_vector` are None.

         For inference, this function is called with one step at a time since
         embedded targets are the predictions from the previous time step.
         In this case, `hidden` and `prev_att_vector` are fed from the output
         of the previous call of this function (from the 2nd step on).

         `src_mask` is needed to mask out the areas of the encoder states that
         should not receive any attention,
         which is everything after the first <eos>.
neural transformer why no nonlinearities
         The `encoder_output` are the hidden states from the encoder and are
         used as context for the attention.

         The `encoder_hidden` is the last encoder hidden state that is used to
         initialize the first hidden decoder state
         (when `self.init_hidden_option` is "bridge" or "last").

        :param trg_embed: emdedded target inputs,
            shape (batch_size, trg_length, embed_size)
        :param encoder_output: hidden states from the encoder,
            shape (batch_size, src_length, encoder.output_size)
        :param encoder_hidden: last state from the encoder,
            shape (batch_size x encoder.output_size)
        :param kb_keys: knowledgebase keys associated with batch
            knowledgebase[0]: batch.kbsrc: kb_size x 1
        :param src_mask: mask for src states: 0s for padded areas,
            1s for the rest, shape (batch_size, 1, src_length)
        :param unroll_steps: number of steps to unrol the decoder RNN
        :param hidden: previous decoder hidden state,
            if not given it's initialized as in `self.init_hidden`,
            shape (num_layers, batch_size, hidden_size)
        :param prev_att_vector: previous attentional vector,
            if not given it's initialized with zeros,
            shape (batch_size, 1, hidden_size)
        :param kb_keys: knowledgebase keys associated with batch
        :param k_hops: number of kvr attention forward passes to do
        :return:
            - outputs: shape (batch_size, unroll_steps, vocab_size),
            - hidden: last hidden state (num_layers, batch_size, hidden_size),
            - att_probs: attention probabilities
                with shape (batch_size, unroll_steps, src_length),
            - att_vectors: attentional vectors
                with shape (batch_size, unroll_steps, hidden_size)
            - kb_probs: kb att probabilities
                with shape (batch_size, unroll_steps, kb_size)
        """
        with self.timer(f"fwd", logname="fwd"):
            # shape checks
            self._check_shapes_input_forward(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                hidden=hidden,
                prev_att_vector=prev_att_vector)
            

            # initialize decoder hidden state from final encoder hidden state
            if hidden is None:
                hidden = self._init_hidden(encoder_hidden)

            # pre-compute projected encoder outputs
            # (the "keys" for the attention mechanism)
            # this is only done for efficiency
            if hasattr(self.attention, "compute_proj_keys"):
                self.attention.compute_proj_keys(keys=encoder_output)
            
            if kb_keys is not None:

                ### end setup proj keys and dimension reshape helper lists
                if not isinstance(kb_keys, tuple): 
                    assert self.kb_dims == 1, (self.kb_dims, kb_keys)
                    kb_keys = (kb_keys,)

                self._init_proj_keys(kb_keys) # sets curr_kb_size for each

                self.curr_kb_sizes = [self.kvr_attention[dim].curr_kb_size for dim in range(self.kb_dims)]

                self.curr_dims_before = [product(self.curr_kb_sizes[:dim]) for dim in range(self.kb_dims)]
                self.curr_dims_after = [product(self.curr_kb_sizes[:dim:-1]) for dim in range(self.kb_dims)]
                self.curr_kb_total = product(self.curr_kb_sizes)


                ### end setup proj keys and dimension reshape helper lists

            att_vectors = [] 
            att_probs = []
            kb_probs = []

            batch_size = encoder_output.size(0)

            if prev_att_vector is None:
                with torch.no_grad():
                    prev_att_vector = encoder_output.new_zeros(
                        [batch_size, 1, self.hidden_size]
                        )

            prev_kb_utils = None
            prev_kb_feed_hiddens = None
            # unroll the decoder RNN for `unroll_steps` steps

            for i in range(unroll_steps):

                prev_embed = trg_embed[:, i].unsqueeze(1)  # batch, 1, emb

                prev_att_vector, hidden, att_prob, u_t, prev_kb_utils, prev_kb_feed_hiddens = self._forward_step(
                    prev_embed=prev_embed,
                    prev_att_vector=prev_att_vector,
                    encoder_output=encoder_output,
                    src_mask=src_mask,
                    hidden=hidden,
                    kb_mask=kb_mask,
                    kb_utils_dims_cache=prev_kb_utils,
                    kb_feed_hidden_cache=prev_kb_feed_hiddens)
                    
                if not self.kb_input_feeding:
                    prev_kb_utils = None

                att_vectors.append(prev_att_vector)
                att_probs.append(att_prob)
                kb_probs.append(u_t)

            # batch, unroll_steps, hidden_size
            att_vectors = torch.cat(att_vectors, dim=1)
            # batch, unroll_steps, src_length
            att_probs = torch.cat(att_probs, dim=1)
            # batch, unroll_steps, KB
            kb_probs = torch.cat(kb_probs, dim=1)

        return hidden, att_probs, att_vectors, kb_probs, prev_kb_utils, prev_kb_feed_hiddens

    def _init_hidden(self, encoder_final: Tensor = None) \
            -> (Tensor, Optional[Tensor]):
        """
        Returns the initial decoder state,
        conditioned on the final encoder state of the last encoder layer.

        In case of `self.init_hidden_option == "bridge"`
        and a given `encoder_final`, this is a projection of the encoder state.

        In case of `self.init_hidden_option == "last"`
        and a size-matching `encoder_final`, this is set to the encoder state.
        If the encoder is twice as large as the decoder state (e.g. when
        bi-directional), just use the forward hidden state.

        In case of `self.init_hidden_option == "zero"`, it is initialized with
        zeros.

        For LSTMs we initialize both the hidden state and the memory cell
        with the same projection/copy of the encoder hidden state.

        All decoder layers are initialized with the same initial values.

        :param encoder_final: final state from the last layer of the encoder,
            shape (batch_size, encoder_hidden_size)
        :return: hidden state if GRU, (hidden state, memory cell) if LSTM,
            shape (batch_size, hidden_size)
        """
        batch_size = encoder_final.size(0)

        # for multiple layers: is the same for all layers
        if self.init_hidden_option == "bridge" and encoder_final is not None:
            # num_layers x batch_size x hidden_size
            hidden = torch.tanh(
                    self.bridge_layer(encoder_final)).unsqueeze(0).repeat(
                    self.num_layers, 1, 1)
        elif self.init_hidden_option == "last" and encoder_final is not None:
            # special case: encoder is bidirectional: use only forward state
            if encoder_final.shape[1] == 2*self.hidden_size:  # bidirectional
                encoder_final = encoder_final[:, :self.hidden_size]
            hidden = encoder_final.unsqueeze(0).repeat(self.num_layers, 1, 1)
        else:  # initialize with zeros
            with torch.no_grad():
                hidden = encoder_final.new_zeros(
                    self.num_layers, batch_size, self.hidden_size
                    )

        return (hidden, hidden) if isinstance(self.rnn, nn.LSTM) else hidden

    def __repr__(self):
        return "RecurrentDecoder(rnn=%r, attention=%r)" % (
            self.rnn, self.attention)


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class TransformerDecoder(Decoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 kb_task: bool = False,
                 infeedkb: bool = False,
                 outfeedkb: bool = False,
                 double_decoder: bool = False,
                 **kwargs):
        """
        Initialize a Transformer decoder.
        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kb_task: performing kb_task or not? used in layer init
        :param kb_max: maximum kb size
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self.timer = Timer(printout=False)

        self._hidden_size = hidden_size
        self._output_size = hidden_size # FIXME see interface classes at top of this file: vocab layer moved to Generator

        # create num_layers decoder layers and put them in a list
        self.kb_layers = nn.ModuleList([
            TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, kb_task=kb_task, 
                tfstyletf=True)
                     for _ in range(num_layers)])

        self.double_decoder = double_decoder
        if self.double_decoder:
            self.layers = nn.ModuleList([
                TransformerDecoderLayer(
                    size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                    dropout=dropout,
                    tfstyletf=False)
                        for _ in range(num_layers)])
            
        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)

        self.kb_task = kb_task
        if self.kb_task:
            self.kb_energy_layer = nn.Linear(num_heads, 1)
            self.num_feed_layers = int(num_layers//2) # half the GRU layers as tf layers for feeding

        if infeedkb:
            # feed previous kb output into next kb module using LSTM?
            self.feed_in_rnn = nn.LSTM(   
                                    hidden_size, hidden_size, # hidden size must be == decoder hidden size
                                    self.num_feed_layers, batch_first=True, 
                                    dropout=dropout if num_layers > 1 else 0.
            )

        else:
            self.feed_in_rnn = None

        if outfeedkb:
            # feed kb output back into hidden state using this abomination?
            # TODO put this into modulelist optionally (separate feeding module per tf layer)

            self.feed_out_nn = VariableCellLSTM(hidden_size,
                                self.num_feed_layers, batch_first=True,
                                dropout=dropout if self.num_feed_layers > 1 else 0.)
                

        else:
            self.feed_out_nn = None
        
        
        self.rnn = self.feed_in_rnn # NOTE this rnn attribute gets checked in initialization if xavier init is used 

        if freeze:
            freeze_params(self)
        

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                encoder_hidden: Tensor = None,
                src_mask: Tensor = None,
                unroll_steps: int = None,
                hidden: Tensor = None,
                trg_mask: Tensor = None,
                kb_keys: Tensor = None,
                kb_values_embed: Tensor = None,
                kb_mask: Tensor = None,
                **kwargs):
        """
        Transformer decoder forward pass.
        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kb_keys: knowledgebase keys: B x KB x TRG_EMB
        :param kwargs:
        :return:
        """
        with self.timer(f"fwd", logname=f"fwd"):

            # kb_mask = B x M x KB; masks unfilled slots as in calendar
            kb_mask = kb_mask.unsqueeze(1).repeat(1,trg_embed.size(1),1).to(dtype=torch.bool)

            assert trg_mask is not None, "trg_mask required for Transformer"

            x = self.pe(trg_embed) # add position encoding to word embedding
            x = self.emb_dropout(x)
            x_cache = x

            trg_mask = trg_mask & subsequent_mask(
                trg_embed.size(1)).type_as(trg_mask)

            # k fwd pass thru k layers (with k x KVR Multihop Attention)
            with torch.no_grad():
                kb_output = x.new_zeros(x.shape)
                # knowledgebase feeding hiddens at time step k
                if self.feed_in_rnn is not None:
                    kb_feed_in_hidden = x.new_zeros(self.num_feed_layers, x.size(1), self._hidden_size)
                else:
                    kb_feed_in_hidden = None
                if self.feed_out_nn is not None:
                    kb_feed_out_hidden = x.new_zeros(x.size())
                else:
                    kb_feed_out_hidden = None

            with self.timer(f"main loop, num_l={len(self.kb_layers)}, infeedkb={self.feed_in_rnn is not None}, outfeedkb={self.feed_out_nn is not None}", logname=f"main_loop"):
                for j, kb_layer in enumerate(self.kb_layers):
                    x, kb_output, kb_att, kb_feed_in_hidden, kb_feed_out_hidden = kb_layer(
                                x=x, 
                                memory=encoder_output, 
                                src_mask=src_mask, 
                                trg_mask=trg_mask, 
                                kb_mask=kb_mask,
                                kb_keys=kb_keys, 
                                kb_values_embed=kb_values_embed,
                                prev_kb_output=kb_output,
                                kb_feed_in=self.feed_in_rnn,
                                kb_feed_in_hidden=kb_feed_in_hidden,
                                kb_feed_out=self.feed_out_nn,
                                kb_feed_out_hidden=kb_feed_out_hidden,
                                )
                    assert (x==x).all(), (j, (kb_att==float("-inf")).any()) # FIXME remove me (check for NaN)
            x = self.layer_norm(x)
            if self.double_decoder:
                # normal transformer decoder block
                with self.timer(f"side loop, num_l={len(self.layers)}"):
                    for j, layer in enumerate(self.layers):
                        x_side , _, _, _, _ = layer(
                                    x=x_cache, 
                                    memory=encoder_output, 
                                    src_mask=src_mask, 
                                    trg_mask=trg_mask, 
                                    )
                    x_side = self.layer_norm(x_side)
                    x = (x, x_side) # pass tuple of both layer loop's hidden states along for the generator to 'merge'

                    for i, t in enumerate(x):
                        # check for NaN values because of weird torch autograd bug with log softmax
                        assert (t==t).all().item(), (i, t,(t!=t), )

            if self.kb_task:
                assert kb_keys is not None
                # fuse the heads of last kb attentions using energy layer
                # kb_att = B x num_h x M x KB
                kb_probs = self.kb_energy_layer(kb_att.transpose(1,3)) # B x KB x M x 1
                kb_probs = kb_probs.transpose(1,2).squeeze(-1) # B x M x KB
            else:
                kb_probs = None

        # decoder output signature is:
        # return hidden, att_probs, att_vectors, kb_probs, hidden_cache, feed_cache
        return None, None, x, kb_probs, None, None


# pylint: disable=arguments-differ,too-many-arguments
# pylint: disable=too-many-instance-attributes, unused-argument
class TransformerKBrnnDecoder(TransformerDecoder):
    """
    A transformer decoder with N masked layers.
    Decoder layers are masked so that an attention head cannot see the future.
    """

    def __init__(self,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 hidden_size: int = 512,
                 ff_size: int = 2048,
                 dropout: float = 0.1,
                 emb_dropout: float = 0.1,
                 vocab_size: int = 1,
                 freeze: bool = False,
                 kb_task: bool=False,
                 kb_max: Tuple = (256,),
                 kb_key_emb_size: int = 1,
                 k_hops: int = 1,
                 kb_input_feeding: bool = True,
                 kb_feed_rnn: bool = True,
                 same_module_for_all_hops: bool = False,
                 kb_multihead_feed: bool = False,
                 **kwargs):
        """
        Initialize a Transformer decoder.
        :param num_layers: number of Transformer layers
        :param num_heads: number of heads for each layer
        :param hidden_size: hidden size
        :param ff_size: position-wise feed-forward size
        :param dropout: dropout probability (1-keep)
        :param emb_dropout: dropout probability for embeddings
        :param vocab_size: size of the output vocabulary
        :param freeze: set to True keep all decoder parameters fixed
        :param kb_task: performing kb_task or not? used in layer init
        :param kb_max: maximum kb size for each dimension FIXME only implemented for 1 d for ATM
        :param kwargs:
        """
        super(TransformerDecoder, self).__init__()

        self.timer = Timer(printout=False)

        self._hidden_size = hidden_size
        self._output_size = hidden_size # FIXME see interface 

        self._hidden_size = hidden_size
        self._output_size = vocab_size

        # create num_layers decoder layers and put them in a list
        self.layers = nn.ModuleList([TransformerDecoderLayer(
                size=hidden_size, ff_size=ff_size, num_heads=num_heads,
                dropout=dropout, tfstyletf=False) for _ in range(num_layers)])

        self.pe = PositionalEncoding(hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

        self.emb_dropout = nn.Dropout(p=emb_dropout)
        self.output_layer = nn.Linear(hidden_size, vocab_size, bias=False)

        if kb_task:
            if not hasattr(kb_max, "__iter__"): 
                assert type(kb_max) == int, (kb_max, type(kb_max), "specify this differently in config plz")
                kb_max = (kb_max,)
            self.kb_max = kb_max 
            self.kb_feed_rnn = kb_feed_rnn

            self.kb_layer_norm = nn.LayerNorm(hidden_size, eps=1e-6)

            # kvr attention after bahdanau attention:
            # multihops:
            self.k_hops = k_hops # how many kvr attention modules?

            # make sure kb_max is iterable, e.g. tuple of ints, instead of a single int
            if not hasattr(kb_max, "__iter__"): kb_max = (kb_max,)
            self.kb_max = kb_max 
            # tuple of maximum size for each knowledgebase dimension
            # for each dimension, 
            # e.g. subj and relation or city x weekday x weather_attribute
            # stores maximum allowable size for this dataset
            # e.g. (7,3) for 7 weekdays and 3 weather attributes

            self.kb_dims = len(self.kb_max)
            allowableDims = [1,2]
            if self.kb_dims not in allowableDims:
                raise NotImplementedError(f"Multidimensional KB attentions only implemented for n={allowableDims}; n={self.kb_dims}; kb_max:{self.kb_max}")

            # as in joeynmt.search.beam_search, repeat and tile KB dimension along each dimension:
            # dims before: repeat value along flat kb_total dimension for each dimension
            #           (product of all previous dimensions)
            # dims after: tile value along flat kb_total dimension for each dimension
            #           (product of all successive dimensions)

            # precalculate these products; use them for indexing in attention hop
            self.dims_before = [product(self.kb_max[:dim]) for dim in range(self.kb_dims)]
            self.dims_after = [product(self.kb_max[:dim:-1]) for dim in range(self.kb_dims)]
            self.kb_total = product(self.kb_max)

            assert len(self.dims_before) == len(self.dims_after) == self.kb_dims

            self.kb_feed_rnn = bool(kb_feed_rnn) # bool of whether feed with GRU (True) or feed forward network (False) for input feeding (LSTM remembers everything)
            self.kb_layers = int(num_layers//2) # half the GRU layers as tf layers if GRU

            # if yes, use only one att module (per dim) for all hops 
            # and feed its output into itself 
            self.same_module_for_all_hops = same_module_for_all_hops
            if self.same_module_for_all_hops:
                _k_hops = 1
                _k_hops_same_module = self.k_hops 
            else:
                _k_hops = self.k_hops
                _k_hops_same_module = 1 

            self.kb_input_feeding = kb_input_feeding
            self.kb_multihead_feed = bool(kb_multihead_feed)
                
            # list of [kvr for dim 0, kvr for dim 1] * k_hops
            self.kvr_attention = nn.ModuleList([
                                KeyValRetAtt(
                                            hidden_size=hidden_size, # use same hidden size as decoder
                                            key_size=kb_key_emb_size, 
                                            query_size=hidden_size, # queried with decoder hidden
                                            kb_max=self.kb_max, # maximum key size for the attention module for this KB dimension,e.g. subj = 10, rel = 5
                                            dim=i%self.kb_dims,
                                            feed_rnn=self.kb_feed_rnn,
                                            num_layers=self.kb_layers,
                                            dropout=dropout,
                                            pad_keys=True,
                                            multihead_feed=self.kb_multihead_feed,
                                            )

                                for i in range(_k_hops * self.kb_dims)] * _k_hops_same_module
                                )
        if freeze:
            freeze_params(self)

    def _kvr_att_step(self, kb_utils_dims_cache, kb_feed_hidden_cache, query, kb_mask=None):
        return kvr_att_step(
                kb_utils_dims_cache, kb_feed_hidden_cache, query,
                self.kvr_attention, 
                self.k_hops, 
                self.kb_dims, self.curr_dims_before, self.curr_dims_after,
                self.curr_kb_total, self.curr_kb_sizes,
                self.kb_multihead_feed, kb_mask=kb_mask
        )
    
    def _init_proj_keys(self, kb_keys):
        return init_proj_keys(self.kvr_attention, self.kb_dims, self.k_hops, kb_keys)

    def forward(self,
                trg_embed: Tensor = None,
                encoder_output: Tensor = None,
                encoder_hidden: Tensor = None,
                src_mask: Tensor = None,
                unroll_steps: int = None,
                hidden: Tensor = None,
                trg_mask: Tensor = None,
                kb_keys: Union[Tensor, Tuple] = None,
                kb_mask: Tensor = None,
                kb_utils_dims_cache: List[Union[Tensor, None]] = None,
                kb_feed_hidden_cache: List[Union[Tensor,None]] = None,
                **kwargs):
        """
        Transformer decoder forward pass.
        :param trg_embed: embedded targets
        :param encoder_output: source representations
        :param encoder_hidden: unused
        :param src_mask:
        :param unroll_steps: unused
        :param hidden: unused
        :param trg_mask: to mask out target paddings
                         Note that a subsequent mask is applied here.
        :param kb_keys: knowledgebase keys: B x KB x TRG_EMB
        :param kwargs:
        :return:
        """
        with self.timer("fwd", logname=f"fwd"):

            assert trg_mask is not None, "trg_mask required for Transformer"

            x = self.pe(trg_embed)  # add position encoding to word embedding
            x = self.emb_dropout(x)

            trg_mask = trg_mask & subsequent_mask(
                trg_embed.size(1)).type_as(trg_mask)

            with self.timer(f"main loop, num_l={len(self.layers)}", logname=f"main_loop"):
                for i, layer in enumerate(self.layers):
                    x, _, _, _, _ = layer(x=x, memory=encoder_output, src_mask=src_mask, trg_mask=trg_mask)

            x = self.layer_norm(x)

            with self.timer(f"kbatt step;k={self.k_hops},n={len(self.kb_max)},mhead={self.kb_multihead_feed},feedRNN={self.kb_feed_rnn}", logname=f"kbatt_loop"):
                # Multiheaded KVR Attention fwd pass
                if kb_keys is not None:
                    
                    ### setup proj keys and current KB dimensions
                    if not isinstance(kb_keys, tuple): 
                        assert self.kb_dims == 1, (self.kb_dims, kb_keys)
                        kb_keys = (kb_keys,)

                    self._init_proj_keys(kb_keys)

                    self.curr_kb_sizes = [self.kvr_attention[dim].curr_kb_size for dim in range(self.kb_dims)]
                    self.curr_dims_before = [product(self.curr_kb_sizes[:dim]) for dim in range(self.kb_dims)]
                    self.curr_dims_after = [product(self.curr_kb_sizes[:dim:-1]) for dim in range(self.kb_dims)]
                    self.curr_kb_total = product(self.curr_kb_sizes)

                    ### end setup proj keys and current KB dimensions

                    batch_size = x.size(0)

                    ### setup caches for access to different utilities and hiddens of t-1 and aggregate utilities
                    u = []
                    with torch.no_grad():
                        # dectivate autograd before creating zero tensors to avoid
                        # invoking the wrath of cuda-thulhu
                        if kb_utils_dims_cache == None:

                            # initialize with maximum length allowed for each dimension;
                            # curb in add_kb_util... procedure
                            kb_utils_dims_cache = [
                                x.new_zeros((batch_size, 1, dim)).to(device=x.device) # ".to" probably unnecessary
                                for dim in self.kb_max
                            ] 

                        if kb_feed_hidden_cache == None:

                            # if not present initialize with first timestep of encoder_output
                            if self.kb_feed_rnn:
                                kb_feed_hidden_cache = [
                                    encoder_output.new_zeros(self.kb_layers, batch_size, self._hidden_size)
                                ] * (self.kb_dims * self.k_hops)
                            else:
                                kb_feed_hidden_cache = [None] * (self.kb_dims * self.k_hops)

                    ### end setup caches 

                    # Recurrent unroll of KB attentions
                    for t in range(x.size(1)):

                        query = x[:,t,:].unsqueeze(1).clone()

                        # get u_t and update dimension-wise utility caches and module wise hidden caches
                        u_t, kb_utils_dims_cache, kb_feed_hidden_cache = self._kvr_att_step(
                            kb_utils_dims_cache, kb_feed_hidden_cache, query, kb_mask=kb_mask
                            )
                        u += [u_t]

                    kb_probs = torch.cat(u, dim=1)
                    
                else:
                    kb_probs = None
                
        # decoder output signature is:
        # return hidden, att_probs, att_vectors, kb_probs, kb_utils_dims_cache, kb_feed_hidden_cache
        return None, None, x, kb_probs, kb_utils_dims_cache, kb_feed_hidden_cache
    
    def pad_kb_tensor(self, t: Tensor) -> Tensor:
        # pad kb tensor shaped like B x CURR_KB x TRG_EMB => B x KB_MAX x TRG_EMB
        # used for kb_keys & kb_mask

        assert len(t.shape) == 3, t.shape

        curr_kb_size = t.shape[1]
        padding = self.kb_max[0] - curr_kb_size # FIXME
        assert padding >= 0, f"KB dim of KB tensor (keys or mask) {t.shape} appears to be larger than self.kb_max={self.kb_max} => increase self.kb_max"

        pad_ = torch.zeros(t.shape[0], padding, t.shape[-1]).to(device=t.device)

        return torch.cat([t,pad_], dim=1)

    def __repr__(self):
        return "%s(num_layers=%r, num_heads=%r)" % (
            self.__class__.__name__, len(self.layers),
            self.layers[0].trg_trg_att.num_heads)



class Generator(Gen):
    """
    Functions as output layer for both recurrent and transformer decoders.
    For knowledgebase task, this is also where kb probabilities get added to outputs.

    """

    def __init__(self, dec_hidden_size, vocab_size, add_kb_biases_to_output, double_decoder, **kwargs):
        super(Generator, self).__init__()

        self._hidden_size = dec_hidden_size
        self._output_size = vocab_size

        self.double_decoder = bool(double_decoder)
        self.output_layer = nn.Linear(self._hidden_size, self._output_size, bias=False)

        if not self.double_decoder:
            self.output_layer_wrap = lambda x: self.output_layer(x)
        else:
            # add the outputs of two separate decoder blocks after passing through two separate output layer matrices
            self.side_output_layer = nn.Linear(self._hidden_size, self._output_size, bias=False)
            self.output_layer_wrap = lambda x_1_x_2: self.output_layer(x_1_x_2[0]) + self.side_output_layer(x_1_x_2[1])

        self.add_kb_biases_to_output = bool(add_kb_biases_to_output)

    def forward(self, x, kb_values=None, kb_probs=None, **kwargs):
        # x = 
        # transformer: x (hidden state)
        # recurrent: att_vectors (\tilde{h}_t)

        # kb_probs : batch x unroll x kb
        # kb_values : batch x kb 

        if isinstance(x, tuple):
            assert self.double_decoder, [x_.shape for x_ in x]
            assert hasattr(self, "side_output_layer"), f"tuple hidden state passed to generator but generator isnt in double decoder mode; x={tuple([x_.shape for x_ in x])}"
            assert len(x) == 2, len(x)
            assert x[0].shape == x[1].shape, [x_.shape for x_ in x]
            assert x[0].size(-1) == self._hidden_size, (x[0].shape, self._hidden_size)
        else:
            assert isinstance(x, torch.Tensor), type(x)
            assert x.size(-1) == self._hidden_size, (x.shape, self._hidden_size)

        outputs = self.output_layer_wrap(x) # Batch x Time x Voc
        
        if isinstance(x, tuple):
            assert x[0].shape[:-1] == outputs.shape[:-1], [t.shape for t in [x[0], x[1], outputs]]

        if self.add_kb_biases_to_output and kb_values is not None and kb_probs is not None:

            try:
                _batch, _unroll, _kb = kb_probs.shape
            except Exception as e:
                print(kb_probs.shape)
                raise e

            # kb_values: b x kb => b x time x kb
            kb_values = kb_values.unsqueeze(1).repeat((1, _unroll, 1))

            B = torch.arange(_batch).unsqueeze(1).unsqueeze(1)
            U = torch.arange(_unroll).unsqueeze(1).unsqueeze(0)

            # add v_t = kb_probs to outputs (logits vector in Eric et al.)
            outputs[B, U, kb_values] += kb_probs # bias outputs towards kb_probs

        # compute log probs
        log_probs = F.log_softmax(outputs, dim=-1) 
        # in default joeynmt, log softmax isnt always used before taking the argmax over VOC dimension.
        # now it is always called at the end of model.forward right here
        # doesnt make a difference for taking argmax because log softmax is monotone transformation

        return log_probs 


################## TODO move all of these to helpers ##################

def init_proj_keys(kvr_attentions, kb_dims, k_hops, kb_keys):
    """
    TODO document
    """
    assert isinstance(kb_keys, tuple)
    # kb_keys is: b x kb x emb OR tuple of (b x kb_max_i x emb,) * n
    for dim in range(kb_dims):
        # compute proj keys for all hops
        for hop in range(k_hops):
            kvr_attentions[hop * kb_dims + dim].compute_proj_keys(keys=kb_keys[dim])


def reshape_kb_mask_to_keys_size(kb_mask, kb_keys, kb_total):
    """
    TODO document TODO move to helpers
    """
    if not isinstance(kb_keys, tuple):
        kb_keys = (kb_keys,)

    keys_dim = product([keys.shape[1] for keys in kb_keys])
    kb_pad_len = kb_total - keys_dim
    
    assert kb_pad_len >= 0, f"kb dim of mask {kb_mask.shape}, with product of keys ={keys_dim} appears to be larger than self.kb_total={kb_total} => increase self.kb_total[-1] = {self.kb_total}"

    # FIXME why is this sometimes a tuple of filled pad tensors instead of one? TODO
    kb_mask_padding = torch.full((kb_mask.shape[0], kb_pad_len), fill_value=False)

    if type(kb_mask_padding) == tuple:
        assert False, kb_mask_padding
    if len(kb_mask_padding.shape) < 2:
        assert False, ((kb_mask.shape[0], kb_pad_len), kb_mask_padding.shape)

    assert len(kb_mask.shape) == 2, kb_mask.shape

    kb_mask_padded = torch.cat([ kb_mask, kb_mask_padding.to( dtype = kb_mask.dtype, device = kb_mask.device )], dim=1)

    kb_mask = kb_mask_padded.unsqueeze(1)

    ### end setup proj keys and mask dimensions
    return kb_mask

def kvr_att_step(  utils_dims_cache, kb_feed_hidden_cache, query, 
                                kvr_attentions, k_hops, kb_dims, 
                                dims_before, dims_after, curr_kb_total, curr_kb_sizes,
                                multihead_feed, kb_mask=None):
    """
    Do one time step of kb attention (with k hops and n dimensions). 
    TODO move to helpers

    modifies three lists in place:
    :param util_aggregator: list of utilitites that u_t of this step can be added to in place
    :param kb_utils_dims_cache: list of previous kb hiddens for each dimension
    :param kb_feed_hidden_cache: list of previous feeding LSTM hidden states or None
    :param query: query to all kvr attentions for that step (decoder hidden state or transformer output of corresponding step)
    :param kvr_attentions:
    :param kb_total:
    :param kb_hops:
    :param kb_dims:
    :param dims_before:
    :param dims_after:
    """

    # FIXME XXX TODO test: this worked better inside hop loop maybe?

    # multiple attention hops
    for j in range(k_hops): # self.k_hops == 1 <=> Eric et al version

        # for each hop, do an attention pass for each key representation dimension
        # e.g. KB total = subject x relation
        # or   KB total = city x weekday x weather attribute

        with torch.no_grad():
            # batch x kb_total x hidden
            u_t_k = query.new_zeros(
                (query.size(0), 1, curr_kb_total)
            )

        utils_cache_update = [None] * kb_dims

        for dim in range(kb_dims): # self.kb_dims == 1 <=> Eric et al version

            idx = j * kb_dims + dim

            # feed the entire cache of previous utilities ? as input to multihead feed layer
            prev_u = utils_dims_cache[dim] if not multihead_feed else utils_dims_cache

            # u_t_j_m = b x kb_curr[dim] x hidden 

            u_t_j_m, feed_hidden_j_m = kvr_attentions[idx](
                query = query, 
                prev_kb_utilities = prev_u, 
                prev_kb_feed_hidden = kb_feed_hidden_cache[idx],
            )
            
            ### start update caches ###

            # u_t_j_m = b x kb_max[dim] x hidden
            # cache this for same dim on next hop
            utils_cache_update[dim] = u_t_j_m # .unsqueeze(0) # FIXME find out if unsqueeze is needed for beam search stack

            if feed_hidden_j_m is not None:
                # feed_u_t_j_m_j = n_layers x b x hidden
                # cache this for exact same kvr att module on next unroll step
                kb_feed_hidden_cache[idx] = feed_hidden_j_m # .unsqueeze(0) 
            
            ### end update caches ###

            ### insert local hiddens into global hidden in appropriate places ###

            u_t_j_m = u_t_j_m[:,:,:curr_kb_sizes[dim]]

            # repeat as often as the product of dims before this dim
            # torch.Tensor.repeat with count_{dim}=3 repeats dim's entries like so:
            # [1,2,3] => [1,2,3,1,2,3,1,2,3]
            u_t_j_m_blocked = u_t_j_m.repeat(1, 1, dims_before[dim])

            # tile as often as the product of dims after this dim
            # JoeyNMT's tile comes from Open-NMT and with count=3 repeats entries along dim
            # [1,2,3] => [1,1,1,2,2,2,3,3,3]
            u_t_j_m_blocked_tiled = tile(u_t_j_m_blocked, count=dims_after[dim], dim=-1)

            u_t_k += u_t_j_m_blocked_tiled
            ### end insert local hidden ###
        # only update after dimension loop for the multihop feeding case (want to retrieve utilities of *last* hop within dim loop)
        utils_dims_cache = utils_cache_update

    if kb_mask is not None:
        # mask entries that arent actually assigned (happens in scheduling)
        if len(kb_mask.shape)+1 == len(u_t_k.shape):
            kb_mask = kb_mask.unsqueeze(1)

        kb_mask = ~kb_mask # invert mask: 1 = keep (valued entry), 0 = discard (unvalued entry)

        assert kb_mask.shape == u_t_k.shape, (kb_mask.shape, u_t_k.shape)

        with torch.no_grad():
            zeros_like_u_t_k = u_t_k.new_zeros(u_t_k.shape)

        u_t_k = torch.where(
            kb_mask, u_t_k, zeros_like_u_t_k 
        )
        assert (u_t_k == u_t_k).all(), ((u_t_k != u_t_k), kb_mask.dtype)

    return u_t_k, utils_dims_cache, kb_feed_hidden_cache