# coding: utf-8
"""
Attention modules
"""

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F



class AttentionMechanism(nn.Module):
    """
    Base attention class
    """

    def forward(self, *inputs):
        raise NotImplementedError("Implement this.")


class BahdanauAttention(AttentionMechanism):
    """
    Implements Bahdanau (MLP) attention

    Section A.1.2 in https://arxiv.org/pdf/1409.0473.pdf.
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """

        super(BahdanauAttention, self).__init__()

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.proj_keys = None   # to store projected keys
        self.proj_query = None  # projected query

    #pylint: disable=arguments-differ
    def forward(self, query: Tensor = None,
                mask: Tensor = None,
                values: Tensor = None):
        """
        Bahdanau MLP attention forward pass.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, src_length)
        :param values: values (encoder states),
            shape (batch_size, src_length, encoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert mask is not None, "mask is required"
        assert self.proj_keys is not None,\
            "projection keys have to get pre-computed"

        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computed.
        self.compute_proj_query(query)

        # Calculate scores.
        # proj_keys: batch x src_len x hidden_size
        # proj_query: batch x 1 x hidden_size
        scores = self.energy_layer(torch.tanh(self.proj_query + self.proj_keys))
        # scores: batch x src_len x 1

        scores = scores.squeeze(2).unsqueeze(1)
        # scores: batch x 1 x time

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x time

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x value_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.

        :param keys:
        :return:
        """
        self.proj_keys = self.key_layer(keys)

    def compute_proj_query(self, query: Tensor):
        """
        Compute the projection of the query.

        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(self, query: torch.Tensor,
                                    mask: torch.Tensor,
                                    values: torch.Tensor):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.query_layer.in_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "BahdanauAttention"

class KeyValRetAtt(AttentionMechanism):
    """
    Implements Key-Value Retrieval Attention 

    from eric et al.
    """

    def __init__(self, hidden_size=1, key_size=1, query_size=1,
    kb_max=256, feed_rnn=True,
    num_layers=2, dropout=0.):
        """
        Creates key value retrieval attention mechanism.
        hidden refers to attention layer hidden, not decoder or encoder hidden

        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """

        super(KeyValRetAtt, self).__init__()

        # Weights
        self.key_layer = nn.Linear(key_size, hidden_size, bias=False) # key part of W_1 in eric et al
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False) # query part of W_1 in eric et al
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False) # utilities for all the kb entries

        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False) # W_2 in eric et al

        self.proj_keys = None   # to store projected keys
        self.proj_query = None  # projected query
       
        # uniformize kb dimension to kb_max as its used in multihop_feeding (see decoders)
        self.kb_max = kb_max # atm weather kb are max at 203
        self.curr_kb_size = None # this is set during self.compute_proj_keys

        self.feed_rnn = feed_rnn
        # module to feed back concatenated query and previous utilities at hops k > 1
        # either parameterized by LSTM or feed forward NN
        # (LSTM remembers stuff from last decoding step, linear one from last hop of different head (but corresponding dim)
        if self.feed_rnn == True:
            self.memory_network = nn.LSTM(hidden_size + self.kb_max, hidden_size, # hidden size must be == decoder hidden size
            num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0.)
        else:
            self.multihop_feeding = nn.Linear(hidden_size + self.kb_max, hidden_size, bias=False)
            self.memory_network = lambda query, _: (None, self.multihop_feeding(query))

        

    #pylint: disable=arguments-differ
    def forward(self, query: Tensor = None, prev_utilities=None, prev_kb_feed_hidden=None):
        """
        Bahdanau MLP attention forward pass.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param prev__utilities: if not None, pass concatenation of query and this thru self.memory_network
        :param prev_kb_feed_hidden: if self.memory_network is LSTM, this is its previous hidden state; or the first decoder hidden state (first query)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        self._check_input_shapes_forward(query=query)

        assert self.proj_keys is not None,\
            f"projection keys have to get pre-computed/assigned in dec.fwd before def.fwd_step"
        
        if prev_utilities is None: 
            # first attention hop, just use query (dec hidden at time=t)
            query_k = query
        else: 
            # successive att hop (k > 1 in loop), 
            # feed concatenation of
            # previously computed kb entry utilities and query 
            # back into new query

            query_k = torch.cat([prev_utilities, query], dim=-1) # batch x 1 x kb_max + hidden
            _, prev_kb_feed_hidden = self.memory_network(query_k, prev_kb_feed_hidden) # batch x 1 x hidden

            if not self.feed_rnn:
                query_k = prev_kb_feed_hidden
            else:
                query_k = prev_kb_feed_hidden[0][-1].unsqueeze(1) # hidden is first item in LSTM state tuple; take last layer of it
            # in case of LSTM, query_k-1 is used as hidden state

        # variable names refer to eric et al (2017) notation,
        # (see https://arxiv.org/abs/1705.05414)
        # with k added for kth multihop pass

        # We first project the query (the decoder state).
        # The projected keys (the knowledgebase entry sums) were already pre-computed.
        self.compute_proj_query(query_k) 
        # => self.proj_query = batch x 1 x hidden

        # Calculate u_t_k. (kb entry utilities at decoding step t and multihop pass k)

        # proj_query: batch x 1 x hidden
        # proj_keys: batch x kb_max x hidden
        query_conc_keys = self.proj_query + self.proj_keys
        # query_conc_keys: batch x kb_max x hidden
        # ('+' repeats query required number of times (kb_max) along dim 1)

        # (https://arxiv.org/abs/1705.05414 : equation 2)
        afterW1 = torch.tanh(query_conc_keys)
        afterW2 = torch.tanh(self.W2(afterW1))
        
        # u_t_k: batch x kb_max x 1
        u_t_k = self.energy_layer(afterW2)

        # u_t_k: batch x 1 x kb_max
        u_t_k = u_t_k.squeeze(2).unsqueeze(1)
        # this done for consistency in the loop and to make the singleton dimension the unroll steps dim
        # to concatenate 1..t...T together and get the same shape
        # as outputs

        return u_t_k, prev_kb_feed_hidden

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.
        :param keys: batch x kb x trg_emb
        :return:
        """

        padded_keys = self.pad_kb_keys(keys)

        self.proj_keys = self.key_layer(padded_keys) # B x kb_max x hidden

    def pad_kb_keys(self, kb_keys: Tensor) -> Tensor:
        """
        pad kb_keys from B x CURR_KB x TRG_EMB => B x KB_MAX x TRG_EMB
        kb dim will get used in multihop feeding input dimension, so it needs to be the same always
        kb dim is determined by keys input only
        solution: 
        pad key tensor along that dimension up to kb max with 0.0
        and remember true kb size

        """
        # calculated at start of decoder.forward;
        # retrieved during decoder.forward_step to recover actual kb size
        self.curr_kb_size = kb_keys.shape[1]

        padding = self.kb_max - self.curr_kb_size
        assert padding >= 0, f"kb dim of keys {kb_keys.shape} appears to be larger than self.kb_max={self.kb_max} => increase self.kb_max"

        keys_pad = torch.zeros(kb_keys.shape[0], padding, kb_keys.shape[2]).to(device=kb_keys.device)
        return torch.cat([kb_keys,keys_pad], dim=1)


    def compute_proj_query(self, query: Tensor):
        """
        Compute the projection of the k_th query

        query: 1 x kb_size x trg_emb_size

        :param query:
        :return:
        """
        self.proj_query = self.query_layer(query)

    def _check_input_shapes_forward(self, query: torch.Tensor):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :return:
        """
        # assert query.shape[1] == 1, query.shape[1] # TODO for transformer, this will be SRC_LEN, not 1
        assert query.shape[2] == self.query_layer.in_features, (query.shape, self.query_layer.in_features)

    def __repr__(self):
        return "KeyValRetAtt"



class LuongAttention(AttentionMechanism):
    """
    Implements Luong (bilinear / multiplicative) attention.

    Eq. 8 ("general") in http://aclweb.org/anthology/D15-1166.
    """

    def __init__(self, hidden_size: int = 1, key_size: int = 1):
        """
        Creates attention mechanism.

        :param hidden_size: size of the key projection layer, has to be equal
            to decoder hidden size
        :param key_size: size of the attention input keys
        """

        super(LuongAttention, self).__init__()
        self.key_layer = nn.Linear(in_features=key_size,
                                   out_features=hidden_size,
                                   bias=False)
        self.proj_keys = None  # projected keys

    # pylint: disable=arguments-differ
    def forward(self, query: torch.Tensor = None,
                mask: torch.Tensor = None,
                values: torch.Tensor = None):
        """
        Luong (multiplicative / bilinear) attention forward pass.
        Computes context vectors and attention scores for a given query and
        all masked values and returns them.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param mask: mask out keys position (0 in invalid positions, 1 else),
            shape (batch_size, 1, src_length)
        :param values: values (encoder states),
            shape (batch_size, src_length, encoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        self._check_input_shapes_forward(query=query, mask=mask, values=values)

        assert self.proj_keys is not None,\
            "projection keys have to get pre-computed"
        assert mask is not None, "mask is required"

        # scores: batch_size x 1 x src_length
        scores = query @ self.proj_keys.transpose(1, 2)

        # mask out invalid positions by filling the masked out parts with -inf
        scores = torch.where(mask, scores, scores.new_full([1], float('-inf')))

        # turn scores to probabilities
        alphas = F.softmax(scores, dim=-1)  # batch x 1 x src_len

        # the context vector is the weighted sum of the values
        context = alphas @ values  # batch x 1 x values_size

        return context, alphas

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys and assign them to `self.proj_keys`.
        This pre-computation is efficiently done for all keys
        before receiving individual queries.

        :param keys: shape (batch_size, src_length, encoder.hidden_size)
        """
        # proj_keys: batch x src_len x hidden_size
        self.proj_keys = self.key_layer(keys)

    def _check_input_shapes_forward(self, query: torch.Tensor,
                                    mask: torch.Tensor,
                                    values: torch.Tensor):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :param mask:
        :param values:
        :return:
        """
        assert query.shape[0] == values.shape[0] == mask.shape[0]
        assert query.shape[1] == 1 == mask.shape[1]
        assert query.shape[2] == self.key_layer.out_features
        assert values.shape[2] == self.key_layer.in_features
        assert mask.shape[2] == values.shape[1]

    def __repr__(self):
        return "LuongAttention"
