# coding: utf-8
"""
Attention modules
"""

from typing import Iterable, Union, List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from joeynmt.helpers import product



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
                feed_rnn=True, num_layers=2, dropout=0.,
                kb_max: Iterable = [256], dim=0, multihead_feed=False,
                pad_keys=False):
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
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)

        self.W2 = nn.Linear(hidden_size, hidden_size, bias=False) # W_2 in eric et al

        self.proj_keys = None   # to store projected keys
        self.proj_query = None  # projected query
       
        # uniformize kb dimension to kb_total as its used in multihop_feeding (see decoders)
        self.kb_max = kb_max # e.g [256], [16,32]
        assert isinstance(self.kb_max, Iterable)
        self.dim = dim # index to kb_max; the dim index for this module
        self.kb_total = self.kb_max[self.dim] # atm weather kb are max at 203
        self.pad_keys = pad_keys # whether to pad keys
        self.curr_kb_size = None # this is set during self.compute_proj_keys

        self.feed_rnn = feed_rnn
        self.num_feed_layers = num_layers
        self.activation = nn.Tanh()

        self.multihead_feed = multihead_feed
        if self.multihead_feed:
            self.head_size = int(hidden_size//len(self.kb_max))
            assert hidden_size % self.head_size == 0, \
            f"hidden dimension {hidden_size} must be divisible by number of dimensions = {len(self.kb_max)}"

        # module to feed back concatenated query and previous kb hidden at hops k > 1
        # either parameterized by GRU or feed forward NN
        # (GRU remembers stuff from last decoding step, linear one from last hop of different head (but corresponding dim)

        memory_input_dim = 2*hidden_size if self.multihead_feed == True else self.kb_total+hidden_size
        if self.feed_rnn == True:

            # extend input dimension by KB total dimension (e.g. 512, 256)? 
            # dont do this if we have a multihead feeding network before this one 
            self.memory_network = nn.GRU(   
                                            memory_input_dim, hidden_size, # hidden size must be == decoder hidden size
                                            self.num_feed_layers, batch_first=True, 
                                            dropout=dropout if num_layers > 1 else 0.
                                        )
        else:

            # feeds kb_hidden from previous hop's att module or from the last hop on previous step
            # 2 linear layers ?
            self.linear_multihop_feeding_1 = nn.Linear(memory_input_dim, hidden_size, bias=False)
            self.memory_network = lambda query, _: (None, self.linear_multihop_feeding_1((query)))

        if self.multihead_feed == True:

            assert self.feed_rnn == False, NotImplementedError(f"too time costly to have multiple gru feeders")

            # prepend stack of multiple feeding modules before memory network in fwd

            self.feeding_input_network = nn.ModuleList(
                [nn.Linear(self.kb_max[i], self.head_size) for i in range(self.dim)]
            )
            # [B x 1 x kb_max[i]] x n  => B x 1 x head_size * n = B x 1 x H
            self.multihead_network = lambda prev_kb_utils_iterable: \
                    self.activation(
                        torch.cat([self.feeding_input_network[i](u) for i,u in enumerate(prev_kb_utils_iterable)],dim=-1)
            )


    #pylint: disable=arguments-differ
    def forward(self, query: Tensor = None, prev_kb_utilities: Union[Tensor, List[Tensor]] = None, prev_kb_feed_hidden = None):
        """
        Bahdanau MLP attention forward pass.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :param prev_kb_utilities: if not None, pass concatenation of query and this thru self.memory_network
        :param prev_kb_feed_hidden: if self.memory_network is GRU, this is its previous hidden state; or the first decoder hidden state (first query)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        self._check_input_shapes_forward(query=query)

        assert self.proj_keys is not None,\
            f"projection keys have to get pre-computed/assigned in dec.fwd before dec.fwd_step"
        
        # variable names refer to eric et al (2017) notation,
        # (see https://arxiv.org/abs/1705.05414)
        # with k added for k_th multihop pass

        # feed sum of 
        # previously computed kvr module's hidden and decoder hidden
        # back into new query

        # GRU kb_feed_hidden: num_layers x batch x hidden 
        # linear kb_feed_hidden: batch x 1 x hidden

        if self.multihead_feed:
            # concatenation of projection of util list onto n heads
            projected_u = self.multihead_network(prev_kb_utilities)
        else:
            projected_u = prev_kb_utilities 

        feed_input = torch.cat([projected_u, query], dim=-1)
        _, kb_feed_hidden = self.memory_network(feed_input, prev_kb_feed_hidden) 

        # query_feed_hidden: batch x KB x hidden
        if self.feed_rnn:
            query_feed_hidden = kb_feed_hidden[-1].unsqueeze(1) # take last layer of hidden
        else:
            query_feed_hidden = kb_feed_hidden

        

        # We first project the query (the decoder state).
        # The projected keys (the knowledgebase entry sums) were already pre-computed.
        self.compute_proj_query(query_feed_hidden) 
        # => self.proj_query = batch x 1 x hidden

        # TODO find out if projecting like this is correct
        # add kb feed hidden if it exists

        # Calculate kb hidden for decoding step t and multihop pass j and dimension n: kb_hidden_t_j_n

        # proj_query: batch x 1 x hidden
        # proj_keys: batch x kb_total x hidden
        proj_query_memory_keys = query + self.proj_keys
        # proj_query_memory_keys: batch x kb_total x hidden
        # ('+' repeats query required number of times (kb_total) along dim 1)

        # (https://arxiv.org/abs/1705.05414 : equation 2)
        afterW1 = self.activation(proj_query_memory_keys)
        afterW2 = self.activation(self.W2(afterW1))

        if self.feed_rnn and kb_feed_hidden is not None:
            assert kb_feed_hidden.shape[0] == self.num_feed_layers, \
                (kb_feed_hidden.shape, query.shape, prev_kb_utilities.shape)

        # this is done for consistency in the loop and to make the singleton dimension the unroll steps dim
        # to concatenate 1..t...T together and get the same shape as outputs

        # only apply energy layer after multihop fwd pass

        # u_t: batch x kb_total x 1
        u_t = self.energy_layer(afterW2)

        # u_t = batch x 1 x product(kb_total)=kb_total
        u_t = u_t.squeeze(2).unsqueeze(1)

        return u_t, kb_feed_hidden

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.
        :param keys: batch x kb x trg_emb
        :return:
        """

        self.curr_kb_size = keys.shape[1]

        if self.pad_keys:
            keys = self.pad_kb_keys(keys)

        self.proj_keys = self.key_layer(keys) # B x kb_total x hidden

    def pad_kb_keys(self, kb_keys: Tensor) -> Tensor:
        """
        pad kb_keys from 

        B x CURR_KB x TRG_EMB => B x KB_MAX x TRG_EMB

        kb dim will get used in multihop feeding input dimension, so it needs to be the same always
        kb dim is determined by keys input only
        solution: 
        pad key tensor along that dimension up to kb max with 0.0
        and remember true kb size

        """
        # calculated at start of decoder.forward;
        # retrieved during decoder.forward_step to recover actual kb size

        padding = self.kb_max[self.dim] - self.curr_kb_size
        assert padding >= 0, f"kb dim of keys {kb_keys.shape} appears to be larger than self.kb_total={self.kb_total} => increase self.kb_total"

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
