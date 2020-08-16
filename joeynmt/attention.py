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

    def __init__(self, hidden_size=1, key_size=1, query_size=1, k_hops=1, kb_max=256):
        """
        Creates key value retrieval attention mechanism.
        hidden refers to attention layer hidden, not decoder or encoder hidden

        :param hidden_size: size of the projection for query and key
        :param key_size: size of the attention input keys
        :param query_size: size of the query
        """

        super(KeyValRetAtt, self).__init__()


        self.key_layer = nn.Linear(key_size, hidden_size, bias=False) # part 1 of W_1 in eric et al

        self.proj_keys = None   # to store projected keys
        self.proj_query = None  # projected query
       
        # uniformize kb dimension to kb_max as its used in multihop_feeding
        self.kb_max = kb_max # atm weather kb are max at 203
        self.curr_kb_size = None # this is set during self.compute_proj_keys

        self.k_hops = k_hops

        # FIXME TODO instead of doing all of these modulelist, just
        # do a modulelist of keyValRetAttentions in the decoder with input feeding???
        # FIXME

        # code readablity helper function (repeat module k times)
        k_layers = lambda module: nn.ModuleList([module for _ in range(self.k_hops)])

        # multihop attention
        self.query_layers = k_layers(
            nn.Linear(query_size, hidden_size, bias=False) # part 2 of W_1 in eric et al (here: k times)
        )
        self.multihop_feeding = k_layers(
            nn.Linear(query_size+kb_max, query_size, bias=False) # modules to feed back concatenated query and keys at hop k
        )
        self.W = k_layers(
            nn.Linear(hidden_size, hidden_size, bias=False) # W_2 in eric et al (repeated k hops)
        )
        self.energy_layers = k_layers(
            nn.Linear(hidden_size, 1, bias=False) # utilities for all the kb entries
        )
        

    #pylint: disable=arguments-differ
    def forward(self, query: Tensor = None):
        """
        Bahdanau MLP attention forward pass.

        :param query: the item (decoder state) to compare with the keys/memory,
            shape (batch_size, 1, decoder.hidden_size)
        :return: context vector of shape (batch_size, 1, value_size),
            attention probabilities of shape (batch_size, 1, src_length)
        """
        self._check_input_shapes_forward(query=query)

        assert self.proj_keys is not None,\
            "projection keys have to get pre-computed"
        assert self.curr_kb_size is not None,\
            "projection keys have to get pre-computed"
        
        u_t_k = None

        for k in range(self.k_hops):

            # u_t_k = batch x 1 x kb_max
            # query = batch x 1 x dec.hidden

            if u_t_k is None: # first attention pass
                query_k = query
            else: # successive multihop attention passes
                query_k = torch.cat([u_t_k, query], dim=-1) # query_k = batch x 1 x dec.hidden + kb_max
                query_k = self.multihop_feeding[k](query_k) # query_k = batch x 1 x dec.hidden

            # We first project the query (the decoder state).
            # The projected keys (the knowledgebase entry sums) were already pre-computed.
            self.compute_proj_query(query_k, k=k) # => self.proj_query = batch x 1 x hidden

            # Calculate u_t_k. (kb entry utilities at decoding step t and multihop pass k)

            ff = lambda x: torch.tanh(self.W[k](torch.tanh(x))) 

            # FIXME find out if normalizing by num of attention hops helps

            # proj_keys: batch x kb_max x hidden_size
            # proj_query: batch x 1 x hidden
            u_t_k = self.energy_layers[k](ff(self.proj_query + self.proj_keys)) / self.k_hops
            # u_t_k: batch x kb_max x 1
            # ('+' repeats query required number of times (kb_max) along dim 1)

            u_t_k = u_t_k.squeeze(2).unsqueeze(1)
            # u_t_k: batch x 1 x kb_max

            # this done for consistency in the loop and to make the singleton dimension the unroll steps dim
            # to concatenate 1..t...T together and get the same shape
            # as outputs

        u_t = u_t_k[:,:,:self.curr_kb_size] # recover only attention values for non pad knowledgebase entries

        return u_t

    def compute_proj_keys(self, keys: Tensor):
        """
        Compute the projection of the keys.
        Is efficient if pre-computed before receiving individual queries.

        :param keys: batch x kb x trg_emb
        :return:
        """
        # kb dim will get used in multihop feeding input dimension, so it needs to be the same always
        # kb dim is determined by keys input only
        # solution: 
        # pad key tensor along that dimension up to kb max with 0.0
        # and remember true kb size

        self.curr_kb_size = keys.shape[1]

        padding = self.kb_max - self.curr_kb_size
        assert padding >= 0, f"kb dim of keys {keys.shape} appears to be larger than self.kb_max={self.kb_max} => increase self.kb_max"

        keys_pad = torch.zeros(keys.shape[0],padding,keys.shape[2]).to(device=keys.device)
        keys = torch.cat([keys,keys_pad], dim=1)

        self.proj_keys = self.key_layer(keys) # B x kb_max x hidden

    def compute_proj_query(self, query: Tensor, k: int = 1):
        """
        Compute the projection of the k_th query

        query: 1 x kb_size x trg_emb_size

        :param query:
        :param k: index to query layers
        :return:
        """
        self.proj_query = self.query_layers[k](query)

    def _check_input_shapes_forward(self, query: torch.Tensor):
        """
        Make sure that inputs to `self.forward` are of correct shape.
        Same input semantics as for `self.forward`.

        :param query:
        :return:
        """
        assert query.shape[1] == 1 
        assert query.shape[2] == self.query_layers[0].in_features

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
