# -*- coding: utf-8 -*-

import math
import torch
import torch.nn as nn
from torch import Tensor
from joeynmt.attention import KeyValRetAtt






# pylint: disable=arguments-differ
class MultiHeadedAttention(nn.Module):
    """
    Multi-Head Attention module from "Attention is All You Need"

    Implementation modified from OpenNMT-py.
    https://github.com/Opeclass MultiHeadedAttention(nn.Module):
nNMT/OpenNMT-py
    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0, (size, num_heads)

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size,size)
        self.v_layer = nn.Linear(size,size)
        self.q_layer = nn.Linear(size,size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """   
        Computes multi-headed attention.

        :param k: keys   [B, M, D] with M being the sentence length.
        :param v: values [B, M, D]
        :param q: query  [B, M, D]
        :param mask: optional mask [B, 1, M]
        :return:
        """
        batch_size = k.size(0)
        head_size = self.head_size
        num_heads = self.num_heads

 
        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        # using num_heads * head_size == size
        k = k.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x key_len   x head_size
        v = v.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x val_len   x head_size
        q = q.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x query_len x head_size

        # scale query because it helps, no idea why 
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        # compute scores
        scores = q @ k.transpose(2, 3)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention) # batch x num_h x M (query) x M (key)

        # get context vector (select values with attention) 
        # and reshape back to [B, M, D]

        context = attention @ v
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * head_size
        )

        output = self.output_layer(context)

        return output 


# pylint: disable=arguments-differ
class MultiHeadedKbAttention(MultiHeadedAttention):
    """
    Multi-Head Kb Attention module similar to "Eric et al.'s" which used bahdanau attention

    """

    def __init__(self, num_heads: int, size: int, dropout: float = 0.1, kb_max: int = 256):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0, (size, num_heads)

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size,size)
        self.v_layer = nn.Linear(size,size)
        self.q_layer = nn.Linear(size,size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)


    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed KB attention.

        :param k: keys   [B, M, D] with M being the sentence length (Max)
        :param v: values [B, M, D] values
        :param q: query  [B, M, D]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        head_size = self.head_size
 
        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..., head_size]
        # using num_heads * head_size == size
        k = k.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x key_len   x head_size
        v = v.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x val_len   x head_size
        q = q.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x query_len x head_size

        # scale query because it helps, no idea why 
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        # compute scores
        scores = q @ k.transpose(2, 3)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention) # batch x num_h x M (query) x KB (key)

        context = attention @ v 
        # context: B x n x M x h => B x M x D
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * head_size
        )

        # output: B x M x D
        output = self.output_layer(context)

        # get context vector (select values with attention) 
        # and reshape back to [B, M, D]

        # query: B x M x D
        # => 
        # proj_query: B x M x 1 x D
        # keys: B x M x KB x D 

        # output:
        # feed into next layer along with query

        # attention : B x n x M x KB
        # and after last layer:  
        # kb_probs = B x M x KB = energy_layer(attention)

        # B x M x KB; B x n x M x KB
        return output, attention
    


# pylint: disable=arguments-differ
class MultiHeadedLinearAttention(MultiHeadedAttention):
    """
    Multi-Headed Linear Attention module ( https://arxiv.org/abs/2006.16236 )
    """
    def __init__(self, num_heads: int, size: int, dropout: float = 0.1):
        """
        Create a multi-headed attention layer.
        :param num_heads: the number of heads
        :param size: model size (must be divisible by num_heads)
        :param dropout: probability of dropping a unit
        """
        super(MultiHeadedAttention, self).__init__()

        assert size % num_heads == 0, (size, num_heads)

        self.head_size = head_size = size // num_heads
        self.model_size = size
        self.num_heads = num_heads

        self.k_layer = nn.Linear(size,size)
        self.v_layer = nn.Linear(size,size)
        self.q_layer = nn.Linear(size,size)

        self.output_layer = nn.Linear(size, size)
        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)
        
        # linear feature map, choice is yours
        self.KernelModule = nn.ELU()
        self.kernel = lambda x: self.KernelModule(x) + 1

    def forward(self, k: Tensor, v: Tensor, q: Tensor, mask: Tensor = None):
        """
        Computes multi-headed KB attention.

        :param k: keys   [B, M, D] with M being the sentence length (Max)
        :param v: values [B, M, D] values
        :param q: query  [B, M, D]
        :return:
        """
        batch_size = k.size(0)
        num_heads = self.num_heads
        head_size = self.head_size
 
        # project the queries (q), keys (k), and values (v)
        k = self.k_layer(k)
        v = self.v_layer(v)
        q = self.q_layer(q)

        # reshape q, k, v for our computation to [batch_size, num_heads, ..]
        # using num_heads * head_size == size
        k = k.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x key_len   x head_size
        v = v.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x val_len   x head_size
        q = q.view(batch_size, -1, num_heads, head_size).transpose(1, 2) # batch x num_h x query_len x head_size

        # scale query because it helps, no idea why 
        q = q / math.sqrt(self.head_size)

        # batch x num_heads x query_len x key_len
        # compute scores
        scores = q @ k.transpose(2, 3)

        # apply the mask (if we have one)
        # we add a dimension for the heads to it below: [B, 1, 1, M]
        if mask is not None:
            scores = scores.masked_fill(~mask.unsqueeze(1), float('-inf'))

        # apply attention dropout and compute context vectors.
        attention = self.softmax(scores)
        attention = self.dropout(attention) # batch x num_h x M (query) x M (key)

        # get context vector (select values with attention) 
        # and reshape back to [B, M, D]

        context = attention @ v
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, num_heads * head_size
        )

        output = self.output_layer(context)

        # u_k in analogy to u_t_k in joeynmt.attention. kb attention:
        # utilities at attention hop (=transf layer) number k
        # B x M x KB
        return output



# pylint: disable=arguments-differ
class PositionwiseFeedForward(nn.Module):
    """
    Position-wise Feed-forward layer
    Projects to ff_size and then back down to input_size.
    """

    def __init__(self, input_size, ff_size, dropout=0.1):
        """
        Initializes position-wise feed-forwarlayer.
        :param input_size: dimensionality of   the input.
        :param ff_size: dimensionality of intermediate representation
        :param dropout:
        """ 
        super(PositionwiseFeedForward, self).__init__()
        self.layer_norm = nn.LayerNorm(input_size, eps=1e-6)
        self.pwff_layer = nn.Sequential(
            nn.Linear(input_size, ff_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_size, input_size),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.layer_norm(x)
        return self.pwff_layer(x_norm) + x


# pylint: disable=arguments-differ
class PositionalEncoding(nn.Module):
    """
    Pre-compute position encodings (PE).
    In forward pass, this adds he position-encodings to the
    input for as many time step s as necessary.

                
            # KVR attention
            h2 = self.kb_trg_att(kb_keys, kb_values_embed, query_k) # TODO find out if I have to apply src_mask here too
            assert False, h2.shape

        # final position-wise feed-forward layer
        o = self.feed_forward(h2)
        
        return o
    Implementation based on OpenNMT-py.
    https://github.com/OpenNMT/OpenNMT-py
    """
    def __init__(self,
                 size: int = 0,
                 max_len: int = 5000,
                 e: int = 4):
        """
        Positional Encoding with maximum length max_len
        :param size:
        :param max_len:
        :param dropout:
        """
        if size % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                             "odd dim (got dim={:d})".format(size))
        pe = torch.zeros( max_len, size )
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            (torch.arange(0, size, 2, dtype=torch.float) * -(math.log(10**e) / size))
                              )
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, size, max_len]
        super(PositionalEncoding, self).__init__()
        self.register_buffer('pe', pe)
        self.dim = size

    def forward(self, emb):
        """Embed inputs.
        Args:
            emb (FloatTensor): Sequence of word vectors
                ``(seq_len, batch_size, self.dim)``
        """
        # Add position encodings
        return emb + self.pe[:, :emb.size(1)]


class TransformerEncoderLayer(nn.Module):
    """
    One Transformer encoder layer has a Multi-head attention layer plus
    a position-wise feed-forward layer.
    """

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1):
        """
        A single Transformer layer.
        :param size:
        :param ff_size:
        :param num_heads:
        :param dropout:
        """
        super(TransformerEncoderLayer, self).__init__()

        self.layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.src_src_att = MultiHeadedAttention(num_heads, size, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)
        self.dropout = nn.Dropout(dropout)
        self.size = size

    # pylint: disable=arguments-differ
    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        """
        Forward pass for a single transformer encoder layer.
        First applies layer norm, then self attention,
        then dropout with residual connection (adding the input to the result),
        and then a position-wise feed-forward layer.
        :param x: layer input
        :param mask: input mask
        :return: output tensor
        """
        x_norm = self.layer_norm(x)
        h = self.src_src_att(x_norm, x_norm, x_norm, mask)
        h = self.dropout(h) + x
        o = self.feed_forward(h)
        return o


class TransformerDecoderLayer(nn.Module):
    """
    Transformer decoder layer.
    Consists of self-attention, source-attention, and feed-forward.
    """

    def __init__(self,
                 size: int = 0,
                 ff_size: int = 0,
                 num_heads: int = 0,
                 dropout: float = 0.1,
                 kb_task: bool = False,
                 kb_max: int = 256,
                 tfstyletf: bool = False,
                 feed_kb_hidden: bool = False,
    ):
        """
        Represents a single Transformer decoder layer.
        It attends to the source representation and the previous decoder states.
        :param size: model dimensionality
        :param ff_size: size of the feed-forward intermediate layer
        :param num_heads: number of heads
        :param kb_task: performing kb task or not?
        :param kb_max: maximum knowledgebase size, used in att init
        :param dropout: dropout to apply to input
        """
        super(TransformerDecoderLayer, self).__init__()
        self.size = size

        self.trg_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)

        self.src_trg_att = MultiHeadedAttention(num_heads, size,
                                                dropout=dropout)

        self.feed_forward = PositionwiseFeedForward(size, ff_size=ff_size)

        self.x_layer_norm = nn.LayerNorm(size, eps=1e-6)
        self.dec_layer_norm = nn.LayerNorm(size, eps=1e-6)


        self.tfstyletf = tfstyletf 
        if kb_task :
            self.kb_trg_att = MultiHeadedKbAttention(num_heads, size, dropout=dropout)
            self.kb_layer_norm = nn.LayerNorm(size, eps=1e-6)
            self.kb_max = kb_max
            self.feed_kb_hidden = feed_kb_hidden

            if self.tfstyletf:
                self.multihop_feeding = nn.Linear(size + self.kb_max, size, bias=False)

        self.dropout = nn.Dropout(dropout)

    # pylint: disable=arguments-differ
    def forward(self,
                x: Tensor = None,
                memory: Tensor = None,
                kb_keys: Tensor = None, # determine if just kb keys are enough
                src_mask: Tensor = None,
                trg_mask: Tensor = None,
                kb_values_embed: Tensor = None,
                prev_kb_hidden: Tensor = None,
                kb_mask: Tensor = None,
                ) -> Tensor:
        """
        Forward pass of a single Transformer decoder layer.
        :param x: inputs
        :param memory: source representations
        :param src_mask: source mask
        :param trg_mask: target mask (so as to not condition on future steps)
        :param kb_keys: knowledgebase keys: B x KB_MAX x TRG_EMB
        :param prev_utilities: B x M x KB_MAX previous kb entry utilities for kb att input feeding
        :param kb_mask: bruh
        :return: output tensor
        """
        # decoder/target self-attention
        x_norm = self.x_layer_norm(x)
        h1 = self.trg_trg_att(x_norm, x_norm, x_norm, mask=trg_mask)
        h1 = self.dropout(h1) + x

        # source-target attention
        h1_norm = self.dec_layer_norm(h1)
        h2 = self.src_trg_att(memory, memory, h1_norm, mask=src_mask) 

        h2 = self.dropout(h2) + h1

        if self.tfstyletf and kb_keys is not None:

            assert kb_values_embed is not None

            h2_norm = self.kb_layer_norm(h2)

            if self.feed_kb_hidden:
                # feed this query at kth hop
                query_k = h2_norm + prev_kb_hidden 
            else:
                query_k = h2_norm

            # KVR attention
            kb_hidden, kb_att = self.kb_trg_att(kb_keys, kb_values_embed, query_k, mask=kb_mask) # TODO find out if I have to apply src_mask here too

            # TODO is there a different way to update the main hidden state
            # with transformer info without completely trashing it?
            h2 = self.dropout(kb_hidden) + h2

        else:
            kb_hidden, kb_att = None, None

        # final position-wise feed-forward layer
        o = self.feed_forward(h2)
        
        return o, kb_hidden, kb_att