# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple
from copy import deepcopy

from joeynmt.decoders import Decoder, TransformerDecoder, Gen
from joeynmt.embeddings import Embeddings
from joeynmt.helpers import tile


__all__ = ["greedy", "transformer_greedy", "beam_search"]


def greedy(src_mask: Tensor, embed: Embeddings, bos_index: int,
           max_output_length: int, decoder: Decoder, generator: Gen,
           encoder_output: Tensor, encoder_hidden: Tensor,
           knowledgebase: Tuple[Tensor] = None)\
        -> (np.array, np.array, np.array):
    """
    Greedy decoding. Select the token word highest probability at each time
    step. This function is a wrapper that calls recurrent_greedy for
    recurrent decoders and transformer_greedy for transformer decoders.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param generator: generator to use as output layer 
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :param knowledgebase: knowledgebase tuple containing keys, values and true values for decoding:
    :return:
    """

    if isinstance(decoder, TransformerDecoder):
        # Transformer greedy decoding
        greedy_fun = transformer_greedy

        return greedy_fun(
        src_mask, embed, bos_index, max_output_length,
        decoder, generator, encoder_output, encoder_hidden, knowledgebase)

    else:
        # Recurrent greedy decoding
        greedy_fun = recurrent_greedy

        return greedy_fun(
            src_mask, embed, bos_index, max_output_length,
            decoder, generator, encoder_output, encoder_hidden, knowledgebase)

def recurrent_greedy(
        src_mask: Tensor, embed: Embeddings, bos_index: int,
        max_output_length: int, decoder: Decoder, generator: Gen,
        encoder_output: Tensor, encoder_hidden: Tensor,
        knowledgebase: Tuple = None) -> (np.array, np.array, np.array):
    """
    Greedy decoding: in each step, choose the word that gets highest score.
    Version for recurrent decoder. 
 
    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param generator: generator to use as output layer 
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder last state for decoder initialization
    :param knowledgebase: knowledgebase tuple containing keys, values and true values for decoding:
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None

    if knowledgebase != None:
        # knowledgebase is (kb_keys, kb_values) (see model.run_batch)

        # kept as Tuple since dynamic typing easily allows shoving more objects into this
        # so I dont have to rewrite tons of function signatures 

        kb_att_scores = []
        kb_keys = knowledgebase[0]
        kb_values = knowledgebase[1]
    else:
        kb_values = None

    # pylint: disable=unused-variable
    for t in range(max_output_length):
        # decode one single step
        if knowledgebase != None:
            hidden, att_probs, prev_att_vector, kb_att_probs = decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=embed(prev_y),
                hidden=hidden,
                prev_att_vector=prev_att_vector,
                unroll_steps=1,
                kb_keys=kb_keys)

        else:
           hidden, att_probs, prev_att_vector, kb_att_probs = decoder(
                encoder_output=encoder_output,
                encoder_hidden=encoder_hidden,
                src_mask=src_mask,
                trg_embed=embed(prev_y),
                hidden=hidden,
                prev_att_vector=prev_att_vector,
                unroll_steps=1)

        logits = generator(prev_att_vector, kb_values=kb_values, kb_probs=kb_att_probs)

        # logits: batch x time=1 x vocab (logits)
        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(logits, dim=-1)  # batch x time=1
        # NOTE:              ^ find idx over ordered vocab embeddings 
        # created by 2nd dimension of decoder.output_layer.weight ...
        # Q:
        # how is the association with the vocabulary done there???
        # the output_layer has no information about which of its output neurons
        # should be associated with which index of the trg_vocab
        # A: the model just randomly guesses and soon learns the connection here 
        output.append(next_word.squeeze(1).cpu().numpy())
        print(output)
        prev_y = next_word
        attention_scores.append(att_probs.squeeze(1).cpu().numpy())
        if kb_att_probs is not None:
            kb_att_scores.append(kb_att_probs.squeeze(1).cpu().numpy())
        # batch, max_src_lengths
    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_attention_scores = np.stack(attention_scores, axis=1)
    if kb_att_probs is not None:
        stacked_kb_att_scores = np.stack(kb_att_scores, axis=1)
    else:
        stacked_kb_att_scores = None
    return stacked_output, stacked_attention_scores, stacked_kb_att_scores


# pylint: disable=unused-argument
def transformer_greedy(
        src_mask: Tensor, embed: Embeddings,
        bos_index: int, max_output_length: int, decoder: Decoder, generator: Gen,
        encoder_output: Tensor, encoder_hidden: Tensor, knowledgebase:Tuple=(None,None,None)) -> (np.array, np.array):

    """
    Special greedy function for transformer, since it works differently.
    The transformer remembers all previous states and attends to them.

    :param src_mask: mask for source inputs, 0 for positions after </s>
    :param embed: target embedding
    :param bos_index: index of <s> in the vocabulary
    :param max_output_length: maximum length for the hypotheses
    :param decoder: decoder to use for greedy decoding
    :param generator: generator to use as output layer 
    :param encoder_output: encoder hidden states for attention
    :param encoder_hidden: encoder final state (unused in Transformer)
    :param knowledgebase: knowledgebase tuple containing keys, values and true values for decoding:
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    if knowledgebase == None: # not kb task
        knowledgebase = (None,)*3
    
    batch_size = src_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])

    for _ in range(max_output_length):

        trg_embed = embed(ys)  # embed the BOS-symbol

        # pylint: disable=unused-variable
        with torch.no_grad():
            _ , _, out, kb_probs = decoder(
                trg_embed=trg_embed,
                encoder_output=encoder_output,
                encoder_hidden=None,
                src_mask=src_mask,
                unroll_steps=None,
                hidden=None,
                trg_mask=trg_mask,
                kb_keys=knowledgebase[0],
            )
            # warning kb_values before: B x KB
            logits = generator(out, kb_values=knowledgebase[1], kb_probs=kb_probs)
            # warning kb_values after: B x 1 x KB

            logits = logits[:, -1] # TODO FIXME what does this do? what dims are this
            _, next_word = torch.max(logits, dim=1)
            next_word = next_word.data
            ys = torch.cat([ys, next_word.unsqueeze(-1)], dim=1)
            
    
    if kb_probs is not None: # last returned kb_probs (maximum length)
        stacked_kb_att_scores = kb_probs.cpu().numpy() # B x M x KB
    else:
        stacked_kb_att_scores = None

    ys = ys[:, 1:]  # remove BOS-symbol
    return ys, None, stacked_kb_att_scores


# pylint: disable=too-many-statements,too-many-branches
def beam_search(
        decoder: Decoder,
        generator: Gen,
        size: int,
        bos_index: int, eos_index: int, pad_index: int,
        encoder_output: Tensor, encoder_hidden: Tensor,
        src_mask: Tensor, max_output_length: int, alpha: float,
        embed: Embeddings, n_best: int = 1,
        knowledgebase: Tuple = None) -> (np.array, np.array, np.array):
    """
    Beam search with size k.
    Inspired by OpenNMT-py, adapted for Transformer.

    In each decoding step, find the k most likely partial hypotheses.

    :param decoder:
    :param generator:
    :param size: size of the beam
    :param bos_index:
    :param eos_index:
    :param pad_index:
    :param encoder_output:
    :param encoder_hidden:
    :param src_mask:
    :param max_output_length:
    :param alpha: `alpha` factor for length penalty
    :param embed:
    :param n_best: return this many hypotheses, <= beam
    :param knowledgebase: knowledgebase tuple containing keys, values and true values for decoding
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
        - stacked_kb_att_scores: kb attention scores (3d array)
    """

    # init
    transformer = isinstance(decoder, TransformerDecoder)
    batch_size = src_mask.size(0)
    att_vectors = None  # not used for Transformer

    # Recurrent models only: initialize RNN hidden state
    # pylint: disable=protected-access
    if not transformer:
        hidden = decoder._init_hidden(encoder_hidden)
    else:
        hidden = None

    # tile encoder states and decoder initial states beam_size times
    if hidden is not None:
        hidden = tile(hidden, size, dim=1)  # layers x batch*k x dec_hidden_size

    encoder_output = tile(encoder_output.contiguous(), size,
                          dim=0)  # batch*k x src_len x enc_hidden_size
    src_mask = tile(src_mask, size, dim=0)  # batch*k x 1 x src_len

    # Transformer only: create target mask
    if transformer:
        trg_mask = src_mask.new_ones([1, 1, 1])  # transformer only
    else:
        trg_mask = None

    # numbering elements in the batch
    batch_offset = torch.arange(
        batch_size, dtype=torch.long, device=encoder_output.device)

    # numbering elements in the extended batch, i.e. beam size copies of each
    # batch element
    beam_offset = torch.arange(
        0,
        batch_size * size,
        step=size,
        dtype=torch.long,
        device=encoder_output.device)

    # keeps track of the top beam size hypotheses to expand for each element
    # in the batch to be further decoded (that are still "alive")
    alive_seq = torch.full(
        [batch_size * size, 1],
        bos_index,
        dtype=torch.long,
        device=encoder_output.device)

    # Give full probability to the first beam on the first step.
    # pylint: disable=not-callable
    topk_log_probs = (torch.tensor([0.0] + [float("-inf")] * (size - 1),
                                   device=encoder_output.device).repeat(
                                    batch_size))

    # Structure that holds finished hypotheses.
    hypotheses = [[] for _ in range(batch_size)]

    results = {}
    results["predictions"] = [[] for _ in range(batch_size)]
    results["scores"] = [[] for _ in range(batch_size)]
    results["gold_score"] = [0] * batch_size

    # kb task: also tile kb tensors along batch dimension as done with other inputs above
    if knowledgebase != None:
        kb_keys = tile(knowledgebase[0], size, dim=0)
        kb_values = tile(knowledgebase[1], size, dim=0)
    else:
        kb_keys, kb_values = None, None

    for step in range(max_output_length):

        # This decides which part of the predicted sentence we feed to the
        # decoder to make the next prediction.
        # For Transformer, we feed the complete predicted sentence so far.
        # For Recurrent models, only feed the previous target word prediction
        if transformer:  # Transformer
            decoder_input = alive_seq  # complete prediction so far
        else:  # Recurrent
            decoder_input = alive_seq[:, -1].view(-1, 1)  # only the last word

        # expand current hypotheses
        # decode one single step
        # logits: logits for final softmax
        # pylint: disable=unused-variable
        trg_embed = embed(decoder_input)

        hidden, att_scores, att_vectors, kb_probs = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=trg_embed,
            hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            trg_mask=trg_mask, # subsequent mask for Transformer only
            kb_keys=kb_keys # None by default 
        )

        logits = generator(att_vectors, kb_values=kb_values, kb_probs=kb_probs)

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        if transformer:
            logits = logits[:, -1]  # keep only the last time step
            hidden = None           # we don't need to keep it for transformer

        # batch * k x trg_vocab
        log_probs = F.log_softmax(logits, dim=-1).squeeze(1)

        # multiply probs by the beam probability ( = add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * decoder.output_size)

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1)

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(decoder.output_size)
        topk_ids = topk_ids.fmod(decoder.output_size)

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices),
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        is_finished = topk_ids.eq(eos_index)
        if step + 1 == max_output_length:
            is_finished.fill_(1)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(1)

        # save finished hypotheses
        if is_finished.any():
            predictions = alive_seq.view(-1, size, alive_seq.size(-1))
            for i in range(is_finished.size(0)):
                b = batch_offset[i]
                if end_condition[i]:
                    is_finished[i].fill_(1)
                finished_hyp = is_finished[i].nonzero().view(-1)
                # store finished hypotheses for this batch
                for j in finished_hyp:
                    hypotheses[b].append((
                        topk_scores[i, j],
                        predictions[i, j, 1:])  # ignore start_token
                    )
                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    best_hyp = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)
                    for n, (score, pred) in enumerate(best_hyp):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)
            non_finished = end_condition.eq(0).nonzero().view(-1)
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            if len(non_finished) == 0:
                break
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices)

        if hidden is not None and not transformer:
            if isinstance(hidden, tuple):
                # for LSTMs, states are tuples of tensors
                h, c = hidden
                h = h.index_select(1, select_indices)
                c = c.index_select(1, select_indices)
                hidden = (h, c)
            else:
                # for GRUs, states are single tensors
                hidden = hidden.index_select(1, select_indices)

        if att_vectors is not None:
            att_vectors = att_vectors.index_select(0, select_indices)

    def pad_and_stack_hyps(hyps, pad_value):
        filled = np.ones((len(hyps), max([h.shape[0] for h in hyps])),
                         dtype=int) * pad_value
        for j, h in enumerate(hyps):
            for k, i in enumerate(h):
                filled[j, k] = i
        return filled

    # from results to stacked outputs
    assert n_best == 1
    # only works for n_best=1 for now
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)
    # final_outputs = batch x time
    # stacked_output, stacked_attention_scores, stacked_kb_att_scores
    return final_outputs, None, None
