# coding: utf-8
import torch
import torch.nn.functional as F
from torch import Tensor
import numpy as np
from typing import Tuple
from copy import deepcopy

import random

from joeynmt.decoders import Decoder, TransformerDecoder, Gen
from joeynmt.embeddings import Embeddings
from joeynmt.helpers import tile


__all__ = ["greedy", "transformer_greedy", "beam_search"]


def greedy(src_mask: Tensor, embed: Embeddings, bos_index: int,
           max_output_length: int, decoder: Decoder, generator: Gen,
           encoder_output: Tensor, encoder_hidden: Tensor,
           knowledgebase: Tuple[Tensor] = None,
           trg_input: Tensor=None, e_i: float=1.)\
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
    :param trg_input: batch.trg_input for scheduled sampling
    :param e_i: probability of taking the true token as input to next time step at each step (self doubt of the model)
    :return:
    """

    if isinstance(decoder, TransformerDecoder):
        # Transformer greedy decoding
        greedy_fun = transformer_greedy

        return greedy_fun(
        src_mask, embed, bos_index, max_output_length,
        decoder, generator, encoder_output, encoder_hidden, knowledgebase,
        trg_input, e_i)

    else:
        # Recurrent greedy decoding
        greedy_fun = recurrent_greedy

        return greedy_fun(
            src_mask, embed, bos_index, max_output_length,
            decoder, generator, encoder_output, encoder_hidden, knowledgebase,
            trg_input, e_i)

def recurrent_greedy(
        src_mask: Tensor, embed: Embeddings, bos_index: int,
        max_output_length: int, decoder: Decoder, generator: Gen,
        encoder_output: Tensor, encoder_hidden: Tensor,
        knowledgebase: Tuple = None,
        trg_input: Tensor=None, e_i: float=0.) -> (np.array, np.array, np.array):
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
    :param trg_input: batch.trg_input for scheduled sampling
    :param e_i: probability of taking the true token as input to next time step at each step (self doubt of the model)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
        - stacked_log_probs: stepwise output log_probs
    """

    batch_size = src_mask.size(0)
    prev_y = src_mask.new_full(size=[batch_size, 1], fill_value=bos_index,
                               dtype=torch.long)
    output = []
    attention_scores = []
    hidden = None
    prev_att_vector = None

    print(f"in search.recurrent_greedy; knowledgebase is \
    {[t.shape if isinstance(t, torch.Tensor) else [t_dim.shape for t_dim in t] for t in knowledgebase]}")

    if knowledgebase != None:
        # knowledgebase is (kb_keys, kb_values) (see model.run_batch)

        # kept as Tuple since dynamic typing easily allows shoving more objects into this
        # so I dont have to rewrite tons of function signatures 

        kb_keys = knowledgebase[0]
        kb_values = knowledgebase[1]
        kb_mask = knowledgebase[2]
        utils_dims_cache = None # equivalent of prev_att_vector but for each KB dim
        kb_feed_hidden_cache = None  # equivalent of hidden but for each kvr module

        kb_att_scores = []
        all_log_probs = []
    else:
        kb_keys = kb_values = kb_mask = None
        stacked_log_probs = None

    # pylint: disable=unused-variable
    for t in range(max_output_length):
        # decode one single step
        hidden, att_probs, prev_att_vector, kb_att_probs, utils_dims_cache, kb_feed_hidden_cache = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=embed(prev_y),
            hidden=hidden,
            prev_att_vector=prev_att_vector,
            unroll_steps=1,
            kb_keys=kb_keys,
            kb_mask=kb_mask,
            utils_dims_cache=utils_dims_cache,
            kb_feed_hidden_cache=kb_feed_hidden_cache)

        log_probs = generator(prev_att_vector, kb_values=kb_values, kb_probs=kb_att_probs)

        # logits: batch x time=1 x vocab
        # greedy decoding: choose arg max over vocabulary in each step
        next_word = torch.argmax(log_probs, dim=-1)  # batch x time=1
        # NOTE:              ^ find idx over ordered vocab embeddings 
        # created by 2nd dimension of decoder.output_layer.weight ...
        # Q:
        # how is the association with the vocabulary done there???
        # the output_layer has no information about which of its output neurons
        # should be associated with which index of the trg_vocab
        # A: the output layer just randomly guesses and soon learns the connection via bp 

        if e_i > 0.0 and trg_input is not None:
            # do scheduled sampling (https://arxiv.org/abs/1506.03099 Section 2.4)
            true_y = trg_input[:,t].unsqueeze(1)

            assert true_y.shape == next_word.shape, (true_y.shape, next_word.shape)

            feed_true_y = random.random() < e_i
            prev_y = true_y if feed_true_y else next_word 
        else:
            prev_y = next_word

        dump = lambda t: t.squeeze(1).cpu().detach().numpy() # helper func

        # batch, max_src_lengths
        output.append(dump(next_word))
        attention_scores.append(dump(att_probs))

        if kb_att_probs is not None:
            kb_att_scores.append(dump(kb_att_probs))

            # need to return log probs because in KB task,
            # greedy search is used to get model loss during training
            all_log_probs.append(log_probs)

    # batch, max_src_lengths, time
    stacked_output = np.stack(output, axis=1)  # batch, time
    stacked_attention_scores = np.stack(attention_scores, axis=1)

    if kb_att_probs is not None:
        stacked_kb_att_scores = np.stack(kb_att_scores, axis=1)
        stacked_log_probs = torch.cat(all_log_probs, dim=1)
    else:
        stacked_kb_att_scores = None
        stacked_log_probs = None

    # FIXME fix kb shape being 5 (see notes)
    # assert stacked_log_probs.requires_grad, stacked_kb_att_scores.shape # FIXME remove me

    return stacked_output, stacked_attention_scores, stacked_kb_att_scores, stacked_log_probs


# pylint: disable=unused-argument
def transformer_greedy(
        src_mask: Tensor, embed: Embeddings,
        bos_index: int, max_output_length: int, decoder: Decoder, generator: Gen,
        encoder_output: Tensor, encoder_hidden: Tensor, 
        knowledgebase:Tuple=(None,None,None), trg_input: Tensor=None, e_i:float = None) -> (np.array, np.array):

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
    :param trg_input: batch.trg_input for scheduled sampling
    :param e_i: probability of taking the true token as input to next time step at each step (self doubt of the model)
    :return:
        - stacked_output: output hypotheses (2d array of indices),
        - stacked_attention_scores: attention scores (3d array)
    """
    if knowledgebase == None: # not kb task
        knowledgebase = (None,)*3
    if knowledgebase[0] is not None:
        all_log_probs = []
    
    batch_size = src_mask.size(0)

    # start with BOS-symbol for each sentence in the batch
    ys = encoder_output.new_full([batch_size, 1], bos_index, dtype=torch.long)
    # batch x max_output_length

    utils_dims_cache, kb_feed_hidden_cache = None, None

    # a subsequent mask is intersected with this in decoder forward pass
    trg_mask = src_mask.new_ones([1, 1, 1])

    for t in range(max_output_length):

        trg_embed = embed(ys)  # embed the BOS-symbol

        _ , _, out, kb_scores, utils_dims_cache, kb_feed_hidden_cache = decoder(
            trg_embed=trg_embed,
            encoder_output=encoder_output,
            encoder_hidden=None,
            src_mask=src_mask,
            unroll_steps=None,
            hidden=None,
            trg_mask=trg_mask,
            kb_keys=knowledgebase[0],
            kb_values=knowledgebase[1],
            kb_mask=knowledgebase[2],
            utils_dims_cache=utils_dims_cache,
            kb_feed_hidden_cache=kb_feed_hidden_cache
        )

        # kb_values : B x KB
        log_probs = generator(out, kb_values=knowledgebase[1], kb_probs=kb_scores)

        # TODO FIXME understand the difference between log_probs[:,-1] vs log_probs.squeeze(1)
        log_probs = log_probs[:,-1] # remove singleton time dimension => B x VOC 
        next_word = torch.argmax(log_probs, dim=-1) # B (select index of top value along VOC dim)
        next_word = next_word.unsqueeze(1) # B x time=1

        assert len(next_word.shape) == 2, next_word.shape

        if e_i > 0.0 and trg_input is not None:
            # do scheduled sampling (https://arxiv.org/abs/1506.03099 Section 2.4)
            true_y = trg_input[:,t].unsqueeze(-1) # B x time=1
            
            assert true_y.shape == next_word.shape, (true_y.shape, next_word.shape)

            feed_true_y = random.random() < e_i
            prev_y = true_y if feed_true_y else next_word 
        else:
            prev_y = next_word

        ys = torch.cat([ys, prev_y], dim=-1)

        if knowledgebase[0] is not None:
            all_log_probs.append(log_probs.unsqueeze(1)) # re-add time dimension for later concatenation => B x time=1 x VOC
    
    if kb_scores is not None:  
        stacked_kb_att_scores = kb_scores.detach().cpu().numpy() # B x M x KB
        stacked_log_probs = torch.cat(all_log_probs,dim=1) # B x M x VOC
    else:
        stacked_kb_att_scores = None
        stacked_log_probs = None

    ys = ys[:, 1:]  # remove BOS-symbol
    # return stacked_output, stacked_attention_scores, stacked_kb_att_scores, stacked_log_probs
    return ys, None, stacked_kb_att_scores, stacked_log_probs


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

    # Structure that holds finished hypotheses in order of completion.
    hypotheses = [[] for _ in range(batch_size)]

    results = {}
    results["predictions"] = [[] for _ in range(batch_size)]
    results["scores"] = [[] for _ in range(batch_size)]
    results["att_scores"] = [[] for _ in range(batch_size)]
    results["kb_att_scores"] = [[] for _ in range(batch_size)]

    # kb task: also tile kb tensors along batch dimension as done with other inputs above
    if knowledgebase != None:
        kb_values = tile(knowledgebase[1], size, dim=0)
        kb_mask = tile(knowledgebase[2], size, dim=0)

        kb_size = kb_values.size(1)
        kb_keys = knowledgebase[0]
        if isinstance(kb_keys, tuple):
            kb_keys = tuple([tile(key_dim, size, dim=0) for key_dim in kb_keys])
        else:
            kb_keys = tile(kb_keys, size, dim=0)

        att_alive = torch.Tensor( # batch*k x src x time
            [[[] for _ in range(encoder_output.size(1))] for _ in range(batch_size * size)]
        ).to(dtype=torch.float32, device=encoder_output.device)
        
        # print(f"att_alive.shape: {att_alive.shape}")
        
        kb_att_alive = torch.Tensor( # batch*k x KB x time
            [[[] for _ in range(kb_size)] for _ in range(batch_size * size)]
        ).to(dtype=torch.float32, device=encoder_output.device)
        
        print(f"kb_att_alive.shape: {kb_att_alive.shape}")
        
        stacked_attention_scores = [[] for _ in range(batch_size)]
        stacked_kb_att_scores = [[] for _ in range(batch_size)]

        util_dims_cache = None
        kb_feed_hidden_cache = None

    else:
        kb_keys, kb_values, kb_mask = None, None, None
        kb_size = None
        att_alive = None 
        kb_att_alive = None 
        stacked_attention_scores, stacked_kb_att_scores = None, None
       
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
        # pylint: disable=unused-variable
        trg_embed = embed(decoder_input)

        hidden, att_scores, att_vectors, kb_scores, util_dims_cache, kb_feed_hidden_cache = decoder(
            encoder_output=encoder_output,
            encoder_hidden=encoder_hidden,
            src_mask=src_mask,
            trg_embed=trg_embed,
            hidden=hidden,
            prev_att_vector=att_vectors,
            unroll_steps=1,
            trg_mask=trg_mask, # subsequent mask for Transformer only
            kb_keys=kb_keys, # None by default 
            kb_mask=kb_mask,
            util_dims_cache=util_dims_cache,
            kb_feed_hidden_cache=kb_feed_hidden_cache
        )

        # generator applies output layer, biases towards KB values, then applies log_softmax
        log_probs = generator(att_vectors, kb_values=kb_values, kb_probs=kb_scores)

        # hidden = ?? x batch*k x dec hidden       #FIXME why 3 ??????
        # att_scores = batch*k x 1 x src_len       #TODO Find correct beam in dim 0 at every timestep.
        # att_vectors = batch*k x 1 x dec hidden
        # kb_scores = batch*k x 1 x KB              #TODO find correct beam in dim 0 at every timestep
        # log_probs = batch*k x 1 x trg_voc

        # For the Transformer we made predictions for all time steps up to
        # this point, so we only want to know about the last time step.
        if transformer:
            logits = logits[:, -1]  # keep only the last time step
            hidden = None           # we don't need to keep it for transformer

        # batch * k x trg_vocab
        log_probs = log_probs.squeeze(1)

        # multiply probs by the beam probability ( = add logprobs)
        log_probs += topk_log_probs.view(-1).unsqueeze(1)
        curr_scores = log_probs

        # compute length penalty
        if alpha > -1:
            length_penalty = ((5.0 + (step + 1)) / 6.0) ** alpha
            curr_scores /= length_penalty

        # flatten log_probs into a list of possibilities
        curr_scores = curr_scores.reshape(-1, size * generator.output_size) # batch x k * voc FIXME

        # pick currently best top k hypotheses (flattened order)
        topk_scores, topk_ids = curr_scores.topk(size, dim=-1) # each: batch x k * k

        if alpha > -1:
            # recover original log probs
            topk_log_probs = topk_scores * length_penalty

        # reconstruct beam origin and true word ids from flattened order
        topk_beam_index = topk_ids.div(generator.output_size) # NOTE why divide by voc size??
        topk_ids = topk_ids.fmod(generator.output_size) # NOTE why mod voc size? isnt every entry < voc size?

        # map beam_index to batch_index in the flat representation
        batch_index = (
            topk_beam_index
            + beam_offset[:topk_beam_index.size(0)].unsqueeze(1))
        select_indices = batch_index.view(-1) # batch * k 

        print(alive_seq.shape)

        # append latest prediction
        alive_seq = torch.cat(
            [alive_seq.index_select(0, select_indices), # index first dim (batch * k) with the beams we want to continue this step
             topk_ids.view(-1, 1)], -1)  # batch_size*k x hyp_len

        if knowledgebase is not None:
            print(f"select_indices: {select_indices}")
            print(f"select_indices: {select_indices.shape}")
            print(f"step: {step}")
            print(f"att_alive.shape: {att_alive.shape}")
            print(f"encoder steps: {encoder_output.size(1)}")
            print(att_alive.index_select(0,select_indices).shape)
            print(att_scores.transpose(1,2).index_select(0,select_indices).shape)
            print(f"kb_att_alive.shape: {kb_att_alive.shape}")
            print(f"kb_size: {kb_size}")
            print(kb_att_alive.index_select(0,select_indices).shape)
            print(kb_scores.transpose(1,2).index_select(0,select_indices).shape)

            att_alive = torch.cat( # batch*k x src len x time
                [
                    att_alive.index_select(0,select_indices),
                    att_scores.transpose(1,2).index_select(0,select_indices)
                ],
            -1 ) 
            kb_att_alive = torch.cat( # batch*k x KB x time
                [
                    kb_att_alive.index_select(0, select_indices),
                    kb_scores.transpose(1,2).index_select(0,select_indices)
                ],
            -1) 

        # which batches are finished? 
        is_finished = topk_ids.eq(eos_index) # batch x k
        if step + 1 == max_output_length:
            # force finish
            is_finished.fill_(1)
        # end condition is whether the top beam is finished
        end_condition = is_finished[:, 0].eq(1)

        # save finished hypotheses
        if is_finished.any():

            predictions = alive_seq.view(-1, size, alive_seq.size(-1)) # batch x k x time

            for i in range(is_finished.size(0)): # iter over batches

                b = batch_offset[i]
                if end_condition[i]:
                    # this batch finished 
                    is_finished[i].fill_(1)

                finished_hyp = is_finished[i].nonzero().view(-1) # k

                # store finished hypotheses for this batch
                # (that doesnt mean the batch is completely finished, hence
                # the list 'hypotheses' being maintained outside the unroll loop)
                for j in finished_hyp: # iter over beams
                    hypotheses[b].append((
                        topk_scores[i, j], # for sorting beams by prob (below)
                        predictions[i, j, 1:])  # ignore start_token
                    )
                    if knowledgebase is not None:
                        # batch x k x src len x time 
                        attentions = att_alive.view(-1, size, att_alive.size(-2), att_alive.size(-1)) 
                        # batch x k x KB x time 
                        kb_attentions = kb_att_alive.view(-1, size, kb_att_alive.size(-2), kb_att_alive.size(-1))

                        stacked_attention_scores[b].append(
                            attentions[i,j].cpu().numpy()
                        )
                        stacked_kb_att_scores[b].append(
                            kb_attentions[i,j].cpu().numpy()
                        )

                # if the batch reached the end, save the n_best hypotheses
                if end_condition[i]:
                    # which beam is best?
                    best_hyps_descending = sorted(
                        hypotheses[b], key=lambda x: x[0], reverse=True)

                    dbg = np.array([hyp[1].cpu().numpy() for hyp in best_hyps_descending])
                    print(dbg.shape, dbg[0])

                    if knowledgebase is not None:
                        print(hypotheses[b][0],type(hypotheses[b][0]))

                        scores, hyps = zip(*hypotheses[b])
                        sort_key = np.array(scores)
                        hyps = np.array([hyp.cpu().numpy() for hyp in hyps])
                        
                        # indices that would sort hyp[b] in descending order of beam score
                        best_hyps_idx = np.argsort(sort_key)[::-1].copy() 
                        best_hyps_d_ = hyps[best_hyps_idx]

                        # unit test implementation
                        assert np.allclose( best_hyps_d_ , dbg)

                        stck_att_np = np.array(stacked_attention_scores[b])
                        stck_kb_att_np = np.array(stacked_kb_att_scores[b])


                        best_atts_d_ = stck_att_np[best_hyps_idx] 
                        best_kb_atts_d_ = stck_kb_att_np[best_hyps_idx]
                    
                    # TODO replace best_hyps_descending with best_hyps_d_ FIXME XXX (after cluster beam test)
                    for n, (score, pred) in enumerate(best_hyps_descending):
                        if n >= n_best:
                            break
                        results["scores"][b].append(score)
                        results["predictions"][b].append(pred)

                        if knowledgebase is not None:
                            results["att_scores"][b].append(best_atts_d_[n])
                            results["kb_att_scores"][b].append(best_kb_atts_d_[n])
                        
            # if all sentences are translated, no need to go further
            # pylint: disable=len-as-condition
            non_finished = end_condition.eq(0).nonzero().view(-1) # batch
            if len(non_finished) == 0:
                break
                        
            # remove finished batches for the next step
            topk_log_probs = topk_log_probs.index_select(0, non_finished)
            batch_index = batch_index.index_select(0, non_finished)
            batch_offset = batch_offset.index_select(0, non_finished)
            alive_seq = predictions.index_select(0, non_finished) \
                .view(-1, alive_seq.size(-1))

            if knowledgebase is not None:

                # TODO: go from 
                # batch x k x src len x time 
                # to
                # batch * k x time x src

                try:
                    att_alive = att_alive.index_select(0, non_finished) \
                        .view(-1, attentions.size(-1), attentions.size(-2))

                    kb_att_alive = kb_att_alive.index_select(0, non_finished) \
                        .view(-1, attentions.size(-1), attentions.size(-2))
                except RuntimeError as e:
                    assert False, (att_alive.shape, attentions.shape)

                    print(e)
                    exit(1)


        # reorder indices, outputs and masks
        select_indices = batch_index.view(-1)
        encoder_output = encoder_output.index_select(0, select_indices)
        src_mask = src_mask.index_select(0, select_indices) # for transformer

        if hidden is not None and not transformer:
            # reshape hidden to correct shape for next step
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
        if knowledgebase is not None:
            kb_values = kb_values.index_select(0, select_indices)
            if isinstance(kb_keys, tuple):
                kb_keys = tuple([key_dim.index_select(0, select_indices) for key_dim in kb_keys])
            else:
                kb_keys = kb_keys.index_select(0, select_indices)
            util_dims_cache = [utils.index_select(0, select_indices) for utils in util_dims_cache]
            kb_feed_hidden_cache = [hidden.index_select(1, select_indices) for hidden in kb_feed_hidden_cache]



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

    # final_outputs = batch x time
    final_outputs = pad_and_stack_hyps([r[0].cpu().numpy() for r in
                                        results["predictions"]],
                                       pad_value=pad_index)

    if knowledgebase is not None:
        # TODO FIXME confirm this implementation

        # stacked_attention_scores: batch x max output len x src len
        stacked_attention_scores = np.stack([
            # select attentions of top (0) beam
            atts[0].T for atts in results["att_scores"]
            ], axis=0)

        # stacked_kb_att_scores: batch x max output len x kb
        stacked_kb_att_scores = np.stack([
            kb_atts[0].T for kb_atts in results["kb_att_scores"]
            ], axis=0)

    return final_outputs, stacked_attention_scores, stacked_kb_att_scores
