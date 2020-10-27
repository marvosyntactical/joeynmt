# coding: utf-8
"""
Module to represent whole models
"""

from typing import Tuple
import random #FIXME remove

from copy import deepcopy
from collections import defaultdict

import numpy as np

import torch.nn as nn
from torch import Tensor, cat, FloatTensor, arange, argmax, float32
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, KeyValRetRNNDecoder, TransformerDecoder, TransformerKBrnnDecoder, Generator
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN, UNASSIGNED_TOKEN, DEFAULT_UNK_ID
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch, Batch_with_KB
from joeynmt.helpers import ConfigurationError, Timer, product, split_tensor_on_pads
from joeynmt.constants import EOS_TOKEN, PAD_TOKEN
from joeynmt.transformer_layers import PositionalEncoding


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 generator: nn.Module,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 kb_key_embed: Embeddings,
                 trv_vocab: Vocabulary=None,
                 k_hops: int = 1,
                 do_postproc: bool = True,
                 canonize = None,
                 kb_att_dims : int = 1,
                 posEncKBkeys: bool = False, 
                 ) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        :param trv_vocab: kb true value lookup vocabulary
        :param k_hops: number of kvr attention forward passes to do
        :param do_postproc: do postprocessing (decode canonical tokens) in KVR task?
        :param canonize: callable canonization object to try to create KB on the fly if none exists; not used by model but piggybacks off it
        :param kb_att_dims: number of dimensions of KB 
        :param posEncdKBkeys: apply positional encoding to KB keys?
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN]
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]

        #kb stuff:
        self.kbsrc_embed = kb_key_embed if kb_key_embed is not None else self.src_embed# optionally use separate embedding table
        if trv_vocab != None:
            self.trv_vocab = trv_vocab #TODO should probably be deleted altogether
        self.pad_idx_kbsrc = self.src_vocab.stoi[PAD_TOKEN] # FIXME used for kb only?
        self.eos_idx_src = self.src_vocab.stoi[EOS_TOKEN]
        self.k_hops = k_hops # FIXME global number of kvr attention forward passes to do
        self.do_postproc = do_postproc
        self.canonize = canonize
        self.kb_att_dims = kb_att_dims
        if posEncKBkeys:
            try:
                decoder_hidden_size = self.decoder.hidden_size
            except AttributeError:
                decoder_hidden_size = self.decoder._hidden_size

            self.posEnc = PositionalEncoding(decoder_hidden_size,e=2) # must be hidden size of attention mechanism actually FIXME (they the same tho atm)
        self.Timer = Timer()


    # pylint: disable=arguments-differ
    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                src_lengths: Tensor, trg_mask: Tensor = None, kb_keys: Tensor = None,
                kb_mask = None, kb_values_embed=None) -> (Tensor, Tensor, Tensor, Tensor, Tensor):
        """
        First encodes the source sentence.
        Then produces the target one word at a time.

        :param src: source input
        :param trg_input: target input
        :param src_mask: source mask
        :param src_lengths: length of source inputs
        :param trg_mask: target mask
        :return: decoder outputs
        """
        encoder_output, encoder_hidden = self.encode(src=src,
                                                     src_length=src_lengths,
                                                     src_mask=src_mask)
        unroll_steps = trg_input.size(1)

        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask,
                           kb_keys=kb_keys,
                           kb_mask=kb_mask,
                           kb_values_embed=kb_values_embed)

    def encode(self, src: Tensor, src_length: Tensor, src_mask: Tensor) \
        -> (Tensor, Tensor):
        """
        Encodes the source sentence.

        :param src:
        :param src_length:
        :param src_mask:
        :return: encoder outputs (output, hidden_concat)
        """
        return self.encoder(self.src_embed(src), src_length, src_mask)

    def decode(self, encoder_output: Tensor, encoder_hidden: Tensor,
               src_mask: Tensor, trg_input: Tensor,
               unroll_steps: int, decoder_hidden: Tensor = None,
               trg_mask: Tensor = None, kb_keys: Tensor = None,
               kb_mask: Tensor=None, kb_values_embed = None) \
        -> (Tensor, Tensor, Tensor, Tensor):
        """
        Decode, given an encoded source sentence.

        :param encoder_output: encoder states for attention computation
        :param encoder_hidden: last encoder state for decoder initialization
        :param src_mask: source mask, 1 at valid tokens
        :param trg_input: target inputs
        :param unroll_steps: number of steps to unrol the decoder for
        :param decoder_hidden: decoder hidden state (optional)
        :param trg_mask: mask for target steps
        :return: decoder outputs (outputs, hidden, att_probs, att_vectors)
        """
        return self.decoder(trg_embed=self.trg_embed(trg_input),
                        encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=src_mask,
                        unroll_steps=unroll_steps,
                        hidden=decoder_hidden,
                        trg_mask=trg_mask,
                        kb_keys=kb_keys,
                        k_hops=self.k_hops,
                        kb_mask=kb_mask,
                        kb_values_embed=kb_values_embed)

    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module,
    max_output_length: int = None, e_i: float = 1., greedy_threshold: float = 0.9) -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :param max_output_length: maximum length of hypotheses
        :param e_i: scheduled sampling probability of taking true label vs model generation at every decoding step
        (https://arxiv.org/abs/1506.03099 Section 2.4)
        :param greedy_threshold: only actually do greedy search once e_i is below this threshold
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """

        print(f"\n{'-'*10}GET LOSS FWD PASS: START current batch{'-'*10}\n")

        assert 0. <= e_i <= 1., f"e_i={e_i} should be a probability"
        do_teacher_force = e_i >= greedy_threshold # prefer to still do teacher forcing when e_i="label taking probability" is high in scheduled sampling

        trg, trg_input, trg_mask = batch.trg, batch.trg_input, batch.trg_mask
        batch_size = trg.size(0)

        if hasattr(batch, "kbsrc"):
            kb_keys, kb_values, kb_values_embed, _, kb_mask = self.preprocess_batch_kb(batch, kbattdims=self.kb_att_dims)
        else:
            kb_keys = None
        
        log_probs = None

        if kb_keys is not None: # kb task
            assert batch.kbsrc != None, batch.kbsrc

            # FIXME hardcoded attribute name
            if hasattr(batch, "trgcanon"): 
                # get loss on canonized target data during validation, see joeynmt.prediction.validate_on_data
                # batch size sanity check
                assert batch.trgcanon.shape[0] == batch.trg.shape[0], [t.shape for t in [batch.trg, batch.trgcanon]]
                # reassign these variables for loss calculation 
                trg, trg_input, trg_mask = batch.trgcanon, batch.trgcanon_input, batch.trgcanon_mask
            
            if not do_teacher_force: # scheduled sampling
                # only use true labels with probability 0 <= e_i < 1; otherwise take previous model generation;
                # => do a greedy search (autoregressive training as hinted at in Eric et al)
                with self.Timer("model training: KB Task: do greedy search"):

                    encoder_output, encoder_hidden = self.encode(
                        batch.src, batch.src_lengths, batch.src_mask)

                    # if maximum output length is not globally specified, adapt to src len
                    if max_output_length is None:
                        max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

                    print(f"in model.glfb; kb_keys are {kb_keys}")
                    stacked_output, stacked_attention_scores, stacked_kb_att_scores, log_probs = greedy(
                            encoder_hidden=encoder_hidden, 
                            encoder_output=encoder_output,
                            src_mask=batch.src_mask,
                            embed=self.trg_embed,
                            bos_index=self.bos_index,
                            decoder=self.decoder, 
                            generator=self.generator,
                            max_output_length=trg.size(-1),
                            knowledgebase = (kb_keys, kb_values, kb_values_embed, kb_mask),
                            trg_input=trg_input,
                            e_i=e_i,
                            )
            else: # take true label at every step => just do fwd pass (normal teacher forcing training)
                with self.Timer("model training: KB Task: model fwd pass"):

                    hidden, att_probs, out, kb_probs, _, _ = self.forward(
                        src=batch.src, trg_input=trg_input,
                        src_mask=batch.src_mask, src_lengths=batch.src_lengths,
                        trg_mask=trg_mask, kb_keys=kb_keys, kb_mask=kb_mask, kb_values_embed=kb_values_embed)

        else: # vanilla, not kb task
            if not do_teacher_force:
                raise NotImplementedError("scheduled sampling only works for KB task atm")

            hidden, att_probs, out, kb_probs, _, _ = self.forward(
                src=batch.src, trg_input=trg_input,
                src_mask=batch.src_mask, src_lengths=batch.src_lengths,
                trg_mask=trg_mask)
            kb_values = None

        if log_probs is None:
            # same generator fwd pass for KB task and no KB task if teacher forcing
            # pass output through Generator and add biases for KB entries in vocab indexes of kb values
            log_probs = self.generator(out, kb_probs=kb_probs, kb_values=kb_values)
        
        if hasattr(batch, "trgcanon"):
            # only calculate loss on this field of the batch during validation loss calculation 
            assert not log_probs.requires_grad, "using batch.trgcanon shouldnt happen / be done during training (canonized data is used in the 'trg' field there)"


        # check number of classes equals prediction distribution support
        # (can otherwise lead to nasty CUDA device side asserts that dont give a traceback to here)
        assert log_probs.size(-1) == self.generator.output_size, (log_probs.shape, self.generator.output_size)

        # compute batch loss
        try:
            batch_loss = loss_function(log_probs, trg)
        except Exception as e:
            print(f"batch_size: {batch_size}")
            print(f"log_probs= {log_probs.shape}")
            print(f"trg = {trg.shape}")
            print(f"")
            print(f"")
            raise e

        # confirm trg is actually canonical:
        # input(f"loss is calculated on these sequences: {self.trv_vocab.arrays_to_sentences(trg.cpu().numpy())}")

        with self.Timer("debugging: greedy hypothesis:"):
            mle_tokens = argmax(log_probs, dim=-1) # torch argmax
            mle_tokens = mle_tokens.cpu().numpy()

            print(f"proc_batch: Hypothesis: {self.trg_vocab.arrays_to_sentences(mle_tokens)[-1]}")

        print(f"\n{'-'*10}GET LOSS FWD PASS: END current batch{'-'*10}\n")

        # batch loss = sum xent over all elements in batch that are not pad
        return batch_loss


    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: 
            stacked_output: hypotheses for batch,
            stacked_attention_scores: attention scores for batch
        """

        encoder_output, encoder_hidden = self.encode(
            batch.src, batch.src_lengths,
            batch.src_mask)

        # if maximum output length is not globally specified, adapt to src len
        if max_output_length is None:
            max_output_length = int(max(batch.src_lengths.cpu().numpy()) * 1.5)

        if hasattr(batch, "kbsrc"):
            # B x KB x EMB; B x KB; B x KB
            kb_keys, kb_values, kb_values_embed, kb_trv, kb_mask = self.preprocess_batch_kb(batch, kbattdims=self.kb_att_dims)
            if kb_keys is None:
                knowledgebase = None
            else:
                knowledgebase = (kb_keys, kb_values, kb_values_embed, kb_mask)
        else:
            knowledgebase = None

        # greedy decoding
        if beam_size == 0:
            stacked_output, stacked_attention_scores, stacked_kb_att_scores, _ = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.src_mask, embed=self.trg_embed,
                    bos_index=self.bos_index, decoder=self.decoder, generator=self.generator,
                    max_output_length=max_output_length,
                knowledgebase = knowledgebase)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores, stacked_kb_att_scores = \
                    beam_search(
                        decoder=self.decoder,
                        generator=self.generator,
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=batch.src_mask, embed=self.trg_embed,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        knowledgebase = knowledgebase)
        
        if knowledgebase != None and self.do_postproc:
            with self.Timer("postprocessing hypotheses"):
                # replace kb value tokens with actual values in hypotheses, e.g. 
                # ['your','@event','is','at','@meeting_time'] => ['your', 'conference', 'is', 'at', '7pm']
                # assert kb_values.shape[1] == 1, kb_values.shape
                stacked_output = self.postprocess_batch_hypotheses(stacked_output, stacked_kb_att_scores, kb_values, kb_trv)

            print(f"proc_batch: Hypotheses: {self.trv_vocab.arrays_to_sentences(stacked_output)}")
        else:
            print(f"proc_batch: Hypotheses: {self.trg_vocab.arrays_to_sentences(stacked_output)}")

        return stacked_output, stacked_attention_scores, stacked_kb_att_scores
        
    def preprocess_batch_kb(self, batch: Batch_with_KB, detailed_debug=True, kbattdims=1, posEnc=False) -> \
        (Tensor, Tensor, Tensor, Tensor):

        kb_keys = batch.kbsrc
        kb_values = batch.kbtrg
        kb_true_vals = batch.kbtrv.T.contiguous()

        """
        # use <UNK> token among targets to determine if we should just return None
        # true values should not contain unknown words (trv vocab should be initialized from concatenation of train/dev/test)
        if (kb_true_vals == DEFAULT_UNK_ID()).any():
            use_dummy_kb = True
        """

        kb_keys[kb_keys == self.eos_idx_src] = self.pad_idx_kbsrc #replace eos with pad

        with self.Timer("converting arrays to sentences for current batch"):

            idx = batch.src.shape[0]-1 # print last example

            print(f"proc_batch: batch.src: {self.src_vocab.arrays_to_sentences(batch.src.cpu().numpy())[idx]}")
            print(f"proc_batch: batch.trg: {self.trv_vocab.arrays_to_sentences(batch.trg.cpu().numpy())[idx]}")
            print(f"proc_batch: kbkeys: {self.src_vocab.arrays_to_sentences(kb_keys.cpu().numpy())}")
            print(f"proc_batch: kbvalues: {self.trg_vocab.arrays_to_sentences(kb_values[:,1].unsqueeze(1).cpu().numpy())}")

            print(f"debug: batch.kbtrv:{self.trv_vocab.arrays_to_sentences(batch.kbtrv[:,1].unsqueeze(1).cpu().numpy())}")
        
        # remove bos, eos tokens
        kb_values = kb_values[:, 1] 
        kb_true_vals = kb_true_vals[1, :]

        # ADD BATCH DIMENSION
        # correct dimensions and make contiguous in memory
        # (put in specifically allocated contiguous memory slot)
        kb_values.unsqueeze_(0)
        kb_values = kb_values.repeat((batch.trg.shape[0], 1)).contiguous() # batch x kb

        kb_true_vals.unsqueeze_(0)
        kb_true_vals = kb_true_vals.repeat((batch.trg.shape[0], 1)).contiguous() # batch x kb

        # embed kb values for transformer style transformer implementation (with multihead KB att instead of RNN stuff)
        kb_values_embed = self.trg_embed(kb_values)

        # (also add batch dim to keys below)

        if detailed_debug:
            with self.Timer("kb_true_vals checks"):
                print(f"debug: kb_true_vals :{kb_true_vals.shape}")
                print(f"debug: kb_true_vals content:{kb_true_vals}")

        with self.Timer(f"knowledgebase attention dimensions == {kbattdims}"):
            if kbattdims > 1:
                # reshape kb_keys in kb dimension from KB to SUBJ x REL or KB to kb1 x kb2 x ... x kbn
                # |keys| == KB x key_repr

                # go along dim 1 (key_repr) and find first column where all entries are <PAD> (before are subjects, after are relation)
                pad_val = self.pad_idx_kbsrc
                kb_size, key_repr_size = kb_keys.shape
                kb_entries = [] 
                dims = [] # dimensions of each kb entry
                for entry in kb_keys:

                    entry_dim_vals = split_tensor_on_pads(entry, pad_val=pad_val)
                    dims.append(len(entry_dim_vals))

                    kb_entries.append(entry_dim_vals) # before first <PAD> is subj repr

                kb_dim_entries = list(zip(*kb_entries)) # [list of subject tensors, list of relation tensors]

                assert set(dims) in [{kbattdims},{1}], (dims,set(dims), {kbattdims}) # dimensions wrong or different num of dimensions
                assert set([len(dim) for dim in kb_entries]) in [{kbattdims},{1}], \
                     (set([len(dim) for dim in kb_entries]), {kbattdims})
                assert len(kb_dim_entries) in [kbattdims, 1], (len(kb_dim_entries), kbattdims)

                kb_repr = []
                steps = [1]
                dim_embeds = []
                dim_embeddings_word_wise = [] 

                for dim, entries in enumerate(kb_dim_entries): 

                    # e.g. relation = [tensor(997),tensor(998),tensor(999),...] (repeat)

                    max_repr = max([len(repr) for repr in entries])

                    entries_padded = [F.pad(entry, (pad_val, max_repr-entry.shape[0])).unsqueeze(1) for entry in entries]
                    entries_padded_stacked = cat(entries_padded, dim=1)

                    # sum embeddings for each dim
                    word_wise_embed_of_dim = self.kbsrc_embed(entries_padded_stacked) # num_entry_tokens x kb_size x emb 
                    kb_dim_embed = word_wise_embed_of_dim.sum(dim=0) # kb_size x emb 
                    dim_embeds += [kb_dim_embed]
                    dim_embeddings_word_wise += [word_wise_embed_of_dim]

                    if len(kb_dim_embed) == 1:
                        steps += [1]
                        continue

                    # find out size of block (num of entries for first subject)
                    # and step (num of entries until relation same as first relation)

                    i = step = 1
                    found_second_entry = False
                    block_flag, step_flag = False, False
                    first_entry = word_wise_embed_of_dim[1,i] # embed of first token of first dim of first KB entry 

                    # find out if first entry is repeated
                    # multiple times in a block (subjects)
                    # or multiple times every so many steps (relations)

                    while i < kb_size-1:
                        # FIXME doing the step & block calc like this is extremely inefficient
                        # do this before embed & sum?
                        if not word_wise_embed_of_dim[1,i+1].allclose(first_entry):
                            # continue step  (found different entry)
                            step_flag = True
                        else:
                            # finish step
                            if step_flag == True:
                                step = i # did see different entry in between first and this same one; this dimension is arranged in steps of i (relations)
                                found_second_entry = True
                                break
                            elif step_flag == False:
                                step = 1 # didnt see different entry in between first and this same one; no step size; all entries are entrered in a contiguous block (subjects)
                                found_second_entry = True
                                break
                        i += 1
                    if found_second_entry == False: #KB has just one subj, but possibly several attr (on the fly KB)
                        step = i+1 # set number of attributes to KB size (i is at end of while loop)

                    assert kb_size % step == 0, (kb_size, step, self.src_vocab.arrays_to_sentences(entries))

                    steps += [step]

                # steps = [1,1,5,15,60] => step_i = steps[i+1]//steps[i], block_i = kb_size/step_i+1
                # dim_sizes = [5,3,4]

                # attempt at FIXME ing VALINKEY bug:
                steps = sorted(steps)

                dim_sizes = [int(steps[i]/steps[i-1]) for i in range(1,len(steps))] 
                block_sizes = dim_sizes[1:]+[kb_size] 

                dim_sizes = dim_sizes[::-1] # 5,1
                block_sizes = block_sizes[::-1] # 40, 5

                for i in range(kbattdims):
                    step_i = dim_sizes[i] 
                    block_i = block_sizes[i]

                    kb_dim_entry_set = dim_embeds[i][:block_i:step_i]

                    kb_dim_entry_set = kb_dim_entry_set.unsqueeze(0) # add batch dimension to keys
                    kb_dim_entry_set = kb_dim_entry_set.repeat((batch.trg.shape[0],1,1)).contiguous() # batch x kb_dim_i x embed

                    kb_repr.append(kb_dim_entry_set)

                if len(kb_repr) == kbattdims:
                    kb_keys = tuple(kb_repr)
                else: # just one dummy entry, use it for both dims
                    assert kb_size == 1, kb_size
                    kb_keys = tuple(kb_repr[0], deepcopy(kb_repr[0]))

                if detailed_debug:
                    print(steps, kb_size, [t.shape for t in kb_repr],\
                        self.src_vocab.arrays_to_sentences(entries[:block_i:step_i]))

                shape_check_keys = kb_keys[0]

                # make sure product of dims is equal to flat representation
                assert product([key_dim.shape[1] for key_dim in kb_keys]) == kb_size,\
                    [key_dim.shape[1] for key_dim in kb_keys]
                # make sure none of the dims is 1 if KB can be decomposed

                # FIXME should check this: rarely one dimension is just 1 and the other has all the info

            else: # normal (1D) mode 
                # option for positonal encoding here

                # NOTE: values dont need to be embedded! they are only used for indexing
                kb_keys = self.kbsrc_embed(kb_keys)

                if posEnc:
                    # positional encoding needs this format:
                    # ``(seq_len, batch_size, self.dim)`` (batch size is KB size here)
                    kb_keys = self.posEnc(kb_keys.transpose(0,1)).transpose(0,1)

                kb_keys = kb_keys.sum(dim=1) # sum embeddings of subj, rel (pad is all 0 in embedding!)
            
                # add batch dimension to keys
                kb_keys.unsqueeze_(0)
                kb_keys = kb_keys.repeat((batch.src.shape[0], 1, 1)).contiguous() # batch x kb x emb

                shape_check_keys = kb_keys
        
        assert len(kb_values.shape) == 2, kb_values.shape

        assert len(kb_values_embed.shape) == 3, kb_values_embed.shape
        
        assert_msg = (shape_check_keys.shape, kb_values.shape, kb_true_vals.shape, kb_true_vals)

        # batch dim equal
        assert shape_check_keys.shape[0] == kb_values.shape[0] == kb_true_vals.shape[0], assert_msg
        # kb dim equal 
        assert kb_values.shape[1] == kb_true_vals.shape[1], assert_msg

        # mask entries that arent assigned (empty calendar scheduling fields) in flat representation
        kb_mask = kb_true_vals == self.trv_vocab.stoi[UNASSIGNED_TOKEN]
        if type(self.decoder) == TransformerDecoder:
            kb_mask = kb_mask.to(float32) 
        
        assert len(kb_mask.shape) == 2, kb_mask.shape
        
        # FIXME this should hopefully trigger at some point for partially assigned scheduling dialogues
        # assert (kb_mask==0.).all(), kb_mask 

        return kb_keys, kb_values, kb_values_embed, kb_true_vals, kb_mask



    
    def postprocess_batch_hypotheses(self, stacked_output, stacked_kb_att_scores, kb_values, kb_truval, pad_value=float("-inf")) -> np.array:

        """
        called in self.run_batch() during knowledgebase task

        postprocesses batch hypotheses
        replaces kb value tokens such as @meeting_time with 7pm

        Arguments:
        :param stacked_output: array
        :param stacked_kb_att_scores: array
        :param kb_values: Tensor
        :param kb_truval: Tensor
        :param attention_pad_value: float indicating what parts of each attention matrix in the stacked_kb_att_score array to cut off in case of beam search
        :return: post_proc_stacked_output
        """

        print(stacked_kb_att_scores.shape, kb_values.shape, kb_truval.shape)

        #                          dimensions:  # (recurrent)         # (transf)    # use:
        kb_trv = kb_truval.cpu().numpy()[0,:]   # kb                  # kb          # used as replacement
        kb_val = kb_values.cpu().numpy()[0,:]   # kb                  # kb          # used for filtering non matching tokens
        kb_att = stacked_kb_att_scores          # batch x unroll x kb # B x M x KB  # local attention ordering info (used for indexing)

        # assert kb_values.shape[2] == 1, (kb_values.shape, kb_val.shape, kb_att.shape, kb_truval.shape, stacked_output.shape)

        print("[[[[[[[[[[[[[[ START POSTPROC VALID/TEST BATCH ]]]]]]]]]]]]]]")

        # for debugging/code readability :
        trvSent = self.trv_vocab.array_to_sentence

        post_proc_stacked_output = []
        outputs = stacked_output.tolist()

        for i, hyp in enumerate(outputs):
            post_proc_hyp = []
            for step, token in enumerate(hyp): # go through i_th hypothesis
                # (token is integer index in self.trg_vocab)
                if token == self.eos_index:
                    break
                if token >= self.trg_vocab.canon_onwards: # this token is a canonical token (@traffic\_info) => replace it

                    kb_att_i = kb_att[i]
                    """
                    if (kb_att_i == pad_value).any():
                        # try to remove padded columns

                        idx_first_pad_ = (kb_att_i==pad_val).non_zero(as_tuple)[:,0].min().item()
                        kb_att_i = kb_att_i[:idx_first_pad_,:]
                    """

                    str_tok = trvSent([token])
                    hypotSent = self.trg_vocab.array_to_sentence(hyp)

                    print(f"\npp: {'='*10} DECIDING REPLACEMENT FOR CANONICAL: {str_tok} {'='*10}\n")
                    print(f"pp: while deciding for hypothesis:\n{hypotSent}")
                    print(f"pp: decoded hypothesis thus far:\n{trvSent(post_proc_hyp)}")

                    assert str_tok[0] in hypotSent, (hyp, str_tok, hypotSent)

                    matching_trv_candidates = np.where(kb_val==token, kb_trv, -1) 
                    #1 dim array of kb true values if belonging to same canonical category (time/distance) as token
                    # only dim: kb: [-1,-1,-1,998,-1,-1,-1,973,-1,-1,-1,1058,-1,...,-1]

                    print(f"pp: matching_trv_candidates tokens (should belong to same canonical):\n \
                        {trvSent(matching_trv_candidates[matching_trv_candidates != -1].tolist())}")

                    if matching_trv_candidates[matching_trv_candidates != -1].shape[0]: # match(es) found!

                        print(f"pp: SUCCESS! Found matches for canonical: {str_tok}")

                        try:
                            # now order matching != -1 by corresponding attention values
                            matching_scores = np.where(matching_trv_candidates != -1, kb_att_i[step,:], float("-inf"))
                        except Exception as e:
                            print(stacked_kb_att_scores.shape)
                            raise e

                        print(f"pp: matching_scores (should have no '-1's):\n{matching_scores}") # should not contain '-1's

                        top_matching = np.argsort(matching_scores)[::-1].copy()
                        # reverse index array in descending order of score

                        top_match_candids = matching_trv_candidates[top_matching] # only for printing
                        print(f"pp: matching_trv_candidates in descending order of attention:\n\
                            {trvSent(top_match_candids[top_match_candids!=-1].tolist())}")

                        top1_match = matching_trv_candidates[top_matching[0]]
                        print(f"pp: top1_match:\n\
                            {trvSent([top1_match])}")

                        assert top1_match != -1, "somehow selected true value with non matching canonical category, shouldnt happen" 

                        post_proc_hyp.append(int(top1_match)) # append this true value instead of the token

                    else:
                        # something went wrong: look at highest attended options:

                        print(f"pp: FAILURE! Found no matches for canonical: {str_tok}")

                        scores = kb_att_i[step,:]
                        hi_scores = np.argsort(scores)[::-1].copy()
                        print(f"pp: failure debug: highest attended tokens overall:\n\
                            {trvSent(kb_trv[hi_scores].tolist())}")

                        print(f"pp: CURRENT POLICY: REPLACING FOUND CANONICAL {str_tok} WITH NON-MATCHING HIGHEST ATTENDED")

                        top1_but_not_matching = kb_trv[hi_scores[0]]

                        post_proc_hyp.append(top1_but_not_matching) # didnt find a match for this, policy: append highest attended but non matching token
                    
                    print(f"\npp: {'+'*10} DECIDED REPLACEMENT FOR CANONICAL: {str_tok}: {trvSent([post_proc_hyp[-1]])} {'+'*10}\n")
                else: 
                    post_proc_hyp.append(token) # append normal non canonical token as it was found in hypothesis
            print(f"pp: finished hyp: {trvSent(post_proc_hyp)}, hyp past first <EOS> would be:\
                {trvSent(post_proc_hyp, cut_at_eos=False)}")
            post_proc_stacked_output.append(post_proc_hyp)
        print()
        print(f"pp: raw hyps:\n{self.trg_vocab.arrays_to_sentences(outputs)}")
        print()
        print(f"pp: post processed hyps:\n{self.trv_vocab.arrays_to_sentences(post_proc_stacked_output)}")
        print()
        print(f"pp: knowledgebase: {trvSent(kb_trv.tolist()[:40])}")
        print()
        print("[[[[[[[[[[[[[[ END POSTPROC VALID/TEST BATCH ]]]]]]]]]]]]]]")

        post_proc_stacked_output = np.array(post_proc_stacked_output)

        return post_proc_stacked_output


    def __repr__(self) -> str:
        """
        String representation: a description of encoder, decoder and embeddings

        :return: string representation
        """
        return "%s(\n" \
               "\tencoder=%s,\n" \
               "\tdecoder=%s,\n" \
               "\tsrc_embed=%s,\n" \
               "\ttrg_embed=%s)" % (self.__class__.__name__, self.encoder,
                   self.decoder, self.src_embed, self.trg_embed)


def build_model(cfg: dict = None,
                src_vocab: Vocabulary = None,
                trg_vocab: Vocabulary = None,
                trv_vocab: Vocabulary = None,
                canonizer = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
    :param trv_vocab: kb true value lookup vocabulary
    :return: built and initialized model
    """
    src_padding_idx = src_vocab.stoi[PAD_TOKEN]
    trg_padding_idx = trg_vocab.stoi[PAD_TOKEN]
    
    if "embedding_files" in cfg.keys(): #init from pretrained
        assert not cfg.get("tied_embeddings", False), "TODO implement tied embeddings along with pretrained initialization"
        raise NotImplementedError("TODO implement kbsrc embed loading for embedding files")
        weight_tensors = []
        for weight_file in cfg["embedding_files"]:
            with open(weight_file, "r") as f:
                weight = []
                for line in f.readlines():
                    line = line.split()
                    line = [float(x) for x in line]
                    weight.append(line)

            weight = FloatTensor(weight)
            weight_tensors.append(weight)
        # Set source Embeddings to Pretrained Embeddings
        src_embed = Embeddings(int(weight_tensors[0][0].shape[0]),
                                    False, #TODO transformer: change to True
                                    len(weight_tensors[0]),
                                    )
        src_embed.lut.weight.data = weight_tensors[0]

        # Set target Embeddings to Pretrained Embeddings
        trg_embed = Embeddings(int(weight_tensors[1][0].shape[0]),
                                    False, #TODO transformer: change to True
                                    len(weight_tensors[1]),
                                    )
        trg_embed.lut.weight.data = weight_tensors[1]
    else:
        src_embed = Embeddings(
            **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
            padding_idx=src_padding_idx)
        if cfg.get("kb_embed_separate", False):
            kbsrc_embed = Embeddings(
                **cfg["encoder"]["embeddings"], vocab_size=len(src_vocab),
                padding_idx=src_padding_idx)
        else:
            kbsrc_embed = src_embed 

        # this ties source and target embeddings
        # for softmax layer tying, see further below
        if cfg.get("tied_embeddings", False):
            if src_vocab.itos == trg_vocab.itos:
                # share embeddings for src and trg
                trg_embed = src_embed
            else:
                raise ConfigurationError(
                    "Embedding cannot be tied since vocabularies differ.")
        else:
            # Latest TODO: init embeddings with vocab_size = len(trg_vocab joined with kb_vocab)
            trg_embed = Embeddings(
                **cfg["decoder"]["embeddings"], vocab_size=len(trg_vocab),
                padding_idx=trg_padding_idx)
    # build encoder
    enc_dropout = cfg["encoder"].get("dropout", 0.)
    enc_emb_dropout = cfg["encoder"]["embeddings"].get("dropout", enc_dropout)
    if cfg["encoder"].get("type", "recurrent") == "transformer":
        assert cfg["encoder"]["embeddings"]["embedding_dim"] == \
               cfg["encoder"]["hidden_size"], \
               "for transformer, emb_size must be hidden_size"

        encoder = TransformerEncoder(**cfg["encoder"],
                                     emb_size=src_embed.embedding_dim,
                                     emb_dropout=enc_emb_dropout)
    else:
        encoder = RecurrentEncoder(**cfg["encoder"],
                                   emb_size=src_embed.embedding_dim,
                                   emb_dropout=enc_emb_dropout)
       

    
    # retrieve kb task info
    kb_task = bool(cfg.get("kb", False))
    k_hops = int(cfg.get("k_hops", 1)) # k number of kvr attention layers in decoder (eric et al/default: 1)
    same_module_for_all_hops = bool(cfg.get("same_module_for_all_hops", False))
    do_postproc = bool(cfg.get("do_postproc", True))
    copy_from_source = bool(cfg.get("copy_from_source", True))
    canonization_func = None if canonizer is None else canonizer(copy_from_source=copy_from_source) 
    kb_input_feeding = bool(cfg.get("kb_input_feeding", True))
    kb_feed_rnn = bool(cfg.get("kb_feed_rnn", True))
    kb_multihead_feed = bool(cfg.get("kb_multihead_feed", False))
    posEncKBkeys = cfg.get("posEncdKBkeys", False)
    tfstyletf = cfg.get("tfstyletf", True)
    infeedkb = bool(cfg.get("infeedkb", False))
    outfeedkb = bool(cfg.get("outfeedkb", False))
    add_kb_biases_to_output = bool(cfg.get("add_kb_biases_to_output", True))
    kb_max_dims = cfg.get("kb_max_dims", (16,32)) # should be tuple
    double_decoder = cfg.get("double_decoder", False)
    tied_side_softmax = cfg.get("tied_side_softmax", False) # actually use separate linear layers, tying only the main one

    if hasattr(kb_max_dims, "__iter__"):
        kb_max_dims = tuple(kb_max_dims)
    else:
        assert type(kb_max_dims) == int, kb_max_dims
        kb_max_dims = (kb_max_dims,)

    assert cfg["decoder"]["hidden_size"]
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)

    if cfg["decoder"].get("type", "recurrent") == "transformer":
        if tfstyletf:
            decoder = TransformerDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout,
                kb_task=kb_task, kb_key_emb_size=kbsrc_embed.embedding_dim,
                feed_kb_hidden=kb_input_feeding, infeedkb=infeedkb, outfeedkb=outfeedkb,
                double_decoder=double_decoder
                )
        else:
            decoder = TransformerKBrnnDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, 
                emb_dropout=dec_emb_dropout,
                kb_task=kb_task, 
                k_hops=k_hops, 
                kb_max=kb_max_dims,
                same_module_for_all_hops=same_module_for_all_hops,
                kb_key_emb_size=kbsrc_embed.embedding_dim,
                kb_input_feeding=kb_input_feeding, 
                kb_feed_rnn=kb_feed_rnn,
                kb_multihead_feed=kb_multihead_feed)
    else:
        if not kb_task:
            decoder = RecurrentDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = KeyValRetRNNDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout, 
                k_hops=k_hops, kb_max=kb_max_dims,
                same_module_for_all_hops=same_module_for_all_hops,
                kb_key_emb_size=kbsrc_embed.embedding_dim, 
                kb_input_feeding=kb_input_feeding, 
                kb_feed_rnn=kb_feed_rnn,
                kb_multihead_feed=kb_multihead_feed)
    
    # specify generator which is mostly just the output layer
    generator = Generator(
        dec_hidden_size= cfg["decoder"]["hidden_size"],
        vocab_size=len(trg_vocab),
        add_kb_biases_to_output=add_kb_biases_to_output,
        double_decoder=double_decoder
    )

    model = Model(
                  encoder=encoder, decoder=decoder, generator=generator,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab,\
                  kb_key_embed=kbsrc_embed,\
                  trv_vocab=trv_vocab,
                  k_hops=k_hops, 
                  do_postproc=do_postproc,
                  canonize=canonization_func, 
                  kb_att_dims=len(kb_max_dims),
                  posEncKBkeys=posEncKBkeys
                  )

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.generator.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.generator.output_layer.weight = trg_embed.lut.weight
            if model.generator.double_decoder:
                # (also also) share trg embeddings and side softmax layer
                assert hasattr(model.generator, "side_output_layer")
                if tied_side_softmax:
                    # because of distributivity this becomes O (x_1+x_2) instead of O_1 x_1 + O_2 x_2
                    model.generator.side_output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
