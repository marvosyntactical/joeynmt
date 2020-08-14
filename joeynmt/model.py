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
from torch import Tensor, cat, FloatTensor
from torch import argmax
import torch.nn.functional as F


from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, KeyValRetRNNDecoder, TransformerDecoder, Generator
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch, Batch_with_KB
from joeynmt.helpers import ConfigurationError, Timer
from joeynmt.constants import EOS_TOKEN, PAD_TOKEN


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
                 trv_vocab: Vocabulary=None,
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
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.generator = generator
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN] #TODO find out when these are used
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        #kb stuff:
        self.kb_embed = self.trg_embed 
        if trv_vocab != None:
            self.trv_vocab = trv_vocab #TODO should probably be deleted altogether

        self.pad_idx_src = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_idx_src = self.src_vocab.stoi[EOS_TOKEN]

        self.Timer = Timer()


    # pylint: disable=arguments-differ
    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                src_lengths: Tensor, trg_mask: Tensor = None, kb_keys: Tensor = None) -> (
        Tensor, Tensor, Tensor, Tensor, Tensor):
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
                           kb_keys=kb_keys)

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
               trg_mask: Tensor = None, kb_keys: Tensor = None) \
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
        if kb_keys == None:
            return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask)
        else:
            return self.decoder(trg_embed=self.trg_embed(trg_input),
                            encoder_output=encoder_output,
                            encoder_hidden=encoder_hidden,
                            src_mask=src_mask,
                            unroll_steps=unroll_steps,
                            hidden=decoder_hidden,
                            trg_mask=trg_mask,
                            kb_keys=kb_keys)


    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """

        print(f"\n{'-'*10}TRN FWD PASS: START current batch{'-'*10}\n")

        # pylint: disable=unused-variable
        if not hasattr(batch, "kbsrc"): # no kb task
            hidden, att_probs, att_vectors , _ = self.forward(
                src=batch.src, trg_input=batch.trg_input,
                src_mask=batch.src_mask, src_lengths=batch.src_lengths,
                trg_mask=batch.trg_mask)

            kb_keys, kb_values, kb_trv, kb_probs = None, None, None, None # for uniform generator call 
        else: #kb task
            assert batch.kbsrc != None

            kb_keys, kb_values, kb_trv = self.preprocess_batch_kb(batch)

            with self.Timer("model fwd pass"):
                hidden, att_probs, att_vectors, kb_probs = self.forward(
                    src=batch.src, trg_input=batch.trg_input,
                    src_mask=batch.src_mask, src_lengths=batch.src_lengths,
                    trg_mask=batch.trg_mask, kb_keys=kb_keys)
        
        # pass att_vectors through Generator

        log_probs = self.generator(att_vectors, kb_values=kb_values, kb_probs=kb_probs)


        # ----- debug start
        mle_tokens = argmax(log_probs, dim=-1) # torch argmax
        mle_tokens = mle_tokens.cpu().numpy()

        print(f"proc_batch: Hypothesis: {self.trg_vocab.arrays_to_sentences(mle_tokens)[-1]}")
        # ----- debug end

        # compute batch loss
        batch_loss = loss_function(log_probs, batch.trg)

        print(f"\n{'-'*10}TRN FWD PASS: END current batch{'-'*10}\n")

        # return batch loss = sum over all elements in batch that are not pad
        return batch_loss

    def preprocess_batch_kb(self, batch: Batch_with_KB, detailed_debug=True)-> (Tensor, Tensor):

        kb_keys = batch.kbsrc
        kb_values = batch.kbtrg
        kb_true_vals = batch.kbtrv.T.contiguous()

        # TODO to save a little time, figure out how to avoid putting eos here
        # during init
        kb_keys[kb_keys == self.eos_idx_src] = self.pad_idx_src #replace eos with pad # TODO why was this after embed and working before?

        with self.Timer("converting arrays to sentences for current batch"):

            idx = batch.src.shape[0]-1 #print last example

            print(f"proc_batch: batch.src: {self.src_vocab.arrays_to_sentences(batch.src.cpu().numpy())[idx]}")
            print(f"proc_batch: batch.trg: {self.trg_vocab.arrays_to_sentences(batch.trg.cpu().numpy())[idx]}")
            print(f"proc_batch: kbkeys: {self.src_vocab.arrays_to_sentences(kb_keys.cpu().numpy())}")
            print(f"proc_batch: kbvalues: {self.trg_vocab.arrays_to_sentences(kb_values[:,1].unsqueeze(1).cpu().numpy())}")

            print(f"debug: batch.kbtrv:{self.trv_vocab.arrays_to_sentences(batch.kbtrv[:,1].unsqueeze(1).cpu().numpy())}")
        
        # remove bos, eos tokens
        kb_values = kb_values[:, 1] 
        kb_true_vals = kb_true_vals[1, :]

        if detailed_debug:
            with self.Timer("kb_true_vals checks"):
                print(f"debug: kb_true_vals :{kb_true_vals.shape}")
                print(f"debug: kb_true_vals content:{kb_true_vals}")

        kb_keys = self.src_embed(kb_keys)
        # NOTE: values dont need to be embedded!

        kb_keys = kb_keys.sum(dim=1) # sum embeddings of subj, rel (pad is all 0 in embedding!)


        # correct dimensions and make contiguous in memory
        # (put in specifically allocated contiguous memory slot)
        kb_keys.unsqueeze_(0)
        kb_keys = kb_keys.repeat((batch.src.shape[0], 1, 1)).contiguous() # batch x 1 x kb
        kb_values.unsqueeze_(0)
        kb_values = kb_values.repeat((batch.trg.shape[0], 1)).contiguous() # batch x kb
        kb_true_vals.unsqueeze_(0)
        kb_true_vals = kb_true_vals.repeat((batch.trg.shape[0], 1)).contiguous() # batch x kb

        if detailed_debug:
            print(f"proc_batch: kbkeys.shape: {kb_keys.shape}")
            print(f"proc_batch: kbvalues.shape: {kb_values.shape}") # FIXME no embedding dim??

        # FIXME FIXME TODO ????
        # kb_values are most of the time here like so:
        # batch x kb_size # 3 x 33
        # but sometimes have one extra dim???:
        # 2 x 1 x 1
        # where does this sometimes come from?
        
        # NOTE shape debug; TODO add to decoder check shapes fwd
        """
        print(f"debug: dir(batch):{[s for s in dir(batch) if s.startswith(('kb', 'src', 'trg'))]}")
        print(f"debug: batch.src.shape:{batch.src.shape}")
        print(f"debug: batch.trg.shape:{batch.trg.shape}")
        #assert batch.src.shape[0] == 3,batch.src.shape[0] # Latest TODO find where this happens???
        
        print(f"debug: model.trg_embed attributes={dir(self.trg_embed)}")
        """

        assert_msg = (kb_keys.shape, kb_values.shape, kb_true_vals.shape, kb_true_vals)

        #batch dim equal
        assert kb_keys.shape[0] == kb_values.shape[0] == kb_true_vals.shape[0], assert_msg
        #kb dim equal
        assert kb_keys.shape[1] == kb_values.shape[1] == kb_true_vals.shape[1], assert_msg

        return kb_keys, kb_values, kb_true_vals


    def run_batch(self, batch: Batch, max_output_length: int, beam_size: int,
                  beam_alpha: float) -> (np.array, np.array):
        """
        Get outputs and attentions scores for a given batch

        :param batch: batch to generate hypotheses for
        :param max_output_length: maximum length of hypotheses
        :param beam_size: size of the beam for beam search, if 0 use greedy
        :param beam_alpha: alpha value for beam search
        :return: stacked_output: hypotheses for batch,
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
            kb_keys, kb_values, kb_trv = self.preprocess_batch_kb(batch)
            oldshape = kb_values.shape
            # assert kb_values.shape[1] == 1, kb_values.shape 
            knowledgebase = (kb_keys, kb_values)
        else:
            knowledgebase = None

        # greedy decoding
        if beam_size == 0:
            stacked_output, stacked_attention_scores, stacked_kb_att_scores = greedy(
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

        if knowledgebase != None:
            with self.Timer("postprocessing hypotheses"):
                # replace kb value tokens with actual values in hypotheses, e.g. 
                # ['your','conference','is','at','@meeting_time'] => ['your', 'conference', 'is', 'at', '7pm']
                # assert kb_values.shape[1] == 1, kb_values.shape
                stacked_output = self.postprocess_batch_hypotheses(stacked_output, stacked_kb_att_scores, kb_values, kb_trv)

        return stacked_output, stacked_attention_scores, stacked_kb_att_scores
    
    def postprocess_batch_hypotheses(self, stacked_output, stacked_kb_att_scores, kb_values, kb_truval) -> np.array:

        """
        called in self.run_batch() during knowledgebase task

        postprocesses batch hypotheses
        replaces kb value tokens such as @meeting_time with 7pm

        Arguments:
        :param stacked_output: Tensor
        :param stacked_kb_att_scores: Tensor
        :param kb_values: Tensor
        :param kb_truval: Tensor
        :return: post_proc_stacked_output
        """


        #                          dimensions:  # (recurrent)         # (transf)    # use:
        kb_trv = kb_truval.cpu().numpy()[0,:]   # kb                  # kb          # used as replacement
        kb_val = kb_values.cpu().numpy()[0,0,:] # kb                  # kb          # used for indexing
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

                if token >= self.trg_vocab.canon_onwards: # this token is a canonical token (@traffic\_info) => replace it

                    str_tok = trvSent([token])
                    hypotSent = self.trg_vocab.array_to_sentence(hyp)

                    print(f"\npp: {'='*10} DECIDING REPLACEMENT FOR CANONICAL: {str_tok} {'='*10}\n")
                    print(f"pp: while deciding for hypothesis:\n{hypotSent}")
                    print(f"pp: decoded hypothesis thus far:\n{trvSent(post_proc_hyp)}")

                    # assert str_tok[0] in hypotSent, (str_tok, hypotSent)

                    matching_trv_candidates = np.where(kb_val==token, kb_trv, -1) 
                    #1 dim array of kb true values if belonging to same canonical category (time/distance) as token
                    # only dim: kb: [-1,-1,-1,998,-1,-1,-1,973,-1,-1,-1,1058,-1,...,-1]

                    print(f"pp: matching_trv_candidates tokens (should belong to same canonical):\n \
                        {trvSent(matching_trv_candidates[matching_trv_candidates!=-1].tolist())}")

                    if matching_trv_candidates[matching_trv_candidates!=-1].shape[0]: # match(es) found!

                        print(f"pp: SUCCESS! Found matches for canonical: {str_tok}")

                        # now order matching != -1 by corresponding attention values
                        matching_scores = np.where(matching_trv_candidates!=-1, kb_att[i,step,:], float("-inf"))

                        print(f"pp: matching_scores (should have no '-1's):\n{matching_scores}") # should not contain '-1's

                        top_matching = np.argsort(matching_scores)[::-1].copy() # reverse index array in descending order of score


                        top_match_candids = matching_trv_candidates[top_matching] # only for printing
                        print(f"pp: matching_trv_candidates in descending order of attention:\n\
                            {trvSent(top_match_candids[top_match_candids!=-1].tolist())}")


                        top1_match = matching_trv_candidates[top_matching[0]]
                        print(f"pp: top1_match:\n\
                            {trvSent([top1_match])}")

                        assert top1_match != -1, "somehow selected true value with non matching canonical category, shouldnt happen" 

                        post_proc_hyp.append(int(top1_match)) # append this true value instead of the token

                    else:
                        # what went wrong: look at highest attended options:

                        print(f"pp: FAILURE! Found no matches for canonical: {str_tok}")

                        scores = kb_att[i,step,:]
                        hi_scores = np.argsort(scores)[::-1].copy()
                        print(f"pp: failure debug: highest attended tokens overall:\n\
                            {trvSent(kb_trv[hi_scores].tolist())}")

                        print(f"pp: CURRENT POLICY: REPLACING FOUND CANONICAL {str_tok} WITH NON-MATCHING HIGHEST ATTENDED")

                        top1_but_not_matching = kb_trv[hi_scores[0]]

                        post_proc_hyp.append(top1_but_not_matching) # didnt find a match for this, policy: append highest attended but non matching token
                    
                    print(f"\npp: {'+'*10} DECIDED REPLACEMENT FOR CANONICAL: {str_tok}: {trvSent([post_proc_hyp[-1]])} {'+'*10}\n")
                else: 
                    post_proc_hyp.append(token) # append normal non canonical token as it was found in hypothesis
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
                trv_vocab: Vocabulary = None) -> Model:
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
       

    
    # build decoder
    kb = bool(cfg.get("kb", False))
    assert cfg["decoder"]["hidden_size"]
    dec_dropout = cfg["decoder"].get("dropout", 0.)
    dec_emb_dropout = cfg["decoder"]["embeddings"].get("dropout", dec_dropout)
    if cfg["decoder"].get("type", "recurrent") == "transformer":
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout, kb_task=kb)
    else:
        if not kb:
            decoder = RecurrentDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = KeyValRetRNNDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    
    # specify generator which is mostly just the output layer
    generator = Generator(
        dec_hidden_size=cfg["decoder"]["hidden_size"],
        vocab_size=len(trg_vocab)
    )

    model = Model(encoder=encoder, decoder=decoder, generator=generator,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab,\
                  trv_vocab=trv_vocab)

    # tie softmax layer with trg embeddings
    if cfg.get("tied_softmax", False):
        if trg_embed.lut.weight.shape == \
                model.decoder.output_layer.weight.shape:
            # (also) share trg embeddings and softmax layer:
            model.decoder.output_layer.weight = trg_embed.lut.weight
        else:
            raise ConfigurationError(
                "For tied_softmax, the decoder embedding_dim and decoder "
                "hidden_size must be the same."
                "The decoder must be a Transformer.")

    # custom initialization of model parameters
    initialize_model(model, cfg, src_padding_idx, trg_padding_idx)

    return model
