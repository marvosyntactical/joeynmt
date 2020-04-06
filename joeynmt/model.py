# coding: utf-8
"""
Module to represent whole models
"""

from typing import Tuple

import numpy as np

import torch.nn as nn
from torch import Tensor, cat, FloatTensor
from torch import argmax
import torch.nn.functional as F

from joeynmt.initialization import initialize_model
from joeynmt.embeddings import Embeddings
from joeynmt.encoders import Encoder, RecurrentEncoder, TransformerEncoder
from joeynmt.decoders import Decoder, RecurrentDecoder, KeyValRetRNNDecoder, TransformerDecoder
from joeynmt.constants import PAD_TOKEN, EOS_TOKEN, BOS_TOKEN
from joeynmt.search import beam_search, greedy
from joeynmt.vocabulary import Vocabulary
from joeynmt.batch import Batch, Batch_with_KB
from joeynmt.helpers import ConfigurationError
from joeynmt.constants import EOS_TOKEN, PAD_TOKEN


class Model(nn.Module):
    """
    Base Model class
    """

    def __init__(self,
                 encoder: Encoder,
                 decoder: Decoder,
                 src_embed: Embeddings,
                 trg_embed: Embeddings,
                 src_vocab: Vocabulary,
                 trg_vocab: Vocabulary,
                 kb_vocab: Vocabulary = None) -> None:
        """
        Create a new encoder-decoder model

        :param encoder: encoder
        :param decoder: decoder
        :param src_embed: source embedding
        :param trg_embed: target embedding
        :param src_vocab: source vocabulary
        :param trg_vocab: target vocabulary
        """
        super(Model, self).__init__()

        self.src_embed = src_embed
        self.trg_embed = trg_embed
        self.encoder = encoder
        self.decoder = decoder
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.bos_index = self.trg_vocab.stoi[BOS_TOKEN] #TODO find out when these are used
        self.pad_index = self.trg_vocab.stoi[PAD_TOKEN]
        self.eos_index = self.trg_vocab.stoi[EOS_TOKEN]
        #kb stuff:
        self.kb_embed = self.trg_embed 
        self.kb_vocab = self.trg_vocab #TODO should probably be deleted altogether

        self.pad_idx_src = self.src_vocab.stoi[PAD_TOKEN]
        self.eos_idx_src = self.src_vocab.stoi[EOS_TOKEN]



    # pylint: disable=arguments-differ
    def forward(self, src: Tensor, trg_input: Tensor, src_mask: Tensor,
                src_lengths: Tensor, trg_mask: Tensor = None, knowledgebase=None) -> (
        Tensor, Tensor, Tensor, Tensor):
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
        print("decoder.forward() unroll_steps in model.forward")
        print(f"is {unroll_steps}, which is 1st dim of trg_input={trg_input.shape}")
        print(f"batch.trg_input: {self.trg_vocab.arrays_to_sentences(trg_input)[-1]}")
        
        return self.decode(encoder_output=encoder_output,
                           encoder_hidden=encoder_hidden,
                           src_mask=src_mask, trg_input=trg_input,
                           unroll_steps=unroll_steps,
                           trg_mask=trg_mask,
                           knowledgebase=knowledgebase) #tuple of kbsrc, kbtrg

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
               trg_mask: Tensor = None, knowledgebase=None) \
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
        if knowledgebase == None:
            assert False
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
                            knowledgebase=knowledgebase)


    def get_loss_for_batch(self, batch: Batch, loss_function: nn.Module) \
            -> Tensor:
        """
        Compute non-normalized loss and number of tokens for a batch

        :param batch: batch to compute loss for
        :param loss_function: loss function, computes for input and target
            a scalar loss for the complete batch
        :return: batch_loss: sum of losses over non-pad elements in the batch
        """
        # pylint: disable=unused-variable
        if not hasattr(batch, "kbsrc"):
            out, hidden, att_probs, _ = self.forward(
                src=batch.src, trg_input=batch.trg_input,
                src_mask=batch.src_mask, src_lengths=batch.src_lengths,
                trg_mask=batch.trg_mask)
        else:
            assert batch.kbsrc != None
            #kb embedding /preproc

            # TODO: Find out how to reconstruct words here for debugging
            # involves trg_vocab 
            # self.trg_vocab.itos

            # TODO: kbvals (list) should be passed through joeynmt funs but doesnt need to be passed on here: 
            # actual values not needed for training

            # TODO: find out why both old attr batch.src/trg and new
            # attr batch.kbsrc/kbtrg are tuples in data.TorchBatchWithKB
            # but only new attributes remain tuples here???

            # TODO: move this shape changing to different model function
            # or write new model.reshape_kb_batch function
            # with options for lookup list and without lookup list
            # (respectively inference and training)

            print(f"\n{'-'*10}TRN FWD PASS: START current batch{'-'*10}\n")

            kb_keys, kb_values, kb_trv = self.process_batch_kb(batch)
            knowledgebase = (kb_keys, kb_values)


            out, hidden, att_probs, _ = self.forward(
                src=batch.src, trg_input=batch.trg_input,
                src_mask=batch.src_mask, src_lengths=batch.src_lengths,
                trg_mask=batch.trg_mask, knowledgebase=knowledgebase)
            
            
        
        # add u_t to log_probs indexed by kb_values here!

        # compute log probs
        log_probs = F.log_softmax(out, dim=-1)

        # ----- debug
        # Latest TODO: decode to sentences for debugging:
        mle_tokens = argmax(log_probs, dim=-1)
        mle_tokens = mle_tokens.cpu().numpy()
        print(mle_tokens.shape)
        print(f"proc_batch: Hypothesis: {self.trg_vocab.arrays_to_sentences(mle_tokens)[-1]}")
        # ----- debug


        # compute batch loss
        batch_loss = loss_function(log_probs, batch.trg)
        # return batch loss = sum over all elements in batch that are not pad
        print(f"\n{'-'*10}TRN FWD PASS: END current batch{'-'*10}\n")

        return batch_loss

    def process_batch_kb(self, batch: Batch_with_KB)-> (Tensor, Tensor):

        kb_keys = batch.kbsrc[0]
        kb_values = batch.kbtrg[0]

        idx = batch.src.shape[0]-1 #plot last example
        
        print(f"proc_batch: batch.src: {self.src_vocab.arrays_to_sentences(batch.src.cpu().numpy())[idx]}")
        print(f"proc_batch: batch.trg: {self.trg_vocab.arrays_to_sentences(batch.trg.cpu().numpy())[idx]}")
        print(f"proc_batch: kbkeys: {self.trg_vocab.arrays_to_sentences(kb_keys.cpu().numpy())}")
        print(f"proc_batch: kbvals: {self.trg_vocab.arrays_to_sentences(kb_values.cpu().numpy())}")

        kb_keys = self.src_embed(kb_keys)
        # NOTE: values dont even need to be embedded!!
        kb_values = kb_values[:, 1] # remove bos, eos tokens

        kb_keys[kb_keys == self.eos_idx_src] = self.pad_idx_src 
        # TODO to save a little time, figure out how to avoid putting eos here
        # during init
        kb_keys = kb_keys.sum(dim=1) # sum embeddings of subj, rel

        assert batch.src.shape[0] == batch.trg.shape[0]
        kb_keys.unsqueeze_(0)
        kb_keys = kb_keys.repeat((batch.src.shape[0], 1, 1))
        kb_values.unsqueeze_(0)
        kb_values = kb_values.repeat((batch.trg.shape[0], 1))
        # assert len(kb_values) == 2 # TODO super important sanity check unit test:
        # kb_values are most of the time here like so:
        # batch x kb_size # 3 x 33
        # but sometimes have one extra dim???:
        # 2 x 1 x 1
        # where does this sometimes come from?
        kb_true_vals = batch.kbtrv.T.unsqueeze(1)

        # NOTE shape debug; TODO add to decoder check shapes fwd
        """
        print(f"kb_keys.shape:{kb_keys.shape}")#batch x kb_size x emb_dim
        print(f"kb_values.shape:{kb_values.shape}")#batch x kb_size
        print(f"debug: dir(batch):{[s for s in dir(batch) if s.startswith(('kb', 'src', 'trg'))]}")
        print(f"debug: batch.src.shape:{batch.src.shape}")
        print(f"debug: batch.trg.shape:{batch.trg.shape}")
        #assert batch.src.shape[0] == 3,batch.src.shape[0] # Latest TODO find where this happens???
        print(f"debug: batch.kbtrv.shape:{batch.kbtrv.shape}")
        print(f"debug: batch.kbtrv:{batch.kbtrv}")
        print(f"debug: kb_true_vals :{kb_true_vals.shape}")
        # NOTE kbtrv.shape should be same as u_t for
        # replacement!
        # u_t == kbtrv: batch x 1 x kb_size
        print(f"debug: model.trg_embed attributes={dir(self.trg_embed)}")
        """

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

        kb_keys, kb_values, kb_trv = self.process_batch_kb(batch)
        knowledgebase = (kb_keys, kb_values)

        # greedy decoding
        if beam_size == 0:
            stacked_output, stacked_attention_scores = greedy(
                    encoder_hidden=encoder_hidden,
                    encoder_output=encoder_output,
                    src_mask=batch.src_mask, embed=self.trg_embed,
                    bos_index=self.bos_index, decoder=self.decoder,
                    max_output_length=max_output_length,
                    knowledgebase = knowledgebase)
            # batch, time, max_src_length
        else:  # beam size
            stacked_output, stacked_attention_scores = \
                    beam_search(
                        size=beam_size, encoder_output=encoder_output,
                        encoder_hidden=encoder_hidden,
                        src_mask=batch.src_mask, embed=self.trg_embed,
                        max_output_length=max_output_length,
                        alpha=beam_alpha, eos_index=self.eos_index,
                        pad_index=self.pad_index,
                        bos_index=self.bos_index,
                        decoder=self.decoder,
                        knowledgebase = knowledgebase)

        return stacked_output, stacked_attention_scores

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
                trg_vocab: Vocabulary = None) -> Model:
    """
    Build and initialize the model according to the configuration.

    :param cfg: dictionary configuration containing model specifications
    :param src_vocab: source vocabulary
    :param trg_vocab: target vocabulary
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
        assert not kb, "Transformer with kb not implemented yet"
        decoder = TransformerDecoder(
            **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
            emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
    else:
        if not kb:
            decoder = RecurrentDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)
        else:
            decoder = KeyValRetRNNDecoder(
                **cfg["decoder"], encoder=encoder, vocab_size=len(trg_vocab),
                emb_size=trg_embed.embedding_dim, emb_dropout=dec_emb_dropout)



    model = Model(encoder=encoder, decoder=decoder,
                  src_embed=src_embed, trg_embed=trg_embed,
                  src_vocab=src_vocab, trg_vocab=trg_vocab)

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
