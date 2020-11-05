# coding: utf-8
"""
This modules holds methods for generating predictions from a model.
"""
import os
import sys
import logging
from typing import List, Optional, Tuple
import numpy as np

import torch
from torchtext.data import Dataset, Field

from joeynmt.helpers import bpe_postprocess, load_config, \
    get_latest_checkpoint, load_checkpoint, store_attention_plots
from joeynmt.metrics import bleu, chrf, token_accuracy, sequence_accuracy, calc_ent_f1_and_ent_mcc 
from joeynmt.model import build_model, Model
from joeynmt.batch import Batch, Batch_with_KB
from joeynmt.data import load_data, make_data_iter, make_data_iter_kb, MonoDataset
from joeynmt.constants import UNK_TOKEN, PAD_TOKEN, EOS_TOKEN
from joeynmt.vocabulary import Vocabulary


# pylint: disable=too-many-arguments,too-many-locals,no-member
def validate_on_data(model: Model, 
                     data: Dataset,
                     batch_size: int,
                     use_cuda: bool, 
                     max_output_length: int,
                     level: str, 
                     eval_metric: Optional[str],
                     loss_function: torch.nn.Module = None,
                     beam_size: int = 0, 
                     beam_alpha: int = -1,
                     batch_type: str = "sentence",
                     kb_task = None,
                     valid_kb: Dataset = None,
                     valid_kb_lkp: list = [], 
                     valid_kb_lens:list=[],
                     valid_kb_truvals: Dataset = None,
                     valid_data_canon: Dataset = None,
                     report_on_canonicals: bool = False,
                     ) \
        -> (float, float, float, List[str], List[List[str]], List[str],
            List[str], List[List[str]], List[np.array]):
    """
    Generate translations for the given data.
    If `loss_function` is not None and references are given,
    also compute the loss.

    :param model: model module
    :param data: dataset for validation
    :param batch_size: validation batch size
    :param use_cuda: if True, use CUDA
    :param max_output_length: maximum length for generated hypotheses
    :param level: segmentation level, one of "char", "bpe", "word"
    :param eval_metric: evaluation metric, e.g. "bleu"
    :param loss_function: loss function that computes a scalar loss
        for given inputs and targets
    :param beam_size: beam size for validation.
        If 0 then greedy decoding (default).
    :param beam_alpha: beam search alpha for length penalty,
        disabled if set to -1 (default).
    :param batch_type: validation batch type (sentence or token)
    :param kb_task: is not None if kb_task should be executed
    :param valid_kb: MonoDataset holding the loaded valid kb data
    :param valid_kb_lkp: List with valid example index to corresponding kb indices
    :param valid_kb_len: List with amount of triples per kb 
    :param valid_data_canon: TranslationDataset of valid data but with canonized target data (for loss reporting)


    :return:
        - current_valid_score: current validation score [eval_metric],
        - valid_loss: validation loss,
        - valid_ppl:, validation perplexity,
        - valid_sources: validation sources,
        - valid_sources_raw: raw validation sources (before post-processing),
        - valid_references: validation references,
        - valid_hypotheses: validation_hypotheses,
        - decoded_valid: raw validation hypotheses (before post-processing),
        - valid_attention_scores: attention scores for validation hypotheses
        - valid_ent_f1: TODO FIXME
    """

    print(f"\n{'-'*10} ENTER VALIDATION {'-'*10}\n")

    print(f"\n{'-'*10}  VALIDATION DEBUG {'-'*10}\n")

    print("---data---")
    print(dir(data[0]))
    print([[getattr(example, attr) for attr in dir(example) if hasattr(getattr(example, attr), "__iter__") and "kb" in attr or "src" in attr or "trg" in attr] for example in data[:3] ])
    print(batch_size)
    print(use_cuda)
    print(max_output_length)
    print(level)
    print(eval_metric)
    print(loss_function)
    print(beam_size)
    print(beam_alpha)
    print(batch_type)
    print(kb_task)
    print("---valid_kb---")
    print(dir(valid_kb[0]))
    print([[getattr(example, attr) for attr in dir(example) if hasattr(getattr(example, attr), "__iter__") and "kb" in attr or "src" in attr or "trg" in attr] for example in valid_kb[:3] ])
    print(len(valid_kb_lkp), valid_kb_lkp[-5:])
    print(len(valid_kb_lens), valid_kb_lens[-5:])
    print("---valid_kb_truvals---")
    print(len(valid_kb_truvals), valid_kb_lens[-5:])
    print([[getattr(example, attr) for attr in dir(example)  if hasattr(getattr(example, attr), "__iter__") and "kb" in attr or "src" in attr or "trg" in attr or "trv" in attr]for example in valid_kb_truvals[:3]])
    print("---valid_data_canon---")
    print(len(valid_data_canon), valid_data_canon[-5:])
    print([[getattr(example, attr) for attr in dir(example) if hasattr(getattr(example, attr), "__iter__") and"kb" in attr or "src" in attr or "trg" in attr or "trv" or "can" in attr]for example in valid_data_canon[:3] ])
    print(report_on_canonicals)

    print(f"\n{'-'*10} END VALIDATION DEBUG {'-'*10}\n")

    if not kb_task:
        valid_iter = make_data_iter(
            dataset=data, batch_size=batch_size, batch_type=batch_type,
            shuffle=False, train=False)
    else:
        # knowledgebase version of make data iter and also provide canonized target data
        # data: for bleu/ent f1 
        # canon_data: for loss 
        valid_iter = make_data_iter_kb(
            data, valid_kb, valid_kb_lkp, valid_kb_lens, valid_kb_truvals,
            batch_size=batch_size,
            batch_type=batch_type,
            shuffle=False, train=False,
            canonize=model.canonize,
            canon_data=valid_data_canon)

    valid_sources_raw = data.src
    pad_index = model.src_vocab.stoi[PAD_TOKEN]

    # disable dropout
    model.eval()
    # don't track gradients during validation
    with torch.no_grad():
        all_outputs = []
        valid_attention_scores = []
        valid_kb_att_scores = []
        total_loss = 0
        total_ntokens = 0
        total_nseqs = 0
        for valid_batch in iter(valid_iter):
            # run as during training to get validation loss (e.g. xent)

            batch = Batch(valid_batch, pad_index, use_cuda=use_cuda) \
                                if not kb_task else \
                Batch_with_KB(valid_batch, pad_index, use_cuda=use_cuda)

            assert hasattr(batch, "kbsrc") == bool(kb_task)

            # sort batch now by src length and keep track of order
            if not kb_task:
                sort_reverse_index = batch.sort_by_src_lengths()
            else:
                sort_reverse_index = list(range(batch.src.shape[0]))

            # run as during training with teacher forcing
            if loss_function is not None and batch.trg is not None:

                ntokens = batch.ntokens
                if hasattr(batch, "trgcanon") and batch.trgcanon is not None:
                    ntokens = batch.ntokenscanon # normalize loss with num canonical tokens for perplexity
                # do a loss calculation without grad updates just to report valid loss
                # we can only do this when batch.trg exists, so not during actual translation/deployment
                batch_loss = model.get_loss_for_batch(
                    batch, loss_function=loss_function)
                # keep track of metrics for reporting
                total_loss += batch_loss
                total_ntokens += ntokens # gold target tokens
                total_nseqs += batch.nseqs

            # run as during inference to produce translations
            output, attention_scores, kb_att_scores = model.run_batch(
                batch=batch, beam_size=beam_size, beam_alpha=beam_alpha,
                max_output_length=max_output_length)

            # sort outputs back to original order
            all_outputs.extend(output[sort_reverse_index])
            valid_attention_scores.extend(
                attention_scores[sort_reverse_index]
                if attention_scores is not None else [])
            valid_kb_att_scores.extend(
                kb_att_scores[sort_reverse_index]
                if kb_att_scores is not None else [])

        assert len(all_outputs) == len(data)

        if loss_function is not None and total_ntokens > 0:
            # total validation loss
            valid_loss = total_loss
            # exponent of token-level negative log likelihood
            # can be seen as 2^(cross_entropy of model on valid set); normalized by num tokens; 
            # see https://en.wikipedia.org/wiki/Perplexity#Perplexity_per_word
            valid_ppl = torch.exp(valid_loss / total_ntokens)
        else:
            valid_loss = -1
            valid_ppl = -1

        # decode back to symbols
        decoding_vocab = model.trg_vocab if not kb_task else model.trv_vocab

        decoded_valid = decoding_vocab.arrays_to_sentences(arrays=all_outputs,
                                                            cut_at_eos=True)

        print(f"decoding_vocab.itos: {decoding_vocab.itos}")
        print(decoded_valid)


        # evaluate with metric on full dataset
        join_char = " " if level in ["word", "bpe"] else ""
        valid_sources = [join_char.join(s) for s in data.src]
        # TODO replace valid_references with uncanonicalized dev.car data ... requires writing new Dataset in data.py
        valid_references = [join_char.join(t) for t in data.trg]
        valid_hypotheses = [join_char.join(t) for t in decoded_valid]

        # post-process
        if level == "bpe":
            valid_sources = [bpe_postprocess(s) for s in valid_sources]
            valid_references = [bpe_postprocess(v) for v in valid_references]
            valid_hypotheses = [bpe_postprocess(v) for v in valid_hypotheses]

        # if references are given, evaluate against them
        if valid_references:
            assert len(valid_hypotheses) == len(valid_references)

            print(list(zip(valid_sources, valid_references, valid_hypotheses)))

            current_valid_score = 0
            if eval_metric.lower() == 'bleu':
                # this version does not use any tokenization
                current_valid_score = bleu(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'chrf':
                current_valid_score = chrf(valid_hypotheses, valid_references)
            elif eval_metric.lower() == 'token_accuracy':
                current_valid_score = token_accuracy(
                    valid_hypotheses, valid_references, level=level)
            elif eval_metric.lower() == 'sequence_accuracy':
                current_valid_score = sequence_accuracy(
                    valid_hypotheses, valid_references)

            if kb_task:
                valid_ent_f1, valid_ent_mcc = calc_ent_f1_and_ent_mcc(valid_hypotheses, valid_references,
                    vocab=model.trv_vocab,
                    c_fun=model.canonize,
                    report_on_canonicals=report_on_canonicals
                    )
                
            else:
                valid_ent_f1, valid_ent_mcc = -1, -1
        else:
            current_valid_score = -1

    print(f"\n{'-'*10} EXIT VALIDATION {'-'*10}\n")
    return current_valid_score, valid_loss, valid_ppl, valid_sources, \
        valid_sources_raw, valid_references, valid_hypotheses, \
        decoded_valid, valid_attention_scores, valid_kb_att_scores, \
        valid_ent_f1, valid_ent_mcc

# pylint: disable-msg=logging-too-many-args
def test(cfg_file,
         ckpt: str,
         output_path: str = None,
         save_attention: bool = False,
         logger: logging.Logger = None) -> None:
    """
    Main test function. Handles loading a model from checkpoint, generating
    translations and storing them and attention plots.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    :param output_path: path to output
    :param save_attention: whether to save the computed attention weights
    :param logger: log output to this logger (creates new logger if not set)
    """

    if logger is None:
        logger = logging.getLogger(__name__)
        FORMAT = '%(asctime)-15s - %(message)s'
        logging.basicConfig(format=FORMAT)
        logger.setLevel(level=logging.DEBUG)

    cfg = load_config(cfg_file)

    if "test" not in cfg["data"].keys():
        raise ValueError("Test data must be specified in config.")

    # when checkpoint is not specified, take latest (best) from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)
        if ckpt is None:
            raise FileNotFoundError("No checkpoint found in directory {}."
                                    .format(model_dir))
        try:
            step = ckpt.split(model_dir+"/")[1].split(".ckpt")[0]
        except IndexError:
            step = "best"

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"]["batch_size"])
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    eval_metric = cfg["training"]["eval_metric"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # load the data
    _, dev_data, test_data,\
    src_vocab, trg_vocab,\
    _, dev_kb, test_kb,\
    _, dev_kb_lookup, test_kb_lookup, \
    _, dev_kb_lengths, test_kb_lengths,\
    _, dev_kb_truvals, test_kb_truvals, \
    trv_vocab, canon_fun,\
         dev_data_canon, test_data_canon \
        = load_data(
        data_cfg=cfg["data"]
    )

    report_entf1_on_canonicals = cfg["training"].get("report_entf1_on_canonicals", False)

    kb_task = (test_kb!=None)

    data_to_predict = {"dev": dev_data, "test": test_data}

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab, trv_vocab=trv_vocab, canonizer=canon_fun)
    model.load_state_dict(model_checkpoint["model_state"])

    # FIXME for the moment, for testing, try overriding model.canonize with canon_fun from test functions loaded data
    # should hopefully not be an issue with gridsearch results...

    if use_cuda:
        model.cuda() # move to GPU

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 0
        beam_alpha = -1

    for data_set_name, data_set in data_to_predict.items():
        
        if data_set_name == "dev":
            kb_info = [dev_kb, dev_kb_lookup, dev_kb_lengths, dev_kb_truvals, dev_data_canon]
        elif data_set_name == "test":
            kb_info = [test_kb, test_kb_lookup, test_kb_lengths, test_kb_truvals, test_data_canon]
        else:
            raise ValueError((data_set_name,data_set))
        
        #pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores, kb_att_scores, ent_f1, ent_mcc = validate_on_data(
            model,
            data=data_set,
            batch_size=batch_size,
            batch_type=batch_type,
            level=level,
            max_output_length=max_output_length,
            eval_metric=eval_metric,
            use_cuda=use_cuda,
            loss_function=None,
            beam_size=beam_size,
            beam_alpha=beam_alpha,
            kb_task = kb_task,
            valid_kb=kb_info[0],
            valid_kb_lkp=kb_info[1], 
            valid_kb_lens=kb_info[2],
            valid_kb_truvals=kb_info[3],
            valid_data_canon=kb_info[4],
            report_on_canonicals=report_entf1_on_canonicals
            )
        """
                batch_size=self.eval_batch_size,
                data=valid_data,
                eval_metric=self.eval_metric,
                level=self.level, 
                model=self.model,
                use_cuda=self.use_cuda,
                max_output_length=self.max_output_length,
                loss_function=self.loss,
                beam_size=0,  
                batch_type=self.eval_batch_type,
                kb_task=kb_task,
                valid_kb=valid_kb,
                valid_kb_lkp=valid_kb_lkp,
                valid_kb_lens=valid_kb_lens,
                valid_kb_truvals=valid_kb_truvals
        """
        #pylint: enable=unused-variable

        if "trg" in data_set.fields:
            decoding_description = "Greedy decoding" if beam_size == 0 else \
                "Beam search decoding with beam size = {} and alpha = {}".\
                    format(beam_size, beam_alpha)

            logger.info("%4s %s: %6.2f f1: %6.2f mcc: %6.2f [%s]",
                        data_set_name, eval_metric, score, ent_f1, ent_mcc, decoding_description)
        else:
            logger.info("No references given for %s -> no evaluation.",
                        data_set_name)

        if save_attention:
            if attention_scores:
                attention_name = "{}.{}.att".format(data_set_name, step)
                attention_path = os.path.join(model_dir, attention_name)

                logger.info("Saving attention plots. This might take a while..")
                store_attention_plots(attentions=attention_scores,
                                      targets=hypotheses_raw,
                                      sources=data_set.src,
                                      indices=range(len(hypotheses)),
                                      output_prefix=attention_path)
                logger.info("Attention plots saved to: %s", attention_path)
            if kb_att_scores:
                kb_att_name = "{}.{}.kbatt".format(data_set_name, step)
                kb_att_path = os.path.join(model_dir, kb_att_name)
                store_attention_plots(
                    attentions=kb_att_scores,
                    targets=hypotheses_raw,
                    sources=list(data_set.kbsrc),#TODO
                    indices=range(len(hypotheses)),
                    output_prefix=kb_att_path,
                    kb_info = (dev_kb_lookup, dev_kb_lengths, list(data_set.kbtrg)))
                logger.info("KB Attention plots saved to: %s", attention_path)
    
            else:
                logger.warning("Attention scores could not be saved. "
                               "Note that attention scores are not available "
                               "when using beam search. "
                               "Set beam_size to 0 for greedy decoding.")

        if output_path is not None:
            output_path_set = "{}.{}".format(output_path, data_set_name)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            logger.info("Translations saved to: %s", output_path_set)


def translate(cfg_file, ckpt: str, output_path: str = None) -> None:
    # TODO FIXME XXX this function needs to be adapted to the KB case
    """
    Interactive translation function.
    Loads model from checkpoint and translates either the stdin input or
    asks for input to translate interactively.
    The input has to be pre-processed according to the data that the model
    was trained on, i.e. tokenized or split into subwords.
    Translations are printed to stdout.

    :param cfg_file: path to configuration file
    :param ckpt: path to checkpoint to load
    """

    def _load_line_as_data(line):
        """ Create a dataset from one line via a temporary file. """
        # write src input to temporary file
        tmp_name = "tmp"
        tmp_suffix = ".src"
        tmp_filename = tmp_name+tmp_suffix
        with open(tmp_filename, "w") as tmp_file:
            tmp_file.write("{}\n".format(line))

        test_data = MonoDataset(path=tmp_name, ext=tmp_suffix,
                                field=src_field)

        # remove temporary file
        if os.path.exists(tmp_filename):
            os.remove(tmp_filename)

        return test_data

    def _translate_data(test_data):
        """ Translates given dataset, using parameters from outer scope. """
        # pylint: disable=unused-variable
        score, loss, ppl, sources, sources_raw, references, hypotheses, \
        hypotheses_raw, attention_scores = validate_on_data(
            model, data=test_data, batch_size=batch_size,
            batch_type=batch_type, level=level,
            max_output_length=max_output_length, eval_metric="",
            use_cuda=use_cuda, loss_function=None, beam_size=beam_size,
            beam_alpha=beam_alpha,
            )
        return hypotheses

    cfg = load_config(cfg_file)

    # when checkpoint is not specified, take oldest from model dir
    if ckpt is None:
        model_dir = cfg["training"]["model_dir"]
        ckpt = get_latest_checkpoint(model_dir)

    batch_size = cfg["training"].get(
        "eval_batch_size", cfg["training"].get("batch_size", 1))
    batch_type = cfg["training"].get(
        "eval_batch_type", cfg["training"].get("batch_type", "sentence"))
    use_cuda = cfg["training"].get("use_cuda", False)
    level = cfg["data"]["level"]
    max_output_length = cfg["training"].get("max_output_length", None)

    # read vocabs
    src_vocab_file = cfg["data"].get(
        "src_vocab", cfg["training"]["model_dir"] + "/src_vocab.txt")
    trg_vocab_file = cfg["data"].get(
        "trg_vocab", cfg["training"]["model_dir"] + "/trg_vocab.txt")
    src_vocab = Vocabulary(file=src_vocab_file)
    trg_vocab = Vocabulary(file=trg_vocab_file)

    data_cfg = cfg["data"]
    level = data_cfg["level"]
    lowercase = data_cfg["lowercase"]

    tok_fun = lambda s: list(s) if level == "char" else s.split()

    src_field = Field(init_token=None, eos_token=EOS_TOKEN,
                      pad_token=PAD_TOKEN, tokenize=tok_fun,
                      batch_first=True, lower=lowercase,
                      unk_token=UNK_TOKEN,
                      include_lengths=True)
    src_field.vocab = src_vocab

    # load model state from disk
    model_checkpoint = load_checkpoint(ckpt, use_cuda=use_cuda)

    # build model and load parameters into it
    model = build_model(cfg["model"], src_vocab=src_vocab, trg_vocab=trg_vocab)
    model.load_state_dict(model_checkpoint["model_state"])

    if use_cuda:
        model.cuda()

    # whether to use beam search for decoding, 0: greedy decoding
    if "testing" in cfg.keys():
        beam_size = cfg["testing"].get("beam_size", 0)
        beam_alpha = cfg["testing"].get("alpha", -1)
    else:
        beam_size = 0
        beam_alpha = -1

    if not sys.stdin.isatty():
        # file given
        test_data = MonoDataset(path=sys.stdin, ext="", field=src_field)
        hypotheses = _translate_data(test_data)

        if output_path is not None:
            output_path_set = "{}".format(output_path)
            with open(output_path_set, mode="w", encoding="utf-8") as out_file:
                for hyp in hypotheses:
                    out_file.write(hyp + "\n")
            print("Translations saved to: {}".format(output_path_set))
        else:
            for hyp in hypotheses:
                print(hyp)

    else:
        # enter interactive mode
        batch_size = 1
        while True:
            try:
                src_input = input("\nPlease enter a source sentence "
                                  "(pre-processed): \n")
                if not src_input.strip():
                    break

                # every line has to be made into dataset
                test_data = _load_line_as_data(line=src_input)

                hypotheses = _translate_data(test_data)
                print("JoeyNMT: {}".format(hypotheses[0]))

            except (KeyboardInterrupt, EOFError):
                print("\nBye.")
                break
