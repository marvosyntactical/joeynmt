# coding: utf-8
"""
Collection of helper functions
"""
import copy
import glob
import os
import os.path
import errno
import shutil
import random
import logging
from logging import Logger
from typing import Callable, Optional, List, Tuple
from contextlib import contextmanager
import time
import numpy as np

import torch
from torch import nn, Tensor

from torch.utils.tensorboard import SummaryWriter

from torchtext.data import Dataset
import yaml
from joeynmt.vocabulary import Vocabulary
from joeynmt.plotting import plot_heatmap
from joeynmt.data import create_KB_on_the_fly


class ConfigurationError(Exception):
    """ Custom exception for misspecifications of configuration """


def make_model_dir(model_dir: str, overwrite=False) -> str:
    """
    Create a new directory for the model.

    :param model_dir: path to model directory
    :param overwrite: whether to overwrite an existing directory
    :return: path to model directory
    """
    if os.path.isdir(model_dir):
        if not overwrite:
            raise FileExistsError(
                "Model directory exists and overwriting is disabled.")
        # delete previous directory to start with empty dir again
        shutil.rmtree(model_dir)
    os.makedirs(model_dir)
    return model_dir


def make_logger(model_dir: str, log_file: str = "train.log") -> Logger:
    """
    Create a logger for logging the training process.

    :param model_dir: path to logging directory
    :param log_file: path to logging file
    :return: logger object
    """
    logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.DEBUG)
    fh = logging.FileHandler(
        "{}/{}".format(model_dir, log_file))
    fh.setLevel(level=logging.DEBUG)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(message)s')
    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logging.getLogger("").addHandler(sh)
    logger.info("Hello! This is Joey-NMT.")
    return logger


def log_cfg(cfg: dict, logger: Logger, prefix: str = "cfg") -> None:
    """
    Write configuration to log.

    :param cfg: configuration to log
    :param logger: logger that defines where log is written to
    :param prefix: prefix for logging
    """
    for k, v in cfg.items():
        if isinstance(v, dict):
            p = '.'.join([prefix, k])
            log_cfg(v, logger, prefix=p)
        else:
            p = '.'.join([prefix, k])
            logger.info("{:34s} : {}".format(p, v))


def clones(module: nn.Module, n: int) -> nn.ModuleList:
    """
    Produce N identical layers. Transformer helper function.

    :param module: the module to clone
    :param n: clone this many times
    :return cloned modules
    """
    return nn.ModuleList([copy.deepcopy(module) for _ in range(n)])


def subsequent_mask(size: int) -> Tensor:
    """
    Mask out subsequent positions (to prevent attending to future positions)
    Transformer helper function.

    :param size: size of mask (2nd and 3rd dim)
    :return: Tensor with 0s and 1s of shape (1, size, size)
    """
    mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    return torch.from_numpy(mask) == 0


def set_seed(seed: int) -> None:
    """
    Set the random seed for modules torch, numpy and random.

    :param seed: random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def log_data_info(train_data: Dataset, valid_data: Dataset, test_data: Dataset,
                  src_vocab: Vocabulary, trg_vocab: Vocabulary,
                  logging_function: Callable[[str], None]) -> None:
    """
    Log statistics of data and vocabulary.

    :param train_data:
    :param valid_data:
    :param test_data:
    :param src_vocab:
    :param trg_vocab:
    :param logging_function:
    """
    logging_function(
        "Data set sizes: \n\ttrain %d,\n\tvalid %d,\n\ttest %d",
            len(train_data), len(valid_data),
            len(test_data) if test_data is not None else 0)

    logging_function("First training example:\n\t[SRC] %s\n\t[TRG] %s",
        " ".join(vars(train_data[0])['src']),
        " ".join(vars(train_data[0])['trg']))

    logging_function("First 10 words (src): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(src_vocab.itos[:10])))
    logging_function("First 10 words (trg): %s", " ".join(
        '(%d) %s' % (i, t) for i, t in enumerate(trg_vocab.itos[:10])))

    logging_function("Number of Src words (types): %d", len(src_vocab))
    logging_function("Number of Trg words (types): %d", len(trg_vocab))


def load_config(path="configs/default.yaml") -> dict:
    """
    Loads and parses a YAML configuration file.

    :param path: path to YAML configuration file
    :return: configuration dictionary
    """
    with open(path, 'r') as ymlfile:
        cfg = yaml.safe_load(ymlfile)
    return cfg


def bpe_postprocess(string) -> str:
    """
    Post-processor for BPE output. Recombines BPE-split tokens.

    :param string:
    :return: post-processed string
    """
    return string.replace("@@ ", "")


def store_attention_plots(attentions: np.array, targets: List[List[str]],
                          sources: List[List[str]],
                          output_prefix: str, indices: List[int],
                          tb_writer: Optional[SummaryWriter] = None,
                          steps: int = 0,
                          kb_info: Tuple[List[int]] = None,
                          on_the_fly_info: Tuple = None) -> str:
    """
    Saves attention plots.

    :param attentions: attention scores
    :param targets: list of tokenized targets
    :param sources: list of tokenized sources
    :param output_prefix: prefix for attention plots
    :param indices: indices selected for plotting
    :param tb_writer: Tensorboard summary writer (optional)
    :param steps: current training steps, needed for tb_writer
    :param dpi: resolution for images
    :param kbinfo: tuple of the valid set's kb_lkp, kb_lens, kb_truvals
    :param on_the_fly_info: tuple containing valid_data.src field, valid_kb, canonization function, model.trg_vocab
    """
    success, failure = 0,0
    for i in indices:
        i -= 1
        if i < 0:
            i = len(indices)+i
        if i >= len(sources):
            continue
        plot_file = "{}.{}.pdf".format(output_prefix, i)

        attention_scores = attentions[i].T # => KB x UNROLL
        print(f"PLOTTING: shape of {i}th attention matrix from print_valid_sents: {attention_scores.shape}")
        trg = targets[i]
        if kb_info is None:
            src = sources[i]
        else:
            kbkey = sources
            kb_lkp, kb_lens, kbtrv = kb_info
            kbtrv_fields = kbtrv.fields # needed for on the fly creation below
            kbtrv = list(kbtrv)
            print(f"KB PLOTTING: kb_lens: {kb_lens}")

            # index calculation (find batch in valid/test files using kb lookup indices and length info)
            kb_num = kb_lkp[i]
            lower = sum(kb_lens[:kb_num])
            upper = lower+kb_lens[kb_num]+1
            calcKbLen = upper-lower

            if calcKbLen == 1 and attention_scores.shape[0] > 1:
                # FIXME make this an option in the cfg
                # this is a scheduling KB created on the fly in data.batch_with_kb
                # TODO which fields are needed to recreate it on the fly here
                # valid_kb: has fields kbsrc, kbtrg; valid_kbtrv
                valid_src, valid_kb, canon_func, trg_vocab = on_the_fly_info
                v_src = list(valid_src)

                on_the_fly_kb, on_the_fly_kbtrv = create_KB_on_the_fly(
                    # FIXME perhaps matchup issues are due to generator to list issues?
                    v_src[i], trg_vocab, valid_kb.fields, kbtrv_fields, canon_func
                )

                keys = [entry.kbsrc for entry in on_the_fly_kb]
                vals = on_the_fly_kbtrv

                calcKbLen = len(keys) # update with length of newly created KB

                print(f"KB PLOTTING: on the fly recreation:")
                print(keys, [v.kbtrv for v in vals])
                print(f"calcKbLen={calcKbLen}")
            else:

                keys = kbkey[lower:upper]
                vals = kbtrv[lower:upper]

                # in the normal case (non_empty KB),
                # the kb lengths (i) summed and (ii) looked up 
                # should match up
                assert calcKbLen == kb_lens[kb_num]+1, (calcKbLen, kb_lens[kb_num]+1)

            assertion_str = f"plotting idx={i} with kb_num={kb_num} and kb_len={kb_lens[kb_num]+1},\n\
                kb_before: {kb_lens[kb_num-1]+1}, kb_after: {kb_lens[kb_num+1]+1};\n\
                att_scores.shape={attention_scores.shape};\n\
                calcKbLen={calcKbLen};\n\
                kb_lens[kb_num]+1={kb_lens[kb_num]+1};"

            # make sure attention plots have the right shape
            if not calcKbLen == attention_scores.shape[0]:
                print(f"Couldnt plot example {i} because knowledgebase was created on the fly")
                print(f"actual shape mismatch: retrieved: {calcKbLen} vs att matrix: {attention_scores.shape[0]}")
                print(assertion_str)
                # FIXME FIXME FIXME FIXME im doing something wrong with the vocab lookup in the code above
                failure += 1
                continue
                



            print(f"KB PLOTTING: calcKbLen: {calcKbLen}")
            print(f"KB PLOTTING: calcKbLen should be != 0 often!!: {assertion_str}")

            # index application 
            DUMMY = "@DUMMY=@DUMMY"

            src = [DUMMY]+["+".join(key)+"="+val.kbtrv[0] for key, val in zip(keys, vals)]

        try:
            fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                               row_labels=src, output_path=plot_file,
                               dpi=100)
            if tb_writer is not None:
                # lower resolution for tensorboard
                fig = plot_heatmap(scores=attention_scores, column_labels=trg,
                                   row_labels=src, output_path=None, dpi=50)
                tb_writer.add_figure("attention/{}.".format(i), fig,
                                     global_step=steps)
            print("plotted example {}: src len {}, trg len {}, "
            "attention scores shape {}".format(i, len(src), len(trg), attention_scores.shape))
        # pylint: disable=bare-except
            success += 1
        except:
            print("Couldn't plot example {}: src len {}, trg len {}, "
                  "attention scores shape {}".format(i, len(src), len(trg),
                                                     attention_scores.shape))
            failure += 1
            continue

    assert success+failure == len(indices), f"plotting success:{success}, failure:{failure}, indices:{len(indices)}"
    return f"{success}/{len(indices)}"



def get_latest_checkpoint(ckpt_dir: str) -> Optional[str]:
    """
    Returns the latest checkpoint (by time) from the given directory.
    If there is no checkpoint in this directory, returns None

    :param ckpt_dir:
    :return: latest checkpoint file
    """
    list_of_files = glob.glob("{}/*.ckpt".format(ckpt_dir))
    latest_checkpoint = None
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getctime)
    return latest_checkpoint


def load_checkpoint(path: str, use_cuda: bool = True) -> dict:
    """
    Load model from saved checkpoint.

    :param path: path to checkpoint
    :param use_cuda: using cuda or not
    :return: checkpoint (dict)
    """
    assert os.path.isfile(path), "Checkpoint %s not found" % path
    checkpoint = torch.load(path, map_location='cuda' if use_cuda else 'cpu')
    return checkpoint


product = lambda iterable : 1 if len(iterable) == 0 else product(iterable[:-1]) * iterable[-1]

# from onmt
def tile(x: Tensor, count: int, dim=0) -> Tensor:
    """
    Tiles x on dimension dim count times. From OpenNMT. Used for beam search.

    :param x: tensor to tile
    :param count: number of tiles
    :param dim: dimension along which the tensor is tiled
    :return: tiled tensor
    """
    if isinstance(x, tuple):
        h, c = x
        return tile(h, count, dim=dim), tile(c, count, dim=dim)

    perm = list(range(len(x.size())))
    if dim != 0:
        perm[0], perm[dim] = perm[dim], perm[0]
        x = x.permute(perm).contiguous()
    out_size = list(x.size())
    out_size[0] *= count
    batch = x.size(0)
    x = x.view(batch, -1) \
        .transpose(0, 1) \
        .repeat(count, 1) \
        .transpose(0, 1) \
        .contiguous() \
        .view(*out_size)
    if dim != 0:
        x = x.permute(perm).contiguous()
    return x


def freeze_params(module: nn.Module) -> None:
    """
    Freeze the parameters of this module,
    i.e. do not update them during training

    :param module: freeze parameters of this module
    """
    for _, p in module.named_parameters():
        p.requires_grad = False


def symlink_update(target, link_name):
    try:
        os.symlink(target, link_name)
    except FileExistsError as e:
        if e.errno == errno.EEXIST:
            os.remove(link_name)
            os.symlink(target, link_name)
        else:
            raise e


class Timer(object):
    def __init__(self, *args, **kwargs):
        super(object, self).__init__(*args, **kwargs)

    @contextmanager
    def __call__(self, activity, x=True):
        t = time.time()
        if x: yield
        else: yield None
        dt = time.time()-t
        print(f"Time spent on {str(activity)}: {str(dt)}")

def split_tensor_on_pads(tensor, pad_val):
    # ["cafe", "central", "<PAD>", "distance", "<PAD>", "<PAD>"]
    # -> [["cafe", "central"], ["distance"]]
    # but used with voc idx ints instead of strings
    r = [[]]

    for i, elem in enumerate(tensor):
        if elem != pad_val:
            r[-1].append(elem.unsqueeze(0)) # continue adding elems to list
        else: 
            # pad value reached
            r[-1] = torch.cat(r[-1], dim=0) # finish up previous list by concatenating tensors

            # stop if only <PAD> values are leftover now (at end of list)
            if (tensor[i:] == pad_val).all():
                break

            r.append([]) # make new list to hold coming elems after this PAD value
    return r

