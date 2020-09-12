# coding: utf-8
"""
This module holds various MT evaluation metrics.
"""

import sacrebleu
from joeynmt.data import pkt_tokenize
from typing import List


def chrf(hypotheses, references):
    """
    Character F-score from sacrebleu

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.corpus_chrf(hypotheses=hypotheses, references=references)


def bleu(hypotheses, references):
    """
    Raw corpus BLEU from sacrebleu (without tokenization)

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    return sacrebleu.raw_corpus_bleu(sys_stream=hypotheses,
                                     ref_streams=[references]).score


def token_accuracy(hypotheses, references, level="word"):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :param level: segmentation level, either "word", "bpe", or "char"
    :return:
    """
    correct_tokens = 0
    all_tokens = 0
    split_char = " " if level in ["word", "bpe"] else ""
    assert len(hypotheses) == len(references)
    for hyp, ref in zip(hypotheses, references):
        all_tokens += len(hyp)
        for h_i, r_i in zip(hyp.split(split_char), ref.split(split_char)):
            # min(len(h), len(r)) tokens considered
            if h_i == r_i:
                correct_tokens += 1
    return (correct_tokens / all_tokens)*100 if all_tokens > 0 else 0.0


def sequence_accuracy(hypotheses, references):
    """
    Compute the accuracy of hypothesis tokens: correct tokens / all tokens
    Tokens are correct if they appear in the same position in the reference.

    :param hypotheses: list of hypotheses (strings)
    :param references: list of references (strings)
    :return:
    """
    assert len(hypotheses) == len(references)
    correct_sequences = sum([1 for (hyp, ref) in zip(hypotheses, references)
                             if hyp == ref])
    return (correct_sequences / len(hypotheses))*100 if hypotheses else 0.0

def ent_f1(hyps: List[str], refs: List[str], vocab, c_fun, report_on_canonicals: bool = False, tok_fun=pkt_tokenize):
    """
    :param hyps: list of string sentences to be tokenized by tok_fun
    :param refs: list of string sentences to be tokenized by tok_fun
    :param vocab: vocab to turn tokenized strings to indices; provides vocab.stoi; vocab.is_unk
    :param c_fun: canonization function to turn a raw sequence List[str] canonical
    :param report_on_canonicals: wether to calculate on canonical entities (~cheating) or surface occurrences
      - True: cheat by comparing in @time == @time; vocab must be model.trg_vocab !
      - False: compare in 7 pm == 8 pm;  vocab must be model.trv_vocab !

    :return:
     - f1_avg: float f1 score == 2* (prec*rec) / (prec + rec) ('harmonic mean'); averaged over all examples
    """

    # requires internal knowledge of the entire universe
    # specifically of names of kb_fields, kbtrv_fields

    # FIXME create_KB_on_the_fly and pkt_tokenize 
    # are referred to separately in 
    # * helpers.py
    # * metrics.py
    # fix this by 
    #  adding them to cfg
    # or 
    #  giving it as attribute to the model 


    # define helper functions
    ## eval metrics
    def precision(predictions: List[int], gold_labels: List[int]):
        # TP/(TP+FP) => iterate over positives (predicted)
        positives = len(predictions)
        if positives:
            tp, fp = zip(*[(1,0) if pred in gold_labels else (0,1) for pred in predictions])
            return sum(tp)/positives
        else:
            return 0.

    def recall(predictions: List[int], gold_labels: List[int]):
        # TP/(TP+FN) => iterate over ground truths (gold labels)
        truths = len(gold_labels)
        if truths:
            tp, fn  = zip(*[(1,0) if gold in predictions else (0,1) for gold in gold_labels])
            return sum(tp)/truths
        else:
            return 0.

    harm_mean = lambda p, r: 2 * (p*r)/(p+r) if p+r != 0. else 0.

    # compare ent f1 in trv => lookup vocab indices

    f1s = [] # accumulate scores
    for i, (hyp,ref) in enumerate(zip(hyps, refs)):

        hyp_ents_ref_ents = [] # will hold entity vocabulary indices in the order hyp,ref
        debug = []

        for seq in (hyp,ref):
            seq_tokzd = tok_fun(seq)
            canons, indices = c_fun(seq_tokzd) # turn to canonical tokens and return indices that raw tokens were mapped to

            entities = [\
                " ".join([raw for map_idx, raw in zip(indices,seq_tokzd) if map_idx==i])
                     for i in range(len((canons)))
                         ]

            # Filter out tokens that werent changed (noncanonical)
            try:
                canonical_entities, surface_entities = list(zip(*[(c,t) for c,t in zip(canons,entities) if c!=t]))
            except ValueError:
                canonical_entities, surface_entities = [], []
            
            if report_on_canonicals: 
                entities = canonical_entities
            else:
                entities = surface_entities

            # Filter out unk IDs
            entities = [tok for tok in entities if not vocab.is_unk(tok)]
            # turn to vocab indices (int)
            seq_enty_voc_indices = [vocab.stoi[entity] for entity in entities]

            hyp_ents_ref_ents.append(seq_enty_voc_indices)
            debug.append(entities)

        assert False, (debug, hyp, ref)
        p, t = hyp_ents_ref_ents
        # calc f1 score for this pair
        f1_score = harm_mean(precision(p,t), recall(p,t))
        f1s.append(f1_score)
            
    assert len(hyps) == len(refs) == len(f1s), (len(hyps), len(refs), len(f1s))
    f1_avg = sum(f1s) / len(f1s)
    return f1_avg
