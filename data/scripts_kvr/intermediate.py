import os
import sys
import shutil
from typing import List, Dict, Union, Tuple
import json
from collections import defaultdict

from canonize import *


def select_subject(src_seq, indices, can_seq, matches, attr_idx):
    pass




def main(args):

    # args must have signature: 0 or splitpart [, EXT, src, trg]

    # take uncanonized source and target file and canonize to @meeting_time level
    # return matches and build classifier that chooses which of the subjects most likely belongs
    # given field:
    # start out with contexts with just 1 contender, go from there 

    directory = "../kvr/"
    if args==0: #use defaults
        filestub = "train"
        EXT = "FINAL"
        target_ext = ".car"
        source_ext = ".usr"
    else:
        i = 0
        vars_ = ["filestub", "EXT", "source_ext", "target_ext"]
        while True:
            try:
                # extremely unsafe, evaling from stdin; will probably kill me
                eval("vars_[i] = args.pop()")
            except IndexError:
                break
            i += 1

    lower = True
    tok_fun = pkt_tokenize

    with open(directory+filestub+source_ext, "r") as source:
        X = source.readlines()
    with open(directory+filestub+target_ext, "r") as target:
        Y = target.readlines()

    raw_X = [seq[:-1] for seq in X] # remove "\n" i think?
    raw_Y = [seq[:-1] for seq in Y] # remove "\n" i think?

    if lower:
        raw_X = [seq.lower() for seq in X]
        raw_Y = [seq.lower() for seq in Y]

    tok_X = [tok_fun(seq) for seq in X]
    tok_Y = [tok_fun(seq) for seq in Y]


    entities = load_json() # entity path is default arg
    efficient_entities = preprocess_entity_dict(entities, lower=lower, tok_fun=tok_fun)

    subjs_path = "../kvr/kvret_subjects.json"
    subjects = load_json(fp=subjs_path)

    canonized_seqs = canonize_sequences(tok_X, efficient_entities)

    assert False, canonized_seqs[:3]




    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

