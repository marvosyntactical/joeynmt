import os
import sys
import shutil
from typing import List, Dict, Union, Tuple
import json
from collections import defaultdict, OrderedDict
import numpy as np

from canonize import *

SEP_CHAR = "@DOT"
JOIN_SUBJ_CHAR = "_"
CANON_START_CHAR = "@"


def med(a, b):
    # calc minimum edit distance between strings a, b
    m, n = len(a), len(b)
    distances = np.full((m,n), float("inf"))

    distance = 0
    return distance


def highest_med_match(query, keys):
    query = query.lower()
    keys = [key.lower() for key in keys]
    scores = [med(query,key) for key in keys]

    # best is argmin of scores

    # sorts ascending
    sort_by_scores = sorted(list(zip(scores,keys)), key=lambda score_and_key: score_and_key[0])

    best = sort_by_scores[0][1]
    return best


def find_subjs_in_seq(
    raw_seq: List[str],
    can_seq: List[str],
    indices: List[int],
    subjs: dict,
    ):
    ### START THIS HELPER FUNCTION FROM HERE ??

    entity_indices_local = set() 
    rels_vals = OrderedDict() # {"poi_name": {5: ["dish", "parking"], i: ["surface", "entity"]}, "poi_type": ...}
    prev_y = None
    # x == raw indices; y == mapped indices
    for x, y in enumerate(indices):
        raw = raw_seq[x]
        map_result = can_seq[y]
        if map_result != raw:
            entity_indices_local.add(y)
            if map_result not in rels_vals.keys():
                rels_vals[map_result] = dict()
                rels_vals[map_result][y] = [raw]
            else: # entry already exists 
                if y == prev_y: # multi word expression, e.g. 'the' '11th' => '@date'
                    rels_vals[map_result][y] += [raw]
                else: # multiple occurrences of e.g. @date 
                    rels_vals[map_result] = dict()
                    rels_vals[map_result][y] = [raw]
        prev_y = y
    
    # start heuristic: go through these entities and find subjects for them
    
    domains = dict() # contains for each entity the domain if its subject or None
    entity_indices_local = sorted(list(entity_indices_local))
    for entity_idx in entity_indices_local:
        entity = can_seq[entity_idx]
        domain = subjs[entity]
        domains[entity_idx] = domain

    domains_vals = set(domains.values())
    subj_indices_local = [ent for ent,val in domains.items() if val is not None]

    return entity_indices_local, domains, domains_vals, rels_vals, subj_indices_local





def intermediate( 
            raw_seqs: List[str], 
            can_seqs: List[str], 
            indices: List[int], 
            matches: List[Tuple[str,List[str]]], 
            seq_idx: int, 
            subjs: dict, 
            ents: dict,
            sep_char: str = SEP_CHAR,
            join_char: str = JOIN_SUBJ_CHAR,
            canon_start_char: str = CANON_START_CHAR,
            trg: bool = True,
            ): 
    """
    For a given attribute in a canonized sequence, finds the corresponding subject for that attribute.

    Algo:
    1. check which domain we are in using subjs on all matches => check we dont get multiple domain options
    2. Procedure depends on domain:
    * traffic: look up POI in ents dict => EZ
    * weather:
    * calendar: probably only got one contender most of the time anyways

    :param raw_seqs: last src and last target sequence of given batch (src is concatenation of dialogue history, so last src + trg contain everything) (list of strings)
    :param can_seqs: output of canonize_seq on raw_seqs
    :param indices: surjective but non injective mapping of raw tokens to canonicals
    :param matches: matches output of canonize_seq on raw_seqs
    :param seq_idx: which sequence in dialogue history we interested in?
    :param subjs: subj dict to look up which attributes are contenders for subject
    :param ents: kvret_entities_altered.json dict
    :param join_char:
    :param canon_start_char:
    :param trg: bool whether to look at seq_idx'th trg sequence or, if False, at seq_idx'th source seq of batch
    """

    del matches # TODO correct implementation of matches! dont use them until that :[]
    if not isinstance(subjs, defaultdict):
        assert type(subjs) == dict, type(subjs)
        subjs = defaultdict(lambda: None,subjs)
    for key, val in subjs.items():
        if not key.startswith(CANON_START_CHAR):
            del subjs[key]
            subjs[CANON_START_CHAR+key] = val

    # t(batch) setup
    seqs_raw_separated = [[]]
    seqs_separated_start_indices = [0]
    for i, tok in enumerate(raw_seqs):
        if tok == sep_char:
            seqs_raw_separated += [[]]
            seqs_separated_start_indices += [i+1]
        else:
            seqs_raw_separated[-1] += [tok]
    seqs_separated_start_indices += [len(raw_seqs)]

    cache_trg = trg

    subject_mapping = None # this should be set at end of while loop, otherwise no subject was found

    # procedure: look at sequences in the order seq_idx[trg], seq_idx[subj], seq_idx-1[trg],seq_idx-1[subj],...,0; then ascending afterwards
    look_at_seq = seq_idx
    while look_at_seq < int(len(raw_seqs)//2):

        # try to find subjects and handle cases
        # if definite subject found, break

        raw_seq = seqs_raw_separated[int(trg)::2][look_at_seq]
        raw_seq_start_idx = seqs_separated_start_indices[(look_at_seq*2)+int(trg)]
        raw_seq_end_idx = seqs_separated_start_indices[(look_at_seq*2)+int(trg)+1]-2

        can_seq = can_seqs[indices[raw_seq_start_idx]:indices[raw_seq_end_idx]+1]

        local_indices = [idx - raw_seq_start_idx for idx in indices[raw_seq_start_idx:raw_seq_end_idx+1]]

        # start procedure: try to find subject indices in this sequence
        entity_indices_local, domains, domains_vals, rels_vals, subj_indices_local = find_subjs_in_seq(
            raw_seq=raw_seq,
            can_seq=can_seq,
            indices=local_indices,
            subjs=subjs
        )

        # cache vars for the sequence of interest (first one)
        if trg == cache_trg and look_at_seq == seq_idx:
            can_seq_of_interest = can_seq
            entity_indices_local_of_interest = entity_indices_local
            rels_vals_of_interest = rels_vals

        # heuristic switch case
        # every case needs to set subj to the index of the found subject in the current can_seq
        # in case of success and break
    
        if domains_vals == {None}:

            # TODO confirm subjs are in proper format
            # case 0: there is 0      subjects: extend search to other sequences in batch
            input(("extend search ! No subjects found:", can_seq, raw_seqs, subjs))

            # what order to recurse to other sentences in? probably backward, then forward
            # TODO this method of looking to other sequences in batch as backup is only better if

            # time_f(all_seq) 
            #                    >
            # time_f(curr_seq) + p(no match | trg_seq) * time_f(prev_seq) * p(match|prev_seq) + p(no match | trg_seq) * time_f(prev_seq) * p (no match | prev_seq) * time_f (prevprev_seq) .....
            # depends on constant overhead i think?
            #
            # (heuristic procedure cases 2,3 are greedy in that they assume
            # the correct subject is likely to be in this sentence, and return it
            # instead of extending search to other sentences)

            pass

        elif len(domains_vals) > 2:
            # case 1: there is multiple domains: assert False, whats this
            assert False, ("subjects of different domains found:", domains, can_seq, raw_seq)

        elif len(subj_indices_local) == 1:
            # case 2: there is 1      subjects: take it for all attributes and break
            subject_mapping = {ent: subj_indices_local[0] for ent in entity_indices_local}

            print(f"found exactly one subject {can_seq[subj_indices_local[0]]} for sequence ", can_seq, raw_seq)

            # unit test
            subj_canon = can_seq[subj_indices_local[0]] 
            assert len(rels_vals[subj_canon]) == 1, f"more than one originator for {subj_canon} found in {rels_vals[subj_canon]}" 

            break # found subj; have set it and can stop searching

        else:
            assert len(subj_indices_local) >= 1, domains
            print(f"found multiple subject contenders")
            # case 3: there is more   subjects: heuristics:
            #                   traffic: match POI attributes based on entities dict # what about distance, traffic info
            #                   event:   assert False, when does this ever happen?        
            #                   weather: print out a bunch and figure out something based on collocation

            domain = list({v for k,v in domains.items() if v is not None})[0]

            if domain == "calendar":
                assert False, f"found multiple events: {[can_seq[subj] for subj in subj_indices_local]} in {can_seq}"
            elif domain == "weather":

                # TODO run some kind of dependency parse to match attributes with subjects
                input((can_seq, can_seq_of_interest))
            else:
                assert domain == "traffic"

                # traffic attributes: poi, address, poi_type, distance, traffic_info
                # can lookup address
                # simply annotate distance, traffic info ? how long is poi_list?

                # TODO move all of this before while loop
                pois = ents["poi"]

                pois_by_address = {poi_dict["address"]: {"poi": poi_dict["poi"], "type": poi_dict["type"]} for poi_dict in pois}
                poi_address_list = list(pois_by_address)

                # look up poi info for each subject

                subject_mapping = dict()
                compare_subjects = dict()

                for subj in subj_indices_local:
                    subject_mapping[subj] = subj # set subject mapping to self
                    can_subj = can_seq[subj]

                    subj_raw_list = rels_vals[can_subj][subj] # TODO should probably unit test if this is in ents.values()
                    candidate_subj = " ".join(subj_raw_list)
                    compare_subjects[subj] = candidate_subj


                # TODO do MED match with poi_name_list; could be multiple in case of home_1, home_2 etc
                # => immediately try to match with attributes

                for entity in entity_indices_local_of_interest:
                    can_ent = can_seq[entity]
                    if "address" in can_ent: 
                        address_raw_list = rels_vals[can_ent][entity]
                        address = " ".join(address_raw_list)
                        proper_address_idx = highest_med_match(address, poi_address_list)
                        subj = pois_by_address[poi_address_list[proper_address_idx]]
                        subject_mapping[entity] = subj # hooray
                    
                        
                        
            # first do descending from seq of interest; when hit 0 go back
            direction = -1 if look_at_seq <= seq_idx and not (cache_trg == False and trg == True) else +1
            increment = not trg if direction == -1 else trg
            if increment:
                look_at_seq += direction # first from src sequence to prev sequence, then afterwards
                if look_at_seq < 0: # hit rock bottom; now continue with entries afterward
                    if cache_trg: # started with target sequence, immediately go up one
                        look_at_seq = seq_idx + 1
                    else: # started with source sequence; "go up" by just switching trg below
                        look_at_seq = seq_idx

            # invert trg (alternate between looking at src and trg)
            trg = not trg
        

    # TODO FIXME at end of while loop, 
    # subj should be the index of the subject in the sequence of interest
    # and can_seq, rels_vals, etc should be set by the last processed sequence that also returned subj

    assert subject_mapping is not None, (can_seqs, can_seq_of_interest)

    subject_prefixes = dict()
    for ent, subj in subject_mapping.items():
        subj_canon = can_seq[subj] 
        subj_raw_list = rels_vals[subj_canon][subj] # TODO should probably unit test if this is in ents.values()
        at_subj_raw_joined_ = CANON_START_CHAR + join_char.join(subj_raw_list) + join_char # @dish_parking_
        subject_prefixes[ent] = at_subj_raw_joined_ 

    intermediate_entities = {entity_idx_local: subject_prefixes[entity_idx_local] + can_seq_of_interest[entity_idx_local][1:]\
            for entity_idx_local in entity_indices_local_of_interest}

    intermediate_canonized = [can if i not in entity_indices_local else intermediate_entities[i] for i, can in enumerate(can_seq_of_interest)]

    input(("canonized ",can_seq_of_interest, " to ", intermediate_canonized))

    return intermediate_canonized







def main(args):

    # args must have signature: 0 or splitpart [, EXT, src, trg]

    # take uncanonized source and target file and canonize to @meeting_time level
    # return matches and build classifier that chooses which of the subjects most likely belongs
    # given field:
    # start out with contexts with just 1 contender, go from there 

    directory = "../kvr/"

    filestub = "train"
    lkp_ext = ".lkp"
    EXT = "FINAL"
    source_ext = ".usr"
    target_ext = ".car"

    if args==0: #use defaults
        pass
    else:
        i = 0
        vars_ = ["filestub", "lkp_ext", "EXT", "source_ext", "target_ext"]
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

    tok_X = [tok_fun(seq) for seq in raw_X]
    tok_Y = [tok_fun(seq) for seq in raw_Y]

    entities = load_json() # entity path is default arg
    efficient_entities = preprocess_entity_dict(entities, lower=lower, tok_fun=tok_fun)

    subjs_path = directory+"kvret_subjects.json"
    subjects = load_json(fp=subjs_path)

    lkp_path = directory+filestub+lkp_ext

    with open(lkp_path, "r") as lkp:
        batches = lkp.readlines() # line numbers belonging to same batch have same num entry here

    batches = [l[:-1] for l in batches if l[:-1]]
    # make every entry point to last entry with same num
    lkp_last = []
    for i, lkp in enumerate(batches):
        last_same_batch = lkp
        j = 0
        while True:
            try:
                if batches[i+j] == lkp:
                    j += 1
                else:
                    break
            except IndexError as e:
                break
        lkp_last += [i+j-1]

    lkp_batch_idx = [0] * len(lkp_last)
    i = 0
    while i < len(lkp_last):
        curr = lkp_last[i]
        j = 0
        while True:
            try:
                if lkp_last[i+j] == curr:
                    j+= 1
                else:
                    break
            except IndexError as e:
                break
        lkp_batch_idx[i:i+j+1] = list(range(j)) 
        i += j



    # FIXME TODO XXX
    # use lkp info to look in entire batch context
    # use kvret_entities_altered.json to bootstrap with traffic stuff
    # return list with entries in signature format

    # raw_seqs: List[str], 
    # can_seqs: List[str], 
    # indices: List[int], 
    # matches: List[Tuple[str,List[str]]], 
    # seq_idx: int, 
    # subjs: dict, 
    # ents: dict,
    # sep_char: SEP_CHAR

    # at each index in this list is the corresponding src & trg last for that batch


    src_tgt_last = [(tok_X[lkp_last[i]],tok_Y[lkp_last[i]]) for i in range(len(tok_X))]

    if lower:
        sep_char = SEP_CHAR.lower()
    else:
        sep_char = SEP_CHAR

    src_tgt_last_join = [x + [sep_char] + y for x,y in src_tgt_last]

    # canonize_sequence -> canon_seqs, indices, matches
    context = [(raw_seqs,) + canonize_sequence(raw_seqs, efficient_entities) + (lkp_batch_idx[i],) for i, raw_seqs in enumerate(src_tgt_last_join)]
    subj_indices = [intermediate(*tup, subjs=subjects, ents=entities, sep_char=sep_char) for tup in context]
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

