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
PAD_TOKEN = "<pad>"

def med(a, b):
    # calc minimum edit distance between strings a, b
    m, n = len(a), len(b)
    d = np.zeros((m,n))
    
    for i in range(m):
        d[i,0] = i
    for j in range(n):
        d[0,j] = j
    for j in range(n):
        for i in range(m):
            if a[i] == b[j]:
                substitutionCost = 0
            else:
                substitutionCost = 1
            d[i,j] = min(d[i-1,j]+1, # del
                        d[i,j-1]+1, # ins
                        d[i-1,j-1]+substitutionCost # sub
            )
    return d[m-1,n-1]


def lowest_med_match(query, keys):
    query = query.lower()
    keys = [key.lower() for key in keys]
    scores = [med(query,key) for key in keys]

    # best is argmin of scores

    # sorts ascending
    sort_by_scores = sorted(list(zip(scores,range(len(keys)))), key=lambda score_and_key: score_and_key[0])

    best_idx = sort_by_scores[0][1]
    return best_idx


def find_subjs_in_seq(
    raw_seq: List[str],
    can_seq: List[str],
    indices: List[int],
    subjs: dict,
    ):

    entity_indices_local = set() 
    rels_vals = dict() # {"poi_name": {5: ["dish", "parking"], i: ["surface", "entity"]}, "poi_type": ...}
    # x == raw indices; y == mapped indices
    for x, y in enumerate(indices):
        try:
            raw = raw_seq[x]
        except IndexError as IE:
            print(raw_seq)
            print(x,y)
            print(can_seq)
            print(indices)
            assert False, IE

        map_result = can_seq[y]
        if map_result != raw:
            entity_indices_local.add(y)
            if map_result not in rels_vals.keys():
                rels_vals[map_result] = dict()
            if y not in rels_vals[map_result].keys():
                rels_vals[map_result][y] = []
            rels_vals[map_result][y] += [raw]
    
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
            seq_idx: int, 
            subjs: dict, 
            ents: dict,
            kb: dict = None,
            sep_char: str = SEP_CHAR,
            join_char: str = JOIN_SUBJ_CHAR,
            canon_start_char: str = CANON_START_CHAR,
            trg: bool = True,
            ): 
    """
    # Input
    For a given batch sequence of tokens delimited into individual sentences by sep_char
    and a seq_idx indexing the seq_idx'th trg (or source) sentence "of interest" within them,

    # Function
    tries to change the labels of all entities in that sentence
    from @time or @poi_type to @meeting_time or @pizza_hut_poi_type

    # Search strategy
    by searching greedily from sentence to sentence for subject contenders
    in the order: descend into history first, then ascend to future:
    seq_idx => seq_idx - 1 => ... => 0 => seq_idx + 1 => seq_idx + 2 => ... => max_batch

    once a subject has been found we greedily take it and label all of our entities with it
    once multiple subjects are found we are fucked and need to use more heuristics:
    
    * look up addresses in entity dictionary
    * TODO figure out what to do with weather
    * FIXME why dont I just look at KB?


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

    global_can_seqs_bin = dict()# index this to get num of sequence or hit a None mine if u index at sep_char u noob lmao
    global_offsets = dict() # start offsets of canonical sequences
    rels_vals_per_seq = dict() # dict of seq idx : rels_vals dict for all visited sequences
    subject_mapping = dict() # this should be set at end of while loop; otherwise no subject appeared in entire batch
    subject_dict = None

    # procedure: look at sequences in the order seq_idx[trg], seq_idx[src], seq_idx-1[trg],seq_idx-1[src],...,0[src]; then ascending afterwards
    direction = -1 # start while loop in descending order, then ascend after hitting first src
    cache_trg = trg
    seq_offset = (seq_idx*2)+int(cache_trg)
    seq_offset_cache = seq_offset
    while seq_offset < len(raw_seqs):
        look_at_seq = (seq_offset//2)
        # # input((seq_idx, seq_offset, look_at_seq))

        raw_seq = seqs_raw_separated[seq_offset]
        raw_seq_start_idx = seqs_separated_start_indices[seq_offset]
        raw_seq_end_idx = seqs_separated_start_indices[seq_offset+1]-2 # leave out delimiting “@DOT” sep_char

        can_seq = can_seqs[indices[raw_seq_start_idx]:indices[raw_seq_end_idx]+1] # inklusionserhaltende abb

        local_indices = [idx - indices[raw_seq_start_idx] for idx in indices[raw_seq_start_idx:raw_seq_end_idx+1]]
        assert local_indices[0] == 0, (can_seq, indices[raw_seq_start_idx:raw_seq_end_idx+1], raw_seq_start_idx, raw_seq_end_idx)

        # input((raw_seq, can_seq))

        # start procedure: try to find subject indices in this sequence
        entity_indices_local, domains, domains_vals, rels_vals, subj_indices_local = find_subjs_in_seq(
            raw_seq=raw_seq,
            can_seq=can_seq,
            indices=local_indices,
            subjs=subjs
        )

        # cache vars for all visited sequences:
        global_offsets[seq_offset] = indices[raw_seq_start_idx]
        rels_vals_per_seq[seq_offset] = rels_vals
        for i in range(indices[raw_seq_start_idx], indices[raw_seq_end_idx+1]):
            global_can_seqs_bin[i] = seq_offset
        

        # cache vars for the sequence of interest (first one)
        if trg == cache_trg and look_at_seq == seq_idx:
            can_seq_of_interest = can_seq
            entity_indices_local_of_interest = entity_indices_local
            rels_vals_of_interest = rels_vals

            # try to look up subject mapping in KB

        # heuristic switch case
        # every case needs to set subject_mapping to dict of entity_idx: subj_idx for all entities in the sent
        # in case of success and break

        if len(domains_vals) == 0:
            # sentence contains no entities
            if seq_offset == seq_offset_cache: 
                # break if this is the sequence of interest (could also just return can_seq)
                # return can_seq
                break

    
        elif domains_vals == {None}:

            # TODO confirm subjs are in proper format
            # case 0: there is 0 subjects: extend search to other sequences in batch
            # input(("extend search ! No subjects found in (seq, then batch): ", can_seq, raw_seqs, subjs, look_at_seq, cache_trg, direction))

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
            # case 2: there is 1 subject: take it for all attributes and break
            subject_mapping.update({ent: global_offsets[seq_offset]+subj_indices_local[0] for ent in entity_indices_local_of_interest})

            print(f"found exactly one subject {rels_vals[can_seq[subj_indices_local[0]]][subj_indices_local[0]]} for sequence ", can_seq, raw_seq)

            # unit test
            subj_canon = can_seq[subj_indices_local[0]] 
            assert len(rels_vals[subj_canon]) == 1, f"more than one originator for {subj_canon} found in {rels_vals[subj_canon]}" 

            break # found subj; have set it and can stop searching

        else:
            assert len(subj_indices_local) > 1, (domains,can_seq)
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
                print(("\n"*4)+("\n"*4)+"WEATHER DOMAIN OMG WHATWEDO"+"\n"*4)
                # input((can_seq, can_seq_of_interest))
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

                compare_subjects = dict()

                for subj in subj_indices_local:
                    subject_mapping[subj] = global_offsets[seq_offset]+subj # set subject mapping to self FIXME set value to global subject
                    can_subj = can_seq[subj]

                    subj_raw_list = rels_vals[can_subj][subj] # TODO should probably unit test if this is in ents.values()
                    candidate_subj = " ".join(subj_raw_list)
                    compare_subjects[subj] = candidate_subj


                # TODO do MED match with poi_name_list; could be multiple in case of home_1, home_2 etc
                # => immediately try to match with attributes

                for entity in entity_indices_local_of_interest:
                    can_ent = can_seq[entity]
                    if "address" in can_ent: 
                        address_raw_list = rels_vals_per_seq[seq_idx][can_ent][entity]
                        address = " ".join(address_raw_list)
                        proper_address_idx = lowest_med_match(address, poi_address_list)
                        subj = pois_by_address[poi_address_list[proper_address_idx]]
                        subject_mapping[entity] = global_offsets[seq_offset]+subj # hooray
                        
        # first do descending from seq of interest; when hit 0 go back
        if seq_offset == 0: 
            seq_offset = seq_idx 
            direction *= -1 # start ascending
            if cache_trg == True: # switch one extra time if we started with target because now we goin from src to src once
                trg = not trg
        seq_offset += direction # first from src sequence to prev sequence, then afterwards if seq_offset <= 0 and not trg: # hit first source; now continue with entries afterward
        # inverttrg (alternate between looking at src and trg)
        trg = not trg


    # TODO FIXME at end of while loop, 
    # subject_mapping should be entity: subject dict with  
    # entity: index of entity in local can_seq
    # subject: index of subject in global can_seqs

    
    # (can_seq, rels_vals, etc should be set to the last processed sequence that also returned subject_mapping)

    # assert subject_mapping != {}, (can_seqs, can_seq_of_interest, global_offsets, seq_offset, global_can_seqs_bin)

    subject_prefixes = dict()

    for local_ent, global_subj in subject_mapping.items():

        # FIXME TODO get these variables
        subj_seq = global_can_seqs_bin[global_subj] # index in can_seqs NOTE probably look at seq but just figure out using sep in beginning
        if subj_seq is None: # just gonna let this slide lol
            subj_seq = global_can_seqs_bin[global_subj+1]

        subj = global_subj-global_offsets[subj_seq] # index in its local sequence


        subj_canon = can_seqs[global_subj] # poi_type
        
        subj_raw_list = rels_vals_per_seq[subj_seq][subj_canon][subj] # TODO should probably unit test if this is in ents.values()

        # input((subj_raw_list, rels_vals[subj_canon], subj, subject_mapping, can_seq))

        at_subj_raw_joined_ = CANON_START_CHAR + join_char.join(subj_raw_list) + join_char # @dish_parking_
        subject_prefixes[local_ent] = at_subj_raw_joined_ 
    
    if kb is not None:
        subject_dict = dict() # subject dict with local enitity index: ["dish", "parking"]
        for label_type in rels_vals:
            dict_for_label_type = rels_vals[label_type]
            for instance in dict_for_label_type:
                joined_instance = " ".join(dict_for_label_type[instance])

                label_without_at = label_type if not label_type.startswith("@") else label_type[1:]

                if label_without_at == "poi_name":
                    label_without_at = "poi"
                if label_without_at == "poi_address":
                    label_without_at = "address"
                if label_without_at == "poi_distance":
                    label_without_at = "distance"

                
                closest_entry_idx = lowest_med_match(joined_instance, kb.keys())
                probable_intermediate_label = list(kb.keys())[closest_entry_idx]
                probable_intermediate_label_list = kb[probable_intermediate_label]

                assert False, (label_type, probable_intermediate_label_list)
                # input(f"compare surface form {joined_instance} with probable subj {probable_subj} \
                #     \n\n and dict \n {scenario_entry_list[closest_entry_idx]}")
                subject_dict[instance] = probable_subj.lower()


        for local_ent, subj_joined in subject_dict.items():
            
            at_subj_raw_joined_ = CANON_START_CHAR + join_char.join(subj_joined.lower().split()) + join_char
            subject_prefixes[local_ent] = at_subj_raw_joined_


    intermediate_entities = dict()
    for e_i in entity_indices_local_of_interest:
        try:
            subject_prefix = subject_prefixes[e_i]
        except KeyError as KE:
            # XXX removeme
            print(subject_prefixes)
            print(entity_indices_local_of_interest)
            print(KE)
            print(e_i)
            print(can_seq)
            print(can_seq_of_interest)
            assert False, subject_prefixes[e_i]
        can_without_at = can_seq_of_interest[e_i][1:]
        intermediate_label_i = subject_prefix + can_without_at
        intermediate_entities[e_i] = intermediate_label_i

    intermediate_entities = {i: subject_prefixes[i] + can_seq_of_interest[i][1:] \
            for i in entity_indices_local_of_interest}

    intermediate_canonized = [can if i not in entity_indices_local_of_interest else intermediate_entities[i] for i, can in enumerate(can_seq_of_interest)]

    # input(("canonized ",can_seq_of_interest, " to ", intermediate_canonized))

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
    EXT = "INTERMEDIATE"
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

    kblkp = batches

    scenarios_path = directory+f"scenarios_{filestub}.json"
    scenarios = load_json(fp=scenarios_path)

    lengths_path = directory+f"{filestub}.len"
    with open(lengths_path, "r") as lens:
        lengths_list = lens.readlines()
    lengths_list = [int(l[:-1]) for l in lengths_list if l[:-1]]


    intermediate_val_path = directory+filestub+".kbc"+EXT
    trv_val_path = directory+filestub+".trv"+EXT

    with open(intermediate_val_path, "r") as vals:
        vals_list = vals.readlines()
    vals_list = [l[:-1].lower() for l in vals_list if l[:-1]]
    
    with open(trv_val_path, "r") as trvs:
        trvs_list = trvs.readlines()
    trvs_list = [l[:-1].lower() for l in trvs_list if l[:-1]]

    keys_path = directory+filestub+".kbk"+EXT
    with open(keys_path, "r") as keys:
        keys_list = keys.readlines()
    keys_list = [key[:key.find(PAD_TOKEN)-1].lower() for key in keys_list if key[:-1]]

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

    if lower:
        sep_char = SEP_CHAR.lower()
    else:
        sep_char = SEP_CHAR

    src_tgt_last = [(tok_X[lkp_last[i]],tok_Y[lkp_last[i]]) for i in range(len(tok_X))]

    src_tgt_last_join = [x + [sep_char] + y for x,y in src_tgt_last]


    class Batch_Iter:
        def __init__(self):
            pass

        def __iter__(self, sequences=src_tgt_last_join, lkp_last=lkp_last):

            i = 0
            while i < len(sequences):
                first = i
                last = lkp_last[i]
                seq_idx = lkp_batch_idx[i]
                batch = []

                while i <= last:
                    batch += sequences[i]
                    i += 1
                                
                scenario_entry_list = scenarios[int(kblkp[last])]["kb"]["items"]

                begin_scenario = sum(lengths_list[:int(kblkp[last])])
                end_scenario = begin_scenario + lengths_list[int(kblkp[last])]

                keys_scenario = keys_list[begin_scenario:end_scenario]
                trv_scenario = trvs_list[begin_scenario:end_scenario]
                val_scenario = vals_list[begin_scenario:end_scenario]


                KB = dict() # {"poi_name": {5: ["dish", "parking"], i: ["surface", "entity"]}, "poi_type": ...}
                for i, (trv, val) in enumerate(zip(trv_scenario, val_scenario)):

                    if trv not in KB.keys():
                        KB[trv] = []
                    KB[trv].append(val) 

                sum_kb_vals = sum([len(val) for val in KB.values()])
                assert len(val_scenario) == sum_kb_vals, (len(val_scenario), sum_kb_vals)


                # canonize_sequence -> canon_seqs, indices, matches
                raw_seqs = sequences[last]
                can_seqs, indices, matches = canonize_sequence(raw_seqs, efficient_entities)
                del matches
                # input(("can seqs:", can_seqs))
                # input(("indices:", indices))
                intermediately_canonized_batch = [
                    intermediate(raw_seqs, can_seqs, indices, j, \
                    subjs=subjects, ents=entities, sep_char=sep_char, kb=KB) \
                    for j in range(last-first)]

                yield intermediately_canonized_batch

                i = last+1

    batch_iter = Batch_Iter()

    debug_raw_last_seqs = []
    for raw_seqs in src_tgt_last_join:
        if not debug_raw_last_seqs or debug_raw_last_seqs[-1] != raw_seqs:
            debug_raw_last_seqs += [raw_seqs]
    
    for i, (raw_batch, processed_batch) in enumerate(zip(debug_raw_last_seqs, batch_iter)):
        input(f"processed_batch:\n\n{processed_batch}\n\n\nAll correct?\n\n\n\n{debug_raw_last_seqs[i]}\n\n\n\n Continue ?\n\n\n")
    
    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

