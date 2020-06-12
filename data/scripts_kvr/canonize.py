import os
import sys
import shutil
from typing import List, Dict, Union
import json
from collections import defaultdict


# tokenization function from joeynmt data:

def pkt_tokenize(s)-> List:
    s = s+" "
    pkt = ".,?!-;:()" # NOTE candidates: '
    num = "0123456789"
    space = ["\t", "\n", " "]

    num_flag = False #current token is numerical?
    r = []
    split = list(s)
    curr = [] #running token
    for c in split:
        if c in space:#word finished after infront of next whitespace 
            num_flag = False
            token = "".join(curr)
            r += [token]
            curr = []
        else:
            if c in pkt:#word finished infront of next pkt; also add pkt as token
                num_flag = False
                token = "".join(curr)
                r += [token]
                curr = []
            elif c in num:
                num_flag = True
            else:#should now be alphabetical character or one of @'
                if num_flag==True: #if before this was a number, add number as token (4pm => 4 pm; 13th => 13 th)
                    num_flag=False
                    number = "".join(curr)
                    assert number, (s, curr)
                    r += [number]
                    curr = []
            curr += [c] # add everything but whitespace
    return r


class DefaultFactory:
    def __init__(self):
        pass
    def __call__(self):
        return None 
    def __repr__(self):
        return "n.a."

def load_entity_dict(fp="../kvr/kvret_entities_altered.json"):
    with open(fp, "r") as file:
        entities = json.load(file)
    return entities


def preprocess_entity_dict(ent_d: Dict[str,Union[List[str],List[int], List[Dict[str,str]]]]={}, lower=True, tok_fun=lambda s:s.split()) -> defaultdict:
    """
    input (ent_d) dictionary:

    processes entity dict read from entities.json into very weird dictionary:

    keys are first tokens in input dictionary value string lists;
    values are dictionaries: keys are labels, values are 2D lists; inner lists contain continuations of
    they top level key as token lists, possibly empty

    TODO clashes between @labels should be avoided, e.g. no two labels with empty inner lists
    e.g.
    "conference": {"@room": [["room", "100"],["room","102"],["room", "50"],[]]}
    "the": {"@date": [["13th"], ["15th"]], "@poi_name": [["grand", "hotel"]]}

    if input dictionary value contains list of dictionaries (e.g. poi info with multiple dictionary entries: address, poi, type), output looks like this:

    "593": {"poi", [["Arrowhead", "Way]]}
    "Chef": {"poi", [["Chu's"]]}
    "chinese": {"poi", [["restaurant"]]}

    """
    def add_entry_to_dict(dictionary:defaultdict, entry: str = "", label: str="", lowerize: bool= True, tokenizer=lambda s:s.split(), suffix:str="", default:DefaultFactory=DefaultFactory()) -> None:

        cased_entry = entry.lower() if lowerize else entry
        tokenized_entity = tokenizer(cased_entry) #list of strings

        classification = "@"+label+suffix # e.g. @poi_name or @distance
        key, continuation = tokenized_entity.pop(0), tokenized_entity

        # hardcode 'the' and 'a' duplication for select classes/labels:
        if classification == "@temperature":
            suffixes = ["f"]#a hospital should also match
            if key not in suffixes: # break recursion
                for suffix in suffixes:
                    modified_entry = entry+" "+suffix

                    # recurse with added article:
                    add_entry_to_dict(dictionary, modified_entry, label, lowerize, tokenizer, suffix, default)


        # check if first token exists as key in return dict r:
        if dictionary[key]:
            if not dictionary[key][classification]:
                dictionary[key][classification] = []
            dictionary[key][classification] += [continuation]# add new continuation to dict entry
        else: # create dict entry 
            dictionary[key] = defaultdict(default,{classification:[]})
            dictionary[key][classification] += [continuation]

    default = DefaultFactory()
    r = defaultdict(default, {})
    for label, value_list in ent_d.items():
        for entry in value_list:
            if type(entry) in (type(""), type(42)):
                add_entry_to_dict(r, str(entry), label, lower, tok_fun)
            elif type(entry) == type({}):
                for label_suffix, v in entry.items():
                    if label_suffix == "poi":
                        label_suffix = "name"
                    add_entry_to_dict(r, v, label, lower, tok_fun, "_"+label_suffix)
            else:
                raise TypeError(f"Can't handle type {type(entry)} of entry: {entry}")
    return r

def canonize_sequence(seq: List[str]=[], entities:defaultdict=defaultdict()) -> List[str]: 
    """
    Efficiently canonize with values in dict
    (search further ahead if a token matches first word in canon string)
    """
    #if only these words match, dont replace them
    stopwords = [\
                 "the", "no",#meetings, pois etc\ 
                 "0","1","2","3","4","5","6","7","8","9","10",#distances
                 "20", "30","40","50","60","70","80","90","100",#temperatures
                 "next",#weekly time
                 "four",#poi name
                ]
    r = []
    i = 0
    while i < len(seq):
        token = seq[i]
        next = entities[token]
        if next:
            j = 0
            candidates, partial_matches = [], []
            matching_continuation, partial_match_backup = False, False
            for label, continuations in next.items():
                for continuation in continuations:
                    match = True 
                    for j,further in enumerate(continuation):
                        try:
                            if further != seq[i+j+1]:
                                match = False
                                partial_matches += [([token]+continuation[:j+1], label,j+1)]
                                partial_match_backup = True
                                break
                        except IndexError:
                            match = False
                            partial_matches += [([token]+continuation, label, j+1)]
                            partial_match_backup = True
                            break
                    if match:
                        matching_continuation = True
                        candidates += [([token]+continuation, label)]

            if matching_continuation:
                #winner is simply longest candidate (in sequence with "2", "pm", match time (both tokens) instead of distance (only first token))
                winning_candidate, lbl = max(candidates, key=lambda tup: len(tup[0]))
                r += [lbl]
                i += len(winning_candidate)-1
            elif partial_match_backup:
                #winner is the candidate that matches farthest
                winning_candidate, lbl, upto = max(partial_matches, key=lambda tup: tup[-1])
                if winning_candidate[0] in stopwords:
                    print(f"Warning: Omitting possible match {winning_candidate}; only matched with first token!")
                    r += [token]
                else:
                    r += [lbl]
                    i += upto-1
            else:
                assert RuntimeError("This shouldnt happen")
            print(f"Made entity classification for seq up to here:\n\
                    {seq[:i+j+1]};\ncandidates are:\n\
                    {candidates};\npartial matches are:\n\
                    {partial_matches};\nchoice is:\n\
                    {(winning_candidate, lbl)};\nr is:\n{r}")
        else:
            r += [token] # dont replace the token, theres no match, just add the token to the result
        i += 1
    print("\n"+("="*40))
    print(f"\tFinished up Sequence\n{seq}\nand transformed it to\n{r}")
    print(("="*40)+"\n")
    return r 

def canonize_sequences(seqs: List[List[str]] = [], dictionary: defaultdict = defaultdict()):
    return [canonize_sequence(seq,dictionary) for seq in seqs]

def main(args):

    directory = "../kvr/"
    if args==0: #use defaults
        filestub = "train"
    else:
        filestub = args[0]

    target_ext = ".car"

    lower = True
    tok_fun = pkt_tokenize

    with open(directory+filestub+target_ext, "r") as target:
        gold_standard = target.readlines()
    gold_standard = [seq[:-1] for seq in gold_standard]
    if lower:
        gold_standard = [seq.lower() for seq in gold_standard]
    gold_standard = [tok_fun(seq) for seq in gold_standard]
    print(gold_standard[:5], len(gold_standard))


    entities = load_entity_dict()
    efficient_entities = preprocess_entity_dict(entities, lower=lower, tok_fun=tok_fun)

    canonized_seqs = canonize_sequences(gold_standard, efficient_entities)
    output = [" ".join(out)+"\n" for out in canonized_seqs]

    output_ext = ".carno"
    with open(directory+filestub+output_ext, "w") as out:
        out.writelines(output)


    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

