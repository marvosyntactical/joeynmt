import os
from collections import defaultdict
import json
from typing import List
from split_normalized_scenarios import CANON_JOIN_CHAR, REL_JOIN_CHAR


# hardcoded lookup
kbval_lkp = {
    "DUMMYREL" : "DUMMYREL",
    "distance" : "poi_distance",
    "traffic_info": "traffic_info",
    "poi_type": "poi_type",
    "address": "poi_address",
    "room": "room",
    "time": "time",
    "agenda": "agenda",
    "date": "date",
    "party":"party",
    "weather":"weather_attribute",
    "temperature_low":"temperature",
    "temperature_high":"temperature",
    "location":"location", # SUBJ weather
    "poi":"poi_name", # SUBJ traffic
    "event":"event" # SUBJ calendar
}

kbval_lkp = defaultdict(None, kbval_lkp)

def find_relation(line:str, d, canon_join_char=CANON_JOIN_CHAR, rel_join_char=REL_JOIN_CHAR):
    splitrelation = line.split(canon_join_char)[-1].split(rel_join_char)
    relation = splitrelation[-1]
    # hardcoded because im smart enough to use the same delimiter as appears in traffic_info, poi_type etc
    if relation in ["info", "type", "low", "high"]:
        relation = splitrelation[-2]+"_"+relation
    replacement_raw = d[relation]
    return relation

def replace_lines(lines:List[str], d: defaultdict = kbval_lkp, fine_grained = False):
    #hardcoded; relies on kb entries being represented with underscores
    #with end of string being in kbval_lkp.keys(),e.g. 'meeting_agenda'

    replacements = []
    for i, line in enumerate(lines):
        relation = find_relation(line, d, CANON_JOIN_CHAR)

        if fine_grained == True:
            # this assumes key representation is subject, run normalize_kbs accordingly
            subj = line.split(CANON_JOIN_CHAR)[0]
            # add subj before relation as described in Eric et al. 2017 details
            relation = subj+REL_JOIN_CHAR+relation

        assert relation, f"line {line} with relation {relation} not found in dictionary {d}"
        replacement_refined = "@"+relation
        replacements += [replacement_refined]

    return replacements

def main(args):
    EXT = "FINAL"
    if args[0] == 0:
        # default f:
        f = "../kvr/dev.kbv" + EXT
    elif type(args[0])==type(""):
        f = f"../kvr/{args[0]}.kbv"
        if len(args)>1:
            EXT = args[1]
        f += EXT
    else:
        raise ValueError(f"this shouldnt ever happen...: {args[0]}")

    fine_grained = False # TODO add to args

    f_stump = ".".join(f.split(".")[:-1])+"."
    cluster_ext = "kbc"+EXT
    fp = f_stump + cluster_ext

    with open(f, "r") as fine:
        ll = fine.readlines()
    ll = [l[:-1] if l.endswith("\n") else l for l in ll] #remove newlines at end of lines
    ll = [l for l in ll if l] # remove empty lines

    #code

    processed = replace_lines(ll, fine_grained=fine_grained)
    processed = [line+"\n" for line in processed]
    print(processed[:50])
    with open(fp, "w") as coarse:
        coarse.writelines(processed)


    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:sys.argv.append(0)
    sys.exit(main(sys.argv[1:]))
