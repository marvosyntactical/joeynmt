import os
from collections import defaultdict
import json
from typing import List


def load_entitties(ent_json="../kvr/kvret_entities.json"):
    ent_dict = json.load(ent_json)
    return ent_dict


#hardcoded lookup
kbval_lkp = {
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
    "location":"location",
    "poi":"poi_name",
    "event":"event"
}

kbval_lkp = defaultdict(None, kbval_lkp)

def replace_line(line:str, d:defaultdict = kbval_lkp):
    #hardcoded; relies on kb entries being represented with underscores
    #with end of string being in kbval_lkp.keys(),e.g. 'meeting_agenda'

    splitline = line.split("_")
    ending = splitline[-1]
    #hardcoded
    if ending in ["info", "type","low","high"]:
        ending = splitline[-2]+"_"+ending

    replacement_raw = d[ending]
    assert replacement_raw, f"line {line} with ending {ending} not found in dictionary {d}"
    replacement_refined = "@"+replacement_raw
    return replacement_refined

def replace_lines(lines: List[str], d:defaultdict=kbval_lkp):
    return [replace_line(line,d) for line in lines]

def main(args):
    if args[0] == 0:
        # default f:
        f = "../kvr/dev.kbvNEW"
    elif type(args[0])==type(""):
        f = f"../kvr/{args[0]}.kbvNEW"
    else:
        raise ValueError(f"this shouldnt ever happen...: {args[0]}")

    f_stump = ".".join(f.split(".")[:-1])+"."
    cluster_ext = "kbcNEW"
    fp = f_stump + cluster_ext

    with open(f, "r") as fine:
        ll = fine.readlines()
    ll = [l[:-1] if l.endswith("\n") else l for l in ll]
    ll = [l for l in ll if l]

    #code

    processed = replace_lines(ll)
    processed = [line+"\n" for line in processed]
    print(processed[:50])
    with open(fp, "w") as coarse:
        coarse.writelines(processed)


    return 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:sys.argv.append(0)
    sys.exit(main(sys.argv[1:]))
