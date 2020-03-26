import os
import sys
import shutil

try:
    from joeynmt.constants import PAD_TOKEN
except ImportError:
    PAD_TOKEN = "<pad>"
    print(f"Importing of joeynmt failed in preproc directory \
        {'/'.join(__file__.split('/')[:-1])}, falling back to \
        PAD_TOKEN=={PAD_TOKEN}")

CANON_JOIN_CHAR = "_"

"""
for use without pretrained embeddings:
    Args:
        specify kb file, e.g. 'dev.kb'
    The values (3rd items in dev.kb) are then mapped to their
    corresponding canonical representation (eric et al sec 2.3) 
    i.e. (subj, rel, val) => (subj, rel, subj-rel)
    e.g. (soccer, party, 1FCK) => (soccer, party, soccer-party)

    and produce copy of kb file with values like so

"""

def canonify(kvr_triple):
    """
    turn kb triples subj rel val into

    "pizza joint mario <pad> traffic info",
    "pizza_joint_mario_traffic_info",
    "heavy traffic"
    """

    subj, rel, val = kvr_triple.split("::")
    key_rep = f" {PAD_TOKEN} ".join((subj, rel)).replace("_"," ")
    canon_val = key_rep.replace(f"{PAD_TOKEN} ", "").replace(" ", CANON_JOIN_CHAR)

    return key_rep,canon_val, val

def main(args):

    directory = "../kvr/"
    voc_dir = "../voc/"
    if args==0: #use defaults
        filename = "dev.kb"
        trg_voc_file = "train.en.w2v.40k.map.voc"
    else:
        filename = args[0]
        if len(args) > 1:
            voc_file = args[1]
        else:
            trg_voc_file = "train.en.w2v.40k.map.voc"

    with open(directory+filename, "r") as kb:
        knowledgebase = kb.readlines()

    keys, canons, vals = [], [], []

    for triple in knowledgebase:

        key_rep, canon_val,val = canonify(triple)

        keys.append(key_rep+"\n")
        canons.append(canon_val+"\n")
        vals.append(val)

    # --- add canonical values to vocab
    trg_voc_loc = voc_dir+trg_voc_file
    kb_voc_ext = "kbvoc"
    new_kb_voc_loc = ".".join(trg_voc_loc.split(".")[:-1]+[kb_voc_ext])

    with open(trg_voc_loc, "r") as V:
        trg_vocab = V.readlines()

    new_vocab = trg_vocab+canons

    with open(new_kb_voc_loc, "w") as newV:
        newV.writelines(new_vocab)

    kb_src_ext, kb_trg_ext,kb_proper_val_ext = "kbk","kbv", "trv"
    old = ".".join(filename.split(".")[:-1])
    kb_src, kb_trg, kb_proper_val = old+"."+kb_src_ext, old+"."+kb_trg_ext, old+"."+kb_proper_val_ext

    with open(directory+kb_src, "w") as out:
        out.writelines(keys)
    with open(directory+kb_trg, "w") as out:
        out.writelines(canons)
    with open(directory+kb_proper_val, "w") as out:
        out.writelines(vals)

    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

