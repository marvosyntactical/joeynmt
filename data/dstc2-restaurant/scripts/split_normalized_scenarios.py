import os
import sys
import shutil

try:
    from  joeynmt.constants import PAD_TOKEN
except ImportError:
    PAD_TOKEN = "<pad>"
    print(f"Importing of joeynmt failed in preproc path \
        {'/'.join(__file__.split('/')[:-1])}, falling back to \
        PAD_TOKEN=={PAD_TOKEN}")

KB_DELIMITER = " "

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
    turn kb triples of subj rel val separated by KB_DELIMITER, e.g.:
    "frankie_and_bennys R_price expensive"

    into:

    "frankie_and_bennys <pad> R_price",
    "expensive"
    """

    subj, rel, val = kvr_triple.split(KB_DELIMITER)
    key_rep = f" {PAD_TOKEN} ".join((subj, rel))

    return key_rep, val

def main(args):

    path = "../"
    kb_file_identifier = "kb"
    txt = ".txt"
    kb_filename_template = "dialog-babi-task6-dstc2-{}"
    kb_file_stem = kb_filename_template.format(kb_file_identifier)
    kb_file = kb_file_stem + txt


    with open(path+kb_file, "r") as kb:
        knowledgebase = kb.readlines()

    keys, vals = [], []

    for line in knowledgebase:
        triple = line[2:-1]
        if not triple: continue # skip empty line
        key_rep, val = canonify(triple)

        keys.append(key_rep+"\n")
        vals.append(val+"\n")


    kb_src_ext, kb_trg_ext, = ["kbk","kbv"]
    kb_src, kb_trg = kb_file_stem+"."+kb_src_ext, kb_file_stem+"."+kb_trg_ext

    with open(path+kb_src, "w") as out:
        out.writelines(keys)
    with open(path+kb_trg, "w") as out:
        out.writelines(vals)

    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

