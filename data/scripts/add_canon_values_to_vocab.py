import os
import sys
import shutil

"""
for use without pretrained embeddings:
    Args:
        specify kb file, e.g. 'dev.kb'
    The values (3rd items in dev.kb) are then mapped to their
    corresponding canonical representation (eric et al sec 2.3) 
    i.e. (subj, rel, val) => (subj, rel, subj-rel)
    e.g. (soccer, party, 1FCK) => (soccer, party, soccer-party)

    and produce copy of kb file with values like so

    Wonderings:


"""

def canonify(kvr_triple):
    # Latest TODO different tokenization necessary
    triple = kvr_triple.replace(" ", "-")
    subj, rel, val = triple.split("::")
    canon_val = subj+"_"+rel
    return "::".join((subj, rel, canon_val)), val



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

    canons,vals = [], []

    for triple in knowledgebase:
        canonified,val = canonify(triple)
        canons.append(canonified+"\n")
        vals.append(val)

    trg_voc_loc = voc_dir+trg_voc_file
    kb_voc_ext = "kbvoc"
    new_kb_voc_loc = ".".join(trg_voc_loc.split(".")[:-1]+[kb_voc_ext])

    with open(trg_voc_loc, "r") as V:
        trg_vocab = V.readlines()

    new_vocab = trg_vocab+vals

    with open(new_kb_voc_loc, "w") as newV:
        newV.writelines(new_vocab)






    ext = "can"
    old = ".".join(filename.split(".")[:-1])
    new = old+"."+ext

    with open(directory+new, "w") as out:
        out.writelines(canons)

    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

