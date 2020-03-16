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
    triple = kvr_triple.replace(" ", "-")
    subj, rel, val = triple.split("::")
    canon_val = subj+"_"+rel
    return "::".join((subj, rel, canon_val))



def main(args):

    directory = "../kv_ret_dataset/"
    voc_dir = "../voc/"
    if args==0: #use defaults
        filename = "dev.kb"
        trg_voc_file = "train.en.w2v.40k.map.voc"#Latest TODO
    else:
        filename = args[0]
        if len(args) > 1:
            voc_file = args[1]
    with open(directory+filename, "r") as kb:
        knowledgebase = kb.readlines()
    
    canons = [] 

    for triple in knowledgebase:
        canon_val = canonify(triple)
        canons.append(canon_val+"\n")

    """
    trg_voc_loc = voc_dir+trg_voc_file  

    with open(trg_voc_loc, "r") as V:
        trg_vocab = V.readlines()
    
    
    #print(trg_vocab[:100])
    """
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

