import os
import sys

def canonify(kvr_triple):
    triple = kvr_triple.replace(" ", "-")
    subj, rel, val = triple.split("::")
    canon_val = subj+"_"+rel
    return subj, rel, canon_val



def main(args):

    directory = "../kv_ret_dataset/"
    voc_dir = "../voc/"
    if args==0: #use defaults
        filename = "dev.kb"
        voc_file = "train.en.w2v.40k.map.voc"#Latest TODO
    else:
        filename = args[0]
        if len(args) > 1:
            voc_file = args[1]
    with open(directory+filename, "r") as kb:
        knowledgebase = kb.readlines()
    
    canons = [] 

    for triple in knowledgebase:
        _, _, canon_val = canonify(triple)
        canons.append(canon_val)

    voc_loc = voc_dir+voc_file  

    with open(voc_loc, "r") as V:
        vocab = V.readlines()
    
    print(vocab)

    return 0


if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))

