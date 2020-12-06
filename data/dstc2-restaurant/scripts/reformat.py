#
#
# Bring dstc2 into the format of KVRET:
# 1. discard first greeting
# 2. discard for now lines after api_call until <SILENCE>
# 3. source trg pairs are lines separate by \t tab
# 
#
# example dialogue:
"""
1 <SILENCE>	Hello , welcome to the Cambridge restaurant system . You can ask for restaurants by area , price range or food type . How may I help you ?
2 i want a moderately priced restaurant in the west part of town	api_call R_cuisine west moderate
3 saint_johns_chop_house R_post_code saint_johns_chop_house_post_code
4 saint_johns_chop_house R_cuisine british
5 saint_johns_chop_house R_location west
6 saint_johns_chop_house R_phone saint_johns_chop_house_phone
7 saint_johns_chop_house R_address saint_johns_chop_house_address
8 saint_johns_chop_house R_price moderate
9 saint_johns_chop_house R_rating 3
10 prezzo R_post_code prezzo_post_code
11 prezzo R_cuisine italian
12 prezzo R_location west
13 prezzo R_phone prezzo_phone
14 prezzo R_address prezzo_address
15 prezzo R_price moderate
16 prezzo R_rating 9
17 <SILENCE>	prezzo is a nice restaurant in the west of town in the moderate price range
18 is there anything else	You are looking for a restaurant is that right?
19 give me a different restaurant	prezzo is a nice restaurant in the west of town in the moderate price range
20 goodbye	 you are welcome

"""

SRC_TRG_DELIM = "\t"
API_CALL = "api_call"
SILENCE = "<SILENCE>"
HISTORY_SEP_TOKEN = "@dot"


def main(args):
    path = "../"
    splitpart = "dev"
    txt = ".txt"
    EXT = "RESULT"
    file_template = "dialog-babi-task6-dstc2-{}"
    if args == 0:
        pass
    else:
        splitpart = args[0]
        if len(args) >= 2:
            EXT = args[1]

    if EXT == "RESULT":
        create_local_kbs = True
        try:
            from  joeynmt.constants import PAD_TOKEN
        except ImportError:
            PAD_TOKEN = "<pad>"
            print(f"Importing of joeynmt failed in preproc path \
                {'/'.join(__file__.split('/')[:-1])}, falling back to \
                PAD_TOKEN=={PAD_TOKEN}")

        KB_DELIMITER = " "

        from split_normalized_scenarios import canonify

        # READ main global knowledgebase to init 
        # knowledgebases_keys, knowledgebases_vals

        kb_path = "../"
        kb_file_identifier = "kb"
        kb_txt = ".txt"
        kb_filename_template = "dialog-babi-task6-dstc2-{}"
        kb_file_stem = kb_filename_template.format(kb_file_identifier)
        kb_file = kb_file_stem + txt

        with open(kb_path+kb_file, "r") as kb:
            _knowledgebase = kb.readlines()

        # init for below
        knowledgebases_keys, knowledgebases_vals = [], []

        for line in _knowledgebase:
            triple = line[2:-1]
            if not triple:
                continue # skip empty line
            key_rep, val = canonify(triple)

            knowledgebases_keys.append(key_rep)
            knowledgebases_vals.append(val)
            assert len(knowledgebases_keys) == len(knowledgebases_vals), \
                    (len(knowledgebases_keys) == len(knowledgebases_vals))
            lens = [len(knowledgebases_keys)]

    else:
        create_local_kbs = False

    assert splitpart in ["trn","dev","tst"], splitpart

    file_stem = file_template.format(splitpart)
    filename = file_stem+txt

    # read in file
    with open(path+filename, "r") as file_contents:
        dialogs = file_contents.readlines()

    # remove newlines
    dialogs = [d[:-1] for d in dialogs]

    # split dialogs into lists based on newlines
    convos = [[]]
    for line in dialogs:
        if not line:
            convos += [[]]
        else:
            convos[-1] += [line]

    # assert False, (len(convos), convos[0])

    # process convos into dialog batch of srcs, trgs
    batches = []
    if create_local_kbs:
        lkp = [] # at index of utterance, contains corresponding knowledgebase index
        _empty_count = 0

    # loop over convos

    for kb_idx, convo in enumerate(convos):
        if not convo:
            print("empty? : ", convo)
            continue # empty conversation
        # historical sources: add lines appropriately
        history = ""
        srcs, trgs = [], []
        if create_local_kbs:
            keys, vals = [], []
        for i, line in enumerate(convo):
            # discard/skip first line
            if line.startswith("1 "):
                assert i == 0, (i, line, convo)
                continue
            else:
                no_num = line[line.find(" "):]

                split_line_no_num = no_num.split(SRC_TRG_DELIM)
                if len(split_line_no_num) == 2 or (len(split_line_no_num)==1 and API_CALL in split_line_no_num[0]):
                    """
                    if API_CALL in split_line_no_num[1]:
                        # update history by source and add this to srcs
                        src = split_line_no_num[0]
                        history += src+" "+HISTORY_SEP_TOKEN+" "
                        srcs += [history[:-len(HISTORY_SEP_TOKEN)-2]] # without latest sep token
                    elif SILENCE in split_line_no_num[0]:
                        # update history by trg and add this to trgs
                        trg = split_line_no_num[1]
                        history += trg+" "+HISTORY_SEP_TOKEN+" "
                        trgs += [trg]
                    else:
                    """
                    # default case;
                    # update history by src, trg and add them to srcs, trgs
                    if len(split_line_no_num) == 2:
                        src, trg = split_line_no_num
                    else:
                        assert len(split_line_no_num) == 1, split_line_no_num
                        src = ""
                        trg = split_line_no_num[0]
                    history += src+" "+HISTORY_SEP_TOKEN+" "
                    srcs += [history[:-len(HISTORY_SEP_TOKEN)-2]] # without latest sep token
                    history += trg+" "+HISTORY_SEP_TOKEN+" "
                    trgs += [trg] # without latest sep token
                    if create_local_kbs:
                        lkp += [kb_idx+1]
                else:
                    # split_line_no_num is a list containing a KB line string
                    subj, rel, val = split_line_no_num[0][1:].split(KB_DELIMITER)
                    key_rep = f" {PAD_TOKEN} ".join((subj, rel))
                    keys += [key_rep]
                    vals += [val]
        # input(f"extending batch list for {splitpart} by \n\n {list(zip(srcs,trgs))}")
        # input(f"also extending the knowledgebase list by this corresponding KB:\n\n {list(zip(keys,vals))}")
        batches += [list(zip(srcs,trgs))]

        if create_local_kbs:

            assert len(keys) == len(vals), (len(keys), len(vals))
            if not len(keys):
                # correct lkp entries for this convo to point to lkp[0]
                # which is big global KB <3
                lkp[-len(convo):] = [0]*len(convo) # the src/trg examples in this KBless batch should point to global KB
                lens += [0]
            else:
                lens += [len(keys)]
                knowledgebases_keys += keys
                knowledgebases_vals += vals

    if create_local_kbs:
        # number of KB length entries + number of skipped KBs should be == #batches +1(+1 for global KB)
        assert len(lens) == len(batches)+1, (len(lens), len(batches))
        assert lkp[-1] == len(batches)
        assert len(knowledgebases_keys) == len(knowledgebases_vals), \
                (len(knowledgebases_keys),len(knowledgebases_vals))


    # NOTE user and system parts below ignore batches: just use one global knowledgebase

    user_parts = [ex[0]+"\n" for b in batches for ex in b]
    system_parts = [ex[1]+"\n" for b in batches for ex in b]

    # write lines to files 
    user, system = ".user",".system"
    filename_usr = path+file_stem+user+EXT
    with open(filename_usr, "w") as usr_out_file:
        usr_out_file.writelines(user_parts)

    filename_sys = path+file_stem+system+EXT
    with open(filename_sys, "w") as sys_out_file:
        sys_out_file.writelines(system_parts)

    if create_local_kbs:
        kb_src_ext, kb_trg_ext, = ["kbk","kbv"]

        # KB src, trg
        kb_src_full_path = path+file_stem+"."+kb_src_ext+EXT
        print("writing to ",kb_src_full_path)
        with open(kb_src_full_path, "w") as kbsrc:
            kbsrc.writelines([k+"\n" for k in knowledgebases_keys])

        kb_trg_full_path = path+file_stem+"."+kb_trg_ext+EXT
        print("writing to ",kb_trg_full_path)
        with open(kb_trg_full_path, "w") as kbtrg:
            kbtrg.writelines([v+"\n" for v in knowledgebases_vals])

        # lkp, length info
        with open(path+file_stem+".lkp"+EXT, "w") as lkp_file:
            lkp_file.writelines([str(KB_IDX)+"\n" for KB_IDX in lkp])
        with open(path+file_stem+".len"+EXT, "w") as len_file:
            len_file.writelines([str(KB_LEN)+"\n" for KB_LEN in lens])


if __name__ ==  "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else:
        print("Preproc script got no args.")
        main(0)



