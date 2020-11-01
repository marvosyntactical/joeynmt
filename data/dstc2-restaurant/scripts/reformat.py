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
    EXT = ""
    file_template = "dialog-babi-task6-dstc2-{}"
    if args == 0:
        pass
    else:
        splitpart = args[0]
        if len(args) >= 2:
            EXT = args[1]

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
    for convo in convos:
        if not convo: continue # empty conversation
        # historical sources: add lines appropriately
        history = ""
        srcs, trgs = [], []
        lines_after_api_call = False
        for i, line in enumerate(convo):
            # discard/skip first line
            if line.startswith("1 "):
                assert i == 0, (i, line, convo)
                continue
            else:
                no_num = line[line.find(" "):]

                split_line_no_num = no_num.split(SRC_TRG_DELIM)
                if len(split_line_no_num) == 2:
                    """
                    if API_CALL in split_line_no_num[1]:
                        # update history by source and add this to srcs
                        src = split_line_no_num[0]
                        history += src+" "+HISTORY_SEP_TOKEN+" "
                        srcs += [history[:-len(HISTORY_SEP_TOKEN)-2]] # without latest sep token
                        lines_after_api_call = True
                    elif SILENCE in split_line_no_num[0]:
                        # update history by trg and add this to trgs
                        trg = split_line_no_num[1]
                        history += trg+" "+HISTORY_SEP_TOKEN+" "
                        trgs += [trg]
                    else:
                    """
                    # default case;
                    # update history by src, trg and add them to srcs, trgs
                    src, trg = split_line_no_num
                    history += src+" "+HISTORY_SEP_TOKEN+" "
                    srcs += [history[:-len(HISTORY_SEP_TOKEN)-2]] # without latest sep token
                    history += trg+" "+HISTORY_SEP_TOKEN+" "
                    trgs += [trg] # without latest sep token
                else:
                    # this a Knowledgebase line
                    # assert lines_after_api_call
                    pass
        lines_after_api_call = False
        batches += [list(zip(srcs,trgs))]

    # NOTE user and system parts below ignore batches: just use one global knowledgebase

    user_parts = [ex[0]+"\n" for b in batches for ex in b]
    system_parts = [ex[1]+"\n" for b in batches for ex in b]

    # assert False, user_parts[:10]

    # write lines to files 
    user, system = ".user",".system"
    filename_usr = path+file_stem+user+EXT
    with open(filename_usr, "w") as usr_out_file:
        usr_out_file.writelines(user_parts)

    filename_sys = path+file_stem+system+EXT
    with open(filename_sys, "w") as sys_out_file:
        sys_out_file.writelines(system_parts)

if __name__ ==  "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else:
        print("Preproc script got no args.")
        main(0)



