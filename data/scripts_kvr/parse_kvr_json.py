import json
import os
import sys


#usr_part = "\n".join([e for i,e in enumerate(elem) if i%2==0])+"\n"

global DOT_CHAR
DOT_CHAR = "@DOT"


def historify_src(utterances, even=True):
    #helper function to add the entire dialogue history as src
    parity = 0 if even else 1

    assert set([bool(utt.strip()) == True for utt in utterances]) == {True}, utterances

    usr_part = ""
    for i, e in enumerate(utterances):
        if i%2==parity:
            usr_part += f" {DOT_CHAR} ".join(utterances[:i+1])+"\n"
    return usr_part

def main(args):
    #defaults:
    directory = "../kvr/"
    splitpart = "dev" if args == 0 else args[0]
    EXT = "FINAL" if args == 0 or len(args) <= 1 else args[1]
    assert splitpart in ["dev", "train", "test"]
    filename = "kvret_dev_public.json"


    with open(directory+filename, "r") as f:
        data= json.load(f)

    settings=[]
    scenarios=[]

    for idx, setting in enumerate(data):
        continue_ = False
        dialogue = setting["dialogue"]
        scenario = setting["scenario"]
        convo = []
        lastspeaker = "assistant"
        for turn in dialogue:
            utterance = turn["data"]["utterance"]
            if not utterance.strip():
                continue_ = True # skip this dialogue; someone didnt answer
            convo.append(utterance)
            speaker = turn["turn"]
            #assert speaker != lastspeaker, utterance
            lastspeaker = speaker
        if continue_:
            continue
        scenarios.append(scenario)
        settings.append(convo)

    unanswered = ""
    scenario_lkp = ""
    convo_usr, convo_car = "", ""

    for idx, elem in enumerate(settings):

        if len(elem)%2==1:
            unanswered+=elem[-1]+"\n"
            elem = elem[:-1]

        nturns = len(elem)
        assert nturns%2==0

        usr_part = historify_src(elem)
        car_part = "\n".join(
            [e for i,e in enumerate(elem) if i % 2 == 1]
            )+"\n"
        scenario_part = (str(idx)+"\n")*(nturns//2)

        lines = lambda s: len(s.split("\n"))
        assert lines(usr_part) == lines(car_part) == lines(scenario_part), (usr_part, car_part, scenario_part)

        convo_usr += usr_part
        convo_car += car_part
        scenario_lkp += scenario_part

    train_usr, train_car = splitpart+".usr"+EXT, splitpart+".car"+EXT

    with open(directory+train_usr, "w") as usr, open(directory+train_car, "w") as car:
        usr.write(convo_usr)
        car.write(convo_car)

    # for normalize scenarios.py
    scenariofile = "scenarios_"+splitpart+"_"+EXT+".json"

    with open(directory+scenariofile, "w") as scenes:
        json.dump(scenarios, scenes, indent=4)

    # for data.py minibatch
    scenario_lkp_file = splitpart+".lkp"+EXT

    with open(directory+scenario_lkp_file, "w") as lkp:
        lkp.write(scenario_lkp)

    unanswered_file = splitpart+".noans"+EXT

    with open(directory+unanswered_file, "w") as unan:
        unan.write(unanswered) # user thanks etc that were never answered

    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else: main(0)
