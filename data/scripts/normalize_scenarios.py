import json
import os
import sys


def normalize_kb(d):
    """
    takes kb dictionary and decides which type of info to extract

    this is unnecessarily done in separate functions
    because I initially believed the database info
    was different for different tasks
    """
    intent = d["task"]["intent"]
    if d["kb"]["items"] is None:
        return []
    
    return eval("normalize_{}(d)".format(intent)) #practice not advised


def normalize_weather(d):
    """
    takes a kb with task intent "weather" and items with
    locations and normalizes all items into triples
    of the form: 

    (subject , relation , value) #generic
    (location, weekday  , info ) #weather specific
    ("san mateo", "monday", "dry, low of 80F, high of 90F") #example 
    """
    normed_kb = [] #to add (subj, rel, val) triples
    assert d["task"]["intent"]=="weather"
    locations = d["kb"]["items"]
    for location in locations:
        subject = location["location"]
        #do something with "today???"
        for weekday in location.keys():
            if weekday in ("today", "location"): continue #? TODO
            normed_kb.append((subject,weekday,location[weekday]))
    return normed_kb

def normalize_navigate(d):
    """
    takes a kb with task intent "navigate" and items with
    and normalizes all items into triples
    of the form: 

    (subject , relation , value) #generic
    (poi, dist/traff/type/addr , info ) #weather specific
    ("Pizza Hut", "address", "704 El Camino Real") #example

    """
    normed_kb = [] #to add (subj, rel, val) triples
    assert d["task"]["intent"]=="navigate"
    blimps = d["kb"]["items"]
    for blimp in blimps:
        subject = blimp["poi"]
        for relation in blimp.keys():
            if relation != "poi":
                normed_kb.append((subject,relation,blimp[relation]))
    return normed_kb

def normalize_schedule(d):
    """
    takes a kb with task intent "schedule" and items with
    and normalizes all items into triples
    of the form: 

    (subject , relation , value) #generic
    (event, room/agenda/time/date/party, info) #weather specific
    ("lab appointment", "date", "wednesday") #example

    """
    normed_kb = [] #to add (subj, rel, val) triples
    assert d["task"]["intent"]=="schedule"
    appointments = d["kb"]["items"]
    for appointment in appointments:
        event = appointment["event"]
        for relation in appointment.keys():
            if not relation == "event":
                normed_kb.append((event,relation,appointment[relation]))
    return normed_kb

def main(args):

    directory = "../kv_ret_dataset/"
    if args==0: #use defaults
        filename = "dev.json"
        splitpart = "dev"
    else:
        filename = args[0]
        splitpart = args[1] 
    with open(directory+filename, "r") as scenarios:
        settings = json.load(scenarios)

    normed_kbs = [normalize_kb(kb) for kb in settings]
    lens = [str(len(kb))+"\n" for kb in normed_kbs]
    lens[-1] = lens[-1][:-1] #delete last newline
    
    normed_kbs_inner = [triple for scenario in normed_kbs for triple in scenario]
    kb_list = ["::".join(t)+"\n" for t in normed_kbs_inner]
    kb_list[-1] = kb_list[-1][:-1] #delete last newline


    """
    LATEST TODO:
        look at torchtext dataset and make_train_iter_batch_size_1:
        how is data loaded? how to write normed_kbs to file to load simultaneously with train data
        idea: write iterator that makes data iter in parallel that takes a batch size function that is read from lines
        in separate file kb_lengths.txt and has type list [230, 102, 56,...,95]
        for this, here we need to output two things:
            dev.kb: kb items line by line, all knowledgebases one after another
            dev.len: kb lengths line by line
        -> look at torchtext.data.Iterator kwarg batch_size_fn to see if possible
    """
    #line formatted normalized kb
    filestamm = filename.split(".")[0]
    ext = "kb"
    save_as = filestamm+"."+ext
    with open(directory+save_as, "w") as o:
        o.writelines(kb_list)

    # line formatted kb length lookup file
    # * dev.len
    # saves number of lines to put in kb batch in 
    # * dev.kb
    # from start of current line from last example
    # according to 
    # * dev.lkp 

    lengths = "len"
    save_lengths = filestamm + "." + lengths
    with open(directory+save_lengths, "w") as l:
        l.writelines(lens)
    return 0

if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))
