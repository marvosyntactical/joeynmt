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

    return eval("normalize_{}(d)".format(intent))


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
        today = location["today"]

        for weekday in location.keys():

            if weekday == "location": # add ("ohio", "location", "ohio")

                normed_kb.append((subject, weekday, location[weekday]))

            elif weekday != "today": # add ("san francisco", "monday weather", "rain")

                weather_info = location[weekday].split(",")
                weather_attribute, temp_low, temp_high = weather_info

                temp_low = temp_low.split()[-1] # just degree f
                temp_high = temp_high.split()[-1] # just degree f

                if weekday == today:
                    weekday += " today" #add "today" token to whichever day today is

                normed_kb.append((subject,weekday+" weather",weather_attribute))
                normed_kb.append((subject,weekday+" temperature low",temp_low))
                normed_kb.append((subject,weekday+" temperature high",temp_high))

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
        subject = blimp["poi_type"]
        for relation in blimp.keys():
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
            # Latest TODO:
            # possible also check appointment[relation] != "-"
            # to filter out unassigned rooms/agendas/..
            if appointment[relation] != "-":
                normed_kb.append((event,relation,appointment[relation]))
    return normed_kb

def main(args):

    directory = "../kvr/"
    if args==0: #use defaults
        splitpart = "dev"
    else:
        splitpart = args[0]
    filename = splitpart+".json"
    with open(directory+filename, "r") as scenarios:
        settings = json.load(scenarios)

    normed_kbs = [normalize_kb(kb) for kb in settings]
    lens = [str(len(kb))+"\n" for kb in normed_kbs]

    normed_kbs_inner = [triple for scenario in normed_kbs for triple in scenario]
    kb_list = ["::".join(t)+"\n" for t in normed_kbs_inner]


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
    ext = "kbFINAL"
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

    lengths = "lenFINAL"
    save_lengths = filestamm + "." + lengths
    with open(directory+save_lengths, "w") as l:
        l.writelines(lens)

    return 0

if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))
