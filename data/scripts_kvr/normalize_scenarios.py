import json
import os
import sys

DUMMY_SUBJ = "DUMMYSUBJ"
DUMMY_REL = "DUMMYREL"
DUMMY_VAL = "DUMMYVAL"

ADD_DUMMY = True


def normalize_kb(d, eric=False):
    """
    takes kb dictionary and decides which type of info to extract

    this is unnecessarily done in separate functions
    because I initially believed the database info
    was different for different tasks
    """
    intent = d["task"]["intent"]
    if d["kb"]["items"] is None:
        return []

    return eval("normalize_{}(d, eric={})".format(intent, eric))


def normalize_weather(d, eric=False):
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
    location_keys_length = set()
    for location in locations:
        location_keys_length |= {len(location.keys())}

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

                normed_kb.append((subject,weekday+" date",weekday.split()[-1])) # 'monday' or 'today'
                normed_kb.append((subject,weekday+" weather",weather_attribute))
                normed_kb.append((subject,weekday+" temperature low",temp_low))
                normed_kb.append((subject,weekday+" temperature high",temp_high))

    assert len(location_keys_length) == 1, f" {location_keys_length}; all KB items (locations) should have same number of keys (weekdays)"
    # add how many dummy entries using num of entries from first kb example
    if ADD_DUMMY:

        normed_kb_with_dummy_entries = [(DUMMY_SUBJ, entry[1], DUMMY_VAL) for entry in normed_kb if
                                        entry[0]==normed_kb[0][0]]
        normed_kb_with_dummy_entries += normed_kb
        return normed_kb_with_dummy_entries
    else:
        return normed_kb

def normalize_navigate(d, eric=False):
    """
    takes a kb with task intent "navigate" 
    and normalizes all items into triples
    of the form: 

    (subject , relation , value) #generic
    (poi, dist/traff/type/addr , info ) #weather specific
    ("Pizza Hut", "address", "704 El Camino Real") #example

    """
    normed_kb = [] #to add (subj, rel, val) triples
    assert d["task"]["intent"]=="navigate"
    blimps = d["kb"]["items"] # blimps as in map marker blips
    blimp_keys_length = set()
    for blimp in blimps:
        blimp_keys_length |= {len(blimp.keys())}

        poi_type = blimp["poi_type"]
        poi = blimp["poi"]

        if eric:
            subject = poi
        else:
            if poi_type != poi:
                subject = poi_type + " " + poi
            else:
                # avoid "home home"; instead put "home"
                assert poi_type.lower() == "home"
                subject = poi_type

        for relation in blimp.keys():
            normed_kb.append((subject,relation,blimp[relation]))

    assert len(blimp_keys_length) == 1, f"{blimp_keys_length}; all KB items ) should have same number of keys (weekdays)"
    # add how many dummy entries using num of entries from first kb example
    if ADD_DUMMY:

        normed_kb_with_dummy_entries = [(DUMMY_SUBJ, entry[1], DUMMY_VAL) for entry in normed_kb if
                                        entry[0]==normed_kb[0][0]]
        normed_kb_with_dummy_entries += normed_kb

        return normed_kb_with_dummy_entries
    else:
        return normed_kb

def normalize_schedule(d, eric=False):
    """
    eric isnt used here :/
    takes a kb with task intent "schedule"
    and normalizes all items into triples
    of the form: 

    (subject , relation , value) #generic
    (event, room/agenda/time/date/party, info) #weather specific
    ("lab appointment", "date", "wednesday") #example

    """
    normed_kb = [] #to add (subj, rel, val) triples
    assert d["task"]["intent"]=="schedule"
    appointments = d["kb"]["items"]
    appointment_keys_lengths = set()
    for appointment in appointments:
        appointment_keys_lengths |= {len(appointment.keys())}

        event = appointment["event"]
        for relation in appointment.keys():
            # also check appointment[relation] != "-"
            # to filter out unassigned rooms/agendas/..
            # if appointment[relation] != "-":
            normed_kb.append((event,relation,appointment[relation]))

    assert len(appointment_keys_lengths) == 1, f"{appointment_keys_lengths}; all KB items should have same number of keys"
    # add how many dummy entries using num of entries from first kb example
    if ADD_DUMMY:

        normed_kb_with_dummy_entries = [(DUMMY_SUBJ, entry[1], DUMMY_VAL) for entry in normed_kb if
                                        entry[0]==normed_kb[0][0]]
        normed_kb_with_dummy_entries += normed_kb
        return normed_kb_with_dummy_entries
    else:
        return normed_kb



def main(args):

    EXT = "FINAL"
    directory = "../kvr/"

    if args==0: #use defaults
        splitpart = "dev"
    else:
        splitpart = args[0]
        if len(args) > 1:
            EXT = args[1]

    eric = True # TODO add to args

    filename = splitpart+".json"
    with open(directory+filename, "r") as scenarios:
        settings = json.load(scenarios)

    normed_kbs = [normalize_kb(kb, eric=eric) for kb in settings]
    lens = [str(len(kb))+"\n" for kb in normed_kbs]

    normed_kbs_inner = [triple for scenario in normed_kbs for triple in scenario]
    kb_list = ["::".join(t)+"\n" for t in normed_kbs_inner]

    #line formatted normalized kb
    filestamm = filename.split(".")[0]
    ext = "kb"+EXT
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

    lengths = "len"+EXT
    save_lengths = filestamm + "." + lengths
    with open(directory+save_lengths, "w") as l:
        l.writelines(lens)

    return 0

if __name__ == "__main__":
    if len(sys.argv) >1:
        sys.exit(main(sys.argv[1:]))
    else:
        sys.exit(main(0))
