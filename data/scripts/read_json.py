import json
import os
import sys

def read_json(filename):
    contents = json.load(filename)
    return contents

def main(args):
    directory = "../kv_ret_dataset/"
    filename = "kvret_entities.json" if args == 0 else args[0]
    with open(directory+filename, "r") as f:
        dicc = read_json(f)
    if type(dicc) == list:




        print(len(dicc[0]))
        print(dicc[0].keys())
        print(dicc[0]["scenario"].keys())



    else:
        print(len(dicc))
        print(dicc.keys())


    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else: main(0)
