import json
import os
import sys




def main(args):
    directory = "../kv_ret_dataset/"
    filename = "kvret_entities.json" if args == 0 else args[0]
    with open(directory+filename, "r") as f:
        dicc = json.load(f)
    


    return 0

if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else: main(0)
