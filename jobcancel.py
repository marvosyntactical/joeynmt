from subprocess import Popen



def main(from_to):
    from_, to_ = from_to
    for i in range(int(from_), int(to_)):
        Popen([f"scancel {i}"],shell=True).wait()





if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
