from subprocess import Popen

# usage:
# python3 canceljoby.py 349500 349504
# to delete all of my slurm jobs that are in the range 349500 349504

def main(from_to):
    from_, to_ = from_to
    for i in range(int(from_), int(to_)):
        Popen([f"scancel {i}"],shell=True).wait()



if __name__ == "__main__":
    import sys
    sys.exit(main(sys.argv[1:]))
