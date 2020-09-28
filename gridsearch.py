import os
import shutil
from collections import OrderedDict
from subprocess import Popen, check_output
import time

partitions = ["students", "gpulong"] # order == priority
jobbers = {0:"studentsgo", 1: "longgo"} # sbatch creator for each partition

me = check_output(["whoami"])[:-1].decode("utf-8") # only worx when nobody else has koss in their name <3
shellext = ".sh"




def wait_for_green_light(partitions=partitions, my_jobs_per_partition=[2,4], update=60):
    """ waits until my squeue has a place"""


    while True:

        print(f"Waiting for another {update} seconds to check if there's an opening on any of: {partitions}")

        time.sleep(update)

        partition_with_slot = -1

        sq = str(check_output(["squeue"])).split("\\n")
        for i, part in enumerate(partitions):

            # check if partition is empty enough
            allowed_num_jobs_here = my_jobs_per_partition[i]

            part_sq = [ job for job in sq if len(job.split()) > 5 and part in job.split()[1] ]
            my_part_sq = [ job for job in part_sq if len(job.split()) > 5 and me in job.split()[3] ] # relies on job name not having whitespace ! FIXME
            
            if len(my_part_sq) < allowed_num_jobs_here:
                partition_with_slot = i
                break

        if partition_with_slot != -1:
            return partition_with_slot

def main(args):

    search_space = OrderedDict({
        "k_hops": [1,2,3],
        "kb_input_feeding": [False, True],
        "teacher_force": [False, True] # add param for this 
    })

    architectures = ["rnn", "tf"] # remember to add if case for tfstyletf
    init = "GridInit"
    path = "configs/kvr/grid/"
    ext  = ".yaml"
    models = "models/"

    for architecture in architectures:

        arch_config = path + architecture + init + ext

        for j in search_space["k_hops"]:
            for no_yes in search_space["kb_input_feeding"]:
                for false_tru in search_space["teacher_force"]:

                        gridcombo = architecture+str(j)+str(int(no_yes))+str(int(false_tru))
                        new_cfg = path+gridcombo+ext
                        # check if model directory exists; if yes then continue
                        existing_models = str(check_output(["ls", models])).split("\\n")
                        skip_job = False
                        for model in existing_models:
                            if gridcombo in model:
                                skip_job = True
                                break

                        if skip_job: continue
                                

                        free_partition = wait_for_green_light()

                        Popen( [f"cp {arch_config} {new_cfg}"], shell=True).wait()
                        assert os.path.isfile(new_cfg), (os.listdir(path), arch_config, new_cfg)

                        for key, value in zip(search_space.keys(), [j, no_yes, false_tru]):

                            param_regex_replace = "s/{key}: [^#]*#/{key}: {value} #/g"
                            Popen(["sed -i", param_regex_replace, new_cfg],shell=True).wait()

                        Popen(f"./{jobbers[free_partition]+shellext} {gridcombo}",shell=True).wait()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else:
        main(0)
