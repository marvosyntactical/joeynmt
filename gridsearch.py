import os
import shutil
from subprocess import Popen, check_output
import time

partitions = ["students", "gpulong"] # order == priority
jobbers = {0:"studentsgo", 1: "longgo"} # sbatch creator for each partition

me = check_output(["whoami"])[:-1].decode("utf-8") # only worx when nobody else has koss in their name <3
shellext = ".sh"

def wait_for_green_light(partitions=partitions, my_jobs_per_partition=[2,6], update=10):
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
    dry = "dry" in args
    clean = "clean" in args

    search_space = {}
    search_space["hops"] = [("k_hops",), \
            [(1,),(2,),(3,)]]
    search_space["feed"] = [("kb_input_feeding","kb_feed_rnn"),\
            [(False, False), (True, False), (True, True)]]
    search_space["enc"] = [("kb_max_dims","posEncKBkeys"),\
            [([256], False), ([16,32], False), ([256],True)]]


    architectures = ["rnn"] # remember to add if case for tfstyletf
    init = "GridInit"
    path = "configs/kvr/grid/"
    ext  = ".yaml"
    models = "models/grid/"

    architecture = architectures[0] # for the moment: only do gridsearch for RNN
    _arch_cfg = architecture + init + ext 
    arch_config = path + _arch_cfg
    
    if clean:
        # to clean all configs from /configs/kvr/grid/
        _tf_cfg = "tf"+init+ext
        tmp = path+"tmp/"
        if not os.path.isdir(tmp):
            os.mkdir(tmp)
        for main_cfg in [ _arch_cfg, _tf_cfg ]:
            if os.path.isfile(path+main_cfg):
                shutil.move(path+main_cfg, tmp+main_cfg)
            else:
                assert os.path.isfile(tmp+main_cfg), f"init config '{main_cfg}' found in neither of {path} or {tmp}"
        Popen("rm "+path+"*", shell=True ).wait()  # remove all generated configs
        for main_cfg in [ _arch_cfg, _tf_cfg ]:
            shutil.move(tmp+main_cfg, path+main_cfg)
        os.rmdir(tmp)
        exit("Cleaned all grid configs from "+path)

    for k in search_space["hops"][1]:
        for false_ff_rnn in search_space["feed"][1]:
            for oneD_twoD_pos in search_space["enc"][1]:

                    gridcombo = architecture+str(k[0])+str(int(false_ff_rnn[0]))+str(int(false_ff_rnn[1]))+"x"+"x".join([str(dim) for dim in oneD_twoD_pos[0]])+"x"+str(int(oneD_twoD_pos[1]))

                    new_cfg = path+gridcombo+ext

                    ### check if model directory exists; if yes then continue ###
                    existing_models = str(check_output(["ls", models])).split("\\n")
                    skip_job = False
                    for model in existing_models:
                        if gridcombo in model:
                            skip_job = True
                            break
                    if skip_job: continue
                    ### end check ###
                    
                    if not dry:
                        free_partition = wait_for_green_light()

                    Popen([f"cp {arch_config} {new_cfg}"], shell=True).wait() # copy the default config for this architecture into a new one
                    assert os.path.isfile(new_cfg), (os.listdir(path), arch_config, new_cfg) # confirm the copying worked lol

                    # k_hops, kb_input_feeding, kb_feed_rnn, kb_max_dims, posEncKBkeys
                    cfg_param_valuations = {}

                    for dimension, options in [("hops", k), ("feed", false_ff_rnn), ("enc", oneD_twoD_pos)]:
                        cfg_params_dim = search_space[dimension][0]
                        for i, p in enumerate(cfg_params_dim):
                            cfg_param_valuations[p] = options[i]

                    for param, setting in cfg_param_valuations.items():

                        param_regex_replace = f"s/{param}: [^#]*#/{param}: {setting} #/g"
                        Popen('sed -i '+f'\"{param_regex_replace}\" '+new_cfg, shell=True).wait()
                    
            
                    Popen("cat "+ new_cfg, shell=True).wait() # REMOVE THIS

                    if not dry:
                        Popen(f"./{jobbers[free_partition]+shellext} {gridcombo}",shell=True).wait()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else:
        main([])
