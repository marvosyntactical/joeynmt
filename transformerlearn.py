import os
import shutil
from subprocess import Popen, check_output
import time

partitions = ["gpulong", "students"] # order == priority
jobbers = {0:"longgo", 1: "studentsgo"} # sbatch creator for each partition

me = check_output(["whoami"])[:-1].decode("utf-8") # only worx when nobody else has koss in their name <3
shellext = ".sh"

def alphanumify(s: str):
    alphanum = "abcdefghijklmnopqrstuvwxyz0123456789"
    r = ""
    for c in s:
        if c.lower() in alphanum:
            r+= c
    return r

def wait_for_green_light(partitions=partitions, my_jobs_per_partition=[4,2], init_update=10):
    """ waits until my squeue has a place"""

    update = init_update
    i = 0
    sleeplonger_once_checked_this_often = 10
    longer_update = init_update * 6

    while True:
        i += 1
        if i == sleeplonger_once_checked_this_often:
            update = longer_update

        print(f"Waiting for another {update} seconds to check if there's an opening on any of: {partitions}")

        time.sleep(init_update)

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

    # inplace dicts of config changes as compared to best grid config (eric et al plus multi dimensional KBs)
    # search space is dict of config name: list of changed param dictionary for each run
    # NOTE remember to add 'param: defaultvalue #' into default configs here NOTE
    search_space = {
            "tfRecurrentKbatt_100x16x32x0_plateau": [
                {'sched_sampl_type': "linear", 'sched_sampl_k':1.0, 'sched_sampl_c_e':[0.0,1.0]}, # teacher force # NOTE default
                {'sched_sampl_type': "invsigmoid", 'sched_sampl_k':10000, 'sched_sampl_c_e':[0.0,0.0]}, # sigmoidal scheduled sampling
                {'sched_sampl_type': "linear", 'sched_sampl_k':0.0, 'sched_sampl_c_e':[0.0,0.0]} # autoregressive 
                ],
            "tfdotKbatt_plateau": [
                {'infeedkb': False}, # NOTE default
                {'outfeedkb': True}, # gated feeding/LSTM, mostly doesnt learn
                {'double_decoder': True},
                {'double_decoder': True, 'tied_side_softmax':True},
                ],
            "tfdot_9enc_3dec_plateau": [
                {'infeedkb': False}, # NOTE default
                {'outfeedkb': True}, # gated feeding/LSTM, mostly doesnt learn
                {'double_decoder': True},
                {'double_decoder': True, 'tied_side_softmax':True},
                ],

            }
    path = "configs/kvr/add/"
    ext  = ".yaml"
    models = "modelsadd/"

    if clean:
        # to clean all configs from /configs/kvr/grid/
        tmp = path+"tmp/"
        if not os.path.isdir(tmp):
            os.mkdir(tmp)
        save_these_cfgs = [key+ext for key in search_space.keys()]
        input(f"These are the only cfgs in {path} that will be saved: {save_these_cfgs}")
        for main_cfg in save_these_cfgs: 
            if os.path.isfile(path+main_cfg):
                shutil.move(path+main_cfg, tmp+main_cfg)
            else:
                assert os.path.isfile(tmp+main_cfg), f"init config '{main_cfg}' found in neither of {path} or {tmp}"
        Popen("rm "+path+"*", shell=True ).wait()  # remove all generated configs
        for main_cfg in save_these_cfgs: # move back
            shutil.move(tmp+main_cfg, path+main_cfg)
        os.rmdir(tmp)
        exit("Cleaned configs from "+path)

    for i, (init_cfg_name,run_list)  in enumerate(list(search_space.items())):
        for config_changes in run_list:
            init_cfg = path+init_cfg_name+ext
            assert os.path.isfile(init_cfg), (os.listdir(path), init_cfg, new_cfg) # confirm the copying worked lol

            # create new config name
            new_cfg_name = init_cfg_name + "."
            new_cfg_name += alphanumify("".join([param[:10]+str(value)[0].upper()+str(value)[1:].lower() for param, value in config_changes.items()])) # NOTE results in damn long filenames

            new_cfg = path+new_cfg_name+ext

            """
            ### check if model directory exists; if yes then continue ###
            existing_models = str(check_output(["ls", models])).split("\\n")

            skip_job = False
            for model in existing_models:
                if new_cfg_name in model: # match with model dirs based on str containment
                    skip_job = True
                    break
            if skip_job: continue
            ### end check ###
            """

            Popen([f"cp {init_cfg} {new_cfg}"], shell=True).wait() # copy the default config for this architecture into a new one
            assert os.path.isfile(new_cfg), (os.listdir(path), init_cfg, new_cfg) # confirm the copying worked 

            for param, setting in config_changes.items():
                # change each parameters setting according to search space entry

                if type(setting) == str:
                    # add escaped string chars
                    param_regex_replace = f"s/{param}: [^#]*#/{param}: \"{setting}\"  #/g"
                else:
                    param_regex_replace = f"s/{param}: [^#]*#/{param}: {setting}  #/g"
                Popen('sed -i '+f'\"{param_regex_replace}\" '+new_cfg, shell=True).wait()
            
            # sanity check: were different parameters actually entered in newly created new_cfg?
            difference_made = False
            try:
                empty_string_on_success = check_output(["diff", init_cfg, new_cfg]).decode("utf-8")
            except Exception as e:
                difference_made = True
                pass
            if not difference_made:
                raise ValueError(f"newly created {new_cfg} not different from init config {init_cfg}")
            
            if not dry: # dry run of this script doesnt actually execute the slurm job
                Popen(f"diff {init_cfg} {new_cfg}", shell=True) # REMOVE THIS
                free_partition = wait_for_green_light()
                Popen(f"./{jobbers[free_partition]+shellext} {new_cfg_name}",shell=True).wait()

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        sys.exit(main(sys.argv[1:]))
    else:
        main([])
