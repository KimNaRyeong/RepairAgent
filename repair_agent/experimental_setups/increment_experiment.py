import os
import sys

# Get model name from command line argument, default to empty string
model_prefix = sys.argv[1] if len(sys.argv) > 1 else ""
if model_prefix:
    model_prefix = model_prefix + "_"

# Get experiments list file path from command line argument, default to experiments_list.txt
experiments_list_file = sys.argv[2] if len(sys.argv) > 2 else "experimental_setups/experiments_list.txt"

with open(experiments_list_file, "r+") as expl:
    exps = expl.read().splitlines()
    #print(exps)
    if exps:
        # Extract the number from the last experiment (supports both "experiment_X" and "model_experiment_X" format)
        last_parts = exps[-1].split("_")
        last_exp = int(last_parts[-1])
    else:
        last_exp = 0

    exp_name = "{}experiment_{}".format(model_prefix, last_exp + 1)
    print("Creating experiment folder:", exp_name)
    expl.write("{}\n".format(exp_name))
    os.mkdir("experimental_setups/{}".format(exp_name))
    os.mkdir("experimental_setups/{}/logs".format(exp_name))
    os.mkdir("experimental_setups/{}/responses".format(exp_name))
    os.mkdir("experimental_setups/{}/external_fixes".format(exp_name))
    os.mkdir("experimental_setups/{}/saved_contexts".format(exp_name))
    os.mkdir("experimental_setups/{}/mutations_history".format(exp_name))
    os.mkdir("experimental_setups/{}/plausible_patches".format(exp_name))