"""
Used as a general training script. Can give command line arguments to specify which json files to use
for hyperparameters and the ranges that those hyperparameters can take on.

$ python3 general_training.py params=hyperparams.json ranges=hyperranges.json

Defaults to hyperparams.json and hyperranges.json if no arguments are provided
"""
import sys
import os
import shutil
import time
import numpy as np
import torch
import select
from sencoder.training import hyper_search
from sencoder.utils import load_json

torch.autograd.set_detect_anomaly(True)

if __name__ == "__main__":
    hyperparams_file = "hyperparams.json"
    hyperranges_file = 'hyperranges.json'
    device = 0
    if len(sys.argv) > 1:
        for i,arg in enumerate(sys.argv[1:]):
            temp = sys.argv[1].split("=")
            if len(temp) > 1:
                if "params" == temp[0]:
                    hyperparams_file = temp[1]
                elif "ranges" == temp[0]:
                    hyperranges_file = temp[1]
            else:
                if i == 0:
                    hyperparams_file = arg
                elif i == 1:
                    hyperranges_file = arg
    print()
    print("Using hyperparams file:", hyperparams_file)
    print("Using hyperranges file:", hyperranges_file)
    print()

    hyps = load_json(hyperparams_file)
    hyp_ranges = load_json(hyperranges_file)
    hyps_str = ""
    for k,v in hyps.items():
        if k not in hyp_ranges:
            hyps_str += "{}: {}\n".format(k,v)
    print("Hyperparameters:")
    print(hyps_str)
    print("\nSearching over:")
    print("\n".join(["{}: {}".format(k,v) for k,v in hyp_ranges.items()]))

    # Random Seeds
    seed = 3
    if "rand_seed" in hyps:
        seed = hyps['rand_seed']
    np.random.seed(seed)
    torch.manual_seed(seed)

    if "shift_labels" in hyps and hyps['shift_labels']:
        print("{0} WARNING: YOU ARE USING SHIFTED LABELS {0}".format("!"*5))

    sleep_time = 8
    if os.path.exists(hyps['exp_name']):
        _, subds, _ = next(os.walk(hyps['exp_name']))
        dirs = []
        for d in subds:
            splt = d.split("_")
            if len(splt) >= 2 and splt[0] == hyps['exp_name']:
                dirs.append(d)
        dirs = sorted(dirs, key=lambda x: int(x.split("_")[1]))
        if len(dirs) > 0:
            s = "Overwrite last folder {}? (No/yes)".format(dirs[-1])
            print(s)
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
            if i and "y" in sys.stdin.readline().strip().lower():
                path = os.path.join(hyps['exp_name'], dirs[-1])
                shutil.rmtree(path, ignore_errors=True)
        else:
            s = "You have {} seconds to cancel experiment name {}:"
            print(s.format(sleep_time, hyps['exp_name']))
            i,_,_ = select.select([sys.stdin], [],[],sleep_time)
    else:
        s = "You have {} seconds to cancel experiment name {}:"
        print(s.format(sleep_time, hyps['exp_name']))
        i,_,_ = select.select([sys.stdin], [],[],sleep_time)
    print()

    keys = list(hyp_ranges.keys())
    start_time = time.time()
    hyper_search(hyps, hyp_ranges, keys, device)
    print("Total Execution Time:", time.time() - start_time)
