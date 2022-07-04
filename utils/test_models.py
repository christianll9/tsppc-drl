################################### USER CONFIG ###################################

# To deactivate a subtest, just set the num_nodes to empty list

# OWN MODELS
models_tests = [
    {
        "model_relpaths": [
            # all models, which should be tested/compared
        ],
        "num_nodes": [200],
        "sampling": [1, 10000], # 1 for greedy
        "beam_search": [False]
    }, # can be appended if other parameters should be used for other models
]

# NEAREST NEIGHBOR
nn_tests = [] #[20, 50] # Specify only the number of nodes

# LKH3
lkh_tests = [{
    "max_trails": [100],
    "runs": [1],
    "num_nodes": [] #[20, 50]
}]

LKH3_REL_PATH = "exec/LKH-3.0.6/LKH" # execution path for linux machine (has to be installed manually)

####################################### CODE ######################################

import os
import sys

import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(file_dir, '..', '..')

from utils import solver, evaluate
import pickle
import csv
import numpy as np
from datetime import datetime
import subprocess



def get_git_revision(base_path):
    """Source: https://stackoverflow.com/a/56245722/11227118"""
    git_dir = os.path.join(base_path, '.git/')
    with open(git_dir + 'HEAD', 'r') as f:
        head_output = f.readline().split(' ')
        ref = head_output[-1].strip()
        if len(head_output) == 1: return ref
        
    with open(os.path.join(git_dir, ref), 'r') as f:
        return f.readline().strip()

if sys.platform == 'linux':
    hostname = subprocess.check_output('hostname').splitlines()[0].decode("ascii")
    username = subprocess.check_output('whoami').splitlines()[0].decode("ascii")
else:
    hostname, username = None, None

commit_hash = get_git_revision(root_dir)
print("Commit hash: " + commit_hash)
csv_filepath = os.path.join(root_dir, "results", "full_evaluation_table.csv")


def split(a, n):
    """into equally large sizes. Source: https://stackoverflow.com/a/2135920/11227118"""
    k, m = divmod(len(a), n)
    return [a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n)]


def write_row(row_data:list):
    with open(csv_filepath, 'a', encoding='UTF8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(row_data)


def main():
    # Own Models
    for idx, subtests in enumerate(models_tests):
        print(f"Model subtests {idx+1}:")
        for num_nodes in subtests["num_nodes"]:
            print(f'Number of nodes: {num_nodes}')
            dataset_relpath = f'results/tsppc_instances/tsppc{num_nodes}.pkl'
            dataset_abspath = os.path.join(root_dir, dataset_relpath)

            with open(dataset_abspath, 'rb') as f:
                tsppcs = pickle.load(f)

            #if data is bigger than 250MB than split it
            n_chunks = int(np.ceil(len(tsppcs) * num_nodes**2 * 64 / 250E6))
            tsppcs = split(tsppcs, n_chunks)
            
            for model_relpath in subtests["model_relpaths"]:
                print(f"Evaluating {model_relpath}...")
                for beam_search_flag in subtests["beam_search"]:
                    print(f"  ...with beam_search={beam_search_flag}")
                    for sampling_val in subtests["sampling"]:
                        print(f"    ...with n_samples={sampling_val if sampling_val!=1 else 'greedy'}")
                        try:
                            model_abspath = os.path.join(root_dir, "trained_models", model_relpath)
                            solver_func = lambda tsppcs: solver.solve_multiple(tsppcs, model_abspath, beam_search=beam_search_flag, n_samples = sampling_val if sampling_val!=1 else None)
                            results = [evaluate.evaluate(solver_multiple=solver_func, tsppcs=chunked_tsppcs) for chunked_tsppcs in tsppcs]
                            tour_length, duration, invalids = list(np.mean(results, axis=0))
                            timestamp = datetime.now().isoformat()
                            results = [timestamp, model_relpath, num_nodes, sampling_val, beam_search_flag, tour_length, duration, int(invalids), dataset_relpath, commit_hash, sys.platform, hostname, username]
                            write_row(results)
                        except Exception as e:
                            print("Exception occured")
                            print(e)
                            print()
            print()
        print()

    # Nearest Neighbor
    if len(nn_tests)> 1: print(f"Nearest Neighbor Baseline")
    for num_nodes in nn_tests:
        print(f'Evaluating number of nodes: {num_nodes}...')
        dataset_relpath = f'results/tsppc_instances/tsppc{num_nodes}.pkl'
        dataset_abspath = os.path.join(root_dir, dataset_relpath)

        with open(dataset_abspath, 'rb') as f:
            tsppcs = pickle.load(f)
        
        try:
            results = evaluate.evaluate(solver=solver.nearest_neighboor, tsppcs=tsppcs)
            timestamp = datetime.now().isoformat()
            results = [timestamp, "Nearest Neighbor", num_nodes, None, None] + list(results) + [dataset_relpath, commit_hash, sys.platform, hostname, username]
            write_row(results)
        except Exception as e:
            print("Exception occured")
            print(e)
            print()
        print()
    print()

    # LKH3
    for idx, lkh_subtests in enumerate(lkh_tests):
        print(f"LKH subtests {idx+1}:")
        for num_nodes in lkh_subtests["num_nodes"]:
            print(f'Number of nodes: {num_nodes}')
            dataset_relpath = f'results/tsppc_instances/tsppc{num_nodes}.pkl'
            dataset_abspath = os.path.join(root_dir, dataset_relpath)

            with open(dataset_abspath, 'rb') as f:
                tsppcs = pickle.load(f)
            
            for run_value in lkh_subtests["runs"]:
                for mt_value in lkh_subtests["max_trails"]:
                    try:
                        model_name = f"LKH3_runs{run_value}_mt{mt_value}"
                        lkh3_abspath = os.path.join(root_dir, LKH3_REL_PATH) if sys.platform == "linux" else None
                        print(f"Evaluating {model_name}...")
                        solver_func = lambda tsppcs: solver.lkh3(tsppcs, lkh3_abspath, max_trials=mt_value, runs=run_value)
                        results = evaluate.evaluate(solver=solver_func, tsppcs=tsppcs)
                        timestamp = datetime.now().isoformat()
                        results = [timestamp, model_name, num_nodes, None, None] + list(results) + [dataset_relpath, commit_hash, sys.platform, hostname, username]
                        write_row(results)
                    except Exception as e:
                        print("Exception occured")
                        print(e)
                        print()
            print()
        print()

main()