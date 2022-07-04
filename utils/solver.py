import numpy as np
import uuid
import time
import utils.tools as tools
import tempfile
import subprocess
import shutil
import utils.converter as conv
from utils.evaluate import tsppc_tour_is_valid
from itertools import permutations


#for using TSPPC evaluator
import sys, os
file_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(file_dir, '..')
sys.path.append(os.path.join(root_dir, 'src'))
import evaluation


def nearest_neighboor(tsppc: np.array):
    """
    Simple Nearest Neighboor Solver, which masks the non-valid solutions
    Node 0 is always starting node
    Node -1 is always end node (should have the same coordinates)
    """

    start_time = time.time() #time measurement

    successors, predecessors = np.where(tsppc[:-1,:-1] == -1)
    nn_sorted = np.argsort(tsppc[:-1,:-1], axis=1)[:,1:]
    next_node = 0
    tour = [next_node]
    for _ in range(len(tsppc)-2):
        filter_mask = predecessors != next_node
        predecessors = predecessors[filter_mask]
        successors = successors[filter_mask]
        next_node = [idx for idx in nn_sorted[next_node] if idx not in tour and idx not in successors][0]
        tour.append(next_node)
    tour.append(len(tsppc)-1)

    total_time = time.time() - start_time
    length = tools.tour_length_by_dist_mat(tsppc, tour)

    return tour, total_time, length

def optimal(tsppc: np.array):
    """
        Finds the optimal solution for a TSPPC by trying out all
        possible routes and returns the smallest valid one. Should
        only be used for extrem small TSPPC matrices.
        Assumes a starting node of 0 and a end node of len(tsppc)-1
    """
    start_time = time.time()
    best_length = np.inf
    best_tour = None
    for tour in permutations(range(1,len(tsppc)-1)):
        tour = [0] + list(tour) + [len(tsppc)-1]
        if tsppc_tour_is_valid(tsppc, tour):
            length = tools.tour_length_by_dist_mat(tsppc, tour)
            if length < best_length:
                best_length = length
                best_tour = tour

    total_time = time.time() - start_time
    
    return best_tour, total_time, best_length


def solve_single(tsppc: np.array, model_file_path:str, coords:np.array=None, **kwargs):
    """
    Solves a single tsppc
    """
    tours, durations, length = solve_multiple([tsppc], model_file_path, [coords], **kwargs)
    return tours[0], durations[0], length[0]



def lkh3(tsppc:np.array, solver_path:str=None, max_trials:int=10000, runs:int=1):
    '''
    Given generated TSPPC data, 
    Outputs the optimal route, tour length, and run time
    solver_path means path to LKH executable
    '''
    import tsplib95 as tsplib
    if solver_path is None:
        if sys.platform == "win32":
            solver_path = os.path.join(root_dir, 'exec', 'LKH-3.exe')
        else:
            raise Exception(f"solver path unknown, there is currently only a default solver path for Windows available")

    if np.issubdtype(type(tsppc[0,0]), np.floating):
        smallest_float = np.min(np.where(tsppc<=0, np.inf, tsppc))
        tsppc_scaled = tools.scale_tsppc(tsppc, 100/smallest_float)
        tsppc_scaled = np.round(tsppc_scaled)

    data = conv.tsppc2tsplib_format(tsppc_scaled)
    problem=tsplib.parse(data) #tsplib.parse reads data file not matrix
    start = time.time()

    ### This part mimics the not yet merged lkh.solve from https://github.com/ben-hudson/pylkh/pull/1 ###
    params = {
        'max_trials': max_trials,
        'runs': runs
    }

    assert shutil.which(solver_path) is not None, f'{solver_path} not found.'
    valid_problem = problem is not None and isinstance(problem, tsplib.models.StandardProblem)
    assert ('problem_file' in params) ^ valid_problem, 'Specify a TSPLIB95 problem object *or* a path.'
    if problem is not None:
        # hack for bug in tsplib
        if len(problem.depots) > 0:
            problem.depots = map(lambda x: f'{x}\n', problem.depots)

        prob_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        problem.write(prob_file)
        prob_file.write('\n')
        prob_file.flush()
        prob_file.close()
        params['problem_file'] = prob_file.name

    # need dimension of problem to parse solution
    problem = tsplib.load(params['problem_file'])

    if 'tour_file' not in params:
        tour_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        params['tour_file'] = tour_file.name
        tour_file.close()

    par_file = tempfile.NamedTemporaryFile(mode='w+', delete=False)
    par_file.write('SPECIAL\n')
    for k, v in params.items():
        par_file.write(f'{k.upper()} = {v}\n')
    par_file.close()

    try:
        # stdin=DEVNULL for preventing a "Press any key" pause at the end of execution
        subprocess.check_output([solver_path, par_file.name], stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        raise Exception(e.output.decode())

    solution = tsplib.load(params['tour_file'])

    os.remove(par_file.name)
    if 'prob_file' in locals(): os.remove(prob_file.name)
    if 'tour_file' in locals(): os.remove(tour_file.name)

    tour = solution.tours[0]
    
    ###################################################################################################
    # This line should be enough, if the merge is finished:
    # tour = lkh.solve(solver_path, problem=problem, max_trials=max_trials, runs=runs)[0]
    
    elapsed_time = time.time() - start

    tour = [node-1 for node in tour] # convert to our tour format

    tour_len = tools.tour_length_by_dist_mat(tsppc, tour)
    
    return tour, elapsed_time, tour_len


def solve_multiple(tsppcs:list, model_file_path:str, coords_lists:list=None, n_samples:int=None,
    beam_search:bool=False, normalize:bool=True):
    """
    Solves multiple TSPPCs at once
    If n_sample is None, greedy is used
    If beam_search is True, n_sample is the beam size
    If normalize is True, the input data is normalized before passed into the model
    """
    tmp_file_path = os.path.join(os.getcwd(), f"tmp_{uuid.uuid4()}.pkl") #in current working directory
    conv.preprocess_tsppcs_for_model(tsppcs, coords_lists, normalize=normalize, pkl_path=tmp_file_path)

    sys.argv = ['evaluation.py', tmp_file_path, '--model', model_file_path, "--no_output_file", "--quite", '--decode_strategy']

    if n_samples is None:
        sys.argv += ['greedy']
    elif beam_search:
        sys.argv += ['bs', "--width", str(n_samples), "--eval_batch_size", "1"]
    else:
        sys.argv += ['sample', "--width", str(n_samples), "--eval_batch_size", "1"]


    opts = evaluation.get_options()
    _, tours, durations = evaluation.eval_dataset(opts.datasets[0], opts.width[0], opts.softmax_temperature, opts)

    os.remove(tmp_file_path)
    
    # add tour start and end node
    tours = [[0] + tour[:-1] + [len(tour)] for tour in tours]

    # recalculate length, due to previous normalization
    length = [tools.tour_length_by_dist_mat(conv.tsppc2dist(tsppc), tour) for tsppc, tour in zip(tsppcs, tours)]

    durations = [time/len(tsppcs) for time in durations]

    return tours, durations, length

