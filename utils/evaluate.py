import numpy as np
import time
import utils.tools as tools
import utils.converter as conv

def evaluate(solver=None, solver_multiple=None, tsppcs:list=None, coords_lists:list=None, verbose_if_invalid=False):
    """
        `solver`: function of the following format: `tour, time, length = solver(tsppc)`
        `solver_multiple`: Sometimes it is faster for a model to evaluate multiple tsppcs at once. If no `solver`
        function is given, this function is used. Format: `tours, times, lengths = solver_multiple(tsppcs)`
        `tsppcs`: list of tsppcs to solve
        `coords_lists`: list of problem coordinates. If this list is given, solver function needs to have following
        format: `tour, time, length = solver(tsppc, coords)`. Some solvers work better with original coordinates.
        
        returns average tour length, average processing time (in s.) and num_of_invalid_tours on list of tsppcs
    """

    start_time = time.time()
    if coords_lists is None:
        tours, _, _ = zip(*[solver(tsppc) for tsppc in tsppcs]) if solver is not None else solver_multiple(tsppcs)
    else:
        tours, _, _ = zip(*[solver(tsppc, coords) for tsppc, coords in zip(tsppcs, coords_lists)]) \
        if solver is not None else solver_multiple(tsppcs, coords_lists)

    # do not rely on self reported times and tour lengths
    avg_time = (time.time() - start_time)/len(tsppcs)
    invalid_tours = [not tsppc_tour_is_valid(tsppc, tour, verbose_if_invalid) for tsppc, tour in zip(tsppcs, tours)]
    # calculate length only for valid tours
    lengths = [tools.tour_length_by_dist_mat(conv.tsppc2dist(tsppc), tour)
        for tsppc, tour, tour_invalid in zip(tsppcs, tours, invalid_tours) if not tour_invalid
    ]

    return np.mean(lengths), avg_time, sum(invalid_tours)


def evaluate_sparse(solver=None, solver_multiple=None, tsppcs_sparse:list=None, tsppcs:list=None, verbose_if_invalid=False):
    """
        Almost the same as evaluate.evaluate, but passes a sparse matrix to the solver and evaluates on the full matrix
    """

    start_time = time.time()
    tours, _, _ = zip(*[solver(tsppc) for tsppc in tsppcs_sparse]) if solver is not None else solver_multiple(tsppcs_sparse)

    # do not rely on self reported times and tour lengths
    avg_time = (time.time() - start_time)/len(tsppcs)
    invalid_tours = [not tsppc_tour_is_valid(tsppc, tour, verbose_if_invalid) for tsppc, tour in zip(tsppcs, tours)]
    # calculate length only for valid tours
    lengths = [tools.tour_length_by_dist_mat(conv.tsppc2dist(tsppc), tour)
        for tsppc, tour, tour_invalid in zip(tsppcs, tours, invalid_tours) if not tour_invalid
    ]

    return np.mean(lengths), avg_time, sum(invalid_tours)


def tsppc_tour_is_valid(tsppc, tour, verbose=False)->bool:
    """ returns True if TSPPC tour is valid """

    if tour == None:
        return False
    # check number of tour steps
    if len(tour) != len(tsppc):
        if verbose:
            print("Number of tour steps is not the same as number of nodes in the TSPPC")
            print(f"len(tsppc): {len(tsppc)}")
            print(f"Tour: {tour}")
        return False
    
    if not all_nodes_visited(tsppc, tour, verbose): return False
    if not all_pred_constr_fullfilled(tsppc, tour, verbose): return False

    return True


# SUB FUNCTIONS

def all_nodes_visited(tsppc, tour, verbose=False):
    """ check if all nodes are visited """

    if not all([node in tour for node in range(len(tsppc))]):
        if verbose: print("Not all nodes were visited")
        return False
    else: return True


def all_pred_constr_fullfilled(tsppc, tour, verbose=False):
    """ check if all precedence constraints are fullfilled """
    successors, predecessors = np.where(tsppc == -1)
    invalid_succ_pred_pairs = [(pred, succ) for pred, succ in zip(predecessors, successors) if tour.index(succ) < tour.index(pred)]

    if len(invalid_succ_pred_pairs) > 0:
        if verbose:
            for pred, succ in invalid_succ_pred_pairs:
                print(f"Successor (node {succ}) is visited before predecessor (node {pred})")
        return False
    else: return True
