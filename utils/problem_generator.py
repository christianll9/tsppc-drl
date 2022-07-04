import numpy as np
import utils.converter as conv

def generate_coords(n:int, distr="uniform", loc=0.5, uni_width=1, normal_scale=0.167) -> np.array:
    """
    Generate 2D coordinates from a "uniform" (default) or a "normal" distribution.

    n is the number of generated instances
    loc is the average center
    """
    if distr == "uniform":
        coords = np.random.uniform(low=loc-uni_width/2, high=loc+uni_width/2, size=(n,2))
    elif distr == "normal":
        coords = np.random.normal(loc=loc, scale=normal_scale, size=(n,2))
    else:
        raise Exception("Distribution is unknown")
    return coords


def generate_dist_matrix(n:int, distr="uniform", loc=0.5, uni_width=1, normal_scale=100, output_coords=False):
    """
    Generates symmetrical distance matrix
    """
    coords = generate_coords(n, distr, loc, uni_width, normal_scale)
    D = conv.coord2sym_dist(coords)
    if not output_coords: return D
    else: return D, coords



def generate_tsppc_matrix(n_nodes:int=20, distr="uniform", loc=0.5, uni_width=1,
    normal_coords_scale=100, n_constraints=0.33, output_coords=False, atsp=False):
    """
    Generates TSPPC matrix from a random permutation solution and than take some random selected precedences as constraints
    If n_constraints is float it is interpreted as fraction of n_nodes. Int will be interpreted as absolute number of constraints

    Returns a (n_nodes+1, n_nodes+1) matrix, because the matrix contains both - starting and end node (which are the same for our experiments)
    """
    if np.issubdtype(type(n_constraints), np.floating):
        n_constraints = round(n_constraints*n_nodes)

    if n_constraints > ((n_nodes - 2) * ((n_nodes-1)/2)): 
        raise Exception("Too many constraints")

    if not output_coords:
        tsppc = generate_dist_matrix(n_nodes, distr, loc, uni_width, normal_coords_scale, output_coords)
    else:
        tsppc, coords = generate_dist_matrix(n_nodes, distr, loc, uni_width, normal_coords_scale, output_coords)

    atsp_matrix = tsppc.copy()
    tsppc = np.append(tsppc, tsppc[:,0:1], axis=1) #insert starting node twice -> starting node = end node
    last_row = np.repeat(-1, len(tsppc[0]))
    last_row[-1] = 0
    tsppc = np.append(tsppc, last_row[np.newaxis, :], axis=0)
    tsppc[1:,0] = -1 # starting node

    a = np.random.permutation(n_nodes-1)
    random_tour= np.concatenate([[0], a+1, [n_nodes]]).tolist()
    
    # Select random precedence constraints
    for _ in range(n_constraints):
        while True: #do-while loop
            t_idx = np.random.randint(1, len(random_tour)-1, size=2)
            a = min(t_idx)
            b = max(t_idx)
            if a != b and tsppc[random_tour[b], random_tour[a]] != -1:
                break

        tsppc[random_tour[b], random_tour[a]] = -1

    if not output_coords and not atsp: return tsppc
    elif output_coords and not atsp: return tsppc, coords
    elif not output_coords and atsp: return tsppc, atsp_matrix
    else: return tsppc, coords, atsp_matrix


def preprocessed_tsppcs(n_problems: int=1, output_path:str=None, n_nodes:int=200, distr:str="uniform",
    normal_coords_scale:int=100, n_constraints:int=0.33):
    """
    Generate data tuples for feeding the Transformer model.
    depot is counted as one of n_nodes
    If output_path is given, the file is also stored
    """
    tsppcs, coords_lists = zip(*[generate_tsppc_matrix(n_nodes, distr, 0.5, 1, normal_coords_scale, n_constraints, output_coords=True) for _ in range(n_problems)])
    
    return conv.preprocess_tsppcs_for_model(tsppcs, coords_lists, normalize=False, pkl_path=output_path)
