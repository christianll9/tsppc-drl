import numpy as np
from scipy.spatial import distance_matrix
import os
import re
import pickle

def cmdscale(D):
    """   
    Classical multidimensional scaling (MDS)
    Copyright © 2014-9 Francis Song (http://www.nervouscomputer.com/hfs/cmdscale-in-python/)

    Parameters
    ----------
    D : (n, n) array
        Symmetric distance matrix.

    Returns
    -------
    Y : (n, p) array
        Configuration matrix. Each column represents a dimension. Only the
        p dimensions corresponding to positive eigenvalues of B are returned.
        Note that each dimension is only determined up to an overall sign,
        corresponding to a reflection.

    e : (n,) array
        Eigenvalues of B.

    """
    # Number of points
    n = len(D)
 
    # Centering matrix
    H = np.eye(n) - np.ones((n, n))/n

    # YY^T
    B = -H.dot(D**2).dot(H)/2
 
    # Diagonalize
    evals, evecs = np.linalg.eigh(B)
 
    # Sort by eigenvalue in descending order
    idx   = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:,idx]
 
    # Compute the coordinates using positive-eigenvalued components only 
    w, = np.where(evals > 0)
    L  = np.diag(np.sqrt(evals[w]))
    V  = evecs[:,w]
    Y  = V.dot(L)
 
    return Y, evals


def coord2sym_dist(coords:list):
    """
    Converts list of 2D coordinates to euclidean distance matrix
    """
    return distance_matrix(coords, coords)


def dist2coord(D):
    """
    Converts symmetrical distance matrix to 2D coordinates
    """
    Y, _ = cmdscale(D)
    return Y[:,0:2]


def tsppc2dist(tsppc):
    """
    Converts TSPPC matrix to distance matrix by removing
    "-1" and replacing them with the corresponding other distance
    """
    mirrored_tsppc = np.rot90(np.fliplr(tsppc))
    return np.where(tsppc==-1, mirrored_tsppc, tsppc)

def norm_coords2unit_sq(coords:list):
    """
    Normalize 2D coords to the unit square
    """
    norm_coords = coords - np.min(coords)
    norm_factor = np.max(norm_coords)
    norm_coords /= norm_factor
    return norm_coords, norm_factor


def norm_tsppc(tsppc, coords=None):
    """
    Normalize symmetric TSPPC to process it with our model (not in-place)
    """
    assert tsppc[0,-1] == 0, "Not compatible TSPPC"

    if coords is None:
        D = tsppc2dist(tsppc)
        coords = dist2coord(D)

    if len(tsppc) == len(coords)+1:
        # add end node coordinate
        coords = np.append(coords, [coords[0]], axis=0)

    norm_coords, norm_factor = norm_coords2unit_sq(coords)
    norm_tsppc = coord2sym_dist(norm_coords)

    if np.round(norm_tsppc[0,-1],6) != 0:
        print("Start and end node differ significantly! Could be problematic")
    norm_tsppc[0,-1] = 0
    norm_tsppc[np.where(tsppc == -1)] = -1
    return norm_tsppc, norm_coords, norm_factor


def preprocess_tsppc_for_model(tsppc, coords=None, normalize:bool=True) -> tuple:
    """
    Converts a single TSPPC matrix to a list of coords, and multiple compatibility masks:

    `att_mask`:   Masking everything except attentions between predecessor and successor
    `group_mask`: Masking everything except attentions within a "precedence chain group"

    Arguments:
    `normalize`: If true, given coords will be normalized to unit square. If coords are not given,
    normalization will always happen.
    """
    if coords is None:
        normalize = True # to compensate effects caused by MDS function cmdscale
        D_asym = tsppc2dist(tsppc)
        coords = dist2coord(D_asym)

    if len(coords) != len(tsppc) and len(coords)+1 != len(tsppc): # Either the coords of the end node are given or  
        print(f"len(coords)={len(coords)}")
        print(f"len(tsppc)={len(tsppc)}")
        raise Exception("The coordinate list has to have a length of len(tsppc)-1 " + 
        "(if the coordinates of the ending node are given) or a length of len(tsppc) " +
        "(if the coordinates of the ending node are retained) Please check your problem.")

    # Checking the assumption that node 0 is the starting node
    # if not -> problem can not be solved (yet)
    assert np.all(tsppc[1:,0] == -1), "Could not find starting node/depot"
    
    # Normalization
    if normalize:
        tsppc, coords, _ = norm_tsppc(tsppc, coords)

    if tsppc[0,-1] == 0 and np.all(tsppc[-1,:-1] == -1):
        #starting node = end node
        tsppc = tsppc[:-1, :-1] # remove end node, because the depot is already the end node
        if len(coords) == len(tsppc) + 1:
            coords = coords[:-1] # remove end node also in coords list
    else:
        if tsppc[0,-1] != 0:
            raise Exception(f"Distance >0 between starting and end node. tsppc[0,-1]={tsppc[0,-1]}")
        elif np.all(tsppc[-1,:-1] == -1):
            raise Exception(f"End node is not successor of all other nodes. tsppc[-1,:-1]={tsppc[-1,:-1]}")


    # Searching for precedence constraints and converting them
    att_mask = tsppc == -1

    # Preprocessing: Creating group masks
    group_mask = np.zeros(tsppc.shape, dtype=bool)
    group_mask[:-1, 1:] = tsppc[:-1, 1:] == -1 # -1 => 1, Rest => 0
    group_mask |= np.rot90(np.fliplr(group_mask)) # mirror every 1
    for row in group_mask:
        ones = np.where(row == True)[0]
        if len(ones) >= 2:
            rows = np.repeat(ones, len(ones)-1)
            cols = np.delete(np.tile(ones, len(ones)), slice(None, None, len(ones)+1))
            group_mask[rows, cols]=True

    return coords, att_mask, group_mask



def preprocess_tsppcs_for_model(tsppcs:list, coords_lists:list=None, normalize:bool=True, pkl_path:str=None) -> tuple:
    """
    Converts multiple TSPPC matrices to multiple lists of coords, and multiple compatibility masks
    If pkl_path is specified, the dataset is stored as a pkl-file
    """
    if coords_lists is None:
        coords_lists = [None]*len(tsppcs) # for passing into zip
    assert type(tsppcs) == list or type(tsppcs) == tuple, "tsppcs is not a list or a tuple"
    datasets = [preprocess_tsppc_for_model(tsppc, coords, normalize) for tsppc, coords in zip(tsppcs, coords_lists)]
    if pkl_path is not None: store_datasets(datasets, pkl_path)
    return datasets



def tsp2tsplib_format(coords: np.array, output_path:str=None) -> str:
    """
    Returns TSP in the TSPLIB TSP format. Coordinates has to be integer!
    Can be stored in `output_path`
    """
    metadata = [
        ("NAME", os.path.basename(output_path) if output_path is not None else "Dummy"),
        ("COMMENT", "Generated by the Student Research Project 2021 - University Hildesheim"),
        ("TYPE", "TSP"),
        ("DIMENSION", len(coords)),
        ("EDGE_WEIGHT_TYPE", "EUC_2D")
    ]
    output_str = ""
    for attr, val in metadata:
        output_str += f"{attr}: {val}\n"
    output_str += "NODE_COORD_SECTION\n"

    assert coords.dtype == np.integer, "coords has to be integers"

    MAX_N_DIGITS = int(np.log10(len(coords))) + 1
    MAX_X_COORD_DIGITS = int(np.log10(np.max(coords, axis=0)[0])) + 1
    MAX_Y_COORD_DIGITS = int(np.log10(np.max(coords, axis=0)[1])) + 1

    for n, coord in enumerate(coords):
        output_str += str(n+1).rjust(MAX_N_DIGITS, " ") + " " + str(coord[0]).rjust(MAX_X_COORD_DIGITS, " ") + " " + str(coord[1]).rjust(MAX_Y_COORD_DIGITS, " ") + "\n"

    output_str += "EOF\n"

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(output_str)

    return output_str


def tsppc2tsplib_format(tsppc: np.array, output_path:str=None) -> str:
    """
    Returns TSPPC matrix in the TSPLIB SOP format
    Can be stored in `output_path`
    """
    metadata = [
        ("NAME", os.path.basename(output_path) if output_path is not None else "Dummy"),
        ("TYPE", "SOP"),
        ("COMMENT", "Generated by the Student Research Project 2021 - University Hildesheim"),
        ("DIMENSION", len(tsppc)),
        ("EDGE_WEIGHT_TYPE", "EXPLICIT"),
        ("EDGE_WEIGHT_FORMAT", "FULL_MATRIX"),
    ]
    output_str = ""
    for attr, val in metadata:
        output_str += f"{attr}: {val}\n"
    output_str += f"EDGE_WEIGHT_SECTION\n{len(tsppc)}\n"

    for row in tsppc:
        output_str += ' '.join(map(lambda x: "%d\t" % (x) if x == int(x) else "%f\t" % (x), row)) + "\n"

    if output_path is not None:
        with open(output_path, "w") as f:
            f.write(output_str)

    return output_str


def store_datasets(datasets: list, output_path: str = "data.pkl"):
    """
    Stores a list of datasets in a .pkl file
    """
    dir_path = os.path.dirname(output_path)
    if len(dir_path)>0 and not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    if not output_path.endswith(".pkl"):
        output_path += ".pkl"

    with open(output_path, 'wb') as f:
        pickle.dump(datasets, f, pickle.HIGHEST_PROTOCOL)


def load_tsppc(filepath: str):
    """
    Loads a .tsppc file in TSPLib format and returns a numpy matrix
    """
    with open(filepath, "r") as f:
        lines = f.readlines()

    # search for useful lines with white spaces
    regex = r" *[-\d.]+ +[-\d.eE+]+" # regex for identifying a matrix line with white spaces
    filtered_lines = [line for line in lines if re.match(regex, line) is not None]
    sep = " "

    if len(filtered_lines) == 0:
        # search for useful lines with tabs
        regex = r"\t*[-\d.eE+]+\t[-\d.eE+]+" # regex for identifying a matrix line with tabs
        filtered_lines = [line for line in lines if re.match(regex, line) is not None]

        if len(filtered_lines) == 0:
            return np.empty(shape=(0,0)) # if no useful lines were found
        sep = "\t"

    filtered_lines = [line for line in filtered_lines if re.match(regex, line).pos == 0] #has to start with the regex

    if any("." in line for line in filtered_lines):
        dtype = np.float
    else:
        dtype = np.int


    arrays = [np.fromstring(line, sep=sep, dtype=dtype) for line in filtered_lines]
    return np.vstack(list(arrays))