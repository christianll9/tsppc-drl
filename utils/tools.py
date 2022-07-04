import numpy as np
import utils.converter as conv

def tour_length_by_coords(coords: list, tour: list):
    total_dist = 0
    pos1 = coords[tour[0]]
    for next_node_idx in tour[1:]:
        pos2 = coords[next_node_idx]
        total_dist += np.linalg.norm(pos2 - pos1)
        pos1 = pos2
        
    return total_dist

def tour_length_by_dist_mat(tsppc: np.array, tour: list):
    """
    Can be used for TSPPCs, ATSPs and TSPs
    TSPPC Feature: if one distance is negative, the tour length is inf
    """ 

    total_dist = 0
    node_idx = tour[0]
    for next_node_idx in tour[1:]:
        dist = tsppc[node_idx, next_node_idx]
        if dist >= 0:
            total_dist += dist
        else:
            return np.inf
        node_idx = next_node_idx
    return total_dist

def scale_tsppc(tsppc: np.array, scale_factor: int):
    """
    This function is needed to solve a TSPPC with small values with LKH-3.
    It scales all values except the -1s
    """
    return np.where(tsppc == -1, -1, tsppc*scale_factor)


def get_all_subtours_from_tour(tour):
    """
    Return all sub tours which start when vehicle enters the deport again.
    Subtour list exclude the depot (start and end node)
    """
    tour_ = np.array(tour)
    depot_indices = np.where(tour_ == 0)[0]
    subtours = np.split(tour_, depot_indices)
    subtours = [subtour[(subtour != tour_[0]) & (subtour != tour_[-1])] for subtour in subtours if len(subtour)>1] # remove depots from subtours
    return subtours

############################# PLOTTING #############################

def _add_tour_to_plot(ax, fig, tour:list, coords:np.array, title:bool=True, color:str="black", animate=False):
    """ Internal method """
    n_nodes = len(coords)-1 if np.sum(coords[-1] - coords[0]) < 1E-10 else len(coords)
    if title:
        ax.set_title(f"{n_nodes} nodes / Total length: {round(tour_length_by_coords(coords, tour),3)}")
    
    def update(i):
        if i == 0: return
        ax.annotate("", xy=coords[tour[i-1]], xycoords='data', xytext=coords[tour[i]], textcoords='data',
            arrowprops=dict(arrowstyle="<-", color=color, connectionstyle="arc3"))
    
    if not animate:
        for i in range(len(coords)): update(i)
        return ax,
    else:
        import matplotlib.animation as animation
        anim = animation.FuncAnimation(fig, update, frames=range(len(coords)), interval=1000)
        return ax, anim


def _add_tsppc_problem_to_plot(ax, tsppc, coords:np.array=None):
    """ Internal method """
    for i in range(len(tsppc)-1):
        for j in range(1,len(tsppc)):
            if tsppc[i,j] == -1:
                ax.annotate("", xy=coords[i], xycoords='data', xytext=coords[j], textcoords='data',
                    arrowprops=dict(arrowstyle="->", linestyle="dotted", linewidth=0.5, color="red", connectionstyle="arc3"))
    return ax


def show_coords(coords:np.array, same_scaling:bool=True):
    """
    coords: (n,2) array containing all 2D coordinates for all n nodes
    same_scaling: if true, x and y have the same scaling
    """
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()

    # don't show start and end node, if they are (almost) the same
    if np.sum(coords[-1] - coords[0]) < 1E-10:
        coords = coords[:-1]

    ax.scatter(coords[1:, 0], coords[1:, 1])
    ax.scatter(coords[0, 0], coords[0, 1], c="r")

    # axis scaling
    if same_scaling:
        ax.set_aspect('equal', adjustable='box')

    for idx, coord in enumerate(coords):
        ax.annotate(str(idx), xy=coord, xytext=(5, -3), fontsize="small", textcoords="offset pixels")
    
    return fig, ax


def show_tsp_tour(tour:list, coords:np.array, same_scaling:bool=True, title:bool=True, animate=False):
    """
    tour: ordered list of node incides, which will be visited
    coords: (n,2) array containing all 2D coordinates for all n nodes
    same_scaling: if true, x and y have the same scaling
    """
    fig, ax = show_coords(coords, same_scaling)
    return fig, *_add_tour_to_plot(ax, fig, tour, coords=coords, title=title, animate=animate)


def show_tsppc_problem(tsppc, orig_coords:np.array=None, same_scaling:bool=True):
    """
    coords: (n,2) array containing all 2D coordinates for all n nodes
    tour: ordered list of node incides, which will be visited

    Source: https://stackoverflow.com/a/46507090/11227118
    """
    if orig_coords is not None:
        coords = orig_coords
        #coords = np.vstack((coords, coords[0])) #starting node = end node
    else:
        coords = conv.dist2coord(conv.tsppc2dist(tsppc))

    fig, ax = show_coords(coords, same_scaling)
    ax = _add_tsppc_problem_to_plot(ax, tsppc, coords)

    return fig, ax


def show_tsppc_tour(tsppc, tour:list, orig_coords:np.array=None, same_scaling:bool=True,
    title:bool=True, animate=False):
    """
    coords: (n,2) array containing all 2D coordinates for all n nodes
    tour: ordered list of node incides, which will be visited

    returns: (fig, ax) if not animate or (fig, ax, anm) if animate

    Source: https://stackoverflow.com/a/46507090/11227118
    """
    if orig_coords is not None:
        coords = orig_coords
        #coords = np.vstack((coords, coords[0])) #starting node = end node
    else:
        coords = conv.dist2coord(conv.tsppc2dist(tsppc))
    
    fig, ax = show_tsppc_problem(tsppc, coords, same_scaling)

    return fig, *_add_tour_to_plot(ax, fig, tour, coords=coords, title=title, animate=animate)