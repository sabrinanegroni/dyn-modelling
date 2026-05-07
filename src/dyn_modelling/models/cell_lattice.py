"""
Building and simulating cell lattice model.

Grid of N = rows x cols cells, each with 3 variables (u, v, s).
Equations:
    du_i/dt = l1 * (-u_i + au/(1+v_i^2) + S_ext(t) * aus/(1 + (sum_j w_ij s_j)^2))
    dv_i/dt = l2 * (-v_i + av/(1+u_i^2))
    ds_i/dt = l3 * (-s_i + as*u_i^2/(1+u_i^2))
"""


import numpy as np
import igraph as ig
from scipy.integrate import solve_ivp


def build_lattice(rows:int,cols:int) -> ig.Graph:
    """Build a 2D grid graph"""
    return ig.Graph.Lattice([rows, cols], circular=False)

def compute_distance_matrix(g: ig.Graph) -> list:
    """Return the shortest-path distance matrix for graph g."""
    return g.shortest_paths()

def external_signal(t:float, t_on:float , t_off:float) -> float:
    """Return a step function that is 1 in [t_on, t_off], 0 otherwise."""
    return 1.0 if (t_on <= t <= t_off) else 0.0

def compute_weights(dist_matrix: list, neigh_order: int) -> np.ndarray:
    """Compute weights w_ij based on distance matrix and neighborhood order."""
    d = np.array(dist_matrix, dtype=float)
    w = np.where((d > 0) & (d <= neigh_order), 1.0 / d, 0.0)
    return w


def compute_rhs(g:ig.Graph,dist_matrix:list, a_params:np.array, l_params:np.array, neigh_order:int , t_on:float, t_off:float) -> callable:
    """
    Return the RHS function f(t, x) for the cell lattice ODE system.

    Parameters
    ----------
    g : ig.Graph
        The lattice graph.
    dist_matrix : list
        Shortest-path distance matrix from compute_distance_matrix().
    a_params : np.ndarray
        [a_u, a_v, a_s, a_us]
    l_params : np.ndarray
        [l_u, l_v, l_s] velocity parameters.
    neigh_order : int
        Neighborhood order.
    """

    #parameters
    a_u, a_v, a_s, a_us = a_params
    l_u, l_v, l_s = l_params

    #number of cells
    N = g.vcount()
    
    #compute weights
    w = compute_weights(dist_matrix, neigh_order)

    def rhs(t:float, x:np.array) -> np.array:
        
        s_vals = x[2::3]  # Extract s_i values for all cells

        S_ext_t = external_signal(t, t_on, t_off) 
     
        dx_dt = np.zeros_like(x)
        
        for i in range(N):
        
            ui = x[3 * i]
            vi = x[3 * i + 1]
            si = x[3 * i + 2]

            dx_dt[3*i] = l_u * (-ui + a_u/(1+vi**2) + S_ext_t * a_us/(1 + (np.sum(w[i]*s_vals))**2))
            dx_dt[3*i + 1] = l_v * (-vi + a_v/(1+ui**2))
            dx_dt[3*i + 2] = l_s * (-si + a_s*ui**2/(1+ui**2))

        return dx_dt

    return rhs


def simulate_cell_lattice(rhs:callable, x0:np.array, t_span:tuple, t_eval:np.array) -> np.array:
    """Simulate the cell lattice ODE system."""
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method='RK45')
    return sol.y