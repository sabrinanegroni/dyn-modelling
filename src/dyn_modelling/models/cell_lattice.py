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
import jax.numpy as jnp

def build_lattice(rows:int,cols:int) -> ig.Graph:
    """Build a 2D grid graph"""
    return ig.Graph.Lattice([rows, cols], circular=False)

def compute_distance_matrix(g: ig.Graph) -> list:
    """Return the shortest-path distance matrix for graph g."""
    return g.shortest_paths()

def external_signal(t:float, t_on:float , t_off:float) -> float:
    """Return a step function that is 1 in [t_on, t_off], 0 otherwise."""
    return jnp.where((t >= t_on) & (t <= t_off), 1.0, 0.0)

def compute_weights(dist_matrix: list, neigh_order: int) -> jnp.ndarray:
    """Compute weights w_ij based on distance matrix and neighborhood order."""
    d = jnp.array(dist_matrix, dtype=float)
    mask = (d > 0) & (d <= neigh_order)
    return jnp.where(mask, 1.0 / jnp.where(mask, d, 1.0), 0.0)


def compute_rhs(g:ig.Graph,dist_matrix:list, l_params:np.array, neigh_order:int , t_on:float, t_off:float,a_params:np.array = None) -> callable:
    """
    Return the RHS function f(t, x) for the cell lattice ODE system.

    If a_params is provided  → returns rhs(t, x) for solve_ivp (a_params fixed).
    If a_params is None      → returns rhs(a_params, y, t) for PINN (a_params free).

    Parameters:
    
    g : ig.Graph
        The lattice graph.
    dist_matrix : list
        Shortest-path distance matrix from compute_distance_matrix().
    l_params : np.ndarray
        [l_u, l_v, l_s] velocity parameters.
    neigh_order : int
        Neighborhood order.
    t_on : float
        Time when external signal turns on.
    t_off : float
        Time when external signal turns off.
    a_params : np.ndarray, optional
        [a_u, a_v, a_s, a_us] model parameters.
    """

    #parameters
    l_u, l_v, l_s = l_params

    #number of cells
    N = g.vcount()
    
    #compute weights
    w = compute_weights(dist_matrix, neigh_order)

    def rhs(a_params, t, x):
        a_u, a_v, a_s, a_us = a_params
        x = jnp.array(x)

        u = x[0::3]  # shape (N,)
        v = x[1::3]
        s = x[2::3]

        S_ext_t = external_signal(t, t_on, t_off)

        du = l_u * (-u + a_u/(1+v**2) + S_ext_t * a_us/(1 + (w @ s)**2))
        dv = l_v * (-v + a_v/(1+u**2))
        ds = l_s * (-s + a_s*u**2/(1+u**2))

        return jnp.stack([du, dv, ds], axis=1).reshape(-1)

    # simulation: a_params fixed , wrap for solve_ivp
    if a_params is not None:
        return lambda t, x: np.array(rhs(a_params, t, x))

    # PINN: return generic rhs(a_params, t, x)
    return rhs


def simulate_cell_lattice(rhs:callable, x0:np.array, t_span:tuple, t_eval:np.array) -> np.array:
    """Simulate the cell lattice ODE system."""
    sol = solve_ivp(rhs, t_span, x0, t_eval=t_eval, method='RK45')
    return sol.y