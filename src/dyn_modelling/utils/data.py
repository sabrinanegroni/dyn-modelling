""" Data utilities for cell lattice model and other. """

import numpy as np
from itertools import product
import jax.numpy as jnp
from dyn_modelling.models.cell_lattice import (
    build_lattice, compute_distance_matrix,
    compute_rhs, simulate_cell_lattice, external_signal,
)

def add_noise(data: np.ndarray, noise_level: float) -> np.ndarray:
    """Add Gaussian noise to the data."""
    noise = np.random.normal(0, noise_level, size=data.shape)
    return data + noise



# DEEPONET DATASET GENERATION


def sample_signal_params(n_traj: int, t_span: tuple , rng: np.random.Generator):
    """
    Sample random (t_on, t_off) pairs.
    """
    t_start, t_end = t_span
    min_duration = (t_end - t_start) * 0.05 #minimum duration of signal
    t_on  = rng.uniform(t_start, t_end - min_duration, size=n_traj) #n_traj random start times
    tot_duration = rng.uniform(min_duration, t_end - t_on) #n_traj random durations
    t_off = np.clip(t_on + tot_duration, t_on + min_duration, t_end)
    return t_on, t_off


def build_query_grid(n_cells: int, t_eval: np.ndarray, rescale: bool = True) -> np.ndarray:
    """
    Trunk input for DeepONet: (cell_index, variable_index, time_index).
    If rescale=True, each dimension is rescaled to [0, 1].
    """
    n_vars = 3
    t_min, t_max = t_eval.min(), t_eval.max()

    query_rows = []
    for i, var, step in product(range(n_cells), range(n_vars), range(len(t_eval))):
        if rescale:
            query_rows.append([
                i   / max(n_cells - 1, 1),
                var / (n_vars - 1),
                (t_eval[step] - t_min) / (t_max - t_min),
            ])
        else:
            query_rows.append([
                float(i),
                float(var),
                float(t_eval[step]),
            ])
    return np.array(query_rows, dtype=np.float32)


def generate_deeponet_dataset(
    n_traj: int, rows: int, cols: int,
    l_params: np.ndarray, a_params: np.ndarray,
    neigh_order: int, x0: np.ndarray,
    t_span: tuple, t_eval: np.ndarray,
    seed: int = 0,
    rescale: bool = True,
) -> dict:
    """
    Returns
    -------
    signal  : jnp.ndarray (n_traj, m)           branch input (S_ext sampled at sensor times)
    queries : jnp.ndarray (n_queries, 3)        trunk input (cell_index, variable_index, time_index)
    outputs : jnp.ndarray (n_traj, n_queries)   target values at each query point
    """
    rng = np.random.default_rng(seed)
    g           = build_lattice(rows, cols)
    dist_matrix = compute_distance_matrix(g)
    n_cells     = g.vcount()

    queries   = build_query_grid(n_cells, t_eval, rescale=rescale)
    n_queries = queries.shape[0]
    m         = len(t_eval)

    t_on_all, t_off_all = sample_signal_params(n_traj, t_span, rng)

    signal  = np.zeros((n_traj, m), dtype=np.float32)
    outputs = np.zeros((n_traj, n_queries), dtype=np.float32)

    for k in range(n_traj):
        t_on, t_off = t_on_all[k], t_off_all[k]

        signal[k] = np.array(external_signal(jnp.array(t_eval), t_on, t_off))

        rhs   = compute_rhs(g, dist_matrix, l_params, neigh_order, t_on, t_off, a_params)
        sol_y = simulate_cell_lattice(rhs, x0, t_span, t_eval)

        outputs[k] = sol_y.flatten()

    return {
        "signal":  jnp.array(signal),
        "queries": jnp.array(queries),
        "outputs": jnp.array(outputs),
    }