"""PINN implementation for solving ODEs and predicting parameters of the model."""


import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
import igraph as ig


# ---------------------------------------------------------------------------
# ANN
# ---------------------------------------------------------------------------

def _activation(x: Array) -> Array:
    return jnp.tanh(x)

# Helper function to initialize weights for ANN layers
def _init_matrix(row: int, col: int) -> Array:
    return np.random.randn(row, col) * 1. / np.sqrt(row + col)  # Xavier initialization

# Helper function to initialize biases
def _init_bias(row: int) -> Array:
    return np.zeros(row)


def ANN(params: list, x: Array) -> Array:
    """Forward pass of the ANN."""
    W = params[::2]
    B = params[1::2]
    layer = x.copy()
    for w, b in zip(W[:-1], B[:-1]):
        layer = _activation(w @ layer + b)
    return W[-1] @ layer + B[-1]


def init_params(topology: list[int]) -> list:
    """Initialize ANN parameters for a given topology."""
    params = []
    for t, t_next in zip(topology[:-1], topology[1:]):
        W = _init_matrix(t_next, t)
        b = _init_bias(t_next)
        params.append(W)
        params.append(b)
    return params

# ---------------------------------------------------------------------------
# LOSS FUNCTION
# ---------------------------------------------------------------------------

def _loss(params: list, a_params: np.ndarray, t: Array, y_true: Array,
         lmbd: float, A: Array, N: int) -> Array:
    """
    Compute the PINN loss for a single time point.

    Parameters:
    
    params : list
        ANN parameters.
    a_params : np.ndarray
        Model parameters [a_u, a_v, a_s, a_us].
    t : Array
        Single time point, shape (1,).
    y_true : Array
        True state at time t, shape (3*N,).
    lmbd : float
        Weighting between data loss and physics loss.
        lmbd=1 -> pure data loss, lmbd=0 -> pure physics loss.
    A : Array
        Adjacency matrix of the graph, shape (N, N).
    N : int
        Number of cells.

    Returns:
    
    Array
        Scalar loss value.
    """
    a_u, a_v, a_s, a_us = a_params

    # ANN prediction at time t
    y = ANN(params, t).flatten()                                    # shape (3*N,)

    # Time derivative from ANN (automatic differentiation)
    dy_dt_hat = jax.jacfwd(ANN, argnums=1)(params, t).flatten()    # shape (3*N,)

    # Reshape to (N, 3) -> columns [u, v, s]
    y_matrix = y.reshape((N, 3))
    u = y_matrix[:, 0]
    v = y_matrix[:, 1]
    s = y_matrix[:, 2]

    # Degree of each node
    k = A @ jnp.ones(N)

    # ODE right-hand side
    du_dt_model = -u + a_u / (1.0 + v**2) + a_us / (1.0 + ((A @ s + s) / (k + 1.0))**2)
    dv_dt_model = -v + a_v / (1.0 + u**2)
    ds_dt_model = -s + a_s * (u**2) / (1.0 + u**2)

    # Reconstruct in interleaved layout [u0,v0,s0, u1,v1,s1, ...]
    dy_dt_model = jnp.stack([du_dt_model, dv_dt_model, ds_dt_model], axis=1).reshape(-1)

    # Physics loss + data loss
    phys_loss = jnp.sum((dy_dt_hat - dy_dt_model)**2)
    data_loss = jnp.sum((y - y_true)**2)

    return (1.0 - lmbd) * phys_loss + lmbd * data_loss


# ---------------------------------------------------------------------------
# PINN - batch loss and gradients
# ---------------------------------------------------------------------------

def make_pinn(g: ig.Graph, N: int):
    """
    Batch loss and gradient functions.

    Parameters
    ----------
    g : ig.Graph
        The lattice graph.
    N : int
        Number of cells.

    Returns
    -------
    batch_loss : callable  (params, a, T, Y, lmbd) -> scalar
    grad_fn    : callable  (params, a, T, Y, lmbd) -> (grad_params, grad_a)
    """
    A = jnp.array(g.get_adjacency().data)

    def _loss_single(params, a, t, y_true, lmbd):
        return _loss(params, a, t, y_true, lmbd, A, N)

    @jax.jit
    def batch_loss(params: list, a: np.ndarray, T: Array,
                   Y: Array, lmbd: float) -> Array:
        """Mean loss over a batch of time points."""
        return jax.vmap(lambda t, y: _loss_single(params, a, t, y, lmbd))(T, Y).mean()

    grad_fn = jax.jit(jax.grad(batch_loss, argnums=(0, 1)))

    return batch_loss, grad_fn


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------

def train(ann_params: list, a: np.ndarray, T: Array, Y: Array,
          batch_loss, grad_fn,
          lmbd: float = 0.5, n_epochs: int = 50_000,
          lr_ann: float = 1e-3, lr_model: float = 1e-1,
          momentum: float = 0.9) -> tuple:
    """
    Train the PINN using SGD with momentum.

    Parameters:
    
    ann_params : list
        Initial ANN parameters from init_params().
    a : np.ndarray
        Initial model parameters [a_u, a_v, a_s, a_us].
    T : Array
        Time points, shape (n_samples, 1).
    Y : Array
        Observations, shape (n_samples, 3*N).
    batch_loss : callable
        From make_pinn().
    grad_fn : callable
        From make_pinn().
    lmbd : float
        Loss weighting parameter (0=physics only, 1=data only).
    n_epochs : int
        Number of training steps.
    lr_ann : float
        Learning rate for ANN parameters.
    lr_model : float
        Learning rate for model parameters a.
    momentum : float
        Momentum coefficient for SGD.

    Returns:
    
    ann_params : list   trained ANN parameters
    a_list : list       history of model parameters
    loss_history : list loss at each step
    """
    mom = [p * 0 for p in ann_params]
    loss_history = []
    a_list = []

    for i in range(n_epochs):
        g_params, g_model = grad_fn(ann_params, a, T, Y, lmbd)

        for j in range(len(g_params)):
            mom[j] = momentum * mom[j] - lr_ann * g_params[j]
            ann_params[j] += mom[j]

        a -= lr_model * g_model
        a_list.append(a.copy())
        loss_history.append(float(batch_loss(ann_params, a, T, Y, lmbd)))

        if i % max(1, n_epochs // 10) == 0: #print every 10% of epochs
            print(f"[{i:>6}] loss = {loss_history[-1]:.6f} | a = {a_list[-1]}")

    return ann_params, a_list, loss_history