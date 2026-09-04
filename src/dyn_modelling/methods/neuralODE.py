import jax
import jax.numpy as jnp
import numpy as np
from jax import Array
import equinox as eqx
import diffrax
import optax


# ---------------------------------------------------------------------------
# VECTOR FIELD - no parameters prediction
# ---------------------------------------------------------------------------

def init_vf_params(data_size: int, width: int, depth: int) -> list:
    """
    Initialize vector field network parameters.

    Parameters
    ----------
    data_size : int     state dimension
    width : int         hidden layer size
    depth : int         number of hidden layers
    """
    input_size = data_size + 1 # state + time
    topology = [input_size] + [width] * depth + [data_size]

    params = []
    for t_in, t_out in zip(topology[:-1], topology[1:]):
        W = np.random.randn(t_out, t_in) * 0.01  # small random weights
        b = np.zeros(t_out)
        params.append(W)
        params.append(b)
    return params


def vf_forward(vf_params: list, x: Array, t: Array) -> Array:
    
    """
    Forward pass of the vector field network.
    """
    W = vf_params[::2]
    B = vf_params[1::2]

    # concatenate state + time as input
    layer = jnp.concatenate([x, t.reshape(1)])

    for w, b in zip(W[:-1], B[:-1]):
        layer = jnp.tanh(w @ layer + b)
    return W[-1] @ layer + B[-1]



# ---------------------------------------------------------------------------
# ODE SOLVER
# ---------------------------------------------------------------------------

def solve_node(vf_params: list, y_0: Array, ts: Array) -> Array:
    """
    Integrate dx/dt = vf(x, t) from ts[0] to ts[-1].

    Returns
    -------
    ys : Array  shape (len(ts), data_size)
    """
    def vector_field(t, y, args):
        return vf_forward(vf_params, y, jnp.array([t]))

    solution = diffrax.diffeqsolve(
        diffrax.ODETerm(vector_field),
        diffrax.Tsit5(),
        t0=ts[0],
        t1=ts[-1],
        dt0 = None, 
        y0=y_0,
        stepsize_controller=diffrax.PIDController(rtol=1e-3, atol=1e-6),
        saveat=diffrax.SaveAt(ts=ts),
        max_steps= 1000
    )
    return solution.ys


# ---------------------------------------------------------------------------
# LOSS AND GRADIENT
# ---------------------------------------------------------------------------

def _loss(vf_params: list, y0: Array, ts: Array, y_true: Array) -> Array:
    """MSE between predicted and true trajectories."""
    y_pred = solve_node(vf_params, y0, ts)
    return jnp.mean((y_pred - y_true) ** 2)


def make_neural_ode():
    loss_fn = eqx.filter_jit(_loss)
    grad_fn = eqx.filter_jit(jax.grad(_loss, argnums=0))
    return loss_fn, grad_fn


# ---------------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------------
def train_adam(vf_params, y0, ts, y_true, loss_fn, grad_fn, n_epochs, lr):
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adam(lr)
    )
    opt_state = optimizer.init(vf_params)
    loss_history = []

    for i in range(n_epochs):
        grads = grad_fn(vf_params, y0, ts, y_true)
        updates, opt_state = optimizer.update(grads, opt_state)
        vf_params = optax.apply_updates(vf_params, updates)

        loss_history.append(float(loss_fn(vf_params, y0, ts, y_true)))

        if i % max(1, n_epochs // 10) == 0:
            print(f"[{i:>6}] loss = {loss_history[-1]:.6f}")

    return vf_params, loss_history

def train_SGD(vf_params: list, y0: np.ndarray, ts: np.ndarray, y_true: np.ndarray,
          loss_fn, grad_fn,
          n_epochs: int = 10_000,
          lr: float = 1e-3,
          momentum: float = 0.9) -> tuple:
    """
    Train the Neural ODE using SGD with momentum.

    """
    mom = [p * 0 for p in vf_params]
    loss_history = []

    ts = jnp.array(ts)
    y0 = jnp.array(y0)
    y_true = jnp.array(y_true)

    for i in range(n_epochs):
        grads = grad_fn(vf_params, y0, ts, y_true)

        for j in range(len(grads)):
            mom[j] = momentum * mom[j] - lr * grads[j]
            vf_params[j] += mom[j]

        loss_history.append(float(loss_fn(vf_params, y0, ts, y_true)))

        if i % max(1, n_epochs // 10) == 0:
            print(f"[{i:>6}] loss = {loss_history[-1]:.6f}")

    return vf_params, loss_history

#----------------------------------------------------------------------------
# PREDICTION ERRORS
#----------------------------------------------------------------------------

def per_variable_error(y_pred, y_true, N):
    """Relative L2 error per variable (u, v, s)"""
    return {
        name: float(jnp.linalg.norm(y_pred[:, i::3] - y_true[:, i::3])
                     / jnp.linalg.norm(y_true[:, i::3]))
        for i, name in enumerate(['u', 'v', 's'])
    }


def per_cell_error(y_pred, y_true, N):
    """Relative L2 error per cell"""
    per_cell = jnp.zeros(N)
    for cell in range(N):
        idx = jnp.array([3 * cell, 3 * cell + 1, 3 * cell + 2])
        per_cell = per_cell.at[cell].set(
            jnp.linalg.norm(y_pred[:, idx] - y_true[:, idx]) / jnp.linalg.norm(y_true[:, idx])
        )
    return per_cell