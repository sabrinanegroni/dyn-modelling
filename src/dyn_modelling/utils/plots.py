"""Plotting utilities for cell lattice model and methods results."""

import numpy as np
import matplotlib.pyplot as plt
import igraph as ig

# ---------------------------------------------------------------------------
# Cell lattice
# ---------------------------------------------------------------------------

def plot_graph(g: ig.Graph, figsize: tuple = (5, 5)) -> None:
    """Plot the lattice graph."""
    fig, ax = plt.subplots(figsize=figsize)
    ig.plot(g, target=ax)
    ax.set_title(f"Lattice graph — {g.vcount()} nodes")
    plt.show()


def plot_trajectories(t: np.ndarray, y: np.ndarray, figsize: tuple = (18, 5)) -> None:
    """
    Plot u, v, s trajectories for all cells.

    Parameters
    ----------
    t : np.ndarray  shape (n_steps,)
    y : np.ndarray  shape (3*N, n_steps) interleaved layout [u0,v0,s0, u1,v1,s1,...]
    """
    N = y.shape[0] // 3
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)

    for i in range(N):
        ax1.plot(t, y[3 * i, :], "-")
        ax2.plot(t, y[3 * i + 1, :], "-")
        ax3.plot(t, y[3 * i + 2, :], "-")

    for ax, label in zip([ax1, ax2, ax3], ["u", "v", "s"]):
        ax.set_title(f"{label} variables")
        ax.set_xlabel("Time")
        ax.set_ylabel(label)

    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# PINN training
# ---------------------------------------------------------------------------

def plot_loss(loss_history: list, figsize: tuple = (10, 4)) -> None:
    """Plot training loss over iterations."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(loss_history)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Training loss")
    ax.set_yscale("log")
    plt.tight_layout()
    plt.show()


def plot_parameter_evolution(a_list: np.ndarray, a_true: np.ndarray,
                              parameter_names: list[str] = None,
                              title: str = "Parameter evolution",
                              figsize: tuple = (10, 6)) -> None:
    """
    Plot predicted parameter evolution vs true values.

    Parameters
    ----------
    a_list : np.ndarray
        shape (n_epochs, n_params) — history of predicted parameters.
    a_true : np.ndarray
        shape (n_params,) — true parameter values.
    parameter_names : list[str]
        Names for each parameter. Defaults to ['a_u', 'a_v', 'a_s', 'a_us'].
    title : str
        Plot title.
    """
    if parameter_names is None:
        parameter_names = ["a_u", "a_v", "a_s", "a_us"]

    colors = ["blue", "green", "red", "orange"]
    a_list = np.array(a_list)

    fig, ax = plt.subplots(figsize=figsize)

    for i in range(a_list.shape[1]):
        ax.plot(a_list[:, i], color=colors[i], label=f"Predicted {parameter_names[i]}")

    for i in range(len(a_true)):
        ax.plot(np.ones(len(a_list)) * a_true[i], linestyle="--",
                color=colors[i], label=f"True {parameter_names[i]}")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter values")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    plt.show()


def plot_predictions(t: np.ndarray, y_pred: np.ndarray, y_true: np.ndarray,
                        figsize: tuple = (10, 15)) -> None:
    """
    Plot predicted vs true u, v, s trajectories.

    Parameters
    ----------
    t : np.ndarray
        Time points, shape (n_steps,).
    y_pred : np.ndarray
        ANN predictions, shape (n_steps, 3*N) interleaved layout.
    y_true : np.ndarray
        True observations, shape (n_steps, 3*N) interleaved layout.
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    u_pred = y_pred[:, 0::3]
    v_pred = y_pred[:, 1::3]
    s_pred = y_pred[:, 2::3]

    u_true = y_true[:, 0::3]
    v_true = y_true[:, 1::3]
    s_true = y_true[:, 2::3]

    fig, axs = plt.subplots(3, 1, figsize=figsize)

    for label, pred, true, ax in zip(["u", "v", "s"],
                                    [u_pred, v_pred, s_pred],
                                    [u_true, v_true, s_true],
                                    axs):
        for i in range(true.shape[1]):
            ax.plot(t, true[:, i], "--", color=colors[i % len(colors)], label=f"True {label}{i+1}")
            ax.plot(t, pred[:, i], color=colors[i % len(colors)], label=f"Predicted {label}{i+1}")
        ax.set_title(f"True vs Predicted {label}", fontsize=16)
        ax.set_xlabel("Time", fontsize=14)
        ax.set_ylabel("Value", fontsize=14)
        ax.tick_params(axis="both", labelsize=12)
        ax.legend(fontsize=13, loc="best")

    plt.tight_layout()
    plt.show()