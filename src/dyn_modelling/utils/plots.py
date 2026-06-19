"""Plotting utilities for cell lattice model and methods results."""

import numpy as np
import matplotlib.pyplot as plt
import igraph as ig
from matplotlib.colors import LogNorm
from itertools import product

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
# PINN
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

            label_true = "True" if i == 0 else None
            label_pred = "Predicted" if i == 0 else None

            ax.plot(t, true[:, i], "--", color="steelblue", linewidth=1.5, alpha=0.8,label=label_true,)

            ax.plot(t, pred[:, i],"-", color="crimson", linewidth=1.5, alpha=0.8, label=label_pred)

    plt.tight_layout()
    plt.show()

#----------------------------------------------------------------------------
# NEURAL ODE
#----------------------------------------------------------------------------


def plot_phase_portrait(y: np.ndarray, y_pred: np.ndarray = None,
                        n_cells: int = None,
                        figsize: tuple = (18, 12)) -> None:
    """
    Plot phase portrait u vs v for each cell.

    Parameters
    ----------
    y : np.ndarray
        True trajectories
    y_pred : np.ndarray, optional
        Predicted trajectories.
        
    n_cells : int, optional
        Number of cells to plot. Defaults to all.
    """
    N = y.shape[1] // 3
    if n_cells is None:
        n_cells = N

    ncols = 5
    nrows = int(np.ceil(n_cells / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    axes = axes.flatten()

    for i in range(n_cells):
        u_true = y[:, 3 * i]
        v_true = y[:, 3 * i + 1]
        axes[i].plot(u_true, v_true, "-", color="steelblue", label="true")

        if y_pred is not None:
            u_pred = y_pred[:, 3 * i]
            v_pred = y_pred[:, 3 * i + 1]
            axes[i].plot(u_pred, v_pred, "--", color="crimson", label="predicted")

        axes[i].set_title(f"Cell {i+1}", fontsize=9)
        axes[i].set_xlabel("u", fontsize=8)
        axes[i].set_ylabel("v", fontsize=8)
        if i == 0:
            axes[i].legend(fontsize=7)

    # hide unused axes
    for j in range(n_cells, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Phase portrait u vs v", fontsize=14)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# DMD
# ---------------------------------------------------------------------------

def plot_dmd_reconstruction(t: np.ndarray, X_true: np.ndarray, X_rec: np.ndarray, title: str = "DMD reconstruction",
                        figsize: tuple = (18, 5)) -> None:
    """
    Plot DMD reconstruction vs true trajectories.

    Parameters
    ----------
    t : np.ndarray
        Time points, shape (n_steps,).
    X_true : np.ndarray
        True trajectories.
    X_rec : np.ndarray
        DMD reconstruction.
    title : str, optional
        Title for the plot.
    figsize : tuple, optional
        Figure size.
    """
    N = X_true.shape[1] // 3
    fig, (ax1, ax2, ax3) = plt.subplots(ncols=3, figsize=figsize)
    for i in range(N):
        label_true = "True" if i == 0 else None
        label_dmd = "DMD" if i == 0 else None

        ax1.plot(t, X_true[:, 3 * i], "--", color="steelblue", label=label_true)
        ax1.plot(t, X_rec[:, 3 * i], "-", color="crimson", label=label_dmd)

        ax2.plot(t, X_true[:, 3 * i + 1], "--", color="steelblue", label=label_true)
        ax2.plot(t, X_rec[:, 3 * i + 1], "-", color="crimson", label=label_dmd)

        ax3.plot(t, X_true[:, 3 * i + 2], "--", color="steelblue", label=label_true)
        ax3.plot(t, X_rec[:, 3 * i + 2], "-", color="crimson", label=label_dmd)


    for ax in [ax1, ax2, ax3]:
        ax.set_xlabel("Time")
        ax.legend(fontsize=7)


    ax1.set_title("u variables")
    ax2.set_title("v variables")
    ax3.set_title("s variables")

    plt.tight_layout()
    plt.show()


def plot_eigenvalues(eigs: np.ndarray, title: str = "DMD eigenvalues",
                     figsize: tuple = (6, 6)) -> None:
    """Plot DMD eigenvalues in the complex plane."""
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(np.real(eigs), np.imag(eigs), "o", markersize=8, label="eigenvalues")

    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), "k--", linewidth=1, label="unit circle")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)

    ax.set_xlabel("Re(λ)")
    ax.set_ylabel("Im(λ)")
    ax.set_title(title)
    ax.axis("equal")
    ax.legend()
    plt.tight_layout()
    plt.show()


from itertools import product
from matplotlib.colors import LogNorm


def plot_param_heatmap(results: dict, row_values: list, col_values: list, metric_key: str,
                        row_label: str = "row", col_label: str = "svd_rank",
                        title: str = "Reconstruction error", cmap: str = "RdYlGn_r",
                        log_scale: bool = False, vmin: float = None, vmax: float = None,
                        fmt: str = ".3f", figsize: tuple = (10, 5)) -> None:
    """Plot a heatmap of a scalar metric over a (row, col) hyperparameter sweep."""
    grid = np.array([[results[(rv, cv)][metric_key] for cv in col_values] for rv in row_values])
    norm = LogNorm(vmin=vmin or grid.min(), vmax=vmax or grid.max()) if log_scale else None

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(grid, aspect="auto", cmap=cmap, norm=norm,
                    vmin=None if log_scale else vmin, vmax=None if log_scale else vmax)
    ax.set_xticks(range(len(col_values)), col_values)
    ax.set_yticks(range(len(row_values)), row_values)
    ax.set_xlabel(col_label); ax.set_ylabel(row_label); ax.set_title(title)
    plt.colorbar(im, ax=ax, extend="max" if (log_scale and vmax is not None) else None)

    for i, j in product(range(len(row_values)), range(len(col_values))):
        ax.text(j, i, f"{grid[i, j]:{fmt}}", ha="center", va="center", fontsize=7)

    plt.tight_layout()
    plt.show()


def plot_eigenvalue_grid(results: dict, row_values: list, col_values: list,
                          row_label: str = "row", title: str = "Eigenvalues grid",
                          highlight_row: float = None) -> None:
    """Plot a grid of DMD/EDMD eigenvalue scatter plots over a (row, col) sweep."""
    n_rows, n_cols = len(row_values), len(col_values)
    theta = np.linspace(0, 2 * np.pi, 300)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows),
                              sharex=True, sharey=True, squeeze=False)
    fig.suptitle(title, fontsize=13)

    for i, rv in enumerate(row_values):
        for j, cv in enumerate(col_values):
            ax, r = axes[i, j], results[(rv, cv)]
            eigs = r["eigenvalues"]

            ax.plot(np.cos(theta), np.sin(theta), "k--", lw=0.8, alpha=0.4)
            ax.scatter(eigs.real, eigs.imag, c=np.abs(eigs), cmap="RdYlGn_r", vmin=0.98, vmax=1.02, s=25)
            ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
            ax.axhline(0, color="gray", lw=0.4); ax.axvline(0, color="gray", lw=0.4)
            ax.set_title(f"{row_label}={rv} r={cv}\n|λ|={r['max_abs_eig']:.4f}", fontsize=7)

            if rv == highlight_row:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red"); spine.set_linewidth(2)
            if i == n_rows - 1: ax.set_xlabel("Re(λ)", fontsize=8)
            if j == 0:          ax.set_ylabel("Im(λ)", fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_reconstruction_grid(results: dict, row_values: list, col_values: list, reconstruct_fn,
                              row_label: str = "row", title: str = "Reconstruction grid",
                              all_states: bool = False, t_fn=None) -> None:
    """Plot a grid of true vs reconstructed trajectories over a (row, col) sweep."""
    n_rows, n_cols = len(row_values), len(col_values)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows), squeeze=False)
    fig.suptitle(title, fontsize=13)

    for i, rv in enumerate(row_values):
        for j, cv in enumerate(col_values):
            ax, r = axes[i, j], results[(rv, cv)]
            X, X_rec = r["X"], reconstruct_fn(r)
            n_col = min(X.shape[1], X_rec.shape[0])
            t_sub = t_fn(r, n_col) if t_fn else np.arange(n_col) * r.get("dt", 1)

            if all_states:
                for k in range(X.shape[0]):
                    ax.plot(t_sub, X[k, :n_col], lw=0.5, color="steelblue", alpha=0.3)
                    ax.plot(t_sub, X_rec[:n_col, k], lw=0.5, color="tomato", ls="--", alpha=0.3)
            else:
                ax.plot(t_sub, X[0, :n_col], lw=1.0, color="steelblue", label="true")
                ax.plot(t_sub, X_rec[:n_col, 0], lw=0.8, color="tomato", ls="--", label="reconstruction")

            ax.set_title(f"{row_label}={rv} r={cv}\nerr={r.get('recon_error', float('nan')):.3f}", fontsize=7)
            ax.tick_params(labelsize=6)
            if i == n_rows - 1: ax.set_xlabel("time", fontsize=7)
            if j == 0:          ax.set_ylabel("$x$", fontsize=7)

    if not all_states:
        axes[0, -1].legend(fontsize=6)
    plt.tight_layout()
    plt.show()