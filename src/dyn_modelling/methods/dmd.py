import numpy as np
from pydmd import DMD


# ---------------------------------------------------------------------------
# Fit DMD model
# ---------------------------------------------------------------------------

def fit_dmd(X: np.ndarray, svd_rank: int) -> DMD:
    """
    Fit DMD model on trajectory data.

    Parameters
    ----------
    X : np.ndarray
        Snapshot matrix.
    svd_rank : int
        SVD rank truncation. 0 = automatic.

    Returns
    -------
    dmd : DMD   fitted DMD object
    """
    dmd = DMD(svd_rank=svd_rank, exact=True)
    dmd.fit(X)
    return dmd


# ---------------------------------------------------------------------------
# Reconstruct from DMD model
# ---------------------------------------------------------------------------

def reconstruct(dmd: DMD) -> np.ndarray:
    """
    Reconstruct trajectories from fitted DMD.

    Returns
    -------
    X_rec : np.ndarray  shape (n_steps, 3*N)
    """
    return dmd.reconstructed_data.real.T


def reconstruct_analytic(dmd: DMD, X: np.ndarray) -> np.ndarray:
    """
    Analytic reconstruction using DMD modes and eigenvalues.

    Returns
    -------
    X_rec : np.ndarray
    """
    eigs = dmd.eigs
    modes = dmd.modes
    f0 = X[:, 0].astype(complex)
    T = X.shape[1]
    n = np.arange(T)

    b = np.linalg.pinv(modes) @ f0
    F_rec = modes @ (b[:, None] * (eigs[:, None] ** n[None, :]))
    return F_rec.T.real


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def get_eigenvalues(dmd: DMD) -> np.ndarray:
    """Return DMD eigenvalues."""
    return dmd.eigs


def get_modes(dmd: DMD) -> np.ndarray:
    """Return DMD modes."""
    return dmd.modes