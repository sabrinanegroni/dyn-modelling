import numpy as np
from pydmd import DMD
from pydmd import HODMD
from typing import Callable




# ---------------------------------------------------------------------------
# Fit
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


def fit_hodmd(X: np.ndarray, svd_rank: int = 0, d: int = 10) -> HODMD:
    """
    Fit Higher Order DMD model on trajectory data.

    Parameters
    ----------
    X : np.ndarray
        Snapshot matrix.
    svd_rank : int
        SVD rank truncation. 0 = automatic.
    d : int
        Number of consecutive snapshots (time delay order).
    Returns
    -------
    hodmd : HODMD   fitted HODMD object
    """
    hodmd = HODMD(svd_rank=svd_rank, exact=True, d=d)
    hodmd.fit(X)
    return hodmd


def fit_edmd(X: np.ndarray, observables: list[Callable] , svd_rank: int = 0) -> tuple:
    """
    Fit Extended DMD model on trajectory data.

    Parameters
    ----------
    X : np.ndarray
        Snapshot matrix.
    observables : list of callables
        List of observables to apply to state.
    svd_rank : int
        SVD rank truncation. 0 = automatic.

    Returns
    -------
    dmd : DMD   fitted DMD object on lifted space
    """
    X_lifted = np.vstack([obs(X) for obs in observables])
    return fit_dmd(X_lifted, svd_rank) , X_lifted



# ---------------------------------------------------------------------------
# Reconstruct from DMD model
# ---------------------------------------------------------------------------

def reconstruct(dmd: DMD) -> np.ndarray:
    """
    Reconstruct trajectories from fitted DMD.

    Returns
    -------
    X_rec : np.ndarray 
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


def reconstruct_hodmd(hodmd: HODMD) -> np.ndarray:
    """
    Reconstruct trajectories from fitted HODMD.

    Returns
    -------
    X_rec : np.ndarray
    """
    return hodmd.reconstructed_data.real.T


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def get_eigenvalues(dmd: DMD) -> np.ndarray:
    """Return DMD eigenvalues."""
    return dmd.eigs


def get_modes(dmd: DMD) -> np.ndarray:
    """Return DMD modes."""
    return dmd.modes