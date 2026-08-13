import numpy as np
import pysindy as ps


def split_uvs(y):
    n_steps = y.shape[1]
    N = y.shape[0] // 3
    y = y.reshape(N, 3, n_steps)
    u, v, s = y[:, 0, :].T, y[:, 1, :].T, y[:, 2, :].T
    return u, v, s 


def compute_phi(s, w):
    return s @ w.T


def generate_sindy_dataset(trajectories, w, external_signal_fn):
    """
    Build input data for SINDyc: X,U (control), and T (time).
    """
    x_list, ctrl_list, t_list = [], [], []
    for traj in trajectories:
        u, v, s = split_uvs(traj["y"])
        phi = compute_phi(s, w)
        s_ext = np.array(external_signal_fn(traj["t"], traj["t_on"], traj["t_off"]))
        N = u.shape[1]
        for cell in range(N):
            x = np.stack([u[:, cell], v[:, cell], s[:, cell]], axis=1)
            ctrl = np.stack([s_ext, phi[:, cell]], axis=1)
            x_list.append(x)
            ctrl_list.append(ctrl)
            t_list.append(traj["t"])
    return x_list, ctrl_list, t_list


# ---------------------------------------------------------------------
# library construction
# ---------------------------------------------------------------------

def hill_neg(x):
    return 1.0 / (1.0 + x**2)


def hill_pos(x):
    return x**2 / (1.0 + x**2)

def names_function(name):
    """Name-generating function required by CustomLibrary"""
    return lambda *args: name


def build_library():
    """Correct functional forms only, no distractors."""
    functions = [
        lambda u, v, s, sext, phi: u,
        lambda u, v, s, sext, phi: hill_neg(v),
        lambda u, v, s, sext, phi: sext * hill_neg(phi),
        lambda u, v, s, sext, phi: v,
        lambda u, v, s, sext, phi: hill_neg(u),
        lambda u, v, s, sext, phi: s,
        lambda u, v, s, sext, phi: hill_pos(u),
    ]
    names = ["u", "1/(1+v^2)", "Sext/(1+Phi^2)", "v", "1/(1+u^2)", "s", "u^2/(1+u^2)"]
    names_fns = [names_function(name) for name in names]
    return ps.CustomLibrary(library_functions=functions, function_names=names_fns, interaction_only= True)

def build_extended_library():
    """Same candidate set handed uniformly to all three equations."""
    functions = [
lambda u, v, s, sext, phi: u,
        lambda u, v, s, sext, phi: v,
        lambda u, v, s, sext, phi: s,
        lambda u, v, s, sext, phi: hill_neg(u),
        lambda u, v, s, sext, phi: hill_neg(v),
        lambda u, v, s, sext, phi: hill_pos(u),
        lambda u, v, s, sext, phi: hill_pos(v),
        lambda u, v, s, sext, phi: sext,
        lambda u, v, s, sext, phi: sext * hill_neg(phi),
        lambda u, v, s, sext, phi: hill_neg(phi),
        lambda u, v, s, sext, phi: u**2,
        lambda u, v, s, sext, phi: v**2,
        lambda u, v, s, sext, phi: s**2,
    ]
    names = ["u", "v", "s", "1/(1+u^2)", "1/(1+v^2)", "u^2/(1+u^2)",
             "v^2/(1+v^2)", "Sext", "Sext/(1+Phi^2)", "1/(1+Phi^2)",
             "u^2", "v^2", "s^2"]
    name_fns = [names_function(name) for name in names]
    return ps.CustomLibrary(library_functions=functions, function_names=name_fns, interaction_only=True)


# ---------------------------------------------------------------------
# fit
# ---------------------------------------------------------------------

def fit_sindy(x_list, ctrl_list, t_list, library, threshold=0.05):
    model = ps.SINDy(
        feature_library=library,
        optimizer=ps.STLSQ(threshold=threshold),
    )
    model.fit(x_list, u=ctrl_list, t=t_list, feature_names=["u", "v", "s"])
    return model


def print_model(model, control_names=("Sext", "Phi")):
    input_features = ["u", "v", "s"] + list(control_names)
    feat_names = model.feature_library.get_feature_names(input_features)
    coeffs = model.coefficients()
    for row_name, row in zip(["u'", "v'", "s'"], coeffs):
        terms = [f"{c:.3f}*{n}" for c, n in zip(row, feat_names) if abs(c) > 1e-8]
        print(f"{row_name} = {' + '.join(terms) if terms else '0'}")