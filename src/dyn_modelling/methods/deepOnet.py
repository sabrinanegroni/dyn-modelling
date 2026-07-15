import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax import Array


class DeepONet(eqx.Module):
    branch_net: eqx.Module
    trunk_net: eqx.Module
    bias: Array

    def __init__(self, input_size_branch: int, input_size_trunk: int, width_size: int, depth: int, interact_size: int, *, key):

        branch_key, trunk_key = jr.split(key)

        self.branch_net = eqx.nn.MLP(
            in_size=input_size_branch,
            out_size=interact_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=branch_key,
        )
        self.trunk_net = eqx.nn.MLP(
            in_size=input_size_trunk,
            out_size=interact_size,
            width_size=width_size,
            depth=depth,
            activation=jax.nn.tanh,
            key=trunk_key,
        )
        self.bias = jnp.zeros(())

    def __call__(self, branch_input: Array, trunk_input: Array) -> Array:
   
        branch_out = self.branch_net(branch_input)        
        trunk_out = self.trunk_net(trunk_input)            
        return jnp.sum(branch_out * trunk_out) + self.bias


def _loss_fn(model: DeepONet, branch_inputs: Array,
             trunk_inputs: Array, outputs: Array) -> Array:
    predictions = jax.vmap(
        jax.vmap(model, in_axes=(None, 0)),
        in_axes=(0, None)
    )(branch_inputs, trunk_inputs)

    return jnp.mean(jnp.square(predictions - outputs))


@eqx.filter_jit
def update_fn(model, opt_state, optimizer, branch_inputs, trunk_inputs, outputs):
    loss, grad = eqx.filter_value_and_grad(_loss_fn)(model, branch_inputs, trunk_inputs, outputs)
    updates, opt_state = optimizer.update(grad, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


@eqx.filter_jit
def val_loss_fn(model, branch_inputs, trunk_inputs, outputs):
    return _loss_fn(model, branch_inputs, trunk_inputs, outputs)


def train(model, optimizer, branch_inputs_train, trunk_inputs, outputs_train, n_epochs=10_000):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    loss_history = []
    for step in range(n_epochs):
        model, opt_state, loss = update_fn(model, opt_state, optimizer, branch_inputs_train, trunk_inputs, outputs_train)
        loss_history.append(loss)

    return model, jnp.stack(loss_history)


def train_validation(model, optimizer, branch_inputs_train, trunk_inputs, outputs_train,
                      branch_inputs_val, outputs_val, n_epochs=10_000, val_every=100):
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    train_loss_history = []
    val_loss_history    = []
    val_steps           = []

    for step in range(n_epochs):
        model, opt_state, loss = update_fn(model, opt_state, optimizer, branch_inputs_train, trunk_inputs, outputs_train)
        train_loss_history.append(loss)

        if step % val_every == 0 or step == n_epochs - 1:
            v_loss = val_loss_fn(model, branch_inputs_val, trunk_inputs, outputs_val)
            val_loss_history.append(v_loss)
            val_steps.append(step)

    return (model, jnp.stack(train_loss_history),
            jnp.stack(val_loss_history), jnp.array(val_steps))


def evaluate_test_error(model, signal_test, queries, y_test, n_cells, n_vars, n_time):
    y_pred = jax.vmap(
        jax.vmap(model, in_axes=(None, 0)),
        in_axes=(0, None)
    )(signal_test, queries)  # (n_traj, n_queries)

    mse = jnp.mean((y_pred - y_test) ** 2)
    rel_l2 = jnp.linalg.norm(y_pred - y_test) / jnp.linalg.norm(y_test)

    pred_r = y_pred.reshape(y_pred.shape[0], n_cells, n_vars, n_time)
    true_r = y_test.reshape(y_test.shape[0], n_cells, n_vars, n_time)

    def sq_norm(x, axis):
        return jnp.sqrt(jnp.sum(x ** 2, axis=axis))

    per_cell_rel_err = sq_norm(pred_r - true_r, axis=(0, 2, 3)) / sq_norm(true_r, axis=(0, 2, 3))
    per_var_rel_err  = sq_norm(pred_r - true_r, axis=(0, 1, 3)) / sq_norm(true_r, axis=(0, 1, 3))

    return {
        "mse": mse,
        "rel_l2": rel_l2,
        "per_cell_rel_err": per_cell_rel_err,  
        "per_var_rel_err":  per_var_rel_err,   }
