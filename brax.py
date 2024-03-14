import math

import jax
from jax.experimental.host_callback import id_print
import jax.numpy as jnp
import numpy as np
from jax import tree_util
from jax.example_libraries import stax
from jax.flatten_util import ravel_pytree
from jax.lax import stop_gradient
from sdeint import sdeint_ito, sdeint_ito_fixed_grid
from odeint import odeint
import copy

from arch import Layer, build_fx

# Hyperparams
# fx_block_type = "resnet"
# fx_dim = 128
# fx_actfn = "erf"
# fw = stax.serial(s
#     stax.Dense(128),
#     stax.Erf(),

# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------


def PSDEBNN(fx_block_type,
            fx_dim,
            fx_actfn,
            fw,
            diff_coef=1e-4,
            name="psdebnn",
            stl=False,
            xt=False,
            nsteps=20,
            remat=False,
            w_drift=True,
            stax_api=False,
            infer_initial_state=False,
            initial_state_prior_std=0.1, 
            ode_first=False,
            timecut=0.1,
            method_ode="euler",
            fix_w1=False): 
    
    if fix_w1==True and ode_first==True:
        raise ValueError("fix_w1 and ode_first cannot be both True FOR NOW")
    
    if ode_first: 
        s = int(nsteps*(1-timecut))
        ts_ode = jnp.linspace(0, 1-timecut, s+1)
        ts_sde = jnp.linspace(1-timecut, 1, nsteps-s+1)

    else:
        s = int(nsteps*timecut)
        ts_ode = jnp.linspace(timecut, 1.0, nsteps-s+1)
        ts_sde = jnp.linspace(0, timecut, s+1)
    
    def make_layer(input_shape):

        fx = build_fx(fx_block_type, input_shape, fx_dim, fx_actfn)

        # Creates the unflatten_w function.
        rng = jax.random.PRNGKey(0)  # temp; not used.
        x_shape, tmp_w = fx.init(rng, input_shape)
        assert input_shape == x_shape, f"fx needs to have the same input and output shapes but got {input_shape} and {x_shape}"
        flat_w, unflatten_w = ravel_pytree(tmp_w)
        w_shape = flat_w.shape
        del tmp_w

        # x_dim definitely not be negative...
        x_dim = np.abs(np.prod(x_shape))
        w_dim = np.abs(np.prod(w_shape))

        def f_aug_sde(y, t, args):

            # Extractring activations
            x = y[:x_dim].reshape(x_shape)

            # Extracting weights
            flat_w = y[x_dim:x_dim + w_dim].reshape(w_shape)

            # Compute next activations
            dx = fx.apply(unflatten_w(flat_w), (x, t))[0] if xt else fx.apply(unflatten_w(flat_w), x)

            # wtf is w_drift = False
            # wtf is xt = True

            # Computing next weights
            if w_drift:
                fw_params = args
                dw = fw.apply(fw_params, (flat_w, t))[0] if xt else fw.apply(fw_params, flat_w)
            else:
                dw = jnp.zeros(w_shape)

            # Hardcoded OU Process.
            u = (dw - (-flat_w)) / diff_coef if diff_coef != 0 else jnp.zeros(w_shape)

            dkl = u**2

            return jnp.concatenate([dx.reshape(-1), dw.reshape(-1), dkl.reshape(-1)])
            

        def f_aug_ode(y, t, args):

                # Extractring activations
                x = y[:x_dim].reshape(x_shape)

                # Extracting weights
                flat_w = y[x_dim:x_dim + w_dim].reshape(w_shape)

                # Compute next activations
                dx = fx.apply(unflatten_w(flat_w), (x, t))[0] if xt else fx.apply(unflatten_w(flat_w), x)

                # wtf is w_drift = False
                # wff is xt = True

                # Computing next weights
                if w_drift:
                    fw_params = args
                    dw = fw.apply(fw_params, (flat_w, t))[0] if xt else fw.apply(fw_params, flat_w)
                else:
                    dw = jnp.zeros(w_shape)

                # Hardcoded OU Process.
                #u = (dw - (-flat_w)) / diff_coef if diff_coef != 0 else jnp.zeros(w_shape)

                return jnp.concatenate([dx.reshape(-1), dw.reshape(-1)])
                

        def g_aug_sde(y, t, args):

            dx = jnp.zeros(x_shape)
            diff_w = jnp.ones(w_shape) * diff_coef
            dkl = jnp.zeros(w_shape)
            return jnp.concatenate([dx.reshape(-1), diff_w.reshape(-1), dkl.reshape(-1)])


        def init_fun(rng, input_shape):
            
            output_shape, w0 = fx.init(rng, input_shape)
            init_w0, unflatten_w = ravel_pytree(w0)

            if infer_initial_state:
                logstd_w0 = tree_util.tree_map(lambda x: jnp.zeros_like(x) - 4.0, init_w0)
            else:
                logstd_w0 = ()

            if w_drift:
                output_shape, fw_params = fw.init(rng, init_w0.shape)
                assert init_w0.shape == output_shape, "fw needs to have the same input and output shapes"
            else:
                fw_params = ()

            if not fix_w1:
                return input_shape, (init_w0, logstd_w0, fw_params)
            else:
                init_w1 = copy.deepcopy(init_w0)
                return input_shape, (init_w0, init_w1, logstd_w0, fw_params)
        

        def _apply_fun_sde_first(params, inputs, rng, full_output=False, fixed_grid=True, **kwargs):
            
            if not fix_w1:
                init_w0, logstd_w0, fw_params = params
            else:
                init_w0, init_w1, logstd_w0, fw_params = params

            x = inputs
            if infer_initial_state:
                raise ValueError("infer_initial_state not implemented for PSDEBNN - Ask Francesco")
                w0_rng, rng = jax.random.split(rng)
                mean_w0 = init_w0
                init_w0 = jax.random.normal(w0_rng, mean_w0.shape) * jnp.exp(logstd_w0) + mean_w0
                kl = normal_logprob(init_w0, mean_w0, logstd_w0) - \
                    normal_logprob(init_w0, 0., jnp.log(initial_state_prior_std))
                kl = jnp.sum(kl)

            else:
                kl = 0

            y0 = jnp.concatenate([x.reshape(-1), init_w0.reshape(-1), jnp.zeros(init_w0.shape).reshape(-1)])
            rep = w_dim if stl else 0  # STL NOT IMPLEMENTED
            
            if fixed_grid:
                ys = sdeint_ito_fixed_grid(f_aug_sde, g_aug_sde, y0, ts_sde, rng, fw_params, method="euler_maruyama", rep=rep)
            
            else:
                print("using stochastic adjoint")
                ys = sdeint_ito(f_aug_sde, g_aug_sde, y0, ts_sde, rng, fw_params, method="euler_maruyama", rep=rep)

            ys2 = ys[-1]  # Take last time value.
            kl = kl + jnp.sum(ys2[x_dim + w_dim:])

            if not fix_w1:
                ys3 = ys2[:(x_dim+w_dim)]
            else:
                ys3 = jnp.concatenate([ys2[:x_dim].reshape(-1), init_w1.reshape(-1)])

            ys4 = odeint(f_aug_ode, ys3, ts_ode, args=(fw_params,), method=method_ode)

            y = ys4[-1]  # Take last time value.
            x = y[:x_dim].reshape(x_shape)

            # Hack to turn this into a stax.layer API when deterministic.
            if stax_api:
                return x

            if full_output:
                infodict = {name + "_w": ys[:, x_dim:x_dim + w_dim].reshape(-1, *w_shape)}
                return x, kl, infodict

            return x, kl

        def _apply_fun_ode_first(params, inputs, rng, full_output=False, fixed_grid=True, **kwargs):
            
            init_w0, logstd_w0, fw_params = params

            x = inputs
            if infer_initial_state:
                raise ValueError("infer_initial_state not implemented for PSDEBNN - Ask Francesco")
                w0_rng, rng = jax.random.split(rng)
                mean_w0 = init_w0
                init_w0 = jax.random.normal(w0_rng, mean_w0.shape) * jnp.exp(logstd_w0) + mean_w0
                kl = normal_logprob(init_w0, mean_w0, logstd_w0) - \
                    normal_logprob(init_w0, 0., jnp.log(initial_state_prior_std))
                kl = jnp.sum(kl)

            else:
                kl = 0

            y0 = jnp.concatenate([x.reshape(-1), init_w0.reshape(-1)])

            y_ode = odeint(f_aug_ode, y0, ts_ode, args=(fw_params,), method=method_ode)

            y0_sde = jnp.concatenate([y_ode[-1].reshape(-1), jnp.zeros(init_w0.shape).reshape(-1)])

            rep = w_dim if stl else 0  # STL NOT IMPLEMENTED
            
            if timecut > 0.0:
                if fixed_grid:
                    ys = sdeint_ito_fixed_grid(f_aug_sde, g_aug_sde, y0_sde, jnp.linspace(0, timecut, nsteps-s+1), rng, fw_params, method="euler_maruyama", rep=rep)
                    #jax.debug.print("ys - x: {}", ys[-1][:x_dim])
                    #jax.debug.print("ys - weights: {}", ys[-1][x_dim:x_dim+w_dim])
                    #jax.debug.print("ys - kl: {}", ys[-1][x_dim+w_dim:])
                
                else:
                    print("using stochastic adjoint")
                    ys = sdeint_ito(f_aug_sde, g_aug_sde, y0_sde, ts_sde, rng, fw_params, method="euler_maruyama", rep=rep)
            elif timecut == 0.0:
                ys = y_ode
                y = ys[-1]  # Take last time value.
                x = y[:x_dim].reshape(x_shape)
                kl = 0
                
                if stax_api:
                    return x

                if full_output:
                    infodict = {name + "_w": ys[:, x_dim:x_dim + w_dim].reshape(-1, *w_shape)}
                    return x, kl, infodict
                
                return x, kl
                            
            else:
                raise ValueError("timecut must be >= 0.0")

            y = ys[-1]  # Take last time value.
            x = y[:x_dim].reshape(x_shape)
            kl = kl + jnp.sum(y[x_dim + w_dim:])
            
            # Hack to turn this into a stax.layer API when deterministic.
            if stax_api:
                return x

            if full_output:
                infodict = {name + "_w": ys[:, x_dim:x_dim + w_dim].reshape(-1, *w_shape)}
                return x, kl, infodict

            return x, kl

        def apply_fun(params, inputs, rng, full_output=False, fixed_grid=True, **kwargs):
            
            if ode_first:
                return _apply_fun_ode_first(params, inputs, rng, full_output, fixed_grid, **kwargs)
            else:
                return _apply_fun_sde_first(params, inputs, rng, full_output, fixed_grid, **kwargs)
            
        if remat:
            apply_fun = jax.checkpoint(apply_fun, concrete=True)
        return init_fun, apply_fun

    return Layer(*stax.shape_dependent(make_layer))



# -------------------------------------------------------------------------------------------------
# -------------------------------------------------------------------------------------------------

def SDEBNN(fx_block_type,
           fx_dim,
           fx_actfn,
           fw,
           diff_coef=1e-4,
           name="sdebnn",
           stl=False,
           xt=False,
           nsteps=20,
           remat=False,
           w_drift=True,
           stax_api=False,
           infer_initial_state=False,
           initial_state_prior_std=0.1):

    # This controls the number of function evaluations and the step size.
    ts = jnp.linspace(0.0, 1.0, nsteps)

    def make_layer(input_shape):

        fx = build_fx(fx_block_type, input_shape, fx_dim, fx_actfn)

        # Creates the unflatten_w function.
        rng = jax.random.PRNGKey(0)  # temp; not used.
        x_shape, tmp_w = fx.init(rng, input_shape)
        assert input_shape == x_shape, f"fx needs to have the same input and output shapes but got {input_shape} and {x_shape}"
        flat_w, unflatten_w = ravel_pytree(tmp_w)
        w_shape = flat_w.shape
        #jax.debug.print("w_shape {}: ", w_shape)
        del tmp_w

        # x_dim definitely not be negative...
        x_dim = np.abs(np.prod(x_shape))
        w_dim = np.abs(np.prod(w_shape))

        def f_aug(y, t, args):
            x = y[:x_dim].reshape(x_shape)
            flat_w = y[x_dim:x_dim + w_dim].reshape(w_shape)
            dx = fx.apply(unflatten_w(flat_w), (x, t))[0] if xt else fx.apply(unflatten_w(flat_w), x)
            if w_drift:
                fw_params = args
                dw = fw.apply(fw_params, (flat_w, t))[0] if xt else fw.apply(fw_params, flat_w)
            else:
                dw = jnp.zeros(w_shape)

            #THIS IS DIFFERENT FROM TORCH!!!!!!!!!!!!!!!!

            # Hardcoded OU Process.
            u = (dw - (-flat_w)) / \
                diff_coef if diff_coef != 0 else jnp.zeros(w_shape)
            dkl = u**2
            return jnp.concatenate([dx.reshape(-1), dw.reshape(-1), dkl.reshape(-1)])

        def g_aug(y, t, args):
            dx = jnp.zeros(x_shape)
            diff_w = jnp.ones(w_shape) * diff_coef

            if w_drift:
                fw_params = tree_util.tree_map(stop_gradient, args)
                drift_w = fw.apply(fw_params, (flat_w, t))[0] if xt else fw.apply(fw_params, flat_w)
            else:
                drift_w = jnp.zeros(w_shape)

            # Hardcoded OU Process.
            u = (drift_w - (-flat_w)) / \
                diff_coef if diff_coef != 0 else jnp.zeros(w_shape)
            dkl = u if stl else jnp.zeros(w_shape)
            return jnp.concatenate([dx.reshape(-1), diff_w.reshape(-1), dkl.reshape(-1)])

        def init_fun(rng, input_shape):
            output_shape, w0 = fx.init(rng, input_shape)
            init_w0, unflatten_w = ravel_pytree(w0)

            if infer_initial_state:
                logstd_w0 = tree_util.tree_map(lambda x: jnp.zeros_like(x) - 4.0, init_w0)
            else:
                logstd_w0 = ()

            if w_drift:
                output_shape, fw_params = fw.init(rng, init_w0.shape)
                assert init_w0.shape == output_shape, "fw needs to have the same input and output shapes"
            else:
                fw_params = ()

            return input_shape, (init_w0, logstd_w0, fw_params)

        def apply_fun(params, inputs, rng, full_output=False, fixed_grid=True, **kwargs):
            init_w0, logstd_w0, fw_params = params
            x = inputs
            if infer_initial_state:
                w0_rng, rng = jax.random.split(rng)
                mean_w0 = init_w0
                init_w0 = jax.random.normal(w0_rng, mean_w0.shape) * jnp.exp(logstd_w0) + mean_w0
                kl = normal_logprob(init_w0, mean_w0, logstd_w0) - \
                    normal_logprob(init_w0, 0., jnp.log(initial_state_prior_std))
                kl = jnp.sum(kl)
            else:
                kl = 0

            y0 = jnp.concatenate([x.reshape(-1), init_w0.reshape(-1), jnp.zeros(init_w0.shape).reshape(-1)])
            rep = w_dim if stl else 0  # STL
            if fixed_grid:
                ys = sdeint_ito_fixed_grid(f_aug, g_aug, y0, ts, rng, fw_params, method="euler_maruyama", rep=rep)
            else:
                print("using stochastic adjoint")
                ys = sdeint_ito(f_aug, g_aug, y0, ts, rng, fw_params, method="euler_maruyama", rep=rep)
            y = ys[-1]  # Take last time value.
            x = y[:x_dim].reshape(x_shape)
            kl = kl + jnp.sum(y[x_dim + w_dim:])

            # Hack to turn this into a stax.layer API when deterministic.
            if stax_api:
                return x

            if full_output:
                infodict = {name + "_w": ys[:, x_dim:x_dim + w_dim].reshape(-1, *w_shape)}
                return x, kl, infodict

            return x, kl

        if remat:
            apply_fun = jax.checkpoint(apply_fun, concrete=True)
        return init_fun, apply_fun

    return Layer(*stax.shape_dependent(make_layer))

def MeanField(layer, prior_std=0.1, disable=False):

    init_fun, apply_fun = layer

    def wrapped_init_fun(rng, input_shape):
        output_shape, params_mean = init_fun(rng, input_shape)
        params_logstd = tree_util.tree_map(lambda x: jnp.zeros_like(x) - 4.0, params_mean)
        return output_shape, (params_mean, params_logstd)

    def wrapped_apply_fun(params, input, rng, **kwargs):
        params_mean, params_logstd = params

        flat_mean, unflatten = ravel_pytree(params_mean)
        flat_logstd, _ = ravel_pytree(params_logstd)

        rng, next_rng = jax.random.split(rng)
        if disable:
            flat_params = flat_mean
        else:
            flat_params = jax.random.normal(rng, flat_mean.shape) * jnp.exp(flat_logstd) + flat_mean
        params = unflatten(flat_params)

        if disable:
            kl = jnp.zeros_like(flat_params)
        else:
            kl = normal_logprob(flat_params, flat_mean, flat_logstd) - \
                normal_logprob(flat_params, 0., jnp.log(prior_std))
        output = apply_fun(params, input, rng=next_rng, **kwargs)
        return output, jnp.sum(kl)

    return Layer(wrapped_init_fun, wrapped_apply_fun)


def normal_logprob(z, mean, log_std):
    mean = mean + jnp.zeros(1)
    log_std = log_std + jnp.zeros(1)
    # c = jnp.array([math.log(2 * math.pi)])
    c = math.log(2 * math.pi)
    inv_sigma = jnp.exp(-log_std)
    tmp = (z - mean) * inv_sigma
    return -0.5 * (tmp * tmp + 2 * log_std + c)


def bnn_serial(*layers):
    """Combinator for composing layers in serial.

    Args:
      *layers: a sequence of layers, each an (init_fun, apply_fun) pair.

    Returns:
      A new layer, meaning an (init_fun, apply_fun) pair, representing the serial
      composition of the given sequence of layers.
    """
    nlayers = len(layers)
    init_funs, apply_funs = zip(*layers)

    def init_fun(rng, input_shape):
        params = []
        for init_fun in init_funs:
            rng, layer_rng = jax.random.split(rng)
            input_shape, param = init_fun(layer_rng, input_shape)
            params.append(param)
        return input_shape, params

    def apply_fun(params, inputs, **kwargs):
        rng = kwargs.pop('rng', None)
        rngs = jax.random.split(rng, nlayers) if rng is not None else (None,) * nlayers
        total_kl = 0
        infodict = {}
        for fun, param, rng in zip(apply_funs, params, rngs):
            output = fun(param, inputs, rng=rng, **kwargs)
            if len(output) == 2:
                inputs, layer_kl = output
            elif len(output) == 3:
                inputs, layer_kl, info = output
                infodict.update(info)
            else:
                raise RuntimeError(f"Expected 2 or 3 outputs but got {len(output)}.")
            total_kl = total_kl + layer_kl
        return inputs, total_kl, infodict

    return Layer(init_fun, apply_fun)

# -------------------------------------------------------------------------------------------------

def PSDEBNN_H(fx_block_type,
           fx_dim,
           fx_actfn,
           fw1,
           fw2,
           ratio=0.5,
           diff_coef=1e-4,
           name="psdebnn_h",
           stl=False,
           xt=False,
           nsteps=20,
           remat=False,
           w_drift=True,
           stax_api=False,
           infer_initial_state=False,
           initial_state_prior_std=0.1):

    # This controls the number of function evaluations and the step size.
    ts = jnp.linspace(0.0, 1.0, nsteps)

    def make_layer(input_shape):

        fx = build_fx(fx_block_type, input_shape, fx_dim, fx_actfn)

        # Creates the unflatten_w function.
        rng = jax.random.PRNGKey(0)  # temp; not used.
        x_shape, tmp_w = fx.init(rng, input_shape)
        assert input_shape == x_shape, f"fx needs to have the same input and output shapes but got {input_shape} and {x_shape}"

        # first w network
        flat_w, unflatten_w = ravel_pytree(tmp_w)
        w_shape = flat_w.shape

        del tmp_w

        # x_dim definitely not be negative...
        x_dim = np.abs(np.prod(x_shape))
        w_dim = np.abs(np.prod(w_shape))
        w_dim1 = int(np.abs(np.prod(w_shape)) * ratio)
        w_dim2 = int(w_dim - w_dim1)

        w1_shape = (w_dim1,)
        w2_shape = (w_dim2,)

        def f_aug(y, t, args):

            x = y[:x_dim].reshape(x_shape)

            flat_w = y[x_dim:x_dim + w_dim].reshape(w_shape)

            dx = fx.apply(unflatten_w(flat_w), (x, t))[0] if xt else fx.apply(unflatten_w(flat_w), x)
            
            if w_drift:
                flat_w1 = flat_w[:w_dim1]
                flat_w2 = flat_w[w_dim1:]
                
                fw1_params, fw2_params = args
                dw1 = fw1.apply(fw1_params, (flat_w1, t))[0] if xt else fw1.apply(fw1_params, flat_w1)
                dw2 = fw2.apply(fw2_params, (flat_w2, t))[0] if xt else fw2.apply(fw2_params, flat_w2)
            else:
                dw1 = jnp.zeros(w1_shape)
                dw2 = jnp.zeros(w2_shape)

            # Hardcoded OU Process.
            u = (dw1 - (-flat_w1)) / diff_coef if diff_coef != 0 else jnp.zeros(w1_shape) # change here
            dkl = u**2

            dkl2 = jnp.zeros(w2_shape)

            return jnp.concatenate([dx.reshape(-1), dw1.reshape(-1), dw2.reshape(-1), dkl.reshape(-1), dkl2.reshape(-1)])

        def g_aug(y, t, args):
            dx = jnp.zeros(x_shape)
            diff_w1 = jnp.ones(w1_shape) * diff_coef
            diff_w2 = jnp.zeros(w2_shape) 

            dkl = jnp.zeros(w_shape)

            return jnp.concatenate([dx.reshape(-1), diff_w1.reshape(-1), diff_w2.reshape(-1), dkl.reshape(-1)])

        def init_fun(rng, input_shape):
            
            output_shape, w0 = fx.init(rng, input_shape)
            init_w0, unflatten_w = ravel_pytree(w0)

            if infer_initial_state:
                logstd_w0 = tree_util.tree_map(lambda x: jnp.zeros_like(x) - 4.0, init_w0)
            else:
                logstd_w0 = ()

            if w_drift:
                output_shape1, fw_params1 = fw1.init(rng, w1_shape)
                output_shape2, fw_params2 = fw2.init(rng, w2_shape)
                assert w1_shape == output_shape1, "fw needs to have the same input and output shapes"
                assert w2_shape == output_shape2, "fw needs to have the same input and output shapes"
            else:
                fw_params1 = ()
                fw_params2 = ()

            return input_shape, (init_w0, logstd_w0, fw_params1, fw_params2)

        def apply_fun(params, inputs, rng, full_output=False, fixed_grid=True, **kwargs):
            
            init_w0, logstd_w0, fw1_params, fw2_params = params
            #fw1_params = fw_params[:w_dim*ratio]
            #fw2_params = fw_params[w_dim*ratio:]
            x = inputs
            
            if infer_initial_state:
                w0_rng, rng = jax.random.split(rng)
                mean_w0 = init_w0
                init_w0 = jax.random.normal(w0_rng, mean_w0.shape) * jnp.exp(logstd_w0) + mean_w0
                kl = normal_logprob(init_w0, mean_w0, logstd_w0) - \
                    normal_logprob(init_w0, 0., jnp.log(initial_state_prior_std))
                kl = jnp.sum(kl)
            else:
                kl = 0

            y0 = jnp.concatenate([x.reshape(-1), init_w0.reshape(-1), jnp.zeros(init_w0.shape).reshape(-1)])
            rep = w_dim if stl else 0  # STL
            if fixed_grid:
                ys = sdeint_ito_fixed_grid(f_aug, g_aug, y0, ts, rng, (fw1_params, fw2_params), method="euler_maruyama", rep=rep)
            else:
                print("using stochastic adjoint")
                ys = sdeint_ito(f_aug, g_aug, y0, ts, rng, (fw1_params, fw2_params), method="euler_maruyama", rep=rep)
            
            y = ys[-1]  # Take last time value.
            x = y[:x_dim].reshape(x_shape)
            
            kl = kl + jnp.sum(y[x_dim + w_dim:x_dim + w_dim + w_dim1])

            # Hack to turn this into a stax.layer API when deterministic.
            if stax_api:
                return x

            if full_output:
                infodict = {name + "_w": ys[:, x_dim:x_dim + w_dim].reshape(-1, *w_shape)}
                return x, kl, infodict

            return x, kl

        if remat:
            apply_fun = jax.checkpoint(apply_fun, concrete=True)
        return init_fun, apply_fun

    return Layer(*stax.shape_dependent(make_layer))
