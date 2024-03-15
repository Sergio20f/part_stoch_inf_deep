from functools import partial

from jax import jit, lax
import jax.numpy as np
import jax


from brownian import make_brownian_motion
from sde_utils import make_gdg_prod


def sdeint_ito_fixed_grid(f, g, y0, ts, rng, args=(), dt=1e-6, method='milstein', rep=0, 
                          start_idx=None, end_idx=None, ode_method='euler'):
    
    if start_idx is None or end_idx is None:
        b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng=rng, depth=10, rep=rep)
        return ito_integrate(f, g, y0, ts, b, dt, args, method=method, fixed_grid=True)
    
    else: 
        b = make_brownian_motion(ts[0], np.zeros(y0[start_idx:end_idx].shape), ts[-1], rng=rng, depth=10, rep=rep)
        return ito_integrate_v2(f, g, y0, ts, b, dt, args, method=method, fixed_grid=True, 
                                start=start_idx, end=end_idx, ode_method=ode_method)



@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def ito_euler_step_v2(y, t, args, noise, t_delta, f, g_prod, gdg_prod=None, start_idx=0, end_idx=-1):
    # Equation 20 from https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf
    return (y + t_delta * f(y, t, args)).at[start_idx:end_idx].add(g_prod(y[start_idx:end_idx], t, args, noise))


@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def ito_milstein_step_v2(y, t, args, noise, t_delta, f, g_prod, gdg_prod, start_idx=0, end_idx=-1):
    # Equation 20 from https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf
    part1 = y + t_delta * f(y, t, args)
    part2 = g_prod(y[start_idx:end_idx], t, args, noise) + 0.5 * gdg_prod(y[start_idx:end_idx], t, args, noise**2 - t_delta)
    return  part1.at[start_idx:end_idx].add(part2)

@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def ito_euler_step_v2_midpoint(y, t, args, noise, t_delta, f, g_prod, gdg_prod=None, start_idx=0, end_idx=-1):
    # Equation 20 from https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf
    return (y + t_delta * f(y, t, args)).at[start_idx:end_idx].add(g_prod(y[start_idx:end_idx], t, args, noise))

@partial(jit, static_argnums=(5, 6, 7, 8, 9))
def ito_milstein_step_v2_midpoint(y, t, args, noise, t_delta, f, g_prod, gdg_prod, start_idx=0, end_idx=-1):
    # Equation 20 from https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf
    part1 = y + t_delta * f(y, t, args)
    part2 = g_prod(y[start_idx:end_idx], t, args, noise) + 0.5 * gdg_prod(y[start_idx:end_idx], t, args, noise**2 - t_delta)
    return  part1.at[start_idx:end_idx].add(part2)


def ito_integrate_v2(f, g, y0, ts, bm, dt, args=(), gdg=None, g_prod=None, method='milstein', 
                     fixed_grid=False, start=0, end=-1, ode_method='euler'):
    if g_prod is None:
        g_prod = lambda y, t, args, noise: g(y, t, args) * noise

    if ode_method == 'euler':
        if method == 'milstein':
            step = ito_milstein_step_v2
        elif method == 'euler_maruyama':
            step = ito_euler_step_v2
        else:
            raise ValueError('Unknown method: {}'.format(method))
    elif ode_method == 'midpoint':
        if method == 'milstein':
            step = ito_milstein_step_v2_midpoint
        elif method == 'euler_maruyama':
            step = ito_euler_step_v2_midpoint
        else:
            raise ValueError('Unknown method: {}'.format(method))
    else:
        raise ValueError('Unknown method: {}'.format(ode_method))

    return stochastic_integrate_v2(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid, start, end)


def stochastic_integrate_v2(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid=False, start=0, end=-1):
    # drift: f(y, t, args)
    # diffusion times noise: g_prod(y, t, args, noise)
    # initial state: y0
    # evaluation times: ts
    # step size: dt
    # dynamics: args
    # diffusion times its own Jacobian: gdg(y, t, args)
    # step function: step(y, t, args, noise, t_delta, f, g_prod, gdg_prod)
    ts = np.array(ts)

    if gdg is None:
        gdg = make_gdg_prod(g_prod)

    step = jit(step, static_argnums=(5, 6, 7, 8, 9))

    return _stochastic_integrate_v2(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid, start, end)


@partial(jit, static_argnums=(0, 1, 4, 7, 8, 9, 10, 11))
def _stochastic_integrate_v2(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid=False, start=0, end=-1):

    def scan_fun_one_step(carry, target_t):
        curr_t, curr_y = carry
        next_t = target_t - curr_t
        t_delta = next_t - curr_t

        dw = bm(next_t) - bm(curr_t)
        next_y = step(curr_y, curr_t, args, dw, t_delta, f, g_prod, gdg, start, end)
        return (next_t, next_y), next_y

    def scan_fun(carry, target_t):
        # Interpolate through to the next time point, integrating as necessary.

        def cond_fun(inp):
            _t, _y = inp
            return _t < target_t

        def body_fun(inp):
            _t, _y = inp
            next_t = np.minimum(_t + dt, ts[-1])
            t_delta = next_t - _t

            dw = bm(next_t) - bm(_t)
            next_y = step(_y, _t, args, dw, t_delta, f, g_prod, gdg, start, end)
            _y = next_y
            _t = next_t
            return _t, _y

        new_t, new_y = lax.while_loop(cond_fun, body_fun, carry)
        return (new_t, new_y), new_y

    init_carry = ts[0], y0
    if fixed_grid:
        _, ys = lax.scan(scan_fun_one_step, init_carry, ts[1:])
    else:
        _, ys = lax.scan(scan_fun, init_carry, ts[1:])
    return np.concatenate((y0[None], ys))

######################################################################

def sdeint_ito(f, g, y0, ts, rng, args=(), dt=1e-6, method='milstein', rep=0):
    # b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng)
    return _sdeint_ito(f, g, dt, method, rng, rep, y0, ts, args)




@partial(jax.custom_vjp, nondiff_argnums=(0, 1, 2, 3, 5))
def _sdeint_ito(f, g, dt, method, rng, rep, y0, ts, args):
    b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng, depth=10, rep=rep)
    return ito_integrate(f, g, y0, ts, b, dt, args, method=method, fixed_grid=False)


def sdeint_strat(f, g, y0, ts, rng, args=(), dt=1e-6, method='milstein'):
    b = make_brownian_motion(ts[0], np.zeros(y0.shape), ts[-1], rng)
    return stratonovich_integrate(f, g, y0, ts, b, dt, args, method=method)


def linear_interp(t0, y0, t1, y1, t):
    return y0 + (y1 - y0) * (t - t0) / (t1 - t0)
        

@partial(jit, static_argnums=(5, 6, 7))
def strat_milstein_step(y, t, args, noise, t_delta, f, g_prod, gdg_prod):
    # Equation 21 from https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf
    return y + t_delta * f(y, t, args) + g_prod(y, t, args, noise) \
        + 0.5 * gdg_prod(y, t, args, noise**2)


@partial(jit, static_argnums=(5, 6, 7))
def ito_euler_step(y, t, args, noise, t_delta, f, g_prod, gdg_prod=None):
    # Equation 20 from https://infoscience.epfl.ch/record/143450/files/sde_tutorial.pdf
    return y + t_delta * f(y, t, args) + g_prod(y, t, args, noise)





def ito_integrate(f, g, y0, ts, bm, dt, args=(), gdg=None, g_prod=None, method='milstein', fixed_grid=False):
    if g_prod is None:
        g_prod = lambda y, t, args, noise: g(y, t, args) * noise

    if method == 'milstein':
        step = ito_milstein_step
    elif method == 'euler_maruyama':
        step = ito_euler_step
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return stochastic_integrate(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid)


def stratonovich_integrate(f, g, y0, ts, bm, dt, args=(), gdg=None, g_prod=None, method='milstein', fixed_grid=False):
    if g_prod is None:
        g_prod = lambda y, t, args, noise: g(y, t, args) * noise

    if method == 'milstein':
        step = strat_milstein_step
    elif method == 'euler_heun':
        raise NotImplementedError
    else:
        raise ValueError('Unknown method: {}'.format(method))

    return stochastic_integrate(f, g_prod, y0, ts, bm, dt, args, gdg, step, False)


def stochastic_integrate(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid=False):
    # drift: f(y, t, args)
    # diffusion times noise: g_prod(y, t, args, noise)
    # initial state: y0
    # evaluation times: ts
    # step size: dt
    # dynamics: args
    # diffusion times its own Jacobian: gdg(y, t, args)
    # step function: step(y, t, args, noise, t_delta, f, g_prod, gdg_prod)
    ts = np.array(ts)

    if gdg is None:
        gdg = make_gdg_prod(g_prod)

    step = jit(step, static_argnums=(5, 6, 7))

    return _stochastic_integrate(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid)


@partial(jit, static_argnums=(0, 1, 4, 7, 8, 9))
def _stochastic_integrate(f, g_prod, y0, ts, bm, dt, args, gdg, step, fixed_grid=False):

    def scan_fun_one_step(carry, target_t):
        curr_t, curr_y = carry
        next_t = target_t - curr_t
        t_delta = next_t - curr_t

        dw = bm(next_t) - bm(curr_t)
        next_y = step(curr_y, curr_t, args, dw, t_delta, f, g_prod, gdg)
        return (next_t, next_y), next_y

    def scan_fun(carry, target_t):
        # Interpolate through to the next time point, integrating as necessary.

        def cond_fun(inp):
            _t, _y = inp
            return _t < target_t

        def body_fun(inp):
            _t, _y = inp
            next_t = np.minimum(_t + dt, ts[-1])
            t_delta = next_t - _t

            dw = bm(next_t) - bm(_t)
            next_y = step(_y, _t, args, dw, t_delta, f, g_prod, gdg)
            _y = next_y
            _t = next_t
            return _t, _y

        new_t, new_y = lax.while_loop(cond_fun, body_fun, carry)
        return (new_t, new_y), new_y

    init_carry = ts[0], y0
    if fixed_grid:
        _, ys = lax.scan(scan_fun_one_step, init_carry, ts[1:])
    else:
        _, ys = lax.scan(scan_fun, init_carry, ts[1:])
    return np.concatenate((y0[None], ys))
