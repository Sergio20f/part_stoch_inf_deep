from jax import jit, lax
import jax.numpy as np

def euler_step(y, t, t_delta, f, args):
    return y + t_delta * f(y, t, *args)

def midpoint_step(y, t, t_delta, f, args):
    mid_point = y + 0.5 * t_delta * f(y, t, *args)
    return y + t_delta * f(mid_point, t + 0.5 * t_delta, *args)

def integrate(f, y0, ts, dt, method, args=()):
    ts = np.array(ts)

    @jit
    def scan_fun(carry, next_t):
        curr_t, curr_y = carry
        t_delta = next_t - curr_t
        if method == 'euler':
            next_y = euler_step(curr_y, curr_t, t_delta, f, args)
        elif method == 'midpoint':
            next_y = midpoint_step(curr_y, curr_t, t_delta, f, args)
        else:
            raise ValueError("Invalid method")
        return (next_t, next_y), next_y

    init_carry = (ts[0], y0)
    _, ys = lax.scan(scan_fun, init_carry, ts[1:])
    return np.concatenate((y0[None], ys))

def odeint_euler(f, y0, ts, args=(), dt=1e-6):
    return integrate(f, y0, ts, dt, 'euler', args)

def odeint_midpoint(f, y0, ts, args=(), dt=1e-6):
    return integrate(f, y0, ts, dt, 'midpoint', args)

def odeint(f, y0, ts, args=(), dt=1e-6, method='midpoint'):
    if method == 'euler':
        return odeint_euler(f, y0, ts, args, dt)
    elif method == 'midpoint':
        return odeint_midpoint(f, y0, ts, args, dt)
    else:
        raise ValueError("Invalid method")