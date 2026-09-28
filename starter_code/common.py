"""Shared problem definition for ECE276B PR3.

Everything that must be identical across CEC, tabular GPI and RBF GPI lives here:
exact-discretization dynamics (eq. 1), error dynamics (eq. 2), stage cost (eq. 3),
obstacle geometry, and a rollout harness that uses common random numbers so that
controllers are compared on exactly the same noise realizations.
"""
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

import utils

# ----------------------------------------------------------------------------------
# Problem constants
# ----------------------------------------------------------------------------------
DT = utils.time_step  # 0.5 s
T_PERIOD = utils.T  # reference period (100 steps)
SIGMA = np.asarray(utils.sigma, dtype=float)  # [0.04, 0.04, 0.004]
U_LB = np.array([utils.v_min, utils.w_min])
U_UB = np.array([utils.v_max, utils.w_max])
N_STEPS = int(round(utils.sim_time / DT))  # 240 steps, same as main.py

# (x, y, radius)
OBSTACLES = np.array([
    [2.35, 0.95, 0.5],
    [-2.35, -0.95, 0.5],
    [1.0, 0.0, 0.5],
    [-1.0, 0.0, 0.5],
])
ROBOT_RADIUS = 0.3
WORKSPACE = 3.0  # free space is inside [-3, 3]^2

X_INIT = np.array([utils.x_init, utils.y_init, utils.theta_init])

# reference table for one period, shape (T, 3): (x, y, theta)
REF = np.array([utils.lemniscate(k) for k in range(T_PERIOD)], dtype=float)


@dataclass
class CostParams:
    Q: np.ndarray = field(default_factory=lambda: np.diag([10.0, 10.0]))
    q: float = 10.0
    R: np.ndarray = field(default_factory=lambda: np.diag([0.1, 0.1]))
    gamma: float = 0.95


COST = CostParams()


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def ref(k):
    """Reference (x, y, theta) at integer time k (periodic)."""
    return REF[np.asarray(k) % T_PERIOD]


# ----------------------------------------------------------------------------------
# Dynamics
# ----------------------------------------------------------------------------------
def f(x, u, w=None):
    """Exact discretization of the unicycle, eq. (1). x: (..., 3), u: (..., 2)."""
    x = np.asarray(x, dtype=float)
    u = np.asarray(u, dtype=float)
    v, om = u[..., 0], u[..., 1]
    half = om * DT / 2.0
    s = DT * np.sinc(half / np.pi)  # np.sinc(x) = sin(pi x)/(pi x)
    th = x[..., 2]
    nxt = np.stack([
        x[..., 0] + s * np.cos(th + half) * v,
        x[..., 1] + s * np.sin(th + half) * v,
        th + DT * om,
    ], axis=-1)
    if w is not None:
        nxt = nxt + w
    nxt[..., 2] = wrap(nxt[..., 2])
    return nxt


def g(t, e, u, w=None):
    """Error dynamics, eq. (2). e = (p - r_t, theta - alpha_t)."""
    r_t, r_n = ref(t), ref(t + 1)
    x = np.asarray(e, dtype=float) + r_t
    x_next = f(x, u, w)
    e_next = x_next - r_n
    e_next[..., 2] = wrap(e_next[..., 2])
    return e_next


def error_state(t, x):
    e = np.asarray(x, dtype=float) - ref(t)
    e[..., 2] = wrap(e[..., 2])
    return e


def stage_cost(e, u, cp: CostParams = COST):
    e = np.asarray(e, dtype=float)
    u = np.asarray(u, dtype=float)
    p = e[..., :2]
    pc = np.einsum("...i,ij,...j->...", p, cp.Q, p)
    tc = cp.q * (1.0 - np.cos(e[..., 2])) ** 2
    uc = np.einsum("...i,ij,...j->...", u, cp.R, u)
    return pc + tc + uc


# ----------------------------------------------------------------------------------
# Geometry
# ----------------------------------------------------------------------------------
def clearance(p):
    """Signed clearance of the robot ball to the closest obstacle (negative = collision)."""
    p = np.asarray(p, dtype=float)[..., :2]
    d = np.linalg.norm(p[..., None, :] - OBSTACLES[:, :2], axis=-1) - (OBSTACLES[:, 2] + ROBOT_RADIUS)
    return d.min(axis=-1)


def in_free_space(p):
    p = np.asarray(p, dtype=float)[..., :2]
    inside = np.all(np.abs(p) <= WORKSPACE, axis=-1)
    return inside & (clearance(p) >= 0.0)


# ----------------------------------------------------------------------------------
# Rollout harness
# ----------------------------------------------------------------------------------
def noise_sequence(seed, n_steps=N_STEPS, noise_scale=1.0):
    """Common random numbers: the same standard-normal draws for a given seed, scaled by
    sigma * noise_scale. Every controller and every noise level sees the same Z."""
    z = np.random.default_rng(seed).standard_normal((n_steps, 3))
    return z * SIGMA * noise_scale


def rollout(controller, noise_scale=1.0, seed=0, n_steps=N_STEPS, x0=X_INIT, cp: CostParams = COST,
            keep_traj=False, start_t=0):
    """Run one closed-loop episode on the numerical simulator (eq. 1) and return metrics.

    controller(t, cur_state, cur_ref_state) -> [v, w]  (same signature as the starter code)
    """
    W = noise_sequence(seed, n_steps, noise_scale)
    x = np.array(x0, dtype=float)
    xs, us, es, ctimes = [x.copy()], [], [], []
    if hasattr(controller, "reset"):
        controller.reset()
    for t in range(n_steps):
        absolute_t = start_t + t
        r = ref(absolute_t)
        tic = perf_counter()
        u = np.asarray(controller(absolute_t, x.copy(), r.copy()), dtype=float).reshape(2)
        ctimes.append(perf_counter() - tic)
        u = np.clip(u, U_LB, U_UB)
        es.append(error_state(absolute_t, x))
        us.append(u)
        x = f(x, u, W[t])
        xs.append(x.copy())
    xs, us, es, ctimes = np.array(xs), np.array(us), np.array(es), np.array(ctimes)

    ell = stage_cost(es, us, cp)
    disc = cp.gamma ** np.arange(n_steps)
    pos_err = np.linalg.norm(es[:, :2], axis=1)
    clr = clearance(xs)  # includes x_0 ... x_N
    oob = np.any(np.abs(xs[:, :2]) > WORKSPACE, axis=1)
    coll = clr < 0.0
    first_hit = int(np.flatnonzero(coll)[0]) if coll.any() else -1
    # A collision is absorbing for the safety analysis, even though the simulator
    # continues to generate a full trajectory for tracking diagnostics.
    observed_safe_steps = min(first_hit, n_steps) if first_hit >= 0 else n_steps
    # starter main.py metric (error of x_{t+1} against r_t, summed) for comparability
    st_err = xs[1:] - ref(start_t + np.arange(n_steps))
    st_err[:, 2] = wrap(st_err[:, 2])

    out = dict(
        seed=seed,
        start_t=start_t,
        noise_scale=noise_scale,
        mean_pos_err=pos_err.mean(),
        rmse_pos=np.sqrt((pos_err ** 2).mean()),
        mean_abs_th_err=np.abs(es[:, 2]).mean(),
        disc_cost=(disc * ell).sum(),
        mean_stage_cost=ell.mean(),
        collided=bool(coll.any()),
        n_coll_steps=int(coll.sum()),
        first_coll_t=first_hit,
        safe_steps_until_hit=observed_safe_steps,
        hit_by_step_100=bool(coll[:101].any()),
        hit_by_step_200=bool(coll[:201].any()),
        pre_hit_stage_cost=float(ell[:observed_safe_steps].mean()) if observed_safe_steps else np.nan,
        min_clearance=clr.min(),
        oob=bool(oob.any()),
        ctrl_ms_mean=1e3 * ctimes.mean(),
        ctrl_ms_p95=1e3 * np.percentile(ctimes, 95),
        ctrl_ms_max=1e3 * ctimes.max(),
        starter_err_trans=np.linalg.norm(st_err[:, :2], axis=1).sum(),
        starter_err_rot=np.abs(st_err[:, 2]).sum(),
    )
    if hasattr(controller, "stats"):
        out.update(controller.stats())
    if keep_traj:
        out["traj"] = xs
        out["controls"] = us
        out["clearance"] = clr
    return out
