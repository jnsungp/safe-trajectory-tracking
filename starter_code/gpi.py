"""Generalized policy iteration (Part 2) and feature-based GPI (Part 3) on the GPU.

Discretized MDP
---------------
State (t, ex, ey, eth) on a grid of size (100, nx, ny, nth); control (v, w) on a grid (nv, nw).
For a node e and control u, the next error state is g(t, e, u, w). Its expected value is computed as

    E_w[V(t+1, g(t,e,u,w))]  ~=  sum_k  omega_k * V_interp(t+1, g(t, e, u, w_k))

where {w_k, omega_k} is a 3-point Gauss-Hermite rule per noise dimension (27 nodes) and V_interp is
multilinear interpolation between the 8 grid vertices of the cell that contains the point. This is a
proper discretized MDP: from (t, e, u) the chain moves to the 8 vertices of each landing cell with
barycentric probabilities, weighted by omega_k, and all probabilities sum to 1 (the (.., 8, 4)
transition table of the handout, with the Gaussian integrated by quadrature instead of evaluating the
likelihood at neighbors). The transition table is never materialized: it is recomputed on the fly on
the GPU, which needs O(|V|) memory instead of O(|S||A| * neighbors).

Collision handling
------------------
Stage cost  l(t,e,u) = p^T Q p + q (1 - cos eth)^2 + u^T R u + lambda * P_risk(t, e, u)
with P_risk the probability that the *next* position p_{t+1} = mean + w_xy is outside the free space F,
computed at the exact continuous next position (not the grid node) with a half-plane approximation
P(||p + w - c|| < r) ~= Phi(-(||p - c|| - r) / sigma_xy), since r = 0.8 >> sigma. When noise_scale = 0
a small sigma floor keeps the penalty finite-slope.
"""
from dataclasses import dataclass, field, asdict
import json
import math
import os
from time import perf_counter

import numpy as np
import torch

import common as C
import utils
from value_function import ErrorGrid, FeatureValueFunction, GridValueFunction, wrap_t


def pos_grid(n_inner=17, inner=0.8, outer=(1.0, 1.25, 1.6, 2.0, 2.5, 3.0)):
    """Adaptive position-error grid: uniform and fine in [-inner, inner], coarse outside."""
    fine = np.linspace(-inner, inner, n_inner)
    outer = np.asarray(outer, dtype=float)
    return np.r_[-outer[::-1], fine, outer]


def theta_grid(nth=40):
    return -np.pi + 2 * np.pi / nth * np.arange(nth)


GRID_PRESETS = {
    # name: (position grid, n_theta)
    "coarse": (pos_grid(9, 0.8, (1.2, 2.0, 3.0)), 24),
    "medium": (pos_grid(17, 0.8, (1.0, 1.25, 1.6, 2.0, 2.5, 3.0)), 40),
    "fine": (pos_grid(33, 0.8, (1.0, 1.25, 1.6, 2.0, 2.5, 3.0)), 60),
}


@dataclass
class GpiConfig:
    traj: callable = utils.lemniscate
    obstacles: np.ndarray = field(default_factory=lambda: C.OBSTACLES.copy())
    ex_space: np.ndarray = field(default_factory=lambda: GRID_PRESETS["medium"][0])
    ey_space: np.ndarray = field(default_factory=lambda: GRID_PRESETS["medium"][0])
    eth_space: np.ndarray = field(default_factory=lambda: theta_grid(GRID_PRESETS["medium"][1]))
    v_space: np.ndarray = field(default_factory=lambda: np.linspace(0.1, 1.0, 10))
    w_space: np.ndarray = field(default_factory=lambda: np.linspace(-1.0, 1.0, 11))
    Q: np.ndarray = field(default_factory=lambda: C.COST.Q.copy())
    q: float = C.COST.q
    R: np.ndarray = field(default_factory=lambda: C.COST.R.copy())
    gamma: float = C.COST.gamma
    num_evals: int = 10  # number of policy evaluations in each iteration
    collision_margin: float = 0.0  # extra radius added to obstacles in the risk term
    output_dir: str = "results/models"
    # --- additions ---
    noise_scale: float = 1.0  # model noise = noise_scale * sigma (0 -> deterministic model)
    collision_penalty: float = 1000.0  # lambda in the stage cost
    risk_sigma_floor: float = 0.01  # [m] smoothing of the collision indicator when noise_scale -> 0
    risk_sigma: float = -1.0  # [m] if > 0, overrides the risk-term sigma (used by the 2x2 ablation)
    value_type: str = "grid"  # "grid" (Part 2) or "rbf" (Part 3)
    # RBF features (eq. 6): centers on a stride sub-lattice, lengthscales in grid-index units.
    # beta_t = 1 / (2 ls_t^2), beta_e = 1 / (2 ls_e^2); alpha is absorbed into theta.
    rbf_stride_t: int = 1
    rbf_stride_e: int = 2
    rbf_ls_t: float = 0.8
    rbf_ls_e: float = 1.6
    rbf_ridge: float = 1e-4
    rbf_fit: str = "ls"  # "ls" (ridge least squares) or "avg" (normalized-RBF averager, contraction)
    device: str = "cuda"
    max_queries: int = 16_000_000  # GPU chunk size (interpolation queries per kernel call)
    online_mode: str = "lookahead"  # "lookahead" (greedy w.r.t. V at the continuous state) or "lookup"


def _gh3_nodes(sig):
    """Tensor-product 3-point Gauss-Hermite rule for N(0, diag(sig)^2) over dims with sig > 0."""
    base = [(-math.sqrt(3.0), 1 / 6), (0.0, 2 / 3), (math.sqrt(3.0), 1 / 6)]
    per_dim = [[(z * s, w) for z, w in base] if s > 0 else [(0.0, 1.0)] for s in sig]
    nodes, wts = [], []
    for a, wa in per_dim[0]:
        for b, wb in per_dim[1]:
            for c, wc in per_dim[2]:
                nodes.append((a, b, c))
                wts.append(wa * wb * wc)
    return np.array(nodes), np.array(wts)


class GPI:
    def __init__(self, config: GpiConfig):
        self.config = cfg = config
        dev = cfg.device
        if dev.startswith("cuda") and not torch.cuda.is_available():
            dev = cfg.device = "cpu"
        self.device = dev
        self.T = C.T_PERIOD
        f32 = dict(dtype=torch.float32, device=dev)
        self.grid = ErrorGrid(cfg.ex_space, cfg.ey_space, len(cfg.eth_space), dev)
        g = self.grid
        # reference and its increments (r_t - r_{t+1}, wrap(alpha_t - alpha_{t+1}))
        ref = np.array([cfg.traj(k) for k in range(self.T)], dtype=float)
        dref = ref - np.roll(ref, -1, axis=0)
        dref[:, 2] = C.wrap(dref[:, 2])
        self.ref = torch.tensor(ref, **f32)
        self.dref = torch.tensor(dref, **f32)
        # controls
        vv, ww = np.meshgrid(cfg.v_space, cfg.w_space, indexing="ij")
        self.actions = torch.tensor(np.stack([vv.ravel(), ww.ravel()], 1), **f32)  # (nA, 2)
        self.nA = self.actions.shape[0]
        # noise quadrature
        nodes, wts = _gh3_nodes(C.SIGMA * cfg.noise_scale)
        self.w_nodes = torch.tensor(nodes, **f32)
        self.w_wts = torch.tensor(wts, **f32)
        self.nW = len(wts)
        self.sig_risk = max(float(C.SIGMA[0] * cfg.noise_scale), cfg.risk_sigma_floor)
        if cfg.risk_sigma > 0:
            self.sig_risk = float(cfg.risk_sigma)
        self.obs = torch.tensor(cfg.obstacles, **f32)
        # all grid nodes, flattened in (ix, iy, ith) order to match V tables of shape (T, nx, ny, nth)
        EX, EY, ETH = torch.meshgrid(g.ex, g.ey, g.th, indexing="ij")
        self.E_all = torch.stack([EX.ravel(), EY.ravel(), ETH.ravel()], 1)  # (S, 3)
        self.S = self.E_all.shape[0]
        self.Qm = torch.tensor(cfg.Q, **f32)
        self.Rm = torch.tensor(cfg.R, **f32)
        self.V = self.init_value_function()
        self.pi = torch.zeros(self.T, self.S, dtype=torch.long, device=dev)
        self.history = []
        self._online_eval = None

    # ------------------------------------------------------------------------------------
    # Online control
    # ------------------------------------------------------------------------------------
    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        lookahead: u = argmin_u  l(t,e,u) + gamma E[V(t+1, g(t,e,u,w))] at the continuous error state e
                   (V from the offline GPI; one small batched evaluation, no optimization).
        lookup:    u = pi(t, nearest grid node of e)   (pure table lookup).
        """
        e = C.error_state(t, cur_state)
        tt = t % self.T
        if self.config.online_mode == "lookup":
            ix, iy, ith = self.state_metric_to_index(e)
            a = int(self.pi[tt, (ix * self.grid.ny + iy) * self.grid.nth + ith])
            return self.actions[a].cpu().numpy().astype(float)
        if self._online_eval is None:
            self._build_online_evaluator()
        with torch.no_grad():
            E = torch.tensor(e, dtype=torch.float32, device=self.device).expand(1, self.nA, 3)
            Qv = self.q_backup(torch.tensor([tt], device=self.device), E, self.actions[None], self._online_eval)
            a = int(torch.argmin(Qv[0]))
        return self.actions[a].cpu().numpy().astype(float)

    def reset(self):
        pass

    def _build_online_evaluator(self):
        self._online_eval = (self.V.evaluator(exact=True) if isinstance(self.V, FeatureValueFunction)
                             else self.V.evaluator())

    # ------------------------------------------------------------------------------------
    # Index <-> metric conversions (starter API)
    # ------------------------------------------------------------------------------------
    def state_metric_to_index(self, metric_state: np.ndarray) -> tuple:
        m = torch.as_tensor(np.asarray(metric_state, dtype=np.float32), device=self.device)
        ix, iy, ith = self.grid.nearest(m[..., 0], m[..., 1], m[..., 2])
        return int(ix), int(iy), int(ith)

    def state_index_to_metric(self, state_index: tuple) -> np.ndarray:
        ix, iy, ith = state_index
        g = self.grid
        return np.array([float(g.ex[ix]), float(g.ey[iy]), float(g.th[ith])])

    def control_metric_to_index(self, control_metric: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        v: np.ndarray = np.digitize(control_metric[0], self.config.v_space, right=True)
        w: np.ndarray = np.digitize(control_metric[1], self.config.w_space, right=True)
        return v, w

    def control_index_to_metric(self, v: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self.config.v_space[v], self.config.w_space[w]

    # ------------------------------------------------------------------------------------
    # Model pieces
    # ------------------------------------------------------------------------------------
    def compute_transition_matrix(self, t, state_index, a):
        """Explicit transition distribution p(e' | t, e, u) for ONE (state, action) pair, as a dict
        {(ix', iy', ith'): prob}. The GPU solver never materializes this table (it integrates the same
        distribution on the fly); this method exists for inspection and unit tests."""
        g = self.grid
        e = torch.tensor(self.state_index_to_metric(state_index), dtype=torch.float32, device=self.device)
        mean = self._next_mean(torch.tensor([t % self.T], device=self.device), e.view(1, 1, 3),
                               self.actions[a].view(1, 1, 2))
        probs = {}
        for wk, ok in zip(self.w_nodes, self.w_wts):
            fx = float(g.frac_x(mean[0] + wk[0]).squeeze())
            fy = float(g.frac_y(mean[1] + wk[1]).squeeze())
            fth = float(g.frac_th(mean[2] + wk[2]).squeeze())
            ix0, iy0, it0 = int(math.floor(fx)), int(math.floor(fy)), int(math.floor(fth))
            for dx in (0, 1):
                for dy in (0, 1):
                    for dth in (0, 1):
                        wx = (fx - ix0) if dx else 1 - (fx - ix0)
                        wy = (fy - iy0) if dy else 1 - (fy - iy0)
                        wt = (fth - it0) if dth else 1 - (fth - it0)
                        key = (min(ix0 + dx, g.nx - 1), min(iy0 + dy, g.ny - 1), (it0 + dth) % g.nth)
                        probs[key] = probs.get(key, 0.0) + float(ok) * wx * wy * wt
        return probs

    def compute_stage_costs(self, E, U):
        """Tracking + control part of l(e, u) (the collision-risk part depends on t, see q_backup)."""
        Q, R = self.Qm, self.Rm
        ex, ey, eth = E[..., 0], E[..., 1], E[..., 2]
        v, w = U[..., 0], U[..., 1]
        return (Q[0, 0] * ex * ex + 2 * Q[0, 1] * ex * ey + Q[1, 1] * ey * ey
                + self.config.q * (1 - torch.cos(eth)) ** 2
                + R[0, 0] * v * v + 2 * R[0, 1] * v * w + R[1, 1] * w * w)

    def _next_mean(self, tt, E, U):
        """g(t, e, u, 0) for tt (Nt,), E (Nt, M, 3), U (Nt or 1, M, 2) -> 3 tensors (Nt, M)."""
        dt = C.DT
        alpha = self.ref[tt, 2][:, None]
        v, om = U[..., 0], U[..., 1]
        half = om * dt / 2
        s = dt * torch.sinc(half / math.pi)  # torch.sinc(x) = sin(pi x)/(pi x)
        phi = E[..., 2] + alpha + half
        mx = E[..., 0] + s * v * torch.cos(phi) + self.dref[tt, 0][:, None]
        my = E[..., 1] + s * v * torch.sin(phi) + self.dref[tt, 1][:, None]
        mth = wrap_t(E[..., 2] + dt * om + self.dref[tt, 2][:, None])
        return mx, my, mth

    def collision_risk(self, px, py):
        """P(next position outside F) under N(0, sig_risk^2 I) noise (half-plane approximation)."""
        sig = self.sig_risk
        m = self.config.collision_margin
        p_free = torch.ones_like(px)
        for i in range(self.obs.shape[0]):
            cx, cy, r = self.obs[i]
            d = torch.sqrt((px - cx) ** 2 + (py - cy) ** 2) - (r + C.ROBOT_RADIUS + m)
            p_free = p_free * (1 - torch.special.ndtr(-d / sig))
        W = C.WORKSPACE
        p_oob = (torch.special.ndtr(-(W - px) / sig) + torch.special.ndtr(-(W + px) / sig)
                 + torch.special.ndtr(-(W - py) / sig) + torch.special.ndtr(-(W + py) / sig))
        return torch.clamp(1 - p_free + p_oob, max=1.0)

    def q_backup(self, tt, E, U, value_fn):
        """Q(t, e, u) = l(t, e, u) + gamma * E_w V(t+1, g(t, e, u, w)).
        tt: (Nt,) long, E: (Nt, M, 3), U: (Nt or 1, M, 2), value_fn(tn, qx, qy, qth) -> (Nt, M')."""
        mx, my, mth = self._next_mean(tt, E, U)
        tn = (tt + 1) % self.T
        ell = self.compute_stage_costs(E, U)
        px = mx + self.ref[tn, 0][:, None]
        py = my + self.ref[tn, 1][:, None]
        ell = ell + self.config.collision_penalty * self.collision_risk(px, py)
        wn = self.w_nodes
        qx = (mx[..., None] + wn[:, 0]).flatten(1)
        qy = (my[..., None] + wn[:, 1]).flatten(1)
        qth = (mth[..., None] + wn[:, 2]).flatten(1)
        vals = value_fn(tn, qx, qy, qth).view(*mx.shape, self.nW)
        return ell + self.config.gamma * (vals * self.w_wts).sum(-1)

    # ------------------------------------------------------------------------------------
    # GPI
    # ------------------------------------------------------------------------------------
    def init_value_function(self):
        cfg = self.config
        args = (self.T, cfg.ex_space, cfg.ey_space, cfg.eth_space)
        if cfg.value_type == "grid":
            return GridValueFunction(*args, device=self.device)
        if cfg.value_type == "rbf":
            return FeatureValueFunction(*args, device=self.device, stride_t=cfg.rbf_stride_t,
                                        stride_e=cfg.rbf_stride_e, ls_t=cfg.rbf_ls_t, ls_e=cfg.rbf_ls_e,
                                        ridge=cfg.rbf_ridge, fit=cfg.rbf_fit)
        raise ValueError(cfg.value_type)

    def evaluate_value_function(self):
        """Values of the current representation at all grid nodes (renders theta^T phi for RBF)."""
        return self.V.grid_values()

    @torch.no_grad()
    def policy_improvement(self):
        """Greedy policy and its one-step values: pi(t,e) = argmin_u Q(t,e,u), V <- min_u Q."""
        interp = self.V.evaluator() if isinstance(self.V, GridValueFunction) else self.V.evaluator(exact=False)
        S, nA, nW = self.S, self.nA, self.nW
        na = max(1, min(nA, self.config.max_queries // (S * nW)))
        nt = max(1, self.config.max_queries // (S * nW * na))
        Vnew = torch.empty(self.T, S, device=self.device)
        pi = torch.empty(self.T, S, dtype=torch.long, device=self.device)
        for t0 in range(0, self.T, nt):
            tt = torch.arange(t0, min(t0 + nt, self.T), device=self.device)
            best = torch.full((len(tt), S), float("inf"), device=self.device)
            arg = torch.zeros((len(tt), S), dtype=torch.long, device=self.device)
            for a0 in range(0, nA, na):
                acts = self.actions[a0:a0 + na]
                k = acts.shape[0]
                E = self.E_all[None, :, None, :].expand(len(tt), S, k, 3).reshape(len(tt), S * k, 3)
                U = acts[None, None].expand(1, S, k, 2).reshape(1, S * k, 2)
                Qv = self.q_backup(tt, E, U, interp).view(len(tt), S, k)
                qmin, qarg = Qv.min(-1)
                better = qmin < best
                best = torch.where(better, qmin, best)
                arg = torch.where(better, qarg + a0, arg)
            Vnew[tt] = best
            pi[tt] = arg
        return Vnew, pi

    @torch.no_grad()
    def policy_evaluation(self):
        """One Bellman backup under the fixed policy: V <- l_pi + gamma P_pi V (then refit V)."""
        interp = self.V.evaluator() if isinstance(self.V, GridValueFunction) else self.V.evaluator(exact=False)
        S, nW = self.S, self.nW
        nt = max(1, self.config.max_queries // (S * nW))
        Vnew = torch.empty(self.T, S, device=self.device)
        for t0 in range(0, self.T, nt):
            tt = torch.arange(t0, min(t0 + nt, self.T), device=self.device)
            E = self.E_all[None].expand(len(tt), S, 3)
            U = self.actions[self.pi[tt]]
            Vnew[tt] = self.q_backup(tt, E, U, interp)
        return Vnew

    def _shape(self, Vflat):
        g = self.grid
        return Vflat.view(self.T, g.nx, g.ny, g.nth)

    def compute_policy(self, num_iters: int, tol: float = 1e-3, verbose: bool = True) -> None:
        """Modified policy iteration: [improvement -> num_evals evaluation sweeps] x num_iters.
        For the RBF value function every sweep is a projected backup: theta <- LS-fit(Bellman targets)."""
        t_start = perf_counter()
        if self.device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        for it in range(num_iters):
            t0 = perf_counter()
            V_old = self.V.grid_values().clone()
            Vg, pi_new = self.policy_improvement()
            changed = float((pi_new != self.pi).float().mean())
            self.pi = pi_new
            self.V.update_all(self._shape(Vg))
            fit_err = float((self.V.grid_values() - self._shape(Vg)).abs().max())
            for _ in range(self.config.num_evals):
                y = self._shape(self.policy_evaluation())
                self.V.update_all(y)
            delta = float((self.V.grid_values() - V_old).abs().max())
            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            rec = dict(iter=it, delta=delta, policy_changed=changed, fit_err_max=fit_err,
                       V_mean=float(self.V.grid_values().mean()), sec=perf_counter() - t0)
            self.history.append(rec)
            if verbose:
                print(json.dumps(rec), flush=True)
            if it > 0 and changed < 1e-4 and delta < tol:
                break
        self.offline_sec = perf_counter() - t_start
        self.peak_gpu_mb = (torch.cuda.max_memory_allocated() / 2 ** 20) if self.device.startswith("cuda") else 0.0
        self._online_eval = None

    # ------------------------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------------------------
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        cfg = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in asdict(self.config).items()
               if k != "traj"}
        meta = dict(config=cfg, history=self.history, offline_sec=getattr(self, "offline_sec", None),
                    peak_gpu_mb=getattr(self, "peak_gpu_mb", None), n_params=self.V.n_params(),
                    n_states=self.T * self.S, n_actions=self.nA)
        vsd = self.V.state_dict()
        np.savez_compressed(path, meta=json.dumps(meta), pi=self.pi.cpu().numpy().astype(np.int16),
                            **{f"V_{k}": np.asarray(v) for k, v in vsd.items()})

    @classmethod
    def load(cls, path, device="cpu", **overrides):
        d = np.load(path, allow_pickle=False)
        meta = json.loads(str(d["meta"]))
        cfg = dict(meta["config"])
        cfg.update(device=device, **overrides)
        for k in ("obstacles", "ex_space", "ey_space", "eth_space", "v_space", "w_space", "Q", "R"):
            cfg[k] = np.asarray(cfg[k])
        obj = cls(GpiConfig(**cfg))
        vsd = {k[2:]: d[k] for k in d.files if k.startswith("V_")}
        obj.V.load_state_dict({k: (v if v.ndim else v.item()) for k, v in vsd.items()})
        obj.pi = torch.as_tensor(d["pi"].astype(np.int64), device=device)
        obj.meta = meta
        obj._build_online_evaluator()  # eager, so the first timed control step is not an init step
        return obj
