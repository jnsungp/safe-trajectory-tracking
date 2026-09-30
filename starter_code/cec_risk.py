"""Receding-horizon CEC that optimizes GPI's objective: tracking + lambda * collision probability.

The nominal CEC (cec.py) treats the obstacles as hard constraints on noise-free predictions. This
controller keeps the noise-free predictions and adds the stage cost that tabular GPI minimizes,

    l(t, e, u) = p^T Q p + q (1 - cos eth)^2 + u^T R u + lambda * P(p_{t+1} + w in an inflated obstacle),

with w ~ N(0, sigma_risk^2 I). P is the exact disk probability used by GPI (noncentral chi-square CDF,
tabulated every 0.5 mm and smoothed by a cubic B-spline so that IPOPT gets exact derivatives), plus
GPI's workspace term. Along the nominal prediction this is the objective of the "deterministic
transition + 4 cm risk" GPI policy of the 2x2 ablation, truncated to N steps. Options:

hard=True         also keep CEC's obstacle constraints ||p - c_i||^2 + s_ik >= r^2 (L1-penalized slacks).
                  The probability is flat (0 or 1) outside a band of about 3 sigma around each boundary;
                  the constraint supplies a gradient everywhere else.
value_model       terminal cost gamma^N V(t+N, e_N) from a trained GPI model instead of CEC's quadratic
                  terminal cost, as a smooth local spline of GPI's table (LocalValue). IPOPT stalls on
                  the kinks of GPI's multilinear interpolation.
risk_substeps=M   path-aware risk, as in GPI: the risk of step k is the largest risk at the arc points
                  s = 1/M, ..., 1 (noise std s * sigma), written with an epigraph variable r_k. With hard=True
                  the obstacle constraints also apply at those arc points.
policy_seed=True  besides the shifted previous plan, also start IPOPT from N steps of the GPI policy
                  (value_model) on the noise-free model, and keep the better of the two solutions.
"""
import casadi
import numpy as np
from scipy.stats import ncx2

import common as C
from cec import _f_ca, _sinc_ca


def _phi(z):
    return 0.5 * (1 + casadi.erf(z / np.sqrt(2.0)))


def disk_risk_table(radius, sigma, step=0.0005):
    """P(||c + sigma z - center|| <= radius) as a function of the distance ||c - center||."""
    d = np.arange(0.0, radius + 10 * sigma + step, step)
    return d, ncx2.cdf((radius / sigma) ** 2, 2, (d / sigma) ** 2)


def gpi_risk_sigma(cfg):
    """Standard deviation that a GPI model uses in its risk term."""
    return cfg.risk_sigma if cfg.risk_sigma > 0 else max(C.SIGMA[0] * cfg.noise_scale, cfg.risk_sigma_floor)


def _bspline_kernel(u):
    """Uniform cubic B-spline basis function centered at 0 (support [-2, 2], twice differentiable)."""
    a = casadi.fabs(u)
    return casadi.if_else(a < 1, (4 - 6 * a ** 2 + 3 * a ** 3) / 6, casadi.if_else(a < 2, (2 - a) ** 3 / 6, 0))


class LocalValue:
    """GPI's value function around a point, as a smooth spline that IPOPT can differentiate.

    For every solve, GPI's value (its own multilinear interpolation) is sampled on a uniform window of
    nodes around the initial guess's terminal error: 0.1 m in position, which coincides with GPI's grid
    wherever |e| <= 0.8 m, and GPI's 9-degree heading step over the full circle. The samples are the
    control points of a cubic B-spline (Schoenberg's variation-diminishing spline): it stays within the
    range of nearby table values, so the optimizer cannot exploit overshoot, and only three variables
    enter it. Outside the position window a quadratic penalty acts as a trust region; the window moves
    with the plan at every step.
    """

    def __init__(self, path, n_pos=17, h_pos=0.1, trust_penalty=1e3):
        import torch
        from gpi import GPI
        g = GPI.load(path, device="cpu")
        self.cfg = g.config
        self.evaluator = g.V.evaluator()
        self.torch = torch
        n_th = g.grid.nth + 6  # full circle plus three nodes on each side
        self.n = np.array([n_pos, n_pos, n_th])
        self.h = np.array([h_pos, h_pos, 2 * np.pi / g.grid.nth])
        self.size = int(np.prod(self.n))
        e = casadi.SX.sym("e", 3)  # heading error within +-pi of the window center
        origin = casadi.SX.sym("o", 3)
        coef = casadi.SX.sym("c", self.size)
        K = []
        outside = 0
        for d in range(3):
            u = (e[d] - origin[d]) / self.h[d]
            uc = casadi.fmax(casadi.fmin(u, self.n[d] - 2), 1)  # where the basis functions sum to one
            if d < 2:
                outside += (u - uc) ** 2
            K.append(casadi.vertcat(*[_bspline_kernel(uc - i) for i in range(self.n[d])]))
        nx, ny = int(self.n[0]), int(self.n[1])
        value = trust_penalty * outside
        for l in range(int(self.n[2])):
            Cl = casadi.reshape(coef[l * nx * ny: (l + 1) * nx * ny], nx, ny)
            value += K[2][l] * casadi.mtimes([K[0].T, Cl, K[1]])
        self.fn = casadi.Function("V_local", [e, origin, coef], [value])

    def window(self, tn, e_guess):
        """Origin and control points of the window centered on e_guess at time index tn."""
        origin = (np.round(e_guess / self.h) - self.n // 2) * self.h
        axes = [origin[d] + self.h[d] * np.arange(self.n[d]) for d in range(3)]
        qx, qy, qt = np.meshgrid(*axes, indexing="ij")
        torch = self.torch
        with torch.no_grad():
            q = [torch.tensor(a.ravel(order="F")[None], dtype=torch.float32) for a in (qx, qy, qt)]
            vals = self.evaluator(torch.tensor([tn % C.T_PERIOD]), *q).numpy().ravel().astype(float)
        return origin, vals

    def center(self, origin):
        return origin + self.h * (self.n // 2)


class RiskCEC:
    """CEC with GPI's calibrated collision-probability penalty.

    lam: penalty weight lambda (GPI default 1000). risk_sigma: position-noise standard deviation in the
    risk term (GPI: 0.04 * noise scale with a 1 cm floor). hard: keep the nominal obstacle constraints.
    value_model: GPI model whose value function is the terminal cost (lam, risk_sigma and risk_substeps
    must match). risk_substeps: arc points per step in the risk term (1 = next sample only).
    policy_seed: also start from the GPI policy's plan (needs value_model); with warm_start=False, start
    only from it, so the NLP refines GPI's plan instead of choosing between the two.
    """

    def __init__(self, horizon: int = 10, lam: float = 1000.0, risk_sigma: float = 0.04,
                 hard: bool = False, value_model: str | None = None, risk_substeps: int = 1,
                 policy_seed: bool = False, warm_start: bool = True, terminal_weight: float = 5.0,
                 slack_penalty: float = 1e4, cp: C.CostParams = C.COST, max_iter: int = 300) -> None:
        self.N = N = horizon
        self.cp = cp
        self.lam = lam
        self.risk_sigma = risk_sigma
        self.hard = hard
        self.warm_start = warm_start
        M = self.M = risk_substeps
        n_obs = len(C.OBSTACLES)
        self.value = None
        self.policy = None
        if value_model is not None:
            self.value = LocalValue(value_model)
            vcfg = self.value.cfg
            if abs(vcfg.collision_penalty - lam) > 1e-9:
                raise ValueError(f"value model lambda {vcfg.collision_penalty} != {lam}")
            if abs(gpi_risk_sigma(vcfg) - risk_sigma) > 1e-9:
                raise ValueError(f"value model risk sigma {gpi_risk_sigma(vcfg)} != {risk_sigma}")
            if vcfg.risk_substeps != M:
                raise ValueError(f"value model risk_substeps {vcfg.risk_substeps} != {M}")
        if policy_seed:
            if value_model is None:
                raise ValueError("policy_seed needs value_model")
            from gpi import GPI
            self.policy = GPI.load(value_model, device="cpu", online_mode="lookahead")
        X = casadi.SX.sym("X", 3, N + 1)
        U = casadi.SX.sym("U", 2, N)
        S = casadi.SX.sym("S", n_obs if hard else 0, N)
        Rk = casadi.SX.sym("R", N if M > 1 else 0)  # epigraph of the per-step path risk
        n_ref = 3 + 3 * (N + 1)
        n_val = 3 + self.value.size if self.value is not None else 0
        P = casadi.SX.sym("P", n_ref + n_val)  # [x_0, r_tau..r_tau+N (x, y, alpha), window origin, control points]
        x0 = P[0:3]
        Rf = casadi.reshape(P[3:n_ref], 3, N + 1)
        Q = casadi.DM(cp.Q)
        Rm = casadi.DM(cp.R)
        radius = C.OBSTACLES[0, 2] + C.ROBOT_RADIUS
        assert np.allclose(C.OBSTACLES[:, 2] + C.ROBOT_RADIUS, radius)
        fracs = [m / M for m in range(1, M + 1)]
        p_disk = []
        for s in fracs:
            sig = max(s * risk_sigma, 1e-3)
            d_tab, p_tab = disk_risk_table(radius, sig)
            p_disk.append((casadi.interpolant("p_disk", "bspline", [d_tab.tolist()], p_tab.tolist(), {}),
                           float(d_tab[-1]), sig))

        def risk(px, py, level=-1):
            fn, d_max, s = p_disk[level]
            p_obs = 0
            for cx, cy, _ in C.OBSTACLES:
                d = casadi.sqrt((px - cx) ** 2 + (py - cy) ** 2 + 1e-12)
                p_obs += fn(casadi.fmin(d, d_max))
            W = C.WORKSPACE
            p_in = (_phi((W - px) / s) - _phi((-W - px) / s)) * (_phi((W - py) / s) - _phi((-W - py) / s))
            return p_obs + (1 - p_obs) * (1 - p_in)

        def arc_point(k, s):
            """Noise-free position at fraction s of step k (s = 1 is X[:, k + 1] under the dynamics)."""
            if s == 1.0:
                return X[0, k + 1], X[1, k + 1]
            half = U[1, k] * s * C.DT / 2
            step = s * C.DT * _sinc_ca(half) * U[0, k]
            return X[0, k] + step * casadi.cos(X[2, k] + half), X[1, k] + step * casadi.sin(X[2, k] + half)

        def state_cost(k):
            ep = X[0:2, k] - Rf[0:2, k]
            return casadi.mtimes([ep.T, Q, ep]) + cp.q * (1 - casadi.cos(X[2, k] - Rf[2, k])) ** 2

        cost = 0
        cons = [X[:, 0] - x0]
        risk_cons = []
        for k in range(N):
            if M == 1:
                step_risk = risk(X[0, k + 1], X[1, k + 1])
            else:
                step_risk = Rk[k]
                for level, s in enumerate(fracs):
                    risk_cons.append(Rk[k] - risk(*arc_point(k, s), level))
            cost += cp.gamma ** k * (state_cost(k) + casadi.mtimes([U[:, k].T, Rm, U[:, k]]) + lam * step_risk)
            cons.append(X[:, k + 1] - _f_ca(X[:, k], U[:, k]))
        if self.value is None:
            cost += cp.gamma ** N * terminal_weight * state_cost(N)
        else:
            origin = P[n_ref: n_ref + 3]
            center_th = origin[2] + self.value.h[2] * (self.value.n[2] // 2)
            d_th = X[2, N] - Rf[2, N] - center_th
            e_th = casadi.atan2(casadi.sin(d_th), casadi.cos(d_th)) + center_th  # continuous around the window
            eN = casadi.vertcat(X[0, N] - Rf[0, N], X[1, N] - Rf[1, N], e_th)
            cost += cp.gamma ** N * self.value.fn(eN, origin, P[n_ref + 3:])
        obs_cons = []
        if hard:
            for k in range(N):
                for s in fracs:
                    px, py = arc_point(k, s)
                    for i, (cx, cy, r) in enumerate(C.OBSTACLES):
                        rr = r + C.ROBOT_RADIUS
                        obs_cons.append((px - cx) ** 2 + (py - cy) ** 2 + S[i, k] - rr ** 2)
            cost += slack_penalty * casadi.sum1(casadi.vec(S))
        n_eq = 3 * (N + 1)
        g = casadi.vertcat(*cons, *obs_cons, *risk_cons)
        z = casadi.vertcat(casadi.vec(X), casadi.vec(U), casadi.vec(S), Rk)
        self.nX, self.nU, self.nS, self.nR = 3 * (N + 1), 2 * N, S.numel(), Rk.numel()
        n_ineq = len(obs_cons) + len(risk_cons)
        self.lbg = np.r_[np.zeros(n_eq), np.zeros(n_ineq)]
        self.ubg = np.r_[np.zeros(n_eq), np.full(n_ineq, np.inf)]
        self.lbx = np.r_[np.full(self.nX, -np.inf), np.tile(C.U_LB, N), np.zeros(self.nS + self.nR)]
        self.ubx = np.r_[np.full(self.nX, np.inf), np.tile(C.U_UB, N), np.full(self.nS + self.nR, np.inf)]
        opts = {
            "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
            "ipopt.max_iter": max_iter, "ipopt.tol": 1e-6,
        }
        self.solver = casadi.nlpsol("cec_risk", "ipopt", {"x": z, "f": cost, "g": g, "p": P}, opts)
        px_s, py_s = casadi.SX.sym("px"), casadi.SX.sym("py")
        self._risk_levels = [casadi.Function("risk", [px_s, py_s], [risk(px_s, py_s, lv)]) for lv in range(M)]
        self._fracs = fracs
        self.reset()

    # ---------------------------------------------------------------------------------
    def reset(self):
        self.z_prev = None
        self.n_calls = 0
        self.n_fail = 0
        self.n_slack = 0
        self.n_policy = 0  # steps where the policy-seeded solve won
        self.iters = []
        self.pred_risk = []  # risk of the executed first step, as the controller scores it

    def _unpack(self, z):
        N = self.N
        X = z[: self.nX].reshape(N + 1, 3).T
        U = z[self.nX: self.nX + self.nU].reshape(N, 2).T
        S = z[self.nX + self.nU: self.nX + self.nU + self.nS]
        return X, U, S

    def _pack(self, X, U):
        return np.r_[X.T.ravel(), U.T.ravel(), np.zeros(self.nS + self.nR)]

    def _rollout(self, cur_state, U):
        X = np.zeros((3, self.N + 1))
        X[:, 0] = cur_state
        for k in range(self.N):  # roll the guess through the model so it is dynamically consistent
            X[:, k + 1] = C.f(X[:, k], U[:, k])
            X[2, k + 1] = X[2, k] + C.wrap(X[2, k + 1] - X[2, k])  # keep heading continuous
        return X

    def _warm_guess(self, cur_state):
        N = self.N
        if self.z_prev is None:
            U = np.tile([0.5, 0.0], (N, 1)).T
        else:  # shift the previous solution by one step
            _, Up, _ = self._unpack(self.z_prev)
            U = np.c_[Up[:, 1:], Up[:, -1:]]
        return self._pack(self._rollout(cur_state, U), U)

    def _policy_guess(self, t, cur_state):
        N = self.N
        X = np.zeros((3, N + 1))
        U = np.zeros((2, N))
        X[:, 0] = cur_state
        for k in range(N):
            U[:, k] = self.policy(t + k, X[:, k].copy(), C.ref(t + k))  # GPI wraps the heading error
            X[:, k + 1] = C.f(X[:, k], U[:, k])
            X[2, k + 1] = X[2, k] + C.wrap(X[2, k + 1] - X[2, k])
        return self._pack(X, U)

    def _params(self, t, cur_state, z0):
        N = self.N
        p = [cur_state, C.ref(np.arange(t, t + N + 1)).ravel()]
        if self.value is not None:
            X0, _, _ = self._unpack(z0)
            eN = X0[:, N] - C.ref(t + N)
            eN[2] = C.wrap(eN[2])
            origin, coef = self.value.window(t + N, eN)
            p += [origin, coef]
        return np.concatenate(p)

    def _solve(self, t, cur_state, z0):
        p = self._params(t, cur_state, z0)
        sol = self.solver(x0=z0, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, p=p)
        st = self.solver.stats()
        return np.array(sol["x"]).ravel(), float(sol["f"]), bool(st["success"]), st.get("iter_count", -1)

    def plan_risk(self, x, u):
        """Risk of one step from x under u, as the controller scores it."""
        risks = []
        for s, fn in zip(self._fracs, self._risk_levels):
            half = u[1] * s * C.DT / 2
            step = s * C.DT * np.sinc(half / np.pi) * u[0]
            risks.append(float(fn(x[0] + step * np.cos(x[2] + half), x[1] + step * np.sin(x[2] + half))))
        return max(risks)

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        if self.policy is not None and not self.warm_start:  # refine the GPI plan only
            z, f, ok, iters = self._solve(t, cur_state, self._policy_guess(t, cur_state))
            self.n_policy += 1
        else:
            z, f, ok, iters = self._solve(t, cur_state, self._warm_guess(cur_state))
            if self.policy is not None:
                z1, f1, ok1, it1 = self._solve(t, cur_state, self._policy_guess(t, cur_state))
                iters += it1
                if (ok1 or not ok) and f1 < f:
                    z, f, ok = z1, f1, ok1
                    self.n_policy += 1
        self.n_calls += 1
        self.iters.append(iters)
        if not ok:
            self.n_fail += 1
        self.z_prev = z
        X, U, S = self._unpack(z)
        self.n_slack += int(S.size > 0 and S.max() > 1e-6)
        u = np.clip(U[:, 0], C.U_LB, C.U_UB)
        self.pred_risk.append(self.plan_risk(cur_state, u))
        return u

    def stats(self):
        return dict(
            solver_fail_rate=self.n_fail / max(self.n_calls, 1),
            plan_slack_rate=self.n_slack / max(self.n_calls, 1),
            policy_seed_rate=self.n_policy / max(self.n_calls, 1),
            solver_iters_mean=float(np.mean(self.iters)) if self.iters else np.nan,
            pred_risk_sum=float(np.sum(self.pred_risk)),
        )
