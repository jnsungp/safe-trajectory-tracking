import casadi
import numpy as np

import common as C


def _sinc_ca(x):
    """sin(x)/x as a Taylor series. Here |x| = |w| dt / 2 <= 0.25, so the truncation
    error is < 1e-9 and the expression is smooth (no 0/0 at x = 0)."""
    x2 = x * x
    return 1 - x2 / 6 + x2 * x2 / 120 - x2 * x2 * x2 / 5040


def _f_ca(x, u):
    """Noise-free exact-discretization dynamics f(x, u, 0), eq. (1)."""
    half = u[1] * C.DT / 2
    s = C.DT * _sinc_ca(half)
    return casadi.vertcat(
        x[0] + s * casadi.cos(x[2] + half) * u[0],
        x[1] + s * casadi.sin(x[2] + half) * u[0],
        x[2] + C.DT * u[1],
    )


class CEC:
    """Receding-horizon certainty-equivalent control, eq. (4)-(5).

    Decision variables (multiple shooting): X = [x_0..x_N] (3 x N+1), U = [u_0..u_{N-1}] (2 x N),
    and obstacle slacks S >= 0 (n_obs x N). The obstacle constraints
        ||p_k - c_i||^2 + S_ik >= (r_i + r_robot + margin)^2,  k = 1..N
    are enforced with an exact L1 penalty rho * sum(S): with rho large enough this is identical to
    the hard constraint whenever the hard problem is feasible, and it keeps IPOPT well-posed when
    noise has already pushed the robot inside an obstacle (where the hard NLP has no solution).
    The NLP is built once; the current state and the reference window are NLP parameters.
    """

    def __init__(self, horizon: int = 10, terminal_weight: float = 5.0, margin: float = 0.0,
                 slack_penalty: float = 1e4, cp: C.CostParams = C.COST, max_iter: int = 300) -> None:
        self.N = N = horizon
        self.cp = cp
        self.margin = margin
        n_obs = len(C.OBSTACLES)
        X = casadi.SX.sym("X", 3, N + 1)
        U = casadi.SX.sym("U", 2, N)
        S = casadi.SX.sym("S", n_obs, N)
        P = casadi.SX.sym("P", 3 + 3 * (N + 1))  # [x_0, r_tau..r_tau+N (x, y, alpha)]
        x0 = P[0:3]
        Rf = casadi.reshape(P[3:], 3, N + 1)
        Q = casadi.DM(cp.Q)
        Rm = casadi.DM(cp.R)

        def state_cost(k):
            ep = X[0:2, k] - Rf[0:2, k]
            return casadi.mtimes([ep.T, Q, ep]) + cp.q * (1 - casadi.cos(X[2, k] - Rf[2, k])) ** 2

        cost = 0
        cons = [X[:, 0] - x0]
        for k in range(N):
            cost += cp.gamma ** k * (state_cost(k) + casadi.mtimes([U[:, k].T, Rm, U[:, k]]))
            cons.append(X[:, k + 1] - _f_ca(X[:, k], U[:, k]))
        cost += cp.gamma ** N * terminal_weight * state_cost(N)
        obs_cons = []
        for k in range(1, N + 1):
            for i, (cx, cy, r) in enumerate(C.OBSTACLES):
                rr = r + C.ROBOT_RADIUS + margin
                obs_cons.append((X[0, k] - cx) ** 2 + (X[1, k] - cy) ** 2 + S[i, k - 1] - rr ** 2)
        cost += slack_penalty * casadi.sum1(casadi.vec(S))
        n_eq = 3 * (N + 1)
        g = casadi.vertcat(*cons, *obs_cons)
        z = casadi.vertcat(casadi.vec(X), casadi.vec(U), casadi.vec(S))

        self.nX, self.nU, self.nS = 3 * (N + 1), 2 * N, n_obs * N
        self.lbg = np.r_[np.zeros(n_eq), np.zeros(len(obs_cons))]
        self.ubg = np.r_[np.zeros(n_eq), np.full(len(obs_cons), np.inf)]
        lbX = np.tile([-C.WORKSPACE, -C.WORKSPACE, -np.inf], (N + 1, 1))
        ubX = np.tile([C.WORKSPACE, C.WORKSPACE, np.inf], (N + 1, 1))
        lbX[0, :2], ubX[0, :2] = -np.inf, np.inf  # x_0 is pinned by the equality constraint
        self.lbx = np.r_[lbX.ravel(), np.tile(C.U_LB, N), np.zeros(self.nS)]
        self.ubx = np.r_[ubX.ravel(), np.tile(C.U_UB, N), np.full(self.nS, np.inf)]

        opts = {
            "ipopt.print_level": 0, "ipopt.sb": "yes", "print_time": False,
            "ipopt.max_iter": max_iter, "ipopt.tol": 1e-6,
        }
        self.solver = casadi.nlpsol("cec", "ipopt", {"x": z, "f": cost, "g": g, "p": P}, opts)
        self.reset()

    # ---------------------------------------------------------------------------------
    def reset(self):
        self.z_prev = None
        self.n_calls = 0
        self.n_fail = 0
        self.n_slack = 0  # steps whose returned plan uses obstacle slack (plan itself not collision-free)
        self.iters = []
        self.pred_next_clear = []  # clearance of the predicted x_{t+1} (what the plan believes)
        self.pred_min_clear = []  # min predicted clearance over the horizon

    def _unpack(self, z):
        N = self.N
        X = z[: self.nX].reshape(N + 1, 3).T
        U = z[self.nX: self.nX + self.nU].reshape(N, 2).T
        S = z[self.nX + self.nU:].reshape(N, -1).T
        return X, U, S

    def _initial_guess(self, cur_state):
        N = self.N
        if self.z_prev is None:
            U = np.tile([0.5, 0.0], (N, 1)).T
        else:  # shift the previous solution by one step
            _, Up, _ = self._unpack(self.z_prev)
            U = np.c_[Up[:, 1:], Up[:, -1:]]
        X = np.zeros((3, N + 1))
        X[:, 0] = cur_state
        for k in range(N):  # roll the guess through the model so it is dynamically consistent
            X[:, k + 1] = C.f(X[:, k], U[:, k])
            X[2, k + 1] = X[2, k] + C.wrap(X[2, k + 1] - X[2, k])  # keep heading continuous
        S = np.zeros((len(C.OBSTACLES), N))
        return np.r_[X.T.ravel(), U.T.ravel(), S.T.ravel()]

    def __call__(self, t: int, cur_state: np.ndarray, cur_ref_state: np.ndarray) -> np.ndarray:
        """
        Given the time step, current state, and reference state, return the control input.
        Args:
            t (int): time step
            cur_state (np.ndarray): current state
            cur_ref_state (np.ndarray): reference state
        Returns:
            np.ndarray: control input
        """
        N = self.N
        refs = C.ref(np.arange(t, t + N + 1))  # the reference is known in advance
        p = np.r_[cur_state, refs.ravel()]
        sol = self.solver(x0=self._initial_guess(cur_state), lbx=self.lbx, ubx=self.ubx,
                          lbg=self.lbg, ubg=self.ubg, p=p)
        st = self.solver.stats()
        self.n_calls += 1
        self.iters.append(st.get("iter_count", -1))
        z = np.array(sol["x"]).ravel()
        if not st["success"]:
            self.n_fail += 1
        self.z_prev = z
        X, U, S = self._unpack(z)
        self.n_slack += int(S.max() > 1e-6)
        clr = C.clearance(X[:2, 1:].T)
        self.pred_next_clear.append(clr[0])
        self.pred_min_clear.append(clr.min())
        return np.clip(U[:, 0], C.U_LB, C.U_UB)

    def stats(self):
        return dict(
            solver_fail_rate=self.n_fail / max(self.n_calls, 1),
            plan_slack_rate=self.n_slack / max(self.n_calls, 1),
            solver_iters_mean=float(np.mean(self.iters)) if self.iters else np.nan,
            pred_next_clear=np.array(self.pred_next_clear),
        )
