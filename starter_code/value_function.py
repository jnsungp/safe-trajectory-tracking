"""Value-function representations for GPI.

Both classes expose the same small interface used by the GPI engine (gpi.py):
    grid_values()          -> tensor (T, nx, ny, nth) of values at the state-grid nodes
    update_all(targets)    -> fit the representation to Bellman targets given at all nodes
    evaluator()            -> callable (t_next, qx, qy, qth) -> values at continuous states
    n_params()             -> number of stored parameters (memory footprint)

Coordinates: position errors live on rectilinear (possibly non-uniform) grids ex_space, ey_space;
the heading error lives on a uniform periodic grid etheta_space = -pi + k * 2pi/nth.
All interpolation is done in "grid-index coordinates" (fractional index along each axis), which
makes multilinear interpolation on a non-uniform grid identical to interpolation on a uniform one.
"""
import math

import numpy as np
import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------
# Grid geometry helpers
# ----------------------------------------------------------------------------------------
def wrap_t(a):
    return torch.remainder(a + math.pi, 2 * math.pi) - math.pi


class ErrorGrid:
    def __init__(self, ex_space, ey_space, nth, device):
        self.device = device
        self.ex = torch.as_tensor(np.asarray(ex_space), dtype=torch.float32, device=device)
        self.ey = torch.as_tensor(np.asarray(ey_space), dtype=torch.float32, device=device)
        self.nx, self.ny, self.nth = len(self.ex), len(self.ey), int(nth)
        self.dth = 2 * math.pi / self.nth
        self.th = -math.pi + self.dth * torch.arange(self.nth, dtype=torch.float32, device=device)

    @staticmethod
    def _frac(q, grid):
        n = grid.numel()
        i = torch.searchsorted(grid, q.contiguous()) - 1
        i = i.clamp(0, n - 2)
        g0, g1 = grid[i], grid[i + 1]
        return i.to(q.dtype) + ((q - g0) / (g1 - g0)).clamp(0.0, 1.0)  # border clamp

    def frac_x(self, q):
        return self._frac(q, self.ex)

    def frac_y(self, q):
        return self._frac(q, self.ey)

    def frac_th(self, q):
        """Fractional periodic index in [0, nth)."""
        return torch.remainder((wrap_t(q) + math.pi) / self.dth, self.nth)

    def nearest(self, ex, ey, eth):
        ix = self.frac_x(ex).round().long()
        iy = self.frac_y(ey).round().long()
        ith = torch.remainder(self.frac_th(eth).round().long(), self.nth)
        return ix, iy, ith


def pad_volume(Vg):
    """(T, nx, ny, nth) -> (T, 1, nth + 2, ny, nx) with periodic padding along theta, laid out for
    F.grid_sample (D = theta, H = y, W = x)."""
    Vp = torch.cat([Vg[..., -1:], Vg, Vg[..., :1]], dim=-1)
    return Vp.permute(0, 3, 2, 1).unsqueeze(1).contiguous()


def make_grid_interpolator(Vg, grid: ErrorGrid):
    """Multilinear interpolation of node values Vg (T, nx, ny, nth).
    Returns f(tn, qx, qy, qth) with tn: (Nt,) long and q*: (Nt, M) -> (Nt, M)."""
    Vp = pad_volume(Vg)
    nx, ny, nth = grid.nx, grid.ny, grid.nth

    def interp(tn, qx, qy, qth):
        Nt, M = qx.shape
        gx = 2 * grid.frac_x(qx) / (nx - 1) - 1
        gy = 2 * grid.frac_y(qy) / (ny - 1) - 1
        gth = 2 * (grid.frac_th(qth) + 1) / (nth + 1) - 1  # +1: shift by the padding slice
        gr = torch.stack([gx, gy, gth], dim=-1).view(Nt, M, 1, 1, 3)
        out = F.grid_sample(Vp[tn], gr, mode="bilinear", padding_mode="border", align_corners=True)
        return out.view(Nt, M)

    return interp


# ----------------------------------------------------------------------------------------
# Starter-code base class (kept for interface compatibility)
# ----------------------------------------------------------------------------------------
class ValueFunction:
    def __init__(self, T: int, ex_space, ey_space, etheta_space):
        self.T = T
        self.ex_space = np.asarray(ex_space)
        self.ey_space = np.asarray(ey_space)
        self.etheta_space = np.asarray(etheta_space)

    def copy_from(self, other):
        raise NotImplementedError

    def update(self, t, ex, ey, etheta, target_value):
        raise NotImplementedError

    def __call__(self, t, ex, ey, etheta):
        raise NotImplementedError

    def copy(self):
        raise NotImplementedError


class GridValueFunction(ValueFunction):
    """Tabular value function: one number per (t, ex, ey, etheta) node, multilinear in between."""

    def __init__(self, T, ex_space, ey_space, etheta_space, device="cpu"):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.grid = ErrorGrid(ex_space, ey_space, len(etheta_space), device)
        self.device = device
        self.table = torch.zeros(T, self.grid.nx, self.grid.ny, self.grid.nth, device=device)

    # --- interface used by GPI -------------------------------------------------------
    def grid_values(self):
        return self.table

    def update_all(self, targets):
        self.table = targets.to(self.table.dtype).contiguous()

    def evaluator(self):
        return make_grid_interpolator(self.table, self.grid)

    def n_params(self):
        return self.table.numel()

    # --- starter-code API -------------------------------------------------------------
    def copy_from(self, other):
        self.table = other.table.clone()

    def update(self, t, ex, ey, etheta, target_value):
        ix, iy, ith = self.grid.nearest(*(torch.as_tensor(a, dtype=torch.float32, device=self.device)
                                          for a in (ex, ey, etheta)))
        self.table[torch.as_tensor(t, device=self.device) % self.T, ix, iy, ith] = torch.as_tensor(
            target_value, dtype=torch.float32, device=self.device)

    def __call__(self, t, ex, ey, etheta):
        q = [torch.as_tensor(a, dtype=torch.float32, device=self.device).reshape(1, -1) for a in (ex, ey, etheta)]
        tn = torch.as_tensor([int(t) % self.T], device=self.device)
        return self.evaluator()(tn, *q).reshape(np.shape(ex))

    def copy(self):
        c = GridValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space, self.device)
        c.copy_from(self)
        return c

    def state_dict(self):
        return {"table": self.table.cpu().numpy()}

    def load_state_dict(self, d):
        self.table = torch.as_tensor(d["table"], device=self.device)


class FeatureValueFunction(ValueFunction):
    """Linear RBF value function  V(t, e) = theta^T phi(t, e)  (Part 3, eq. 6).

    Features are Gaussian kernels  k = exp(-sum_d (z_d - c_d)^2 / (2 l_d^2))  where z = (t, ix, iy, ith)
    are grid-index coordinates (t and ith periodic) and the centers c are a stride-s sub-lattice of
    the state grid. This is the kernel of eq. (6) with a diagonal metric (beta_d = 1 / 2 l_d^2).

    Because the centers form a lattice, the design matrix over all state-grid nodes is a Kronecker
    product Phi = K_t (x) K_x (x) K_y (x) K_th, so the ridge least-squares projection
        theta = argmin ||Phi theta - y||^2 + lam ||theta||^2
    is solved *exactly* with per-dimension eigendecompositions (no SGD, no dense Phi).

    fit="ls":  plain Gaussian features + ridge least squares (above). The LS hat matrix has negative
               weights and ||Pi||_inf >> 1, so projected value iteration is NOT a contraction; near the
               steep collision-penalty jumps it rings and can converge to a negative-biased fixed point.
    fit="avg": normalized RBF features phi_i(z) = k_i(z) / sum_j k_j(z) (rows of Phi sum to 1) and an
               averaging fit theta_i = sum_s k_i(s) y(s) / sum_s k_i(s) (Gordon 1995 "averager"). Both maps
               are convex combinations, so the projected Bellman operator stays a gamma-contraction in
               sup-norm and V >= 0 is preserved. The product-kernel normalization factorizes per
               dimension, so the Kronecker structure (and the speed) is kept.
    """

    def __init__(self, T, ex_space, ey_space, etheta_space, device="cpu", stride_t=1, stride_e=2,
                 ls_t=None, ls_e=None, ridge=1e-4, fit="ls"):
        super().__init__(T, ex_space, ey_space, etheta_space)
        self.grid = ErrorGrid(ex_space, ey_space, len(etheta_space), device)
        self.device = device
        self.stride_t, self.stride_e = stride_t, stride_e
        self.ls_t = float(ls_t if ls_t is not None else max(0.8 * stride_t, 0.5))
        self.ls_e = float(ls_e if ls_e is not None else 0.8 * stride_e)
        self.ridge = ridge
        self.fit = fit
        g = self.grid
        self.ct = torch.arange(0, T, stride_t, dtype=torch.float32, device=device)
        self.cx = torch.arange(0, g.nx, stride_e, dtype=torch.float32, device=device)
        self.cy = torch.arange(0, g.ny, stride_e, dtype=torch.float32, device=device)
        self.cth = torch.arange(0, g.nth, stride_e, dtype=torch.float32, device=device)
        # 1-D design matrices: rows = grid nodes, cols = centers
        self.Kt = self._k(torch.arange(T, dtype=torch.float32, device=device), self.ct, self.ls_t, period=T)
        self.Kx = self._k(torch.arange(g.nx, dtype=torch.float32, device=device), self.cx, self.ls_e)
        self.Ky = self._k(torch.arange(g.ny, dtype=torch.float32, device=device), self.cy, self.ls_e)
        self.Kth = self._k(torch.arange(g.nth, dtype=torch.float32, device=device), self.cth, self.ls_e,
                           period=g.nth)
        # eigendecompositions of the 1-D Gram matrices (float64 for accuracy)
        self.eig = []
        for K in (self.Kt, self.Kx, self.Ky, self.Kth):
            lam, U = torch.linalg.eigh((K.T @ K).double())
            self.eig.append((lam.clamp_min(0.0), U))
        Ks = (self.Kt, self.Kx, self.Ky, self.Kth)
        self.K_render = tuple(K / K.sum(1, keepdim=True) for K in Ks) if fit == "avg" else Ks
        self.K_fitT = tuple((K / K.sum(0, keepdim=True)).T.contiguous() for K in Ks) if fit == "avg" else None
        self.theta = torch.zeros(len(self.ct), len(self.cx), len(self.cy), len(self.cth), device=device)
        self._cache = None

    @staticmethod
    def _k(z, c, ls, period=None):
        d = z[:, None] - c[None, :]
        if period is not None:
            d = torch.remainder(d + period / 2, period) - period / 2
        return torch.exp(-0.5 * (d / ls) ** 2)

    @staticmethod
    def _modes(X, mats):
        """Multiply tensor X (a, b, c, d) along each mode by the given matrices (rows x cols applied as X x_i M_i)."""
        A, B, Cm, D = mats
        X = torch.einsum("abcd,ia->ibcd", X, A)
        X = torch.einsum("ibcd,jb->ijcd", X, B)
        X = torch.einsum("ijcd,kc->ijkd", X, Cm)
        return torch.einsum("ijkd,ld->ijkl", X, D)

    # --- interface used by GPI -------------------------------------------------------
    def grid_values(self):
        if self._cache is None:
            self._cache = self._modes(self.theta, self.K_render)
        return self._cache

    def update_all(self, targets):
        if self.fit == "avg":
            self.theta = self._modes(targets.float(), self.K_fitT)
            self._cache = None
            return
        y = targets.double()
        Ks = [K.double() for K in (self.Kt, self.Kx, self.Ky, self.Kth)]
        b = self._modes(y, [K.T for K in Ks])  # Phi^T y
        Us = [U for _, U in self.eig]
        c = self._modes(b, [U.T for U in Us])
        lt, lx, ly, lth = (lam for lam, _ in self.eig)
        Lam = lt[:, None, None, None] * lx[None, :, None, None] * ly[None, None, :, None] * lth[None, None, None, :]
        scale = Lam.max()
        c = c / (Lam + self.ridge * scale)
        self.theta = self._modes(c, Us).float()
        self._cache = None

    def fit_residual(self, targets):
        return (self.grid_values() - targets).abs()

    def evaluator(self, exact=False):
        """exact=False: multilinear interpolation of the rendered node values (used offline, cheap);
        exact=True: evaluate the kernel expansion at the continuous query (used online)."""
        if not exact:
            return make_grid_interpolator(self.grid_values(), self.grid)
        g = self.grid

        def ev(tn, qx, qy, qth):
            Nt, M = qx.shape
            kt = self._k(tn.float(), self.ct, self.ls_t, period=self.T)  # (Nt, Tc)
            kx = self._k(g.frac_x(qx).reshape(-1), self.cx, self.ls_e).view(Nt, M, -1)
            ky = self._k(g.frac_y(qy).reshape(-1), self.cy, self.ls_e).view(Nt, M, -1)
            kth = self._k(g.frac_th(qth).reshape(-1), self.cth, self.ls_e, period=g.nth).view(Nt, M, -1)
            if self.fit == "avg":  # normalized features
                kt, kx, ky, kth = (k / k.sum(-1, keepdim=True) for k in (kt, kx, ky, kth))
            th_t = torch.einsum("nc,cxyz->nxyz", kt, self.theta)
            A = torch.einsum("nxyz,nmz->nmxy", th_t, kth)
            A = torch.einsum("nmxy,nmy->nmx", A, ky)
            return torch.einsum("nmx,nmx->nm", A, kx)

        return ev

    def n_params(self):
        return self.theta.numel()

    # --- starter-code API -------------------------------------------------------------
    def copy_from(self, other):
        self.theta = other.theta.clone()
        self._cache = None

    def update(self, t, ex, ey, etheta, target_value):
        raise NotImplementedError("FeatureValueFunction is fitted to all grid targets at once (update_all).")

    def __call__(self, t, ex, ey, etheta):
        q = [torch.as_tensor(a, dtype=torch.float32, device=self.device).reshape(1, -1) for a in (ex, ey, etheta)]
        tn = torch.as_tensor([int(t) % self.T], device=self.device)
        return self.evaluator(exact=True)(tn, *q).reshape(np.shape(ex))

    def copy(self):
        c = FeatureValueFunction(self.T, self.ex_space, self.ey_space, self.etheta_space, self.device,
                                 self.stride_t, self.stride_e, self.ls_t, self.ls_e, self.ridge, self.fit)
        c.copy_from(self)
        return c

    def state_dict(self):
        return {"theta": self.theta.cpu().numpy(), "stride_t": self.stride_t, "stride_e": self.stride_e,
                "ls_t": self.ls_t, "ls_e": self.ls_e, "ridge": self.ridge, "fit": self.fit}

    def load_state_dict(self, d):
        self.theta = torch.as_tensor(d["theta"], device=self.device)
        self._cache = None
