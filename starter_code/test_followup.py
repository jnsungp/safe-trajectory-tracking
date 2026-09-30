"""Checks for the follow-up experiment: path-aware risk, the risk-aware CEC, and the path metric."""
import os
import unittest

import numpy as np
import torch

import common as C
from gpi import GPI, GpiConfig

MODELS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "models")


def tiny_gpi(**kw):
    cfg = GpiConfig(ex_space=np.array([-1., 0., 1.]), ey_space=np.array([-1., 0., 1.]),
                    eth_space=np.linspace(-np.pi, np.pi, 4, endpoint=False),
                    v_space=np.array([0.1, 1.0]), w_space=np.array([-1., 1.]),
                    risk_mode="exact_disks", device="cpu", **kw)
    return GPI(cfg)


def gap_dash_state():
    """A pose 26 cm before the C1-C3 gap center, heading through it at full speed (0.5 m per step)."""
    a, b = C.OBSTACLES[0, :2], C.OBSTACLES[2, :2]
    mid = (a + b) / 2
    along = np.array([-(b - a)[1], (b - a)[0]]) / np.linalg.norm(b - a)  # perpendicular to the center line
    start = mid - 0.26 * along
    return np.r_[start, np.arctan2(along[1], along[0])], np.array([1.0, 0.0])


class PathRiskTest(unittest.TestCase):
    def step_risks(self, model, x, u, t=0):
        e = torch.tensor(C.error_state(t, x), dtype=torch.float32).view(1, 1, 3)
        U = torch.tensor(u, dtype=torch.float32).view(1, 1, 2)
        tt = torch.tensor([t])
        mx, my, _ = model._next_mean(tt, e, U)
        px = mx + model.ref[(tt + 1) % model.T, 0][:, None]
        py = my + model.ref[(tt + 1) % model.T, 1][:, None]
        return float(model.collision_risk(px, py)), float(model.path_risk(tt, e, U, px, py))

    def test_one_substep_is_next_sample_risk(self):
        model = tiny_gpi(noise_scale=1.0, risk_substeps=1)
        x, u = gap_dash_state()
        sample, path = self.step_risks(model, x, u)
        self.assertEqual(sample, path)

    def test_gap_dash_is_cheaper_for_samples_than_for_the_path(self):
        x, u = gap_dash_state()
        # both samples stay about 6 cm clear, the arc passes the gap center with 2.5 cm on each side
        self.assertGreater(min(C.clearance(x), C.clearance(C.f(x, u))), 0.055)
        mid = x[:2] + 0.26 * np.array([np.cos(x[2]), np.sin(x[2])])
        self.assertLess(C.clearance(mid), 0.03)
        sample, path = self.step_risks(tiny_gpi(noise_scale=1.0, risk_substeps=4), x, u)
        self.assertGreater(path, 1.8 * sample)
        _, path8 = self.step_risks(tiny_gpi(noise_scale=1.0, risk_substeps=8), x, u)
        self.assertLess(abs(path8 - path) / path8, 0.1)  # four points per step are enough here
        sample, path = self.step_risks(tiny_gpi(noise_scale=0.5, risk_substeps=4), x, u)
        self.assertGreater(path, 5 * sample)


class RiskCecTest(unittest.TestCase):
    def test_risk_matches_gpi(self):
        from cec_risk import RiskCEC
        rng = np.random.default_rng(0)
        pts = np.c_[rng.uniform(-2.8, 2.8, 200), rng.uniform(-2.8, 2.8, 200)]
        pts[:20] = (C.OBSTACLES[0, :2] + C.OBSTACLES[2, :2]) / 2 + rng.normal(0, 0.03, (20, 2))
        for k in (0.25, 1.0):
            sig = max(0.04 * k, 0.01)
            ctrl = RiskCEC(horizon=2, risk_sigma=sig)
            model = tiny_gpi(noise_scale=k)
            want = model.collision_risk(torch.tensor(pts[:, 0]), torch.tensor(pts[:, 1])).numpy()
            got = np.array([float(ctrl._risk_levels[0](p[0], p[1])) for p in pts])
            np.testing.assert_allclose(got, want, atol=3e-4)

    def test_path_risk_matches_gpi(self):
        from cec_risk import RiskCEC
        x, u = gap_dash_state()
        ctrl = RiskCEC(horizon=2, risk_sigma=0.04, risk_substeps=4)
        _, want = PathRiskTest().step_risks(tiny_gpi(noise_scale=1.0, risk_substeps=4), x, u)
        self.assertAlmostEqual(ctrl.plan_risk(x, u), want, delta=1e-3)

    @unittest.skipUnless(os.path.exists(os.path.join(MODELS, "path_grid_medium_k1.npz")), "model not trained")
    def test_local_value_tracks_gpi_value(self):
        from cec_risk import LocalValue
        path = os.path.join(MODELS, "path_grid_medium_k1.npz")
        lv = LocalValue(path)
        g = GPI.load(path, device="cpu")
        ev = g.V.evaluator()
        rng = np.random.default_rng(1)
        w = np.array([1, 4, 1]) / 6  # cubic B-spline weights at a node and its two neighbors
        for t in (0, 37, 81):
            e0 = np.array([rng.uniform(-0.3, 0.3), rng.uniform(-0.3, 0.3), rng.uniform(-1, 1)])
            origin, coef = lv.window(t, e0)
            C3 = coef.reshape(lv.n, order="F")
            # at a window node the spline is the separable (1, 4, 1) / 6 average of the samples
            for i, j, l in ((8, 8, 23), (6, 9, 20), (10, 7, 26)):
                node = origin + lv.h * np.array([i, j, l])
                want = np.einsum("a,b,c,abc->", w, w, w, C3[i - 1:i + 2, j - 1:j + 2, l - 1:l + 2])
                self.assertAlmostEqual(float(lv.fn(node, origin, coef)), want, delta=1e-6 * max(1.0, want))
            # the samples are GPI's own interpolated values
            with torch.no_grad():
                q = [torch.tensor([[origin[d] + lv.h[d] * idx]], dtype=torch.float32)
                     for d, idx in enumerate((8, 8, 23))]
                self.assertAlmostEqual(float(ev(torch.tensor([t]), *q)), C3[8, 8, 23], delta=1e-3)
            q = e0 + rng.normal(0, [0.1, 0.1, 0.3], (30, 3))
            got = np.array([float(lv.fn(qq, origin, coef)) for qq in q])
            with torch.no_grad():
                want = ev(torch.tensor([t]), *[torch.tensor(q[None, :, i], dtype=torch.float32)
                                               for i in range(3)]).numpy().ravel()
            # smoothing over 0.1 m / 9 degree cells: close to GPI's multilinear value, never below the samples
            self.assertLess(np.median(np.abs(got - want) / np.maximum(want, 1.0)), 0.1)
            self.assertGreaterEqual(got.min(), coef.min() - 1e-6)


class CecPathTest(unittest.TestCase):
    def test_substeps_keep_noise_free_arc_clear(self):
        import analyze as A
        from cec import CEC
        r = C.rollout(CEC(horizon=10, substeps=4), noise_scale=0.0, n_steps=60, keep_traj=True)
        self.assertGreater(A.arc_clearance(r["traj"], r["controls"]), -1e-4)

    def test_path_metric_contains_sampled_metric(self):
        import analyze as A
        from followup_analysis import path_clearance_per_step
        x, u = gap_dash_state()
        traj = np.array([x, C.f(x, u)])
        clr = path_clearance_per_step(traj, u[None])
        self.assertLessEqual(clr[0], min(C.clearance(traj[0]), C.clearance(traj[1])) + 1e-12)
        self.assertLess(clr[0], 0.03)  # the arc passes through the gap
        self.assertAlmostEqual(clr.min(), A.arc_clearance(traj, u[None]))


if __name__ == "__main__":
    unittest.main()
