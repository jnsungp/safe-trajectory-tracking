"""Regression checks for the corrected corridor collision probability."""
import unittest

import numpy as np
import torch
from scipy.stats import ncx2

import common as C
from gpi import GPI, GpiConfig


class RevisedRiskTest(unittest.TestCase):
    def test_corridor_midpoint_matches_exact_disjoint_disk_union(self):
        midpoint = (C.OBSTACLES[0, :2] + C.OBSTACLES[2, :2]) / 2
        distances = np.linalg.norm(C.OBSTACLES[:, :2] - midpoint, axis=1)
        for k in (0.25, 0.5, 1.0, 2.0):
            cfg = GpiConfig(ex_space=np.array([-1., 0., 1.]), ey_space=np.array([-1., 0., 1.]),
                            eth_space=np.linspace(-np.pi, np.pi, 4, endpoint=False),
                            v_space=np.array([0.1, 1.0]), w_space=np.array([-1., 1.]),
                            noise_scale=k, risk_mode="exact_disks", device="cpu")
            model = GPI(cfg)
            got = float(model.collision_risk(torch.tensor(midpoint[0]), torch.tensor(midpoint[1])))
            sigma = C.SIGMA[0] * k
            radius = C.OBSTACLES[:, 2] + C.ROBOT_RADIUS
            want = float(ncx2.cdf((radius / sigma) ** 2, 2, (distances / sigma) ** 2).sum())
            self.assertAlmostEqual(got, want, delta=2e-4)

    def test_rollout_first_hit_and_phase_index(self):
        seen = []

        def controller(t, state, reference):
            seen.append(t)
            return np.array([0.1, 0.0])

        safe = C.rollout(controller, noise_scale=0.0, n_steps=3,
                         x0=np.array([0.0, 0.0, 0.0]), start_t=50)
        self.assertEqual(seen, [50, 51, 52])
        self.assertEqual(safe["first_coll_t"], -1)
        self.assertEqual(safe["safe_steps_until_hit"], 3)

        hit = C.rollout(controller, noise_scale=0.0, n_steps=3,
                        x0=np.array([1.0, 0.0, 0.0]), start_t=0)
        self.assertEqual(hit["first_coll_t"], 0)
        self.assertEqual(hit["safe_steps_until_hit"], 0)


if __name__ == "__main__":
    unittest.main()
