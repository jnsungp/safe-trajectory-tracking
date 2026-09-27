"""Run one closed-loop episode with a chosen controller.

    python main.py --controller cec                     # receding-horizon CEC (CasADi / IPOPT)
    python main.py --controller gpi --model results/models/grid_medium_k1.npz
    python main.py --controller rbf --model results/models/rbfavg_medium_k1.npz
    python main.py --controller p                       # starter P controller (baseline)
    add --noise 0 for the noise-free system, --show to animate, --mujoco to use the MuJoCo car.
"""
import argparse
from time import time

import numpy as np

import common as C
import utils


def make_controller(name, model, horizon):
    if name == "p":
        return lambda t, x, r: utils.simple_controller(x, r)
    if name == "cec":
        from cec import CEC
        return CEC(horizon=horizon)
    from gpi import GPI
    return GPI.load(model, device="cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["p", "cec", "gpi", "rbf"], default="cec")
    ap.add_argument("--model", default="results/models/grid_medium_k1.npz", help="trained GPI / RBF model (.npz)")
    ap.add_argument("--horizon", type=int, default=10, help="CEC horizon")
    ap.add_argument("--noise", type=float, default=1.0, help="noise scale k (w ~ N(0, (k sigma)^2))")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--mujoco", action="store_true", help="step the MuJoCo car instead of the numerical model")
    ap.add_argument("--show", action="store_true", help="animate the rollout (utils.visualize)")
    a = ap.parse_args()

    ctrl = make_controller(a.controller, a.model, a.horizon)
    if a.mujoco:
        from mujoco_car import MujocoCarSim
        sim = MujocoCarSim()
    W = C.noise_sequence(a.seed, C.N_STEPS, a.noise)

    cur_state = C.X_INIT.copy()
    car_states, ref_traj, times = [], [], []
    error_trans = error_rot = 0.0
    main_loop = time()
    for cur_iter in range(C.N_STEPS):
        cur_ref = C.ref(cur_iter)
        ref_traj.append(cur_ref)
        car_states.append(cur_state)
        t1 = time()
        control = np.clip(np.asarray(ctrl(cur_iter, cur_state.copy(), cur_ref.copy()), dtype=float), C.U_LB, C.U_UB)
        times.append(time() - t1)
        if a.mujoco:
            next_state = np.asarray(sim.car_next_state(control), dtype=float)
        else:
            next_state = C.f(cur_state, control, W[cur_iter])  # exact discretization, eq. (1)
        cur_state = next_state
        cur_err = cur_state - cur_ref
        cur_err[2] = C.wrap(cur_err[2])
        error_trans += np.linalg.norm(cur_err[:2])
        error_rot += np.abs(cur_err[2])

    car_states = np.array(car_states)
    clr = C.clearance(car_states)
    print(f"controller={a.controller} noise=x{a.noise:g} seed={a.seed}")
    print(f"total time {time() - main_loop:.2f} s | mean control time {1e3 * np.mean(times):.2f} ms")
    print(f"final error_trans {error_trans:.2f} | final error_rot {error_rot:.2f}")
    print(f"collision steps {(clr < 0).sum()} | min clearance {clr.min():.3f} m")
    if a.mujoco:
        sim.viewer_handle.close()
    if a.show:
        utils.visualize(car_states, np.array(ref_traj), C.OBSTACLES, np.array(times), utils.time_step, save=False)


if __name__ == "__main__":
    main()
