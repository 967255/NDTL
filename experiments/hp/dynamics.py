from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .holling import DEFAULT_DOMAIN, HollingParams, holling_rhs_ecology, unit_to_ecology


@dataclass
class TrajectorySummary:
    status: str
    persistent: bool
    min_tail: list[float]
    mean_tail: list[float]
    std_tail: list[float]
    max_tail: list[float]
    peak_count_c: int
    lyapunov: list[float] | None = None

    def to_dict(self) -> dict:
        return {
            "status": self.status,
            "persistent": self.persistent,
            "min_tail": self.min_tail,
            "mean_tail": self.mean_tail,
            "std_tail": self.std_tail,
            "max_tail": self.max_tail,
            "peak_count_c": self.peak_count_c,
            "lyapunov": self.lyapunov,
        }


def rk4_step(rhs: Callable[[np.ndarray], np.ndarray], state: np.ndarray, dt: float):
    k1 = rhs(state)
    k2 = rhs(state + 0.5 * dt * k1)
    k3 = rhs(state + 0.5 * dt * k2)
    k4 = rhs(state + dt * k3)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def integrate_rhs(
    rhs: Callable[[np.ndarray], np.ndarray],
    init_state=(0.5, 0.2, 8.0),
    dt: float = 0.05,
    steps: int = 120000,
    keep: int = 40000,
    lower_abort: float = -1e-6,
    upper_abort: float = 1e6,
):
    state = np.asarray(init_state, dtype=np.float64).reshape(3)
    keep = int(min(keep, steps))
    traj = np.empty((keep, 3), dtype=np.float64)
    start = int(steps) - keep
    j = 0
    status = "normal"
    for i in range(int(steps)):
        state = rk4_step(rhs, state, dt)
        if (not np.all(np.isfinite(state))) or np.max(np.abs(state)) > upper_abort:
            status = "diverged"
            break
        if np.min(state) < lower_abort:
            status = "negative"
            break
        if i >= start and j < keep:
            traj[j] = state
            j += 1
    return traj[:j], status


def count_local_peaks(x: np.ndarray) -> int:
    if x.size < 3:
        return 0
    return int(np.sum((x[1:-1] > x[:-2]) & (x[1:-1] > x[2:])))


def summarize_trajectory(
    traj: np.ndarray,
    status: str,
    persistence_threshold: float = 1e-3,
    lyapunov=None,
) -> TrajectorySummary:
    if traj.size == 0:
        return TrajectorySummary(
            status=status,
            persistent=False,
            min_tail=[float("nan")] * 3,
            mean_tail=[float("nan")] * 3,
            std_tail=[float("nan")] * 3,
            max_tail=[float("nan")] * 3,
            peak_count_c=0,
            lyapunov=None if lyapunov is None else list(map(float, lyapunov)),
        )
    mn = np.min(traj, axis=0)
    return TrajectorySummary(
        status=status,
        persistent=bool(status == "normal" and np.all(mn > persistence_threshold)),
        min_tail=list(map(float, mn)),
        mean_tail=list(map(float, np.mean(traj, axis=0))),
        std_tail=list(map(float, np.std(traj, axis=0))),
        max_tail=list(map(float, np.max(traj, axis=0))),
        peak_count_c=count_local_peaks(traj[:, 2]),
        lyapunov=None if lyapunov is None else list(map(float, lyapunov)),
    )


def finite_diff_jac(
    rhs: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-6
):
    x = np.asarray(x, dtype=np.float64).reshape(3)
    J = np.zeros((3, 3), dtype=np.float64)
    scale = 1.0 + np.linalg.norm(x)
    h = eps * scale
    for j in range(3):
        dx = np.zeros(3, dtype=np.float64)
        dx[j] = h
        J[:, j] = (rhs(x + dx) - rhs(x - dx)) / (2.0 * h)
    return J


def lyapunov_spectrum(
    rhs: Callable[[np.ndarray], np.ndarray],
    init_state=(0.5, 0.2, 8.0),
    dt: float = 0.03,
    transient_steps: int = 5000,
    measure_steps: int = 10000,
    qr_interval: int = 10,
):
    state = np.asarray(init_state, dtype=np.float64).reshape(3)
    for _ in range(int(transient_steps)):
        state = rk4_step(rhs, state, dt)
        if not np.all(np.isfinite(state)):
            return None

    Q = np.eye(3)
    logs = np.zeros(3, dtype=np.float64)
    elapsed = 0.0
    for i in range(int(measure_steps)):
        J = finite_diff_jac(rhs, state)
        state = rk4_step(rhs, state, dt)
        Q = Q + dt * (J @ Q)
        if (i + 1) % int(qr_interval) == 0:
            Q, R = np.linalg.qr(Q)
            logs += np.log(np.abs(np.diag(R)) + 1e-300)
        elapsed += dt
        if (not np.all(np.isfinite(state))) or np.min(state) < -1e-6:
            return None
    return np.sort(logs / max(elapsed, 1e-12))[::-1]


def summarize_params(params: HollingParams, fast: bool = False):
    rhs = lambda s: holling_rhs_ecology(s, params)
    if fast:
        traj, status = integrate_rhs(rhs, steps=20000, keep=6000, dt=0.05)
        lyap = None
    else:
        traj, status = integrate_rhs(rhs, steps=80000, keep=20000, dt=0.05)
        lyap = lyapunov_spectrum(
            rhs,
            dt=0.03,
            transient_steps=5000,
            measure_steps=10000,
        )
    return summarize_trajectory(traj, status=status, lyapunov=lyap)


class TrilinearVectorField:
    """Interpolate a unit-coordinate vector-field tensor and return ecology RHS."""

    def __init__(self, field_unit: np.ndarray, domain=DEFAULT_DOMAIN):
        self.field = np.asarray(field_unit, dtype=np.float64)
        if self.field.ndim != 4 or self.field.shape[-1] != 3:
            raise ValueError("field_unit must have shape (D,H,W,3)")
        self.n = np.array(self.field.shape[:3], dtype=np.int64)
        self.domain = domain
        _, _, self.spans = self._domain_arrays()

    def _domain_arrays(self):
        domain_arr = np.asarray(self.domain, dtype=np.float64)
        lows = domain_arr[:, 0]
        highs = domain_arr[:, 1]
        spans = highs - lows
        return lows, highs, spans

    def __call__(self, ecology_state: np.ndarray):
        lows, _, spans = self._domain_arrays()
        unit = (np.asarray(ecology_state, dtype=np.float64) - lows) / spans
        unit = np.clip(unit, 0.0, 1.0)
        pos = unit * (self.n - 1)
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, self.n - 1)
        w = pos - i0

        c000 = self.field[i0[0], i0[1], i0[2]]
        c100 = self.field[i1[0], i0[1], i0[2]]
        c010 = self.field[i0[0], i1[1], i0[2]]
        c001 = self.field[i0[0], i0[1], i1[2]]
        c110 = self.field[i1[0], i1[1], i0[2]]
        c101 = self.field[i1[0], i0[1], i1[2]]
        c011 = self.field[i0[0], i1[1], i1[2]]
        c111 = self.field[i1[0], i1[1], i1[2]]

        wx, wy, wz = w
        c00 = c000 * (1 - wx) + c100 * wx
        c01 = c001 * (1 - wx) + c101 * wx
        c10 = c010 * (1 - wx) + c110 * wx
        c11 = c011 * (1 - wx) + c111 * wx
        c0 = c00 * (1 - wy) + c10 * wy
        c1 = c01 * (1 - wy) + c11 * wy
        unit_rhs = c0 * (1 - wz) + c1 * wz
        return unit_rhs * spans
