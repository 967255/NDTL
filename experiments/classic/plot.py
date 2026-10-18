import re

import json

import math

from dataclasses import dataclass

from pathlib import Path

from typing import Dict, List, Tuple, Optional

from ndtl.sampling import DataSampler
from ndtl.fields import solve_ivp

from ndtl.fields import ensemble_rescaling

import ast

import numpy as np

import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

try:
    from scipy.integrate import solve_ivp

    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False


class BlowUpError(RuntimeError):
    """Trajectory escapes / blows up before t_span ends."""

    pass


Exp = Tuple[int, int, int]  # (ex, ey, ez)

_LOG_PREFIX_RE = re.compile(r"^\[[0-9]{4}-[0-9]{2}-[0-9]{2} .*?\]\s+INFO:\s*")
_TERM_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([xyzXYZ]+)?\s*$"
)
_VEC_RE = re.compile(r"\[vec\s+(\d+)\]")
_NAME_RE = re.compile(r"^\s*name\s*=\s*(.+?)\s*$", re.IGNORECASE)


@dataclass
class PolyODE3:
    "Three-dimensional polynomial ODE. terms[i] maps exponent tuples to coefficients for component i."
    name: str
    terms: List[Dict[Exp, float]]
    meta: Dict[str, object] = None

    def __post_init__(self):
        if self.meta is None:
            self.meta = {}
        assert len(self.terms) == 3

        self._compiled = []
        for comp in range(3):
            items = []
            for exp, c in self.terms[comp].items():
                if abs(c) > 0:
                    items.append((exp, float(c)))
            self._compiled.append(items)

    def rhs(self, t: float, y: np.ndarray) -> np.ndarray:
        x, yy, z = float(y[0]), float(y[1]), float(y[2])
        out = np.zeros(3, dtype=float)
        base = (x, yy, z)

        for i in range(3):
            s = 0.0
            for (ex, ey, ez), c in self._compiled[i]:
                # monomial = x^ex * y^ey * z^ez
                m = 1.0
                if ex:
                    m *= base[0] ** ex
                if ey:
                    m *= base[1] ** ey
                if ez:
                    m *= base[2] ** ez
                s += c * m
            out[i] = s
        return out

    # def simulate(
    #     self,
    #     y0: Tuple[float, float, float],
    #     t_span: Tuple[float, float] = (0.0, 50.0),
    #     dt: float = 0.01,
    #     method: str = "RK45",
    #     rtol: float = 1e-7,
    #     atol: float = 1e-9,
    #     max_steps: Optional[int] = None,
    # ) -> Tuple[np.ndarray, np.ndarray]:
    #     """

    #     """
    #     t0, t1 = float(t_span[0]), float(t_span[1])
    #     t_eval = np.arange(t0, t1 + 1e-12, dt, dtype=float)
    #     y0_arr = np.array(y0, dtype=float)

    #     if _HAS_SCIPY:
    #         sol = solve_ivp(
    #             fun=self.rhs,
    #             t_span=(t0, t1),
    #             y0=y0_arr,
    #             t_eval=t_eval,
    #             method=method,
    #             rtol=rtol,
    #             atol=atol,
    #         )
    #         if not sol.success:
    #             raise RuntimeError(f"solve_ivp failed: {sol.message}")
    #         traj = sol.y.T  # (T, 3)

    #     if max_steps is not None and len(t_eval) > max_steps:
    #         t_eval = t_eval[:max_steps]

    #     traj = np.zeros((len(t_eval), 3), dtype=float)
    #     traj[0] = y0_arr
    #     t = t_eval[0]
    #     for i in range(1, len(t_eval)):
    #         h = float(t_eval[i] - t_eval[i - 1])
    #         y = traj[i - 1]

    #         k1 = self.rhs(t, y)
    #         k2 = self.rhs(t + 0.5 * h, y + 0.5 * h * k1)
    #         k3 = self.rhs(t + 0.5 * h, y + 0.5 * h * k2)
    #         k4 = self.rhs(t + h, y + h * k3)

    #         traj[i] = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    #         t = t_eval[i]

    def simulate(
        self,
        y0: Tuple[float, float, float],
        t_span: Tuple[float, float] = (0.0, 50.0),
        dt: float = 0.01,
        method: str = "RK45",
        rtol: float = 1e-7,
        atol: float = 1e-9,
        max_steps: Optional[int] = None,
        blowup_norm: float = 1e6,
    ) -> Tuple[np.ndarray, np.ndarray]:

        t0, t1 = float(t_span[0]), float(t_span[1])
        t_eval = np.arange(t0, t1 + 1e-12, dt, dtype=float)
        y0_arr = np.array(y0, dtype=float)

        if _HAS_SCIPY:
            events = None
            if blowup_norm is not None:

                def _blow_event(t, y):

                    return blowup_norm - float(np.linalg.norm(y))

                _blow_event.terminal = True
                _blow_event.direction = -1
                events = [_blow_event]

            sol = solve_ivp(
                fun=self.rhs,
                t_span=(t0, t1),
                y0=y0_arr,
                t_eval=t_eval,
                method=method,
                rtol=rtol,
                atol=atol,
                events=events,
                # max_step=dt,
            )

            traj = sol.y.T

            if sol.status == 1 and sol.t_events and len(sol.t_events[0]) > 0:
                tb = float(sol.t_events[0][0])
                raise BlowUpError(
                    f"{self.name} blew up: ||y||>{blowup_norm} at t={tb:.6g}"
                )

            if not sol.success:
                raise RuntimeError(f"solve_ivp failed: {sol.message}")

            if not np.isfinite(traj).all():
                raise BlowUpError(f"{self.name} produced non-finite states (inf/nan).")

            return sol.t, traj

        if max_steps is not None and len(t_eval) > max_steps:
            t_eval = t_eval[:max_steps]

        traj = np.zeros((len(t_eval), 3), dtype=float)
        traj[0] = y0_arr
        t = t_eval[0]
        for i in range(1, len(t_eval)):
            h = float(t_eval[i] - t_eval[i - 1])
            y = traj[i - 1]

            k1 = self.rhs(t, y)
            k2 = self.rhs(t + 0.5 * h, y + 0.5 * h * k1)
            k3 = self.rhs(t + 0.5 * h, y + 0.5 * h * k2)
            k4 = self.rhs(t + h, y + h * k3)

            traj[i] = y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            t = t_eval[i]

            if (blowup_norm is not None) and (np.linalg.norm(traj[i]) > blowup_norm):
                raise BlowUpError(
                    f"{self.name} blew up: ||y||>{blowup_norm} at t={t:.6g}"
                )
            if not np.isfinite(traj[i]).all():
                raise BlowUpError(
                    f"{self.name} produced non-finite states (inf/nan) at t={t:.6g}"
                )

        return t_eval, traj

    def to_jsonable(self) -> Dict[str, object]:
        def exp_key(e: Exp) -> str:
            return f"{e[0]},{e[1]},{e[2]}"

        return {
            "name": self.name,
            "terms": [
                {exp_key(k): float(v) for k, v in self.terms[i].items()}
                for i in range(3)
            ],
            "meta": self.meta,
        }

    @staticmethod
    def from_jsonable(d: Dict[str, object]) -> "PolyODE3":
        def parse_key(s: str) -> Exp:
            a, b, c = s.split(",")
            return (int(a), int(b), int(c))

        terms = []
        for comp in d["terms"]:
            terms.append({parse_key(k): float(v) for k, v in comp.items()})
        return PolyODE3(name=d["name"], terms=terms, meta=d.get("meta", {}))


def _strip_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line).rstrip("\n")


def _parse_poly_expr(expr: str) -> Dict[Exp, float]:
    "Parse polynomial terms such as xx, xxy, or xyz into exponent-tuple coefficients."

    parts = expr.split("+")
    out: Dict[Exp, float] = {}

    for raw in parts:
        s = raw.strip()
        if not s:
            continue

        m = _TERM_RE.match(s)
        if not m:

            raise ValueError(f"Cannot parse term: {s!r} from expr: {expr!r}")

        coeff = float(m.group(1))
        mono = m.group(2) or ""
        mono = mono.lower()

        ex = mono.count("x")
        ey = mono.count("y")
        ez = mono.count("z")
        key = (ex, ey, ez)

        out[key] = out.get(key, 0.0) + coeff

    return out


def parse_ode_hat_from_log(log_path: str) -> List[PolyODE3]:
    "Read fitted polynomial systems from all ode_hat blocks, retaining run and vector indices."
    p = Path(log_path)
    if not p.exists():
        raise FileNotFoundError(f"log not found: {log_path}")

    lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()

    systems: List[PolyODE3] = []
    run_id = -1

    i = 0
    while i < len(lines):
        line_raw = lines[i]
        line = _strip_prefix(line_raw)

        if "vecs.shape" in line:
            run_id += 1

        if "ode_hat info:" in line:
            # vec index
            vec_idx = None
            mv = _VEC_RE.search(line)
            if mv:
                vec_idx = int(mv.group(1))

            after = line.split("ode_hat info:", 1)[1].strip()

            if not after.lower().startswith("dx/dt"):
                raise ValueError(f"Unexpected ode_hat line: {line}")

            # dx
            dx_expr = after.split("=", 1)[1].strip()

            dy_expr = None
            dz_expr = None
            name = None

            j = i + 1

            while j < len(lines):
                nxt_raw = lines[j]
                nxt = _strip_prefix(nxt_raw).strip()

                if nxt.lower().startswith("dy/dt"):
                    dy_expr = nxt.split("=", 1)[1].strip()
                elif nxt.lower().startswith("dz/dt"):
                    dz_expr = nxt.split("=", 1)[1].strip()
                else:
                    mn = _NAME_RE.match(nxt)
                    if mn:
                        name = mn.group(1).strip()

                if (
                    (dy_expr is not None)
                    and (dz_expr is not None)
                    and (name is not None)
                ):
                    break
                j += 1

            if dy_expr is None or dz_expr is None or name is None:
                raise ValueError(f"Incomplete ode_hat block near line {i}")

            terms = [
                _parse_poly_expr(dx_expr),
                _parse_poly_expr(dy_expr),
                _parse_poly_expr(dz_expr),
            ]

            sys = PolyODE3(
                name=name,
                terms=terms,
                meta={
                    "run_id": run_id,
                    "vec_idx": vec_idx,
                    "log_path": str(p),
                    "dx_expr": dx_expr,
                    "dy_expr": dy_expr,
                    "dz_expr": dz_expr,
                },
            )
            systems.append(sys)

            i = j
        i += 1

    return systems


def _auto_range(
    a: np.ndarray, b: np.ndarray, pad_ratio: float = 0.05
) -> List[Tuple[float, float]]:
    "Build a padded histogram range from the combined bounds of two (T, 3) trajectories."
    c = np.vstack([a, b])
    mn = c.min(axis=0)
    mx = c.max(axis=0)
    span = mx - mn
    span[span == 0] = 1.0
    mn = mn - pad_ratio * span
    mx = mx + pad_ratio * span
    return [(float(mn[k]), float(mx[k])) for k in range(3)]


def kl_divergence_hist(
    traj_p: np.ndarray,
    traj_q: np.ndarray,
    bins: int = 40,
    hist_range: Optional[List[Tuple[float, float]]] = None,
    eps: float = 1e-12,
) -> float:
    "Approximate trajectory distributions with 3D histograms and compute KL(P || Q)."
    assert traj_p.shape[1] == 3 and traj_q.shape[1] == 3

    if hist_range is None:
        hist_range = _auto_range(traj_p, traj_q)

    P, _ = np.histogramdd(traj_p, bins=bins, range=hist_range)
    Q, _ = np.histogramdd(traj_q, bins=bins, range=hist_range)

    P = P.astype(float)
    Q = Q.astype(float)

    P = P + eps
    Q = Q + eps
    P = P / P.sum()
    Q = Q / Q.sum()

    return float(np.sum(P * np.log(P / Q)))


def js_divergence_hist(
    traj_p: np.ndarray,
    traj_q: np.ndarray,
    bins: int = 40,
    hist_range: Optional[List[Tuple[float, float]]] = None,
    eps: float = 1e-12,
) -> float:
    if hist_range is None:
        hist_range = _auto_range(traj_p, traj_q)

    P, _ = np.histogramdd(traj_p, bins=bins, range=hist_range)
    Q, _ = np.histogramdd(traj_q, bins=bins, range=hist_range)

    P = P.astype(float) + eps
    Q = Q.astype(float) + eps
    P = P / P.sum()
    Q = Q / Q.sum()
    M = 0.5 * (P + Q)

    kl_pm = float(np.sum(P * np.log(P / M)))
    kl_qm = float(np.sum(Q * np.log(Q / M)))
    return 0.5 * (kl_pm + kl_qm)


def compute_divergences_to_reference(
    systems: List[PolyODE3],
    ref: PolyODE3,
    inits: List[Tuple[float, float, float]],
    t_span: Tuple[float, float] = (0.0, 70.0),
    dt: float = 0.01,
    transient: float = 20.0,
    bins: int = 40,
    use_js: bool = True,
) -> Dict[int, Dict[str, object]]:
    "Compare each system trajectory with the reference for each initial condition."

    out: Dict[int, Dict[str, object]] = {}

    for sys in systems:
        vec_idx = sys.meta.get("vec_idx", None)
        key = int(vec_idx) if vec_idx is not None else len(out)

        per_init = []
        for y0 in inits:
            try:
                t_ref, tr_ref = ref.simulate(y0=y0, t_span=t_span, dt=dt)
                t_sys, tr_sys = sys.simulate(y0=y0, t_span=t_span, dt=dt)
            except BlowUpError as e:
                per_init.append(
                    {
                        "init": y0,
                        "value": float("nan"),
                        "metric": "JS" if use_js else "KL",
                        "status": "diverged",
                        "error": str(e),
                    }
                )
                continue
            except Exception as e:
                per_init.append(
                    {
                        "init": y0,
                        "value": float("nan"),
                        "metric": "JS" if use_js else "KL",
                        "status": "failed",
                        "error": str(e),
                    }
                )
                continue
            # t_ref, tr_ref = ref.simulate(y0=y0, t_span=t_span, dt=dt)
            # t_sys, tr_sys = sys.simulate(y0=y0, t_span=t_span, dt=dt)

            mask_ref = t_ref >= transient
            mask_sys = t_sys >= transient
            tr_ref2 = tr_ref[mask_ref]
            tr_sys2 = tr_sys[mask_sys]

            if tr_ref2.shape[0] < 10 or tr_sys2.shape[0] < 10:
                per_init.append(
                    {
                        "init": y0,
                        "value": float("nan"),
                        "metric": "JS" if use_js else "KL",
                        "status": "insufficient_points",
                    }
                )
                continue

            hist_range = _auto_range(tr_sys2, tr_ref2)

            if use_js:
                d = js_divergence_hist(
                    tr_sys2, tr_ref2, bins=bins, hist_range=hist_range
                )
                metric = "JS"
            else:
                d = kl_divergence_hist(
                    tr_sys2, tr_ref2, bins=bins, hist_range=hist_range
                )
                metric = "KL"

            # per_init.append({"init": y0, "value": float(d), "metric": metric})
            per_init.append(
                {"init": y0, "value": float(d), "metric": metric, "status": "ok"}
            )

        values = np.array([x["value"] for x in per_init], dtype=float)
        if np.all(np.isfinite(values)):
            mean_val = float(values.mean())
            std_val = float(values.std())
        else:
            mean_val = float("nan")
            std_val = float("nan")

        # mean_val = float(np.mean([x["value"] for x in per_init]))
        # std_val = float(np.std([x["value"] for x in per_init]))

        out[key] = {
            "name": sys.name,
            "run_id": sys.meta.get("run_id", None),
            "vec_idx": vec_idx,
            "metric": per_init[0]["metric"],
            "per_init": per_init,
            "mean": mean_val,
            "std": std_val,
        }

    return out


def save_systems_json(systems: List[PolyODE3], save_path: str):
    data = [s.to_jsonable() for s in systems]
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(save_path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def keep_latest_run(systems):
    "Keep the systems with the largest recorded run_id."
    max_run = max(s.meta.get("run_id", -1) for s in systems)
    ss = [s for s in systems if s.meta.get("run_id", -1) == max_run]
    ss.sort(key=lambda s: (s.meta.get("vec_idx") is None, s.meta.get("vec_idx", 10**9)))
    return ss


def plot_divergence_compare_bars(
    results_a,
    results_b,
    label_a="refA",
    label_b="refB",
    save_path=None,
    title=None,
    ratios=None,
    fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    annotate_fontsize=10,
    rotation=0,
    annotate_fmt="{:.3f}",
):
    keys = sorted(set(results_a.keys()) & set(results_b.keys()))
    if not keys:
        raise ValueError("No common vec_idx keys to plot.")

    means_a = [results_a[k]["mean"] for k in keys]
    stds_a = [results_a[k]["std"] for k in keys]
    means_b = [results_b[k]["mean"] for k in keys]
    stds_b = [results_b[k]["std"] for k in keys]

    def _finite_or_zero(v):
        return 0.0 if (v is None or (not np.isfinite(v))) else float(v)

    means_a_plot = [_finite_or_zero(v) for v in means_a]
    stds_a_plot = [_finite_or_zero(v) for v in stds_a]
    means_b_plot = [_finite_or_zero(v) for v in means_b]
    stds_b_plot = [_finite_or_zero(v) for v in stds_b]

    metric = results_a[keys[0]]["metric"]  # JS or KL
    x = np.arange(len(keys), dtype=float)
    width = 0.38

    if ratios is None:
        ratios = np.linspace(0.0, 1.0, len(keys))
    else:
        if len(ratios) != len(keys):
            raise ValueError(
                f"len(ratios)={len(ratios)} must equal len(keys)={len(keys)}"
            )

    plt.figure()

    # bars_a = plt.bar(x - width/2, means_a, yerr=stds_a, capsize=3, width=width, label=label_a)
    # bars_b = plt.bar(x + width/2, means_b, yerr=stds_b, capsize=3, width=width, label=label_b)
    bars_a = plt.bar(
        x - width / 2,
        means_a_plot,
        yerr=stds_a_plot,
        capsize=3,
        width=width,
        label=label_a,
    )
    bars_b = plt.bar(
        x + width / 2,
        means_b_plot,
        yerr=stds_b_plot,
        capsize=3,
        width=width,
        label=label_b,
    )

    plt.plot(x - width / 2, means_a, marker="o", linewidth=1.6, label=None)
    plt.plot(x + width / 2, means_b, marker="o", linewidth=1.6, label=None)

    def _annotate_bars(bar_container, values):
        for rect, val in zip(bar_container, values):
            x0 = rect.get_x() + rect.get_width() / 2
            y0 = rect.get_height()
            if not np.isfinite(val):
                plt.text(
                    x0, y0, "NA", ha="center", va="bottom", fontsize=annotate_fontsize
                )
            else:
                plt.text(
                    x0,
                    y0,
                    annotate_fmt.format(val),
                    ha="center",
                    va="bottom",
                    fontsize=annotate_fontsize,
                )

    _annotate_bars(bars_a, means_a)
    _annotate_bars(bars_b, means_b)

    plt.xticks(
        x, [f"{r:.2f}" for r in ratios], fontsize=tick_fontsize, rotation=rotation
    )
    plt.yticks(fontsize=tick_fontsize)

    plt.xlabel(r"$\alpha_c/(\alpha_c+\beta_c)$", fontsize=fontsize)
    plt.ylabel(rf"{metric} Divergence", fontsize=fontsize)

    if title is not None:
        plt.title(title, fontsize=fontsize)

    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=450)
    else:
        plt.show()
    plt.close()


def polyode_from_odebank(system_obj, override_name=None):
    "Convert a classic system and its recorded polynomial coefficients to PolyODE3."
    if hasattr(system_obj, "get_info"):
        info = system_obj.get_info()
    else:
        info = str(system_obj)

    lines = [ln.strip() for ln in str(info).splitlines() if ln.strip()]

    dx_expr = dy_expr = dz_expr = None
    name = None
    grid_limits = None
    sys_vel = None

    for ln in lines:
        if ln.lower().startswith("dx/dt"):
            dx_expr = ln.split("=", 1)[1].strip()
        elif ln.lower().startswith("dy/dt"):
            dy_expr = ln.split("=", 1)[1].strip()
        elif ln.lower().startswith("dz/dt"):
            dz_expr = ln.split("=", 1)[1].strip()
        elif ln.lower().startswith("name"):
            name = ln.split("=", 1)[1].strip()
        elif ln.lower().startswith("grid_limits"):
            rhs = ln.split("=", 1)[1].strip()

            try:
                grid_limits = ast.literal_eval(rhs)
            except Exception:
                grid_limits = rhs
        elif "system_velocity" in ln.lower():
            rhs = ln.split("=", 1)[1].strip()
            try:
                sys_vel = float(rhs)
            except Exception:
                sys_vel = rhs

    if dx_expr is None or dy_expr is None or dz_expr is None:
        raise ValueError("ode_bank get_info() missing dx/dt or dy/dt or dz/dt lines.")

    terms = [
        _parse_poly_expr(dx_expr),
        _parse_poly_expr(dy_expr),
        _parse_poly_expr(dz_expr),
    ]

    final_name = override_name or name or getattr(system_obj, "name", "odebank_ref")

    return PolyODE3(
        name=final_name,
        terms=terms,
        meta={
            "source": "ode_bank",
            "dx_expr": dx_expr,
            "dy_expr": dy_expr,
            "dz_expr": dz_expr,
            "grid_limits": grid_limits,
            "system_velocity(avg)": sys_vel,
        },
    )


def _simulate_traj_for_plot(sys: PolyODE3, y0, t_span, dt, transient, stride=5):
    try:
        t, tr = sys.simulate(y0=y0, t_span=t_span, dt=dt)
    except BlowUpError:
        return np.zeros((0, 3), dtype=float)
    except Exception:

        return np.zeros((0, 3), dtype=float)

    mask = t >= transient
    tr = tr[mask]
    if stride is not None and stride > 1:
        tr = tr[::stride]

    if tr.size == 0 or (not np.isfinite(tr).all()):
        return np.zeros((0, 3), dtype=float)

    return tr


def plot_attractors_row(
    systems: List[PolyODE3],
    y0: Tuple[float, float, float],
    ratios: List[float],
    save_path: str,
    t_span=(0.0, 100.0),
    dt=0.01,
    transient=20.0,
    stride=5,
    fontsize=12,
    tick_fontsize=10,
    title=None,
    pad_ratio: float = 0.05,
):
    "Plot one 3D attractor per panel, using independent trajectory bounds."
    assert len(systems) == len(ratios), "systems and ratios must have the same length"

    trajs = []
    for sys in systems:
        tr = _simulate_traj_for_plot(
            sys, y0=y0, t_span=t_span, dt=dt, transient=transient, stride=stride
        )
        trajs.append(tr)

    n = len(systems)
    fig = plt.figure(figsize=(3.2 * n, 3.2))

    for i, (sys, tr, r) in enumerate(zip(systems, trajs, ratios)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")

        if tr.shape[0] == 0:

            ax.text2D(
                0.5,
                0.5,
                "NA",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=fontsize,
            )
        else:

            ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], linewidth=0.7, color="#BF1E2E")

            mins = tr.min(axis=0)
            maxs = tr.max(axis=0)
            span = maxs - mins
            span[span == 0] = 1.0

            pad = pad_ratio * span
            mins = mins - pad
            maxs = maxs + pad

            ax.set_xlim(mins[0], maxs[0])
            ax.set_ylim(mins[1], maxs[1])
            ax.set_zlim(mins[2], maxs[2])

        ax.tick_params(labelsize=tick_fontsize)
        ax.view_init(elev=22, azim=-60)

        ax.set_xlabel("x", fontsize=tick_fontsize)
        ax.set_ylabel("y", fontsize=tick_fontsize)
        ax.set_zlabel("z", fontsize=tick_fontsize)

        # ax.set_title(rf"$\alpha_c={r:.2f}$", fontsize=fontsize)

    if title is not None:
        fig.suptitle(title, fontsize=fontsize + 2)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=450)
    plt.close()


def plot_dims_ky_id(
    ratios: List[float],
    ky_list: List[float],
    id_list: List[float],
    save_path: str,
    fontsize=14,
    tick_fontsize=12,
    legend_fontsize=12,
    title=None,
):
    assert len(ratios) == len(ky_list) == len(id_list)

    plt.figure()
    plt.plot(ratios, ky_list, marker="o", linewidth=1.6, label="Kaplan-Yorke dim")
    plt.plot(ratios, id_list, marker="o", linewidth=1.6, label="Intrinsic dim")

    plt.xlabel(r"$\alpha_c/(\alpha_c+\beta_c)$", fontsize=fontsize)
    plt.ylabel("Dimension", fontsize=fontsize)
    plt.xticks(ratios, [f"{r:.2f}" for r in ratios], fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    if title is not None:
        plt.title(title, fontsize=fontsize)
    plt.legend(fontsize=legend_fontsize)
    plt.tight_layout()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=450)
    plt.close()


def plot_lyapunov_heatmap(
    ratios: List[float],
    spectra: List[List[float]],
    save_path: str,
    fontsize=14,
    tick_fontsize=12,
    title=None,
    annotate=True,
    annotate_fontsize=10,
    fmt="{:.2f}",
):
    "Plot an (N, 3) Lyapunov spectrum array against the fusion ratios."
    S = np.array(spectra, dtype=float)  # (N, 3)
    assert S.shape[0] == len(ratios) and S.shape[1] == 3

    H = S.T
    H = np.ma.masked_invalid(H)

    fig, ax = plt.subplots()
    im = ax.imshow(H, aspect="equal", origin="lower")
    # fig.colorbar(im, ax=ax)

    # x: ratio
    ax.set_xticks(np.arange(len(ratios)))
    ax.set_xticklabels([f"{r:.2f}" for r in ratios], fontsize=tick_fontsize)

    # y: lambda
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(
        [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"], fontsize=tick_fontsize
    )

    ax.set_xlabel(r"$\alpha_c/(\alpha_c+\beta_c)$", fontsize=fontsize)
    # ax.set_ylabel(r"$\lambda$", fontsize=fontsize)

    if title is not None:
        ax.set_title(title, fontsize=fontsize)

    # if annotate:
    #     for i in range(H.shape[0]):        # 0..2  (lambda index)
    #         for j in range(H.shape[1]):    # 0..N-1 (ratio index)
    #             ax.text(j, i, fmt.format(H[i, j]),
    #                     ha="center", va="center", fontsize=annotate_fontsize)
    if annotate:
        for i in range(H.shape[0]):
            for j in range(H.shape[1]):
                val = H[i, j]
                if np.ma.is_masked(val):
                    s = "NA"
                else:
                    s = fmt.format(float(val))

                # ax.text(j, i, s,
                #         ha="center", va="center",
                #         fontsize=annotate_fontsize,
                #         color=text_color)
                ax.text(j, i, s, ha="center", va="center", fontsize=annotate_fontsize)

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path, dpi=450)
    plt.close(fig)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Plot saved metrics for a classic fusion experiment."
    )
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--initial-index", type=int, default=0)
    parser.add_argument("--trajectories", action="store_true")
    args = parser.parse_args()
    folders = sorted(
        args.result_dir.glob("vec_*"), key=lambda p: int(p.name.split("_")[1])
    )
    if not folders:
        parser.error("result directory has no vec_* records")
    records = []
    for folder in folders:
        path = folder / f"init_{args.initial_index}" / "features" / "results.json"
        records.append(json.loads(path.read_text()) if path.exists() else {})
    if not any(records):
        parser.error("no saved feature records for this initial-index")
    ratios = np.linspace(0, 1, len(folders))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dims_ky_id(
        ratios=ratios,
        ky_list=[r.get("kaplan_yorke_dimension", float("nan")) for r in records],
        id_list=[r.get("intrinsic_dimension_twonn", float("nan")) for r in records],
        save_path=str(args.output_dir / "dims_ky_id.png"),
    )
    plot_lyapunov_heatmap(
        ratios=ratios,
        spectra=[r.get("lyapunov_spectrum", [float("nan")] * 3) for r in records],
        save_path=str(args.output_dir / "lyapunov_heatmap.png"),
    )
    if args.trajectories:
        systems = keep_latest_run(
            parse_ode_hat_from_log(
                str(
                    args.result_dir
                    / (
                        "info.log"
                        if (args.result_dir / "info.log").exists()
                        else "equations.txt"
                    )
                )
            )
        )[: len(folders)]
        initial = next(r["init_point"] for r in records if "init_point" in r)
        plot_attractors_row(
            systems=systems,
            y0=initial,
            ratios=ratios,
            save_path=str(args.output_dir / "attractors_row.png"),
            t_span=(0, 70),
            dt=0.01,
            transient=0.0,
            stride=1,
        )
    (args.output_dir / "plot_inputs.json").write_text(
        json.dumps(
            {
                "result_dir": str(args.result_dir),
                "initial_index": args.initial_index,
                "metric_source": "saved measured results; no new numerical evaluation",
                "missing_record_indices": [i for i, r in enumerate(records) if not r],
            },
            indent=2,
        )
    )
    plt.close("all")


if __name__ == "__main__":
    main()
