import re
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from ode_bank import *
from utils_ode import ensemble_rescaling

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


# ----------------------------
# 0) META INFO
# ----------------------------
META_INFO = 'lorenz+lv'
# META_INFO = 'lv+rossler'
# META_INFO = 'hadley+halvorsen'
# META_INFO = 'nose_hoover+rucklidge'
# META_INFO = 'lorenz+chen'
# META_INFO = 'lorenz+nose_hoover'
# META_INFO = 'lorenz+halvorsen'

if META_INFO == 'lorenz+lv':
    log_path = "./results_display/fusion_results/lorenz_chen_lv/lorenz+lv/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, chen_unit_nv = ensemble_rescaling(chen)
    _, lv_unit_nv = ensemble_rescaling(lv)
    REF_A = lorenz_unit_nv
    REF_A_NAME = "lorenz"
    REF_B = lv_unit_nv
    REF_B_NAME = "lv"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.3, 0.3, 0.5)]

    ID_LIST = [2.9258, 2.2751, 2.0734, 2.1059, 3.1413]
    KY_LIST = [2.1248, 2.1170, 2.0940, 2.0388, 2.0126]

    SPECTRA = [
        [0.71888144, -0.26939292, -3.60310726],
        [0.86931027, -0.24740166, -5.31657486],
        [0.87758803, -0.23135489, -6.87733049],
        [0.53901919, -0.2239613,  -8.12955057],
        [0.21233924, -0.09172644, -9.58515971],
    ]

if META_INFO == 'lv+rossler':
    log_path = "./results_display/fusion_results/lorenz_lv_rossler/lv+rossler/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, lv_unit_nv = ensemble_rescaling(lv)
    _, rossler_unit_nv = ensemble_rescaling(rossler)
    REF_A = lv_unit_nv
    REF_A_NAME = "lv"
    REF_B = rossler_unit_nv
    REF_B_NAME = "rossler"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.3, 0.3, 0.5)]

    NAN = float("nan")
    ID_LIST = [1.5735, NAN, 0.0, 0.0, 3.7493]
    KY_LIST = [2.0179, NAN, 0.0, 0.0, 2.1355]

    SPECTRA = [
        [0.17744025, -0.05502868, -6.82789624],
        [NAN, NAN, NAN],
        [-5.34513517, -5.36881091, -23.36708549],
        [-1.07096323, -1.1046846, -2.69413149],
        [0.77397089, -0.28694381, -3.59303449],
    ]
        
if META_INFO == 'hadley+halvorsen':
    log_path = "./results_display/fusion_results/lorenz_hadley_halvorsen/hadley+halvorsen/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, hadley_unit_nv = ensemble_rescaling(hadley)
    _, halvorsen_unit_nv = ensemble_rescaling(halvorsen)
    REF_A = hadley_unit_nv
    REF_A_NAME = "hadley"
    REF_B = halvorsen_unit_nv
    REF_B_NAME = "halvorsen"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.3, 0.3, 0.5)]

    NAN = float("nan")
    ID_LIST = [2.0804, NAN, 0.0, 2.9087, 2.5260]
    KY_LIST = [2.0809, NAN, 0.0, 1.5964, 1.6943]

    SPECTRA = [
        [0.7539091, -0.27347236, -5.93757028],
        [NAN, NAN, NAN],
        [-0.23401794, -0.32912264, -5.19485799],
        [0.88874899, -1.49026298, -1.79347196],
        [0.61636738, -0.88771309, -1.85144855],
    ]

if META_INFO == 'nose_hoover+rucklidge':
    log_path = "./results_display/fusion_results/lorenz_nose_hoover_rucklidge/nose_hoover+rucklidge/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, nose_hoover_unit_nv = ensemble_rescaling(nose_hoover)
    _, rucklidge_unit_nv = ensemble_rescaling(rucklidge)
    REF_A = nose_hoover_unit_nv
    REF_A_NAME = "nose_hoover"
    REF_B = rucklidge_unit_nv
    REF_B_NAME = "rucklidge"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.3, 0.3, 0.5)]

    NAN = float("nan")
    ID_LIST = [3.4283, NAN, NAN, 2.4458, 4.5831]
    KY_LIST = [2.0073, NAN, NAN, 1.7975, 1.3922]

    SPECTRA = [
        [0.20181586, -0.15669863, -6.21433067],
        [NAN, NAN, NAN],
        [NAN, NAN, NAN],
        [0.23602637, -0.29597183, -0.79495746],
        [0.16524952, -0.42132272, -0.62318845],
    ]

if META_INFO == 'lorenz+chen':
    log_path = "./results_display/fusion_results/lorenz_chen_lv/lorenz+chen/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, chen_unit_nv = ensemble_rescaling(chen)
    _, lv_unit_nv = ensemble_rescaling(lv)
    REF_A = lorenz_unit_nv
    REF_A_NAME = "lorenz"
    REF_B = chen_unit_nv
    REF_B_NAME = "chen"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.3, 0.3, 0.5)]

    NAN = float("nan")
    ID_LIST = [2.4307, 2.1744, 2.9779, 1.9573, 2.4805]
    KY_LIST = [2.1044, 2.1227, 2.0845, 2.0391, 2.0135]

    SPECTRA = [
        [0.45264071, -0.1400754, -2.99340534],
        [0.86406061, -0.25939755, -4.92752731],
        [0.76411775, -0.21144777, -6.54137581],
        [0.50568878, -0.1910169, -8.05177877],
        [0.26277229, -0.13239491, -9.62452781],
    ]

if META_INFO == 'lorenz+nose_hoover':
    log_path = "./results_display/fusion_results/lorenz_nose_hoover_rucklidge/lorenz+nose_hoover/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, nose_hoover_unit_nv = ensemble_rescaling(nose_hoover)
    _, rucklidge_unit_nv = ensemble_rescaling(rucklidge)
    REF_A = lorenz_unit_nv
    REF_A_NAME = "lorenz"
    REF_B = nose_hoover_unit_nv
    REF_B_NAME = "nose_hoover"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.1, 0.2, 0.8)]

    NAN = float("nan")
    ID_LIST = [2.5739, 0.0000, 0.0000, 0.0000, 2.4441]
    KY_LIST = [1.5930, 0.0000, 0.0000, 0.0000, 2.0165]

    SPECTRA = [
        [0.40539718, -0.68366983, -0.72173414],
        [-0.16367586, -0.38327292, -0.43882776],
        [-1.388717, -1.95358485, -1.98989195],
        [-0.14587833, -0.17772562, -7.39569383],
        [0.28000359, -0.11970448, -9.69305392],
    ]

if META_INFO == 'lorenz+halvorsen':
    log_path = "./results_display/fusion_results/lorenz_hadley_halvorsen/lorenz+halvorsen/info.log"

    _, lorenz_unit_nv = ensemble_rescaling(lorenz)
    _, hadley_unit_nv = ensemble_rescaling(hadley)
    _, halvorsen_unit_nv = ensemble_rescaling(halvorsen)
    REF_A = lorenz_unit_nv
    REF_A_NAME = "lorenz"
    REF_B = halvorsen_unit_nv
    REF_B_NAME = "halvorsen"

    RATIO = [0.0, 0.25, 0.50, 0.75, 1.0]

    # inits = [(0.3, 0.3, 0.5), (0.1, 0.2, 0.8), (0.1, 0.0, 0.0)]
    inits = [(0.1, 0.2, 0.8)]

    NAN = float("nan")
    ID_LIST = [1.9968, NAN, 0.0000, 0.0000, 2.0521]
    KY_LIST = [2.1083, NAN, 0.0000, 0.0000, 2.0145]

    SPECTRA = [
        [0.90046686, -0.24415322, -6.05953543],
        [NAN, NAN, NAN],
        [-1.86154787, -1.92371892, -3.58773349],
        [-0.97171501, -0.99383408, -6.55975951],
        [0.28527187, -0.14157845, -9.90618999],
    ]


Exp = Tuple[int, int, int]  # (ex, ey, ez)

@dataclass
class PolyODE3:
    """
    terms[i] : dict[exp_tuple -> coeff]
    """
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
                if ex: m *= base[0] ** ex
                if ey: m *= base[1] ** ey
                if ez: m *= base[2] ** ez
                s += c * m
            out[i] = s
        return out


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
            )

            traj = sol.y.T

            if sol.status == 1 and sol.t_events and len(sol.t_events[0]) > 0:
                tb = float(sol.t_events[0][0])
                raise BlowUpError(f"{self.name} blew up: ||y||>{blowup_norm} at t={tb:.6g}")

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

            traj[i] = y + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
            t = t_eval[i]

            if (blowup_norm is not None) and (np.linalg.norm(traj[i]) > blowup_norm):
                raise BlowUpError(f"{self.name} blew up: ||y||>{blowup_norm} at t={t:.6g}")
            if not np.isfinite(traj[i]).all():
                raise BlowUpError(f"{self.name} produced non-finite states (inf/nan) at t={t:.6g}")

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


_LOG_PREFIX_RE = re.compile(r"^\[[0-9]{4}-[0-9]{2}-[0-9]{2} .*?\]\s+INFO:\s*")

_TERM_RE = re.compile(
    r"^\s*([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*([xyzXYZ]+)?\s*$"
)

_VEC_RE = re.compile(r"\[vec\s+(\d+)\]")
_NAME_RE = re.compile(r"^\s*name\s*=\s*(.+?)\s*$", re.IGNORECASE)

def _strip_prefix(line: str) -> str:
    return _LOG_PREFIX_RE.sub("", line).rstrip("\n")

def _parse_poly_expr(expr: str) -> Dict[Exp, float]:
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

                if (dy_expr is not None) and (dz_expr is not None) and (name is not None):
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


def _auto_range(a: np.ndarray, b: np.ndarray, pad_ratio: float = 0.05) -> List[Tuple[float, float]]:
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

    P = (P.astype(float) + eps)
    Q = (Q.astype(float) + eps)
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
                per_init.append({"init": y0, "value": float("nan"), "metric": "JS" if use_js else "KL",
                                 "status": "diverged", "error": str(e)})
                continue
            except Exception as e:
                per_init.append({"init": y0, "value": float("nan"), "metric": "JS" if use_js else "KL",
                                 "status": "failed", "error": str(e)})
                continue
            # t_ref, tr_ref = ref.simulate(y0=y0, t_span=t_span, dt=dt)
            # t_sys, tr_sys = sys.simulate(y0=y0, t_span=t_span, dt=dt)

            mask_ref = t_ref >= transient
            mask_sys = t_sys >= transient
            tr_ref2 = tr_ref[mask_ref]
            tr_sys2 = tr_sys[mask_sys]

            if tr_ref2.shape[0] < 10 or tr_sys2.shape[0] < 10:
                per_init.append({"init": y0, "value": float("nan"), "metric": "JS" if use_js else "KL",
                                 "status": "insufficient_points"})
                continue

            hist_range = _auto_range(tr_sys2, tr_ref2)

            if use_js:
                d = js_divergence_hist(tr_sys2, tr_ref2, bins=bins, hist_range=hist_range)
                metric = "JS"
            else:
                d = kl_divergence_hist(tr_sys2, tr_ref2, bins=bins, hist_range=hist_range)
                metric = "KL"

            # per_init.append({"init": y0, "value": float(d), "metric": metric})
            per_init.append({"init": y0, "value": float(d), "metric": metric, "status": "ok"})

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
    max_run = max(s.meta.get("run_id", -1) for s in systems)
    ss = [s for s in systems if s.meta.get("run_id", -1) == max_run]
    ss.sort(key=lambda s: (s.meta.get("vec_idx") is None, s.meta.get("vec_idx", 10**9)))
    return ss

def plot_divergence_compare_bars(
    results_a, results_b,
    label_a="refA", label_b="refB",
    save_path=None, title=None,
    ratios=None,
    fontsize=14,          
    tick_fontsize=12,   
    legend_fontsize=12, 
    annotate_fontsize=10, 
    rotation=0,           
    annotate_fmt="{:.3f}" 
):
    keys = sorted(set(results_a.keys()) & set(results_b.keys()))
    if not keys:
        raise ValueError("No common vec_idx keys to plot.")

    means_a = [results_a[k]["mean"] for k in keys]
    stds_a  = [results_a[k]["std"]  for k in keys]
    means_b = [results_b[k]["mean"] for k in keys]
    stds_b  = [results_b[k]["std"]  for k in keys]

    def _finite_or_zero(v):
        return 0.0 if (v is None or (not np.isfinite(v))) else float(v)

    means_a_plot = [_finite_or_zero(v) for v in means_a]
    stds_a_plot  = [_finite_or_zero(v) for v in stds_a]
    means_b_plot = [_finite_or_zero(v) for v in means_b]
    stds_b_plot  = [_finite_or_zero(v) for v in stds_b]


    metric = results_a[keys[0]]["metric"]  # JS or KL
    x = np.arange(len(keys), dtype=float)
    width = 0.38

    if ratios is None:
        ratios = np.linspace(0.0, 1.0, len(keys))
    else:
        if len(ratios) != len(keys):
            raise ValueError(f"len(ratios)={len(ratios)} must equal len(keys)={len(keys)}")

    plt.figure()

    bars_a = plt.bar(x - width/2, means_a_plot, yerr=stds_a_plot, capsize=3, width=width, label=label_a)
    bars_b = plt.bar(x + width/2, means_b_plot, yerr=stds_b_plot, capsize=3, width=width, label=label_b)


    plt.plot(x - width/2, means_a, marker="o", linewidth=1.6, label=None)
    plt.plot(x + width/2, means_b, marker="o", linewidth=1.6, label=None)

    def _annotate_bars(bar_container, values):
        for rect, val in zip(bar_container, values):
            x0 = rect.get_x() + rect.get_width() / 2
            y0 = rect.get_height()
            if not np.isfinite(val):
                plt.text(x0, y0, "NA", ha="center", va="bottom", fontsize=annotate_fontsize)
            else:
                plt.text(x0, y0, annotate_fmt.format(val), ha="center", va="bottom", fontsize=annotate_fontsize)

    _annotate_bars(bars_a, means_a)
    _annotate_bars(bars_b, means_b)


    plt.xticks(x, [f"{r:.2f}" for r in ratios], fontsize=tick_fontsize, rotation=rotation)
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
    assert len(systems) == len(ratios), "systems = ratios "

    trajs = []
    for sys in systems:
        tr = _simulate_traj_for_plot(
            sys, y0=y0, t_span=t_span, dt=dt,
            transient=transient, stride=stride
        )
        trajs.append(tr)

    n = len(systems)
    fig = plt.figure(figsize=(3.2 * n, 3.2))

    for i, (sys, tr, r) in enumerate(zip(systems, trajs, ratios)):
        ax = fig.add_subplot(1, n, i + 1, projection="3d")

        if tr.shape[0] == 0:
            ax.text2D(
                0.5, 0.5, "NA",
                transform=ax.transAxes,
                ha="center", va="center",
                fontsize=fontsize
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
    S = np.array(spectra, dtype=float)           # (N, 3)
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
    ax.set_yticklabels([r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"], fontsize=tick_fontsize)

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

                # text_color = "white" if i == 2 else "black"
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
    systems = parse_ode_hat_from_log(log_path)
    print(f"Parsed {len(systems)} ODE systems from log (may include multiple runs).")

    systems = keep_latest_run(systems)

    systems = systems[:5]
    print("Using systems:", [(s.meta.get("vec_idx"), s.name) for s in systems])

    def divergence_painter():
        ref_a = polyode_from_odebank(REF_A)
        ref_b = polyode_from_odebank(REF_B)

        results_a = compute_divergences_to_reference(
            systems=systems,
            ref=ref_a,
            inits=inits,
            t_span=(0.0, 100.0),
            dt=0.01,
            transient=0.0,
            bins=10,
            use_js=True,
        )

        results_b = compute_divergences_to_reference(
            systems=systems,
            ref=ref_b,
            inits=inits,
            t_span=(0.0, 100.0),
            dt=0.01,
            transient=0.0,
            bins=10,
            use_js=True,
        )

        Path("./painter").mkdir(parents=True, exist_ok=True)
        Path(f"./painter/{META_INFO}").mkdir(parents=True, exist_ok=True)
        # Path("./painter/divergences_refA.json").write_text(json.dumps(results_a, indent=2), encoding="utf-8")
        # Path("./painter/divergences_refB.json").write_text(json.dumps(results_b, indent=2), encoding="utf-8")

        plot_divergence_compare_bars(
            results_a, results_b,
            label_a=f"ref: {REF_A_NAME}", label_b=f"ref: {REF_B_NAME}",
            ratios=RATIO,
            fontsize=25,
            tick_fontsize=25,
            legend_fontsize=15,
            annotate_fontsize=10,
            save_path=f"./painter/{META_INFO}/divergence.png",
        )
        print("divergence pic saved.")

    def attractor_painter():
        plot_attractors_row(
            systems=systems,
            y0=inits[0],
            ratios=RATIO,
            save_path=f"./painter/{META_INFO}/attractors_row.png",
            t_span=(0.0, 70.0),
            dt=0.01,
            transient=0.0, 
            stride=1,
            fontsize=14,
            tick_fontsize=10,
            title=None,
        )
        print("attractors saved.")

    def dimension_painter():
        plot_dims_ky_id(
            ratios=RATIO,
            ky_list=KY_LIST,
            id_list=ID_LIST,
            save_path=f"./painter/{META_INFO}/dims_ky_id.png",
            fontsize=25,
            tick_fontsize=25,
            legend_fontsize=25,
            title=None,
        )
        print("dims_ky_id saved.")

    def lyapunov_painter():
        plot_lyapunov_heatmap(
            ratios=RATIO,
            spectra=SPECTRA,
            save_path=f"./painter/{META_INFO}/lyapunov_heatmap.png",
            fontsize=25,
            tick_fontsize=25,
            title=None,
            annotate=True,
            annotate_fontsize=25,
            fmt="{:.2f}",
        )
        print("lyapunov_heatmap saved.")
    
    divergence_painter()
    attractor_painter()
    dimension_painter()
    lyapunov_painter()



if __name__ == "__main__":
    main()
