"""DYSTS system selection and polynomial representation used by the 52-system benchmark."""

from __future__ import annotations
import ast
import inspect
import json
import re
from pathlib import Path
from typing import Any, Callable
import numpy as np
from experiments.classification.systems import SystemSpec
from ndtl.fields import ODE
from ndtl.paths import ROOT as PROJECT_ROOT

DYSTS_REFERENCE_DIR = PROJECT_ROOT / "third_party/dysts"

FLOWS_PATH = DYSTS_REFERENCE_DIR / "flows.py"

METADATA_PATH = DYSTS_REFERENCE_DIR / "chaotic_attractors.json"

EXCLUDE_NAMES = {"Torus"}

STANDARD_TERMS = ["", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]


def camel_to_snake(name: str) -> str:
    text = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
    text = re.sub("([a-z0-9])([A-Z])", r"\1_\2", text)
    return text.lower()


def standard_basis(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    z = points[:, 2]
    return np.stack(
        [
            np.ones_like(x),
            x,
            y,
            z,
            x * x,
            x * y,
            x * z,
            y * y,
            y * z,
            z * z,
        ],
        axis=1,
    )


def load_dysts_namespace(flows_path: Path) -> dict[str, Any]:
    source = flows_path.read_text(encoding="utf-8")
    source = source.replace(
        "from .base import DynSys, DynSysDelay, staticjit",
        "class DynSys: pass\nclass DynSysDelay: pass\ndef staticjit(f): return staticmethod(f)",
    )
    namespace: dict[str, Any] = {"__name__": "dysts_reference_flows"}
    exec(compile(source, str(flows_path), "exec"), namespace)
    return namespace


def ast_class_has_rhs(flows_path: Path) -> set[str]:
    tree = ast.parse(flows_path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "_rhs":
                    names.add(node.name)
                    break
    return names


def rhs_callable(
    cls: type, params: dict[str, Any]
) -> Callable[[np.ndarray], np.ndarray] | None:
    func = getattr(cls, "_rhs", None)
    if func is None:
        return None

    sig = inspect.signature(func)
    arg_names = list(sig.parameters)
    if "t" not in arg_names:
        return None

    t_index = arg_names.index("t")
    state_args = arg_names[:t_index]
    param_args = arg_names[t_index + 1 :]
    if len(state_args) != 3:
        return None
    if set(param_args) - set(params):
        return None

    param_values = [params[name] for name in param_args]

    def f(point: np.ndarray) -> np.ndarray:
        point = np.asarray(point, dtype=float).reshape(3)
        values = [*point.tolist(), 0.0, *param_values]
        return np.asarray(func(*values), dtype=float).reshape(3)

    return f


def fit_quadratic_ode(
    name: str, rhs: Callable[[np.ndarray], np.ndarray]
) -> tuple[ODE, float]:
    rng = np.random.default_rng(12345)
    fit_points = rng.uniform(-1.3, 1.3, size=(40, 3))
    basis = standard_basis(fit_points)
    values = np.asarray([rhs(point) for point in fit_points], dtype=float)
    coef_t, *_ = np.linalg.lstsq(basis, values, rcond=None)
    coef = coef_t.T

    test_points = rng.uniform(-1.7, 1.7, size=(30, 3))
    predicted = standard_basis(test_points) @ coef.T
    expected = np.asarray([rhs(point) for point in test_points], dtype=float)
    max_error = float(np.max(np.abs(predicted - expected)))
    if max_error > 1e-8:
        raise ValueError(
            f"{name} is not represented by a quadratic polynomial; max_error={max_error:g}"
        )

    ode = ODE(
        name=camel_to_snake(name),
        dimension=3,
        max_order=2,
        coef=coef,
        h=0.01,
        grid_limits=None,
    )
    if list(ode.terms) != STANDARD_TERMS:
        raise ValueError(f"Unexpected ODE basis for {name}: {ode.terms}")
    return ode, max_error


def build_dysts_quadratic_specs(
    flows_path: Path,
    metadata_path: Path,
    limit_systems: int | None = None,
) -> tuple[list[SystemSpec], list[dict[str, Any]]]:
    metadata = json.load(metadata_path.open("r", encoding="utf-8"))
    namespace = load_dysts_namespace(flows_path)
    direct_rhs_classes = ast_class_has_rhs(flows_path)

    specs: list[SystemSpec] = []
    selection_rows: list[dict[str, Any]] = []

    for class_name in sorted(metadata):
        meta = metadata[class_name]
        if class_name in EXCLUDE_NAMES:
            continue
        if int(meta.get("embedding_dimension", -1)) != 3:
            continue
        if bool(meta.get("delay", False)) or bool(meta.get("nonautonomous", False)):
            continue
        cls = namespace.get(class_name)
        if cls is None:
            continue

        params = meta.get("parameters") or {}
        rhs = rhs_callable(cls, params)
        if rhs is None:
            continue

        try:
            ode, fit_error = fit_quadratic_ode(class_name, rhs)
        except Exception as exc:  # noqa: BLE001
            selection_rows.append(
                {
                    "class_name": class_name,
                    "key": camel_to_snake(class_name),
                    "selected": False,
                    "reason": str(exc),
                }
            )
            continue

        initial_conditions = meta.get("initial_conditions")
        if initial_conditions is None or len(initial_conditions) != 3:
            continue

        key = camel_to_snake(class_name)
        dt = float(meta.get("dt") or 0.005)
        spec = SystemSpec(
            key=key,
            label=class_name,
            source="DYSTS",
            ode=ode,
            x0=tuple(float(x) for x in initial_conditions),
            bbox_h=dt,
            bbox_burn=5000,
            bbox_steps=25000,
            bbox_max_abs=1e8,
        )
        specs.append(spec)
        selection_rows.append(
            {
                "class_name": class_name,
                "key": key,
                "selected": True,
                "source": (
                    "direct_rhs"
                    if class_name in direct_rhs_classes
                    else "inherited_rhs"
                ),
                "fit_error": fit_error,
                "dt": dt,
                "maximum_lyapunov_estimated": meta.get("maximum_lyapunov_estimated"),
                "parameters": json.dumps(params, sort_keys=True),
                "initial_conditions": json.dumps(initial_conditions),
            }
        )

    specs.sort(key=lambda spec: spec.key)
    selected_keys = {spec.key for spec in specs}
    selection_rows.sort(
        key=lambda row: (not row.get("selected", False), row.get("key", ""))
    )

    if limit_systems is not None:
        keep = {spec.key for spec in specs[:limit_systems]}
        specs = specs[:limit_systems]
        selection_rows = [
            (
                {**row, "selected": row.get("key") in keep}
                if row.get("key") in selected_keys
                else row
            )
            for row in selection_rows
        ]

    return specs, selection_rows
