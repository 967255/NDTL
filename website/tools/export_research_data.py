#!/usr/bin/env python3
"""Export small, provenance-bearing NDTL website datasets from read-only archives.

Run with the bundled Python runtime (numpy is required). This exporter never
modifies experiments and never trains, fuses, or recalculates reported metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import re
import tarfile
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]
WORKSPACE = SITE.parent
PUBLIC = WORKSPACE / "GitHub_更新预览_20261017/NDTL"
ARCHIVE = WORKSPACE / "ndtl投稿版/2026草稿结果"
OUT = SITE / "dist/data"
COMPANION = WORKSPACE / "GitHub_更新预览_20261017/NDTL_paper_data.tar.gz"


def read_json(path):
    return json.loads(path.read_text())


def source(path):
    return {
        "path": str(path.relative_to(WORKSPACE)),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def write(name, data):
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")


def logged_equations(path):
    """Use each vector index's last occurrence; fail if repeats disagree."""
    blocks = re.findall(
        r"\[vec (\d+)\] ode_hat info: (dx/dt = [^\n]+)\n(dy/dt = [^\n]+)\n(dz/dt = [^\n]+)",
        path.read_text(),
    )
    equations = {}
    for index, *eqs in blocks:
        index = int(index)
        if index in equations and equations[index] != eqs:
            raise ValueError(f"Conflicting equation logs: {path} vec {index}")
        equations[index] = eqs
    return equations


def coefficients(equations):
    rows = []
    basis = ["", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]
    for eq in equations:
        terms = eq.split(" = ", 1)[1].split(" + ")
        row = [0.0] * len(basis)
        for term in terms:
            match = re.fullmatch(r"([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)([xyz]*)", term)
            if match is None or match[2] not in basis:
                raise ValueError(f"Unexpected polynomial: {eq}")
            row[basis.index(match[2])] = float(match[1])
        rows.append(row)
    return np.array(rows)


def integrate_coefficients(coeff, burn_in=500, retained=4500):
    """RK4 integration of a saved polynomial; never interpolate fusion weights."""
    state = np.array([0.3, 0.3, 0.5])
    points = []
    h = 0.01

    def f(q):
        x, y, z = q
        return coeff @ np.array([1, x, y, z, x*x, x*y, x*z, y*y, y*z, z*z])

    for i in range(burn_in + retained):
        with np.errstate(over="ignore", invalid="ignore"):
            k1 = f(state)
            k2 = f(state + h * k1 / 2)
            k3 = f(state + h * k2 / 2)
            k4 = f(state + h * k3)
            state = state + h * (k1 + 2*k2 + 2*k3 + k4) / 6
        if not np.isfinite(state).all() or np.max(np.abs(state)) > 1e5:
            return [], "diverged"
        if i >= burn_in:
            points.append(state.tolist())
    return np.round(np.array(points), 5).tolist(), "normal"


def reintegrate(equations):
    """Fallback geometry only: integrate the 3-decimal logged equations.

    When companion data are available, export_fusion replaces this fallback
    with a full-precision regression from the original fused tensor.
    """
    return integrate_coefficients(coefficients(equations))


def format_equations(coeff):
    terms = ["", "x", "y", "z", "xx", "xy", "xz", "yy", "yz", "zz"]
    return [f"d{variable}/dt = " + " + ".join(f"{value:.6g}{term}" for value, term in zip(row, terms) if abs(value) > 1e-12) for variable, row in zip("xyz", coeff)]


def kaplan_yorke(spectrum):
    spectrum = np.sort(np.asarray(spectrum, dtype=float))[::-1]
    sums = np.cumsum(spectrum)
    indices = np.flatnonzero(sums >= 0)
    if not len(indices):
        return 0.0
    j = int(indices[-1])
    if j == len(spectrum) - 1:
        return float(len(spectrum))
    return float(j + 1 + sums[j] / abs(spectrum[j + 1]))


def upgrade_from_companion(cases):
    """Use actual nine-point benchmark coefficients and original style tensor.

    Benchmark rows come from a separate pair-specific-CAE experiment, so the
    five-point Fig.2/Fig.11 metadata remain in referenceFrames and are never
    combined with the new nine-point sequence.
    """
    coeff_member = "classification_benchmark/results/content_only_paircae_dysts52/fused_coefficients.npz"
    style_member = "results/fusion_20260220_224200spec_exp_lorenz_chen_lv/vecs_lorenz_unit_nv_chen_unit_nv_fusion_20260220_224200spec_exp_lorenz_chen_lv.npy"
    manifest_path = PUBLIC / "data/companion_manifest.json"
    manifest = {row["path"]: row for row in read_json(manifest_path)["files"]}
    member_sources = {}
    with tarfile.open(COMPANION, "r:gz") as tar:
        arrays = {}
        for name in [coeff_member, style_member]:
            blob = tar.extractfile(name).read()
            digest = hashlib.sha256(blob).hexdigest()
            assert len(blob) == manifest[name]["bytes"] and digest == manifest[name]["sha256"], f"Companion integrity mismatch: {name}"
            member_sources[name] = {
                "archive": str(COMPANION.relative_to(WORKSPACE)),
                "member": name, "bytes": len(blob), "sha256": digest,
                "manifestSource": source(manifest_path),
            }
            loaded = np.load(io.BytesIO(blob), allow_pickle=False)
            if name.endswith(".npz"):
                arrays[name] = {key: loaded[key].copy() for key in loaded.files if key.startswith(("000_lorenz__chen__", "001_lorenz__lv__"))}
                loaded.close()
            else:
                arrays[name] = loaded
    bundle = arrays[coeff_member]
    result = PUBLIC / "classification_benchmark/results/content_only_paircae_dysts52"
    eval_path = result / "eval_records.csv"
    eval_rows = list(csv.DictReader(eval_path.open()))
    config_path = result / "config.json"
    config = read_json(config_path)
    for case, prefix in zip(cases[:2], ["001_lorenz__lv", "000_lorenz__chen"]):
        original_frames = case["frames"]
        case["referenceFrames"] = [{k: frame[k] for k in ["weight", "metrics", "equations", "status", "provenance"]} for frame in original_frames]
        case["referencePaperReference"] = case["paperReference"]
        pid = prefix.split("_", 1)[1]
        archived_alphas = bundle[f"{prefix}__alphas"]
        assert np.array_equal(archived_alphas, np.array([0, .125, .25, .375, .5, .625, .75, .875, 1]))
        frames = []
        for i, alpha in enumerate(archived_alphas):
            row = next(r for r in eval_rows if r["pair_id"] == pid and int(r["alpha_index"]) == i)
            assert float(row["alpha"]) == float(alpha) and row["status"] == "OK"
            coeff = bundle[f"{prefix}__alpha_{i:03d}_coef"]
            assert coeff.shape == (3, 10) and np.isfinite(coeff).all()
            trajectory, status = integrate_coefficients(coeff, burn_in=1000)
            if status != "normal" or len(trajectory) != 4500:
                raise ValueError(f"Reintegrated benchmark trajectory failed: {pid} alpha {alpha}")
            spectrum = [float(row[f"lambda{j}"]) for j in [1, 2, 3]]
            frames.append({
                "weight": float(1 - alpha),
                "metrics": {"lyapunov": spectrum, "kaplanYorke": kaplan_yorke(spectrum), "intrinsicDimension": None, "jsA": None, "jsB": None},
                "trajectory": trajectory,
                "equations": format_equations(coeff),
                "coefficients": coeff.tolist(),
                "status": status,
                "trajectoryNote": "Reintegrated from saved full-precision NDTL benchmark equations; 4500 points after burn-in. No interpolation between fusion weights.",
                "provenance": {
                    "status": "paper-benchmark-full-precision-coefficients",
                    "trajectoryStatus": "reintegrated-from-full-precision-archived-coefficients",
                    "coefficientsSource": member_sources[coeff_member],
                    "coefficientArray": f"{prefix}__alpha_{i:03d}_coef",
                    "metricsSource": source(eval_path),
                    "metricsRow": {"pair_id": pid, "alpha_index": i, "alpha": float(alpha)},
                    "metricsAggregation": "single initial condition; original recorded Lyapunov spectrum; Kaplan-Yorke dimension derived from that spectrum",
                    "configSource": source(config_path),
                    "archivedWeight": float(alpha),
                    "displayWeightConvention": "1 - archived alpha, to express the Lorenz content weight",
                    "initialCondition": config["init_point"],
                    "h": .01, "burnIn": 1000, "retainedPoints": 4500,
                    "trajectorySelection": "RK4 h=0.01; 5500 steps; burn-in 1000; stride 1; 4500 consecutive points",
                    "recordedEvaluationStatus": row["status"],
                    "missingMetrics": "Intrinsic dimension and Jensen-Shannon divergences were not part of this benchmark export; null is deliberate.",
                    "equationDisplayPrecision": "6 significant digits for readability; coefficients contains full float64 values",
                },
            })
        case["frames"] = sorted(frames, key=lambda f: f["weight"])
        case["endpointOrder"] = {"firstFrame": case["parentB"], "lastFrame": case["parentA"], "parentAFrameIndex": 8, "parentBFrameIndex": 0}
        case["alphaMeaning"] = "Displayed weight is the Lorenz content fraction: weight=0 is parentB, weight=1 is parentA (Lorenz). It equals 1 minus the archived classification alpha."
        case["metricsDerivation"] = {"kaplanYorke": "Derived analytically from each frame's original archived Lyapunov spectrum", "intrinsicDimension": "Not recorded for this benchmark; null", "jsA": "Not recorded; null", "jsB": "Not recorded; null"}
        case["source"] = "NDTL 52-system classification benchmark · pair-specific CAE"
        case["paperReference"] = "Sec. IV and Appendix E; same benchmark underlying Figs. 4–6 and 18–20"
        case["provenance"].update({
            "status": "paper-benchmark-nine-real-weights",
            "experiment": "content_only_paircae_dysts52 (legacy pilot20 pair reused)",
            "notSameAsReferenceFigure": "The nine frames use independently trained pair-specific CAEs from the classification benchmark. They are not additional weights of the original five-point Fig.2/Fig.11 experiment.",
            "weightConvention": "Display weight increases toward Lorenz content; benchmark alpha increases toward the other parent, so display weight = 1-alpha.",
            "fusionWeightCount": 9,
            "coefficientBasis": ["1", "x", "y", "z", "x²", "xy", "xz", "y²", "yz", "z²"],
        })

    # Regress the original saved style tensor on the exact 48³ grid with an
    # intercept, as in utils_ode.vec2ode(mode='linear', fit_intercept=True).
    axes = [np.linspace(0, 1, 48)] * 3
    mesh = np.meshgrid(*axes, indexing="ij")
    points = np.stack([m.reshape(-1) for m in mesh], axis=1)
    x, y, z = points.T
    features = np.column_stack([x, y, z, x*x, x*y, x*z, y*y, y*z, z*z])
    feature_mean = features.mean(axis=0)
    centered_features = features - feature_mean
    fields = arrays[style_member]
    assert fields.shape == (9, 1, 48, 48, 48, 3)
    for i, frame in enumerate(cases[2]["frames"]):
        # Centered least squares with a recovered intercept is algebraically
        # equivalent to LinearRegression(fit_intercept=True) used by vec2ode.
        target = fields[i].reshape((-1, 3)).astype(np.float64)
        target_mean = target.mean(axis=0)
        fitted, _, _, _ = np.linalg.lstsq(centered_features, target - target_mean, rcond=None)
        intercept = target_mean - feature_mean @ fitted
        coeff = np.column_stack([intercept, fitted.T])
        if not np.allclose(coeff, coefficients(frame["equations"]), atol=.000501, rtol=0):
            raise ValueError(f"Full tensor regression does not agree with archived rounded style equations: frame {i}")
        trajectory, status = integrate_coefficients(coeff)
        if frame["status"] == "diverged":
            trajectory, status = [], "diverged"
        elif status != "normal":
            raise ValueError(f"Full-precision style trajectory unexpectedly diverged: frame {i}")
        frame["trajectory"] = trajectory
        frame["coefficients"] = coeff.tolist()
        frame["equations"] = format_equations(coeff)
        frame["trajectoryNote"] = "Reintegrated from the full-precision polynomial regression of the original saved NDTL vector field; original archived signatures are retained."
        frame["provenance"].update({
            "trajectoryStatus": "reintegrated-from-original-full-precision-vector-field-regression",
            "vectorFieldSource": member_sources[style_member],
            "vectorFieldIndex": i,
            "equationPrecision": "Full-precision float64 regression; six significant digits shown in equations; full values in coefficients",
            "regressionMethod": "Centered ordinary least squares with intercept (numpy.linalg.lstsq), algebraically equivalent to the publication's sklearn LinearRegression(fit_intercept=True); quadratic basis, 48³ points on [0,1]³",
            "regressionNumpyVersion": np.__version__,
            "roundedLogAgreement": "Each fitted coefficient agrees with the archived logged equation to its 3-decimal printed precision (absolute difference <=0.000501)",
            "trajectoryLimitation": "Trajectory is reintegrated from the original fused vector-field tensor's polynomial projection, not read from an archived trajectory file. No rounded log coefficients are used in this integration.",
            "trajectorySelection": "RK4 h=0.01; 5000 steps; burn-in 500; stride 1; 4500 consecutive points; init=(0.3,0.3,0.5)",
        })
    cases[2]["provenance"]["trajectorySource"] = "Original full-precision vector-field tensor from the verified companion package, projected by the publication regression procedure"


def export_fusion():
    definitions = [
        ("lorenz-lv", "Lorenz + Lv", "content-content", "Lorenz", "Lv", "Lorenz content weight", "lorenz_chen_lv/lorenz+lv", [0, .25, .5, .75, 1], "Fig. 2; Sec. III D"),
        ("lorenz-chen", "Lorenz + Chen", "content-content", "Lorenz", "Chen", "Lorenz content weight", "lorenz_chen_lv/lorenz+chen", [0, .25, .5, .75, 1], "Fig. 11; Sec. III D"),
        ("lorenz-content-chen-style", "Lorenz content + Chen style", "content-style", "Lorenz", "Chen", "Lorenz content / (Lorenz content + Chen style)", "lorenz_chen_lv_spec_exp/lorenz_c+chen_s", [0, .01, .1, .25, .5, .75, .9, .99, 1], "Table I; Sec. III D"),
    ]
    cases = []
    audit = []
    for case_id, title, mode, parent_a, parent_b, weight_label, relative, weights, paper_reference in definitions:
        root = PUBLIC / "results_display/fusion_results" / relative
        equation_path = root / "equations.txt"
        eq_map = logged_equations(equation_path)
        frames = []
        for index, weight in enumerate(weights):
            p = root / f"vec_{index}/init_0/features/results.json"
            data = read_json(p)
            config_path = root / f"vec_{index}/config.json"
            config = read_json(config_path)
            if data["init_point"] != [0.3, 0.3, 0.5]:
                raise ValueError(f"Wrong initial condition: {p}")
            eqs = eq_map[index]
            provenance = {
                "status": "paper-aligned-archived-evaluation",
                "metricsSource": source(p),
                "configSource": source(config_path),
                "equationsSource": source(equation_path),
                "equationPrecision": "3 decimals as printed by the archived logger; not full-precision coefficients",
                "metricsInitialCondition": data["init_point"],
                "metricsAggregation": "single init_0; not the three-initial-condition mean",
                "h": data["h"],
                "gridShape": config["grid_shape"],
                "gridLimits": config["grid_limits"],
                "recordedSimulationStatus": data["sim_status"],
            }
            if mode == "content-content":
                original_p = ARCHIVE / "results_display/fusion_results" / relative / f"vec_{index}/init_0/features/results.json"
                if read_json(original_p) != data:
                    raise ValueError(f"Archive metrics do not match verified baseline: {original_p}")
                traj_p = original_p.parent / "traj.npz"
                archive_equations = original_p.parents[2] / "ode_info.txt"
                if archive_equations.read_text().splitlines()[:3] != eqs:
                    raise ValueError(f"Archive equations disagree: {archive_equations}")
                with np.load(traj_p) as ar:
                    trajectory = np.round(ar["traj"][500:], 5).tolist()
                provenance.update({
                    "trajectoryStatus": "original-archived-trajectory",
                    "trajectorySource": source(traj_p),
                    "trajectorySelection": "Original saved traj: burn-in 500, stride 1, all 4500 remaining points from the 5000-point simulation; no interpolation or reintegration",
                    "archiveAgreement": "init_0 metrics are byte-for-byte JSON-equivalent to the verified baseline; logged equations agree",
                })
                status = "normal" if data["sim_status"] == "Normal" else "diverged"
                trajectory_note = "Original archived numerical trajectory (4500 displayed points after burn-in)."
            else:
                trajectory, status = reintegrate(eqs)
                provenance.update({
                    "trajectoryStatus": "illustrative-reintegration-from-rounded-log",
                    "trajectorySelection": "RK4 h=0.01; 5000 steps; burn-in 500; stride 1; all 4500 remaining points; init=(0.3,0.3,0.5)",
                    "trajectoryLimitation": "Original trajectory and full-precision coefficients are not present locally. Display geometry was reintegrated from 3-decimal logged equations; it is not the trajectory used for the reported metrics.",
                })
                trajectory_note = "Illustrative geometry, reintegrated from rounded archived equations; signatures come from the original evaluation."
            if not data["sim_status"] == "Normal":
                status = "diverged"
                trajectory = []
            frame = {
                "weight": weight,
                "metrics": {
                    "lyapunov": data["lyapunov_spectrum"],
                    "kaplanYorke": data["kaplan_yorke_dimension"],
                    "intrinsicDimension": data["intrinsic_dimension_twonn"],
                    "jsA": None,
                    "jsB": None,
                },
                "trajectory": trajectory,
                "equations": eqs,
                "status": status,
                "trajectoryNote": trajectory_note,
                "provenance": provenance,
            }
            if status == "diverged":
                frame["warning"] = "Archived run diverged. Reported spectrum is retained for traceability to Table I; it must not be interpreted as a validated stable attractor. No attractor trajectory is displayed."
            frames.append(frame)
            audit.append({"case": case_id, "weight": weight, "trajectoryPoints": len(trajectory), "status": status, "mle": data["lyapunov_spectrum"][0]})
        cases.append({
            "id": case_id, "title": title, "mode": mode,
            "parentA": parent_a, "parentB": parent_b,
            "weightLabel": weight_label,
            "source": "Archived NDTL numerical evaluations",
            "paperReference": paper_reference,
            "frames": frames,
            "provenance": {
                "status": "paper-aligned",
                "paper": "Dynamics creation through neural dynamical transfer learning, Phys. Rev. Research 8, 033337 (2026)",
                "doi": "10.1103/ccgv-hgjb",
                "auditReference": "NDTL_项目盘点_20261017/盘点报告.txt",
                "weightConvention": "Weight increases toward Lorenz content; for content-style, Chen style weight is 1-weight.",
                "missingMetrics": "Jensen-Shannon divergence values have not been exported; null is deliberate.",
            },
        })
    expected_mle = [-1.054, .007, .220, .234, .271, .247, .247, .305, .288]
    actual_mle = [round(frame["metrics"]["lyapunov"][0], 3) for frame in cases[2]["frames"]]
    if actual_mle != expected_mle:
        raise ValueError(f"Table I check failed: {actual_mle}")
    if COMPANION.exists():
        upgrade_from_companion(cases)
        audit = [{
            "case": case["id"], "weight": frame["weight"],
            "trajectoryPoints": len(frame["trajectory"]), "status": frame["status"],
            "mle": frame["metrics"]["lyapunov"][0],
            "trajectorySourceStatus": frame["provenance"]["trajectoryStatus"],
        } for case in cases for frame in case["frames"]]
    write("fusion.json", {"schemaVersion": 1, "cases": cases})
    return audit


def export_classification():
    root = PUBLIC / "classification_benchmark/results"
    result = root / "content_only_paircae_dysts52"
    score_path = root / "display/paper/dysts52_similarity_density_landscape_scores.csv"
    score_rows = list(csv.DictReader(score_path.open()))
    score_map = {r["system"]: r for r in score_rows}
    labels = {r["key"]: r["label"] for r in csv.DictReader((result / "selected_systems.csv").open())}
    matrix_path = result / "distance_matrix.csv"
    matrix_csv = list(csv.reader(matrix_path.open()))
    ids = matrix_csv[0][1:]
    if [r[0] for r in matrix_csv[1:]] != ids:
        raise ValueError("Distance matrix header and row order disagree")
    matrix = np.array([[float(v) for v in r[1:]] for r in matrix_csv[1:]])
    assert matrix.shape == (52, 52)
    assert np.allclose(matrix, matrix.T) and np.allclose(np.diag(matrix), 0)
    systems = [{
        "id": key, "name": labels[key],
        "x": float(score_map[key]["embedding_1"]), "y": float(score_map[key]["embedding_2"]),
        "cluster": int(score_map[key]["cluster"]),
        "similarityDensity": float(score_map[key]["similarity_density"]),
        "densityPotential": float(score_map[key]["density_potential"]),
    } for key in ids]
    cluster_ids = sorted({r["cluster"] for r in systems})
    clusters = [{"id": k, "name": f"Cluster {k}", "systems": [r["id"] for r in systems if r["cluster"] == k], "size": sum(r["cluster"] == k for r in systems)} for k in cluster_ids]
    pair_rows = list(csv.DictReader((result / "pair_components.csv").open()))
    assert len(pair_rows) == 1326
    config = read_json(result / "config.json")
    selected_path = root / "display/paper/lyapunov_spectrum_landscape_selected_pairs.csv"
    selected_rows = list(csv.DictReader(selected_path.open()))
    selected_ids = {r["pair_id"] for r in selected_rows}
    eval_path = result / "eval_records.csv"
    eval_rows = [r for r in csv.DictReader(eval_path.open()) if r["pair_id"] in selected_ids]
    signature_landscapes = [{
        "id": r["pair_id"], "systemA": r["system_a"], "systemB": r["system_b"],
        "distance": float(r["distance"]),
        "selection": r["selection_rule"],
        "frames": [{"weight": float(e["alpha"]), "lyapunov": [float(e["lambda1"]), float(e["lambda2"]), float(e["lambda3"])], "status": e["status"]} for e in sorted([e for e in eval_rows if e["pair_id"] == r["pair_id"]], key=lambda e: float(e["alpha"]))],
    } for r in selected_rows]
    data = {
        "schemaVersion": 1,
        "systems": systems,
        "distanceMatrix": np.round(matrix, 8).tolist(),
        "clusters": clusters,
        "source": "Original 52-system benchmark; archived paper figure data",
        "paperReference": "Fig. 5 (embedding), Fig. 18 (distance matrix), Fig. 4 (signature landscapes); Sec. IV, Appendix E",
        "signatureLandscapes": signature_landscapes,
        "provenance": {
            "status": "paper-aligned",
            "doi": "10.1103/ccgv-hgjb",
            "embeddingSource": source(score_path),
            "distanceMatrixSource": source(matrix_path),
            "signatureLandscapesSource": source(selected_path),
            "signatureEvaluationSource": source(eval_path),
            "auditReference": "NDTL_项目盘点_20261017/盘点报告.txt",
            "numSystems": 52, "numPairs": 1326,
            "embeddingAlgorithm": "Archived two-dimensional SpectralEmbedding of the NDTL-induced RBF affinity matrix; each axis scaled to [-1,1]. No coordinates were reconstructed from images or recomputed for this website.",
            "clusterSourceNote": "Cluster IDs and memberships are those in the paper Fig.5 density-landscape score export. Other archived plotting versions can use different cluster assignments; they are not mixed here.",
            "distanceDefinition": "Mean of percentile ranks of endpoint gap, barrier, roughness, and chaos loss across all 1326 pairs; a relative dissimilarity, not a strict mathematical metric.",
            "alphas": config["alphas"],
            "lyapunovConfig": config["lyap_config"],
            "initialCondition": config["init_point"],
            "densityPotentialMeaning": "Negative log of mean affinity to the other systems; visualization of similarity density, not physical potential energy.",
            "exportChanges": "Distance values rounded to 8 decimals for browser delivery; original CSVs retain their precision.",
        },
    }
    write("classification.json", data)
    return {"systems": len(systems), "pairs": len(pair_rows), "clusterSizes": [c["size"] for c in clusters]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fusion-only", action="store_true", help="Export fusion and update its audit without touching classification.json")
    args = parser.parse_args()
    audit_path = OUT / "data-audit.json"
    audit = read_json(audit_path) if args.fusion_only and audit_path.exists() else {}
    audit["fusion"] = export_fusion()
    if not args.fusion_only:
        audit["classification"] = export_classification()
    write("data-audit.json", audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
