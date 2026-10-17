#!/usr/bin/env python3
"""Export the complete published classification paths and a display-only mesh.

This is read-only with respect to the research repository. It does not train a
network, optimize a vector field, integrate an ODE, or replace failed results.
Run with the bundled Python runtime (numpy required).
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

SITE = Path(__file__).resolve().parents[1]
WORKSPACE = SITE.parent
DEFAULT_REPO = WORKSPACE / "GitHub_更新预览_20261017/NDTL"
COMPONENTS = ["endpoint_gap", "barrier", "roughness", "chaos_loss"]


def csv_rows(path):
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def maybe_float(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def source(path):
    return {"path": str(path.relative_to(WORKSPACE)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def read_matrix(path):
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))
    assert rows[0][0] == "system"
    names = rows[0][1:]
    assert [row[0] for row in rows[1:]] == names
    matrix = np.asarray([[float(value) for value in row[1:]] for row in rows[1:]])
    assert matrix.shape == (52, 52)
    assert np.isfinite(matrix).all()
    assert np.allclose(matrix, matrix.T, atol=1e-12)
    assert np.allclose(np.diag(matrix), 0)
    return names, matrix


def path_components(spectra, alphas):
    """Same finite-point conventions as classification_benchmark/common.py."""
    values = np.asarray([[np.nan]*3 if row is None else row for row in spectra])
    valid = np.isfinite(values).all(axis=1)
    chaos_loss_mask = [not bool(ok) or values[i, 0] <= 0 for i, ok in enumerate(valid)]
    endpoint, barrier, roughness = None, None, None
    if valid[0] and valid[-1]:
        endpoint = float(np.linalg.norm(values[-1]-values[0]))
        terms = [float(np.linalg.norm(values[i]-((1-alpha)*values[0]+alpha*values[-1])))
                 for i, alpha in enumerate(alphas) if 0 < i < len(alphas)-1 and valid[i]]
        if terms:
            barrier = float(np.mean(terms))
    terms = [float(np.linalg.norm(values[i+1]-2*values[i]+values[i-1]))
             for i in range(1,len(alphas)-1) if valid[i-1:i+2].all()]
    if terms:
        roughness = float(np.mean(terms))
    return [endpoint, barrier, roughness, sum(chaos_loss_mask)/len(alphas)], chaos_loss_mask


def percentile_ranks(values):
    values = np.asarray([np.nan if x is None else x for x in values], dtype=float)
    work = values.copy()
    finite = np.isfinite(work)
    work[~finite] = np.max(work[finite])+1 if np.any(finite) else 1
    order = np.argsort(work, kind="mergesort")
    ranks = np.empty(len(work))
    start = 0
    while start < len(work):
        end = start+1
        while end < len(work) and work[order[end]] == work[order[start]]:
            end += 1
        ranks[order[start:end]] = .5*(start+end-1)
        start = end
    return ranks/(len(work)-1)


def idw_surface(xy, values, grid_size):
    """Original plot script's IDW p=2, padding=.08, squared-distance floor=1e-8."""
    lows, highs = xy.min(axis=0)-.08, xy.max(axis=0)+.08
    x = np.linspace(lows[0], highs[0], grid_size)
    y = np.linspace(lows[1], highs[1], grid_size)
    xx, yy = np.meshgrid(x,y)
    locations = np.stack([xx.ravel(),yy.ravel()],axis=1)
    diff = locations[:,None,:]-xy[None,:,:]
    dist2 = np.sum(diff*diff,axis=2)
    weights = 1/np.maximum(dist2,1e-8)
    zz = np.sum(weights*values[None,:],axis=1)/np.sum(weights,axis=1)
    return {
        "gridSize": grid_size, "x": np.round(x,8).tolist(), "y": np.round(y,8).tolist(),
        "z": np.round(zz.reshape(xx.shape),8).tolist(), "zRange": [float(values.min()),float(values.max())],
        "interpolation": "inverse-distance weighting, p=2", "padding": .08,
        "squaredDistanceFloor": 1e-8, "rowOrder": "z[yIndex][xIndex]",
        "role": "Display interpolation of 52 measured system-level similarity statistics; intermediate positions are not new systems or measured dynamical values.",
        "originalPaperSurfaceGridSize": 165,
    }


def export(repo, output, grid_size):
    result = repo / "classification_benchmark/results/content_only_paircae_dysts52"
    paper = repo / "classification_benchmark/results/display/paper"
    config_path = result / "config.json"
    config = json.loads(config_path.read_text())
    alphas = config["alphas"]
    names, distance = read_matrix(result/"distance_matrix.csv")
    assert len(alphas)==9
    name_index = {name:index for index,name in enumerate(names)}
    labels = {item["key"]:item["label"] for item in config["systems"]}
    records = csv_rows(result/"eval_records.csv")
    components = csv_rows(result/"pair_components.csv")
    scores = {row["system"]:row for row in csv_rows(paper/"dysts52_similarity_density_landscape_scores.csv")}
    selected = csv_rows(paper/"lyapunov_spectrum_landscape_selected_pairs.csv")
    by_pair = collections.defaultdict(dict)
    for row in records:
        index = int(row["alpha_index"])
        assert index not in by_pair[row["pair_id"]]
        assert abs(float(row["alpha"])-alphas[index]) < 1e-12
        by_pair[row["pair_id"]][index] = row
    pairs = []
    max_component_relative_error = 0
    for row in components:
        pair_id = row["pair_id"]
        assert len(by_pair[pair_id])==9
        ordered = [by_pair[pair_id][i] for i in range(9)]
        spectra = [[float(item[k]) for k in ["lambda1","lambda2","lambda3"]]
                   if item["status"]=="OK" else None for item in ordered]
        raw = [maybe_float(row[key]) for key in COMPONENTS]
        reconstructed, lost = path_components(spectra, alphas)
        for saved, calculated in zip(raw,reconstructed):
            assert (saved is None)==(calculated is None)
            if saved is not None:
                error = abs(saved-calculated)/max(1,abs(saved))
                max_component_relative_error = max(max_component_relative_error,error)
                assert error < 1e-8
        ranks = [float(row[key+"_rank"]) for key in COMPONENTS]
        d = float(row["distance"])
        assert abs(d-float(np.mean(ranks))) < 1e-10
        assert abs(distance[name_index[row["system_a"]],name_index[row["system_b"]]]-d) < 1e-10
        pairs.append({"id": pair_id, "a": row["system_a"], "b": row["system_b"],
                      "distance": d, "components": raw, "ranks": ranks,
                      "spectra": spectra, "status": [item["status"] for item in ordered],
                      "chaosLossMask": [bool(x) for x in lost],
                      "validCount": sum(item is not None for item in spectra)})
    assert len(pairs)==1326
    max_rank_error = 0
    for i in range(4):
        recalculated = percentile_ranks([p["components"][i] for p in pairs])
        max_rank_error=max(max_rank_error,float(np.max(np.abs(recalculated-np.array([p["ranks"][i] for p in pairs])))))
    assert max_rank_error < 1e-10

    upper = distance[np.triu_indices_from(distance,k=1)]
    sigma = float(np.median(upper[upper>0]))
    affinity = np.exp(-distance**2/(2*sigma**2))
    without_self = affinity.copy()
    np.fill_diagonal(without_self,0)
    density = without_self.sum(axis=1)/51
    potential = -np.log(density+1e-12)
    systems = []
    max_height_error = 0
    for index,name in enumerate(names):
        row=scores[name]
        max_height_error=max(max_height_error,abs(potential[index]-float(row["density_potential"])))
        assert abs(density[index]-float(row["similarity_density"])) < 1e-9
        systems.append({"id":name,"name":labels[name],"index":index,
                        "x":float(row["embedding_1"]),"y":float(row["embedding_2"]),
                        "cluster":int(row["cluster"]),"similarityDensity":float(row["similarity_density"]),
                        "U":float(row["density_potential"])})
    assert max_height_error < 1e-9
    xy=np.array([[node["x"],node["y"]] for node in systems])
    u=np.array([node["U"] for node in systems])
    cluster_order=sorted(range(52),key=lambda i:(systems[i]["cluster"],systems[i]["x"],systems[i]["y"],names[i]))
    examples=[{"id":row["pair_id"],"role":row["selection_rule"],"source":"Published Fig. 4 selection CSV"} for row in selected]
    examples.append({"id":"lorenz__rossler","role":"interrupted_spectrum","source":"Existing eval_records.csv; includes failed evaluation at alpha=0.75."})
    out={
        "schemaVersion":1,"alphas":alphas,"componentOrder":COMPONENTS,
        "systems":systems,"sigma":sigma,"distanceMatrix":distance.tolist(),
        "affinityMatrix":np.round(affinity,12).tolist(),"clusterOrder":cluster_order,
        "pairs":pairs,"surface":idw_surface(xy,u,grid_size),"examples":examples,
        "defaultPairId":"chen__lv",
        "method":{
            "fusion":"Content-only, pair-specific CAE; L=(1-alpha)L_A+alpha L_B, with three encoder layers weighted equally.",
            "orientation":"alpha=0 is system a; alpha=1 is system b. Reverse the path if a user selects the opposite direction.",
            "normalization":"Attractor box mapped to [0,1]^3; mean vector norm on a 48^3 grid scaled to 5.0.",
            "lyapunovProtocol":{"initialState":config["init_point"],**config["lyap_config"]},
            "componentDefinitions":[
                {"id":"endpoint_gap","label":"Endpoint gap","formula":"||lambda(1)-lambda(0)||_2","meaning":"Difference between parent Lyapunov spectra.","finiteConvention":"Requires both endpoints."},
                {"id":"barrier","label":"Path deviation","formula":"mean ||lambda(alpha)-[(1-alpha)lambda(0)+alpha lambda(1)]||_2","meaning":"How far the fusion spectrum bends away from its linear endpoint reference.","finiteConvention":"Mean over valid interior evaluations; requires both endpoints."},
                {"id":"roughness","label":"Path roughness","formula":"mean ||lambda[i+1]-2lambda[i]+lambda[i-1]||_2","meaning":"Local changes of slope along the spectrum path.","finiteConvention":"Mean only over consecutive triples of valid evaluations; no interpolation across failed weights."},
                {"id":"chaos_loss","label":"Chaos loss","formula":"count[failed evaluation or lambda1<=0] / 9","meaning":"Fraction of the nine evaluations without a positive measured maximal exponent.","finiteConvention":"Includes the two endpoints. Nonpositive finite-time MLE does not by itself distinguish periodic or equilibrium dynamics."},
            ],
            "rankNormalization":"Percentile ranks across all 1326 unordered pairs, with average ranks for exact ties and denominator 1325. Missing raw components are placed after the largest finite value and share the corresponding worst-end average rank.",
            "distance":"d(A,B) = [r(endpoint_gap)+r(barrier)+r(roughness)+r(chaos_loss)] / 4",
            "affinity":"W_ij=exp[-d_ij^2/(2 sigma^2)], sigma=median(nonzero off-diagonal distances)=0.485",
            "similarityDensity":"rho_i=(1/51) sum_{j!=i} W_ij",
            "similarityPotential":"U_i=-ln(rho_i+1e-12)",
            "landscapeInterpretation":"Low U denotes a system with high average affinity to the other systems; high U denotes relative isolation under this fusion protocol. U is not physical energy or a dynamical potential governing the systems.",
            "embeddingInterpretation":"The two coordinates are saved spectral-embedding coordinates, rescaled separately to [-1,1]. They are dimensionless display coordinates; clustering is computed from the full affinity matrix, not from this two-dimensional view.",
            "surfaceInterpretation":"The continuous surface is a display interpolation using the original plotting script's IDW method, not evidence for unmeasured intermediate systems, physical basins, or transition barriers.",
            "failureInterpretation":"All saved FAIL records report numerical trajectory explosion or non-finite values. Render them as gaps or failure markers; never substitute zero Lyapunov exponents.",
        },
        "audit":{
            "numSystems":52,"numPairs":len(pairs),"numWeights":9,"numEvaluations":len(records),
            "evaluationStatusCounts":dict(collections.Counter(row["status"] for row in records)),
            "successfulNonpositiveMleCount":sum(row["status"]=="OK" and float(row["lambda1"])<=0 for row in records),
            "validPointsPerPairHistogram":dict(sorted(collections.Counter(pair["validCount"] for pair in pairs).items())),
            "missingRawComponentCounts":{name:sum(pair["components"][i] is None for pair in pairs) for i,name in enumerate(COMPONENTS)},
            "rawComponentReconstructionMaxRelativeError":max_component_relative_error,
            "rankReconstructionMaxAbsoluteError":max_rank_error,
            "densityPotentialReconstructionMaxAbsoluteError":max_height_error,
            "clusterLabelSource":"Use the same landscape_scores.csv as the existing classification.json. Other archived k7 order files contain a different partition and are deliberately not mixed into this export.",
        },
        "provenance":{
            "paper":"Physical Review Research 8, 033337 (2026), Sec. IV, Fig. 4–6, Appendix E/Fig. 20",
            "doi":"https://doi.org/10.1103/ccgv-hgjb",
            "sources":[source(path) for path in [config_path,result/"eval_records.csv",result/"pair_components.csv",result/"distance_matrix.csv",paper/"dysts52_similarity_density_landscape_scores.csv",paper/"lyapunov_spectrum_landscape_selected_pairs.csv",repo/"classification_benchmark/common.py",repo/"classification_benchmark/plot_dysts52_similarity_density_landscape.py"]],
            "numericPolicy":"Saved spectra, raw components, ranks, distances, coordinates, cluster labels and point heights are retained. Only the display mesh is newly interpolated, at the requested browser resolution.",
        },
    }
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(out,ensure_ascii=False,separators=(",",":"),allow_nan=False)+"\n",encoding="utf-8")
    print(json.dumps({"output":str(output),"bytes":output.stat().st_size,"systems":52,"pairs":len(pairs),"grid":grid_size,"audit":out["audit"]},ensure_ascii=False))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--repo",type=Path,default=DEFAULT_REPO)
    parser.add_argument("--output",type=Path,default=SITE/"dist/data/classification-landscape.json")
    parser.add_argument("--grid-size",type=int,default=51)
    args=parser.parse_args()
    export(args.repo,args.output,args.grid_size)


if __name__=="__main__":
    main()
