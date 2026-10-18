#!/usr/bin/env python3
"""Check the included classification tables without running training."""
import csv
import json
import math
from pathlib import Path

from ndtl.paths import ROOT


def read_rows(group, name):
    with (ROOT / "experiments/classification/results" / group / name).open(
        newline=""
    ) as stream:
        return list(csv.DictReader(stream))


def main():
    reports = []
    for group, systems, pairs in [("systems20", 20, 190), ("systems52", 52, 1326)]:
        components = read_rows(group, "pair_components.csv")
        fusion = read_rows(group, "fusion_records.csv")
        spectra = read_rows(group, "eval_records.csv")
        assert len(components) == pairs
        assert len(fusion) == pairs * 7
        assert len(spectra) == pairs * 9
        assert len({r["pair_id"] for r in fusion}) == pairs
        assert all(r["status"] == "OK" and int(r["iterations"]) > 0 for r in fusion)
        ranks = [
            "endpoint_gap_rank",
            "barrier_rank",
            "roughness_rank",
            "chaos_loss_rank",
        ]
        for row in components:
            expected = sum(float(row[key]) for key in ranks) / 4
            assert math.isclose(float(row["distance"]), expected, abs_tol=1e-10)
        reports.append(
            {
                "systems": systems,
                "pairs": pairs,
                "fusion_records": len(fusion),
                "spectrum_records": len(spectra),
                "ok": True,
            }
        )
    final = {
        (r["pair_id"], r["alpha_index"]): r
        for r in read_rows("systems52", "eval_records.csv")
    }
    for row in read_rows("systems20", "eval_records.csv"):
        saved = final[(row["pair_id"], row["alpha_index"])]
        for key in ["status", "lambda1", "lambda2", "lambda3", "used_steps", "dt"]:
            assert row[key] == saved[key]
    print(
        json.dumps(
            {"ok": True, "tables": reports, "reused_20_system_records": 1710}, indent=2
        )
    )


if __name__ == "__main__":
    main()
