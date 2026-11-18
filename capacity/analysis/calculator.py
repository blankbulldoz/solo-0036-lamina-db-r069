#!/usr/bin/env python3
"""Capacity calculator: tunable inputs -> prediction with error bounds.

Consumes only the fitted coefficients in findings.json, never the raw
measurements, so it is the operational face of the calibrated models.

Two modes:
  1. single point from flags:
     calculator.py --findings ../data/tidy/findings.json \
         --keys 1000 --versions 12000 --index-entries 0 --journal 260 --width 24
  2. batch: JSONL file with one {keys, versions, index_entries, journal_records,
     text_width} object per line; emits one prediction row per object.

Every prediction reports the M1/M2/M3 point estimates and the M2 95%
prediction interval. A --budget-mb flag additionally returns an admit/deny
verdict using the interval's upper bound (deny unless upper bound fits).
"""
import argparse
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import statsx as sx  # noqa: E402


def load(findings_path):
    d = json.load(open(findings_path))
    base = d["baseline"]["heap_alloc_mean"]
    fits = d["model_fits"]
    return d, base, fits


def design(rec, model):
    K = rec["keys"]
    Vr = rec["versions"]
    Vh = rec.get("historical_versions", Vr)
    I = rec.get("index_entries", 0)
    C = rec.get("journal_records", 0)
    w = rec.get("text_width", 24)
    if model == "M1":
        return [1.0, float(Vr)]
    if model == "M2":
        return [1.0, float(Vr), float(Vh), float(K), float(I), float(C)]
    if model == "M3":
        return [1.0, float(Vr), float(Vh), float(K), float(I), float(C),
                float((Vr + Vh) * w)]
    raise ValueError(model)


def predict_with_interval(d, model, rec, base):
    fit = d["model_fits"][f"{model}_ols"]
    if fit.get("beta") is None:
        return {"status": "unidentifiable", "reason": fit.get("status", "")}
    x = design(rec, model)
    point = sum(b * xi for b, xi in zip(fit["beta"], x))
    # Prediction SE needs the covariance matrix; findings stores coefficient SE
    # only, so rebuild covariance from SE under a diagonal approximation and add
    # residual variance. This slightly overstates interval width when
    # predictors are negatively correlated — the conservative direction.
    var_mean = sum((se ** 2) * xi ** 2
                   for se, xi in zip(fit["se_beta"], x) if se is not None)
    rse = fit["rse_bytes"]
    se = math.sqrt(var_mean + rse * rse)
    tcrit = sx.t_critical(fit["dof"])
    return {
        "point_heap_mb": round((base + point) / 1e6, 3),
        "pi95_mb": [round((base + point - tcrit * se) / 1e6, 3),
                    round((base + point + tcrit * se) / 1e6, 3)],
        "interval_uses": "diagonal covariance approximation (conservative)",
    }


def evaluate(d, base, rec, budget_mb=None):
    out = {"inputs": rec,
           "M1": predict_with_interval(d, "M1", rec, base),
           "M2": predict_with_interval(d, "M2", rec, base),
           "M3": predict_with_interval(d, "M3", rec, base)}
    if budget_mb is not None:
        m2 = out["M2"]
        if m2.get("status") == "unidentifiable":
            out["admission"] = {
                "budget_mb": budget_mb, "verdict": "indeterminate",
                "reason": "M2 not identifiable; refit on data covering the "
                          "requested predictors before admitting"}
        else:
            upper = m2["pi95_mb"][1]
            out["admission"] = {
                "budget_mb": budget_mb,
                "upper_bound_mb": upper,
                "verdict": "admit" if upper <= budget_mb else "deny",
                "rule": "admit only when M2 95% upper prediction bound fits",
                "headroom_mb": round(budget_mb - upper, 3)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--findings", default="../data/tidy/findings.json")
    ap.add_argument("--batch", help="JSONL file of input objects")
    ap.add_argument("--keys", type=int)
    ap.add_argument("--versions", type=int,
                    help="retained versions Vr (post-vacuum if applicable)")
    ap.add_argument("--historical-versions", type=int,
                    help="distinct key-commit versions ever committed Vh; "
                         "defaults to --versions (no vacuum)")
    ap.add_argument("--index-entries", type=int, default=0)
    ap.add_argument("--journal-records", type=int, default=0)
    ap.add_argument("--width", type=int, default=24)
    ap.add_argument("--budget-mb", type=float)
    args = ap.parse_args()
    d, base, _ = load(args.findings)

    if args.batch:
        rows = [json.loads(line) for line in open(args.batch) if line.strip()]
        for rec in rows:
            print(json.dumps(evaluate(d, base, rec, args.budget_mb),
                             sort_keys=True))
        return
    if args.keys is None or args.versions is None:
        ap.error("provide --keys/--versions or --batch")
    rec = {"keys": args.keys, "versions": args.versions,
           "historical_versions": (args.historical_versions
                                   if args.historical_versions is not None
                                   else args.versions),
           "index_entries": args.index_entries,
           "journal_records": args.journal_records, "text_width": args.width}
    print(json.dumps(evaluate(d, base, rec, args.budget_mb), indent=2,
                     sort_keys=True))


if __name__ == "__main__":
    main()
