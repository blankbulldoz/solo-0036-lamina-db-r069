"""End-to-end independent analysis of the raw capacity measurements.

Pipeline (no numpy/scipy; stdlib only):
  1. load raw JSONL
  2. independent ledger cross-check every run; refuse to fit failed runs
  3. emit tidy CSVs
  4. fit competing models M0..M3, leave-one-axis-out validation
  5. microbench slope calibration + zero-fit prediction of engine runs
  6. disk models and the journal-never-shrinks invariant
  7. fixed-budget write-set headroom with 95% prediction intervals
  8. machine-readable findings.json consumed by the report generator
"""
import argparse
import csv
import json
import math
import os
import sys

# Isolated mode (-I) omits the script directory from sys.path; add it back so
# the sibling modules load while site-packages and cwd remain excluded.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import statsx as sx
from ledger import cross_check, ledger
import models as md


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def phase(rec, name):
    return md.phase_mem(rec, name)


def tidy_engine(recs, out_path):
    cols = ["run_id", "axis", "scenario", "replicate", "keys", "versions",
            "index_entries_total", "journal_records", "text_width",
            "updates", "hot_fraction", "batch_size", "delete_fraction",
            "index_count", "pin", "vacuum", "vacuum_pinned",
            "vacuum_effective_horizon", "vacuum_removed",
            "heap_post_load", "heap_post_vacuum", "heap_final",
            "rss_final", "journal_bytes", "checkpoint_bytes", "backup_bytes",
            "load_ms", "commit_ms", "commits_observed"]
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for r in recs:
            s = r["scenario"]
            st = r["stats_after"]
            w.writerow([
                r["run_id"], r.get("axis", ""), s["scenario"], r["replicate"],
                st["keys"], st["versions"],
                sum(st["index_entries"].values()), st["journal_records"],
                s["text_width"], s["updates"], s["hot_fraction"],
                s["batch_size"], s["delete_fraction"], s["index_count"],
                s["pin_snapshot"], s["vacuum_fraction"] > 0,
                r["vacuum_pinned"], r["vacuum_effective_horizon"],
                r["vacuum_removed_versions"],
                phase(r, "post_load"), phase(r, "post_vacuum"),
                phase(r, "final"),
                next((p["mem"]["rss_bytes"] for p in r["phases"]
                      if p["phase"] == "final"), None),
                r["journal_file"]["bytes"],
                r["checkpoint_file"]["bytes"] if r["checkpoint_file"]["exists"] else "",
                r["backup_image_bytes_post_vacuum"],
                round(r["load_elapsed_ms"], 3), round(r["commit_total_ms"], 3),
                r["commit_count_observed"],
            ])


def tidy_micro(recs, out_path):
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run_id", "component", "n", "text_width", "params",
                    "delta_heap_alloc", "delta_heap_inuse", "delta_rss",
                    "rse_notes"])
        for r in recs:
            w.writerow([r["run_id"], r["component"], r["n"],
                        r.get("text_width", ""), json.dumps(r.get("params", {})),
                        r["delta"]["heap_alloc_bytes"],
                        r["delta"]["heap_inuse_bytes"],
                        r["delta"]["rss_bytes"], r["notes"]])


def replicate_noise(recs):
    """Per-scenario measurement spread: the empirical error floor."""
    groups = {}
    for r in recs:
        groups.setdefault(r["scenario"]["scenario"], []).append(r)
    rows = []
    for name, rs in sorted(groups.items()):
        finals = [phase(r, "final") for r in rs if phase(r, "final")]
        loads = [r["load_elapsed_ms"] for r in rs]
        rows.append({
            "scenario": name, "replicates": len(rs),
            "heap_final_mean": sx.mean(finals),
            "heap_final_sd": sx.sd(finals),
            "heap_final_cv_pct": (100 * sx.sd(finals) / sx.mean(finals))
            if sx.mean(finals) else 0.0,
            "load_ms_cv_pct": (100 * sx.sd(loads) / sx.mean(loads))
            if sx.mean(loads) else 0.0,
        })
    cvs = [r["heap_final_cv_pct"] for r in rows if r["replicates"] > 1]
    return rows, {
        "max_heap_cv_pct": max(cvs) if cvs else 0.0,
        "median_heap_cv_pct": sx.median(cvs) if cvs else 0.0,
    }


def throughput_table(recs):
    """Observed write throughput and maintenance cost, model-free."""
    by_name = {}
    for r in recs:
        by_name.setdefault(r["scenario"]["scenario"], []).append(r)
    rows = []
    for name in sorted(by_name):
        rs = by_name[name]
        muts = sx.mean([r["ground_truth"]["total_mutations"] for r in rs])
        commits = sx.mean([r["commit_count_observed"] for r in rs])
        load = sx.mean([r["load_elapsed_ms"] for r in rs])
        commit_ms = sx.mean([r["commit_total_ms"] for r in rs])
        vac_values = [r["vacuum_elapsed_ms"] for r in rs
                      if r["vacuum_elapsed_ms"] > 0]
        vac_ms = sx.mean(vac_values) if vac_values else 0.0
        rows.append({
            "scenario": name,
            "axis": rs[0].get("axis", ""),
            "mutations_per_s": round(1000 * muts / load, 1) if load else None,
            "commits_per_s": round(1000 * commits / load, 2) if load else None,
            "mean_commit_ms": round(commit_ms / max(1, commits), 4),
            "vacuum_pause_ms": round(vac_ms, 3) if vac_ms else 0.0,
            "pinned": rs[0]["scenario"]["pin_snapshot"],
            "load_ms_cv_pct": round(100 * sx.sd([r["load_elapsed_ms"] for r in rs])
                                    / load, 2) if load and len(rs) > 1 else 0.0,
        })
    # Headline comparisons using scenario means across all replicates.
    def agg(name):
        return next((r for r in rows if r["scenario"] == name), None)
    headline = {}
    base = agg("u10k_h010_b050")
    b1 = agg("u10k_h010_b001")
    if base and b1:
        headline["batch50_vs_batch1_throughput_ratio"] = round(
            base["mutations_per_s"] / b1["mutations_per_s"], 2)
    pin_no = agg("u10k_pin_novac")
    if base and pin_no:
        headline["pinned_reader_throughput_ratio_vs_unpinned"] = round(
            pin_no["mutations_per_s"] / base["mutations_per_s"], 3)
    vac = agg("u10k_vac_head")
    if vac:
        headline["vacuum_pause_ms"] = vac["vacuum_pause_ms"]
    return rows, headline


def evidence_tables(recs):
    """Direct, model-free evidence for the brief's central claims."""
    by_name = {}
    for r in recs:
        by_name.setdefault(r["scenario"]["scenario"], []).append(r)

    def first(name):
        return sorted(by_name[name], key=lambda r: r["replicate"])[0]

    # Claim 1: identical current row count, wildly different version chains.
    same_rows = []
    for name in ("u02k_h010_b050", "u05k_h010_b050", "u10k_h010_b050",
                 "u10k_h100_b050", "u10k_h005_b050", "u10k_h010_b001",
                 "u10k_h010_b500"):
        if name not in by_name:
            continue
        r = first(name)
        same_rows.append({
            "scenario": name, "keys": r["stats_after"]["keys"],
            "versions": r["stats_after"]["versions"],
            "mean_chain": round(r["stats_after"]["versions"] /
                                r["stats_after"]["keys"], 2),
            "heap_final_mb": round(phase(r, "final") / 1e6, 3),
            "journal_kb": round(r["journal_file"]["bytes"] / 1e3, 1),
        })

    # Claim 2: a long reader makes planned reclamation ineffective.
    pin = []
    for name in ("u10k_vac_head", "u10k_pin_vac", "u10k_pin_novac"):
        if name not in by_name:
            continue
        r = first(name)
        pin.append({
            "scenario": name, "pinned": r["vacuum_pinned"],
            "effective_horizon": r["vacuum_effective_horizon"],
            "versions_removed": r["vacuum_removed_versions"],
            "versions_retained": r["stats_after"]["versions"],
            "heap_final_mb": round(phase(r, "final") / 1e6, 3),
            "journal_kb": round(r["journal_file"]["bytes"] / 1e3, 1),
        })

    # Claim 3: vacuum frees heap but never journal bytes (same workload pair).
    # The only permitted difference is the vacuum frame's own horizon integer
    # encoding (a few digits); every commit frame is byte-identical because the
    # workloads are otherwise identical.
    journal_invariant = []
    if "u10k_pin_vac" in by_name and "u10k_vac_head" in by_name:
        a = first("u10k_pin_vac")
        b = first("u10k_vac_head")
        diff = a["journal_file"]["bytes"] - b["journal_file"]["bytes"]
        journal_invariant.append({
            "pinned_journal_bytes": a["journal_file"]["bytes"],
            "vacuumed_journal_bytes": b["journal_file"]["bytes"],
            "byte_difference": diff,
            "difference_within_vacuum_frame_tolerance": abs(diff) <= 8,
            "same_record_count": (a["stats_after"]["journal_records"]
                                  == b["stats_after"]["journal_records"]),
            "same_mutation_count":
                a["ground_truth"]["total_mutations"]
                == b["ground_truth"]["total_mutations"],
            "pinned_versions": a["stats_after"]["versions"],
            "vacuumed_versions": b["stats_after"]["versions"],
        })
    # Vacuum adds exactly one durable frame relative to the identical workload
    # without a vacuum call; it never deletes any frame.
    if "u10k_h010_b050" in by_name and "u10k_vac_head" in by_name:
        plain = first("u10k_h010_b050")
        vac = first("u10k_vac_head")
        journal_invariant.append({
            "pair": "no_vacuum vs vacuum_head (identical write workload)",
            "plain_records": plain["stats_after"]["journal_records"],
            "vacuum_records": vac["stats_after"]["journal_records"],
            "vacuum_adds_exactly_one_record":
                vac["stats_after"]["journal_records"]
                - plain["stats_after"]["journal_records"] == 1,
            "plain_journal_bytes": plain["journal_file"]["bytes"],
            "vacuum_journal_bytes": vac["journal_file"]["bytes"],
            "plain_versions": plain["stats_after"]["versions"],
            "vacuum_retained_versions": vac["stats_after"]["versions"],
        })

    # Claim 4: index candidates track full history, not current rows.
    index_rows = []
    for name in ("u10k_h010_b050", "u10k_idx1", "u10k_idx2",
                 "u10k_idx1_late"):
        if name not in by_name:
            continue
        r = first(name)
        index_rows.append({
            "scenario": name,
            "indexes": r["scenario"]["index_count"],
            "keys": r["stats_after"]["keys"],
            "index_entries": sum(r["stats_after"]["index_entries"].values()),
            "versions": r["stats_after"]["versions"],
            "heap_final_mb": round(phase(r, "final") / 1e6, 3),
        })
    # Vacuum effect on index candidates: needs an index + vacuum scenario;
    # when present, entries must collapse with retained versions.
    for name in ("u10k_idx1_vac",):
        if name in by_name:
            r = first(name)
            index_rows.append({
                "scenario": name,
                "indexes": r["scenario"]["index_count"],
                "keys": r["stats_after"]["keys"],
                "index_entries": sum(r["stats_after"]["index_entries"].values()),
                "versions": r["stats_after"]["versions"],
                "heap_final_mb": round(phase(r, "final") / 1e6, 3),
            })
    return {"same_rows_different_chains": same_rows,
            "long_reader_pins_horizon": pin,
            "journal_never_truncated": journal_invariant,
            "index_candidates_track_history": index_rows}


def headroom_analysis(recs):
    """Fixed-memory-budget write-set admission under each retention policy.

    M2 predictors: Vr retained versions (store+index), Vh historical versions
    (WAL row copies, immune to vacuum), K live keys, I index candidates and C
    WAL record containers. Coefficient covariance and residual scatter give a
    95% prediction interval on every boundary.
    """
    base = md.mean_baseline_heap(recs)
    recs_f = [r for r in recs if phase(r, "final") is not None]
    y = [phase(r, "final") - base for r in recs_f]
    try:
        full = sx.ols([md.design_row(r, "M2") for r in recs_f], y)
        m2_singular = False
    except ValueError:
        full = sx.ols([md.design_row(r, "M1") for r in recs_f], y)
        m2_singular = True
    beta, dof, rse = full["beta"], full["dof"], full["rse"]
    b0, bVr = beta[0], beta[1]
    if m2_singular:
        bVh = bK = bI = bC = 0.0
    else:
        _, bVr, bVh, bK, bI, bC = beta
    K = 1000
    avg_batch = 50
    c0 = K / avg_batch + 2  # insert commits plus DDL records
    # P1 marginal per added historical version: retained + WAL row copies and
    # one record container per batch. Must be positive for inversion.
    marginal = bVr + bVh + bC / avg_batch
    using_fallback = marginal <= 0
    fit_for_bound = (sx.ols([md.design_row(r, "M1") for r in recs_f], y)
                     if using_fallback else full)

    def point_and_interval(Vr, Vh, I, C):
        if using_fallback:
            x = [1.0, float(Vr)]
        else:
            x = [1.0, float(Vr), float(Vh), float(K), float(I), float(C)]
        yhat = sum(b * xi for b, xi in zip(fit_for_bound["beta"], x))
        se = sx.prediction_se(fit_for_bound, x, fit_for_bound["rse"])
        tcrit = sx.t_critical(fit_for_bound["dof"])
        return yhat, yhat - tcrit * se, yhat + tcrit * se

    def bound_at(Vr, Vh, I, C):
        return point_and_interval(Vr, Vh, I, C)[2]

    def invert(f_of_v):
        """Largest non-negative integer v for which f(v) <= 0."""
        lo, hi = 0, 50_000_000
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if f_of_v(mid) <= 0:
                lo = mid
            else:
                hi = mid - 1
        return lo

    budgets = [64, 128, 256, 512]
    out = {"assumptions": {"keys": K, "avg_commit_batch": avg_batch,
                           "text_width": 24, "indexes": 0,
                           "baseline_heap_mb": round(base / 1e6, 3),
                           "coefficients_bytes": {
                               "per_retained_version_bVr": bVr,
                               "per_historical_wal_version_bVh": bVh,
                               "per_key_bK": bK, "per_index_entry_bI": bI,
                               "per_wal_record_bC": bC, "intercept_b0": b0},
                           "interval": "95% prediction interval, upper bound",
                           "inversion_model": "M1_fallback" if using_fallback else "M2",
                           "rse_mb": round(fit_for_bound["rse"] / 1e6, 3),
                           "dof": fit_for_bound["dof"]},
           "p1_full_history": [], "p2_vacuum_window": [],
           "p3_pin_reserve": [], "width_sensitivity": []}

    # P1: no vacuum. Retained == historical == total versions T; WAL holds
    # T/batch record containers as well as one row copy per version.
    for mb in budgets:
        B = mb * 1024 * 1024 - base

        def p1_bound(t):
            return bound_at(t, t, 0, t / avg_batch + c0) - B

        t_cap = invert(p1_bound)
        C = t_cap / avg_batch + c0
        yhat, lo95, hi95 = point_and_interval(t_cap, t_cap, 0, C)
        out["p1_full_history"].append({
            "budget_mb": mb, "max_versions": t_cap,
            "max_writes_equiv": t_cap - K,
            "predicted_heap_mb": round((base + yhat) / 1e6, 2),
            "pi95_mb": [round((base + lo95) / 1e6, 2),
                        round((base + hi95) / 1e6, 2)]})

    # P2: vacuum keeps a read window W of retained versions while the write
    # volume T keeps growing. The WAL still holds Vh=T row copies and every
    # commit's record container, so the window only saves bVr per reclaimed
    # version; the long-term ceiling is set by the never-truncated journal.
    for mb in budgets:
        B = mb * 1024 * 1024 - base
        p1 = next(r["max_versions"] for r in out["p1_full_history"]
                  if r["budget_mb"] == mb)
        row = {"budget_mb": mb, "planning_horizons": []}
        # Sample inside the feasible band: up to the measured ~1.75x ceiling,
        # where a non-zero read window still fits.
        for mult in (1.25, 1.5, 1.7):
            T = int(p1 * mult)
            C = T / avg_batch + c0
            w_cap = invert(
                lambda w, C=C, T=T: bound_at(w, T, 0, C) - B)
            row["planning_horizons"].append({
                "multiplier_vs_p1": mult,
                "total_versions_planned": T,
                "max_read_window_versions": w_cap,
                "window_covers_pct_of_history": round(100 * w_cap / T, 2)})
        # One deliberately infeasible point to document the WAL ceiling.
        T = p1 * 3
        C = T / avg_batch + c0
        w_over = invert(lambda w, C=C, T=T: bound_at(w, T, 0, C) - B)
        row["infeasible_horizon"] = {
            "total_versions_planned": T,
            "max_read_window_versions": w_over,
            "window_covers_pct_of_history": round(100 * w_over / T, 2)}
        # Volume ceiling when the window collapses to the minimum W=K.
        t_ceiling = invert(
            lambda t: bound_at(K, t, 0, t / avg_batch + c0) - B)
        row["write_volume_ceiling_min_window"] = t_ceiling
        row["volume_multiplier_vs_p1"] = round(t_ceiling / p1, 2) if p1 else None
        out["p2_vacuum_window"].append(row)

    # P3: reserve part of the budget for versions a long reader pins (the
    # pin prevents vacuum below its snapshot, so its protected versions are
    # retained); the writer fills the remainder under P1 rules.
    for mb in budgets:
        B = mb * 1024 * 1024 - base
        row = {"budget_mb": mb, "tiers": []}
        p1_cap = next(r["max_versions"] for r in out["p1_full_history"]
                      if r["budget_mb"] == mb)
        for reserve in (0.10, 0.25, 0.50):
            writer_cap = invert(
                lambda t: bound_at(t, t, 0, t / avg_batch + c0)
                - B * (1 - reserve))
            pin_cap = invert(
                lambda t: bound_at(t, t, 0, c0) - B * reserve)
            row["tiers"].append({
                "reserve_fraction": reserve,
                "writer_version_cap": writer_cap,
                "pinned_version_allowance": pin_cap,
                "writer_cap_vs_unreserved_pct":
                    round(100 * writer_cap / max(1, p1_cap), 1)})
        out["p3_pin_reserve"].append(row)

    # Width sensitivity via M3 (adds the (Vr+Vh)*w text term), refit on all runs.
    try:
        fit3 = sx.ols([md.design_row(r, "M3") for r in recs_f], y)
        b3 = fit3["beta"]
        m3_ok = True
    except ValueError:
        fit3, b3, m3_ok = None, None, False
    for mb in budgets:
        B = mb * 1024 * 1024 - base
        row = {"budget_mb": mb, "by_text_width": []}
        for w in (24, 64, 128, 256):
            if not m3_ok:
                row["by_text_width"].append(
                    {"text_width": w, "max_versions": None})
                continue

            def cost3(t, w=w):
                x = [1.0, float(t), float(t), float(K), 0.0,
                     float(t / avg_batch + c0), float(2 * t * w)]
                yhat = sum(b * xi for b, xi in zip(b3, x))
                se = sx.prediction_se(fit3, x, fit3["rse"])
                return yhat + sx.t_critical(fit3["dof"]) * se
            probe = cost3(1000) - cost3(100)
            cap = invert(lambda t: cost3(t) - B) if probe > 0 else None
            row["by_text_width"].append(
                {"text_width": w, "max_versions": cap})
        out["width_sensitivity"].append(row)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="../data/raw")
    ap.add_argument("--tidy", default="../data/tidy")
    args = ap.parse_args()
    os.makedirs(args.tidy, exist_ok=True)

    engine = load_jsonl(os.path.join(args.raw, "engine_runs.jsonl"))
    micro = load_jsonl(os.path.join(args.raw, "microbench.jsonl"))
    baselines = load_jsonl(os.path.join(args.raw, "baseline.jsonl"))

    # ---- 1. independent cross-checks
    check_rows = []
    failures = 0
    for r in engine:
        checks, L = cross_check(r)
        checks["scenario"] = r["scenario"]["scenario"]
        checks["replicate"] = r["replicate"]
        check_rows.append(checks)
        if not all(v for k, v in checks.items() if isinstance(v, bool)):
            failures += 1
    with open(os.path.join(args.tidy, "cross_checks.jsonl"), "w") as f:
        for c in check_rows:
            f.write(json.dumps(c, sort_keys=True) + "\n")
    if failures:
        raise SystemExit(f"{failures} runs failed independent cross-check; aborting fit")

    base_heap = sx.mean([b["mem"]["heap_alloc_bytes"] for b in baselines])
    base_sd = sx.sd([b["mem"]["heap_alloc_bytes"] for b in baselines])
    md.set_baseline(base_heap)

    # ---- 2. tidy tables
    tidy_engine(engine, os.path.join(args.tidy, "engine_runs_tidy.csv"))
    tidy_micro(micro, os.path.join(args.tidy, "micro_tidy.csv"))

    # ---- 3. competing model fits on final-phase heap above baseline
    fits, recs, y, base = md.fit_engine_models(engine, phase="final")
    loo_table, loo_detail = md.leave_one_axis_out(engine, phase="final")
    with open(os.path.join(args.tidy, "loo_predictions.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(loo_detail[0].keys()))
        w.writeheader()
        w.writerows(loo_detail)

    # ---- 4. micro calibration and zero-fit prediction
    slopes = md.micro_slopes(micro)
    micro_eval, micro_detail = md.evaluate_micro_calibrated(slopes, engine)
    with open(os.path.join(args.tidy, "micro_predictions.csv"), "w",
              newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(micro_detail[0].keys()))
        w.writeheader()
        w.writerows(micro_detail)

    # ---- 5. disk models and invariants
    disk = md.disk_models(engine)

    # ---- 6. evidence, noise, headroom
    evidence = evidence_tables(engine)
    noise_rows, noise_summary = replicate_noise(engine)
    throughput_rows, throughput_headline = throughput_table(engine)
    with open(os.path.join(args.tidy, "throughput.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(throughput_rows[0].keys()))
        w.writeheader()
        w.writerows(throughput_rows)
    headroom = headroom_analysis(engine)

    findings = {
        "schema_version": 1,
        "n_engine_runs": len(engine),
        "n_micro_runs": len(micro),
        "baseline": {"heap_alloc_mean": base_heap, "heap_alloc_sd": base_sd,
                     "rss_mean": sx.mean([b["mem"]["rss_bytes"] for b in baselines]),
                     "go_version": baselines[0]["go_version"],
                     "num_cpu": baselines[0]["num_cpu"]},
        "cross_check_failures": failures,
        "model_fits": {f"{m}_ols": fits[m] for m in fits},
        "leave_one_axis_out": loo_table,
        "micro_slopes": slopes,
        "micro_calibrated_eval": micro_eval,
        "disk_models": disk,
        "replicate_noise": noise_summary,
        "replicate_detail": noise_rows,
        "throughput_headline": throughput_headline,
        "throughput_detail": throughput_rows,
        "evidence": evidence,
        "headroom": headroom,
    }
    with open(os.path.join(args.tidy, "findings.json"), "w") as f:
        json.dump(findings, f, indent=2, sort_keys=True)
    print(json.dumps({
        "runs": len(engine), "cross_check_failures": failures,
        "models": {m: ({"status": fits[m]["status"]} if fits[m]["beta"] is None
                       else {"adj_r2": round(fits[m]["adj_r2"], 4),
                             "rse_kb": round(fits[m]["rse_bytes"] / 1e3, 2)})
                   for m in fits},
        "loo": [(t["model"],
                 None if t["loo_rmse_bytes"] is None
                 else round(t["loo_rmse_bytes"] / 1e3, 2),
                 t["loo_median_abs_pct"]) for t in loo_table],
        "micro_eval": {k: (round(v, 3) if isinstance(v, float) else v)
                       for k, v in micro_eval.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
