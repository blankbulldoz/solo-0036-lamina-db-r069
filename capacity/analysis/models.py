"""Competing capacity models, fit and compared on real measurements.

Models are deliberately nested in explanatory power so the evidence for each
extra term is visible:

  M0  row-count only          H = b0 + bK*K
      (the "one coefficient on current rows" baseline the brief rejects)
  M1  version linear          H = b0 + bV*V
  M2  structural components   H = b0 + bV*V + bK*K + bI*I + bC*C
      V retained versions, K live keys, I index candidates, C WAL records
  M3  structural + payload    M2 + bW*(V*w)   (w = text width, bytes)

Two independent fitting routes are used:
  * micro-calibrated: slopes come from component microbenchmarks in fresh
    processes; the engine runs are then *predicted*, never fit.
  * engine OLS: models fit directly on engine-run observations, validated
    leave-one-axis-out so prediction error is measured on unseen workload
    shapes rather than in-sample residuals.
"""
import math

import statsx as sx


def phase_mem(rec, phase):
    for p in rec["phases"]:
        if p["phase"] == phase:
            return p["mem"]["heap_alloc_bytes"]
    return None


def design_row(rec, model):
    """Predictor row for one engine observation.

    Vr = retained versions (version store + rebuilt index candidates); vacuum
    shrinks this. Vh = versions ever committed (distinct key-commit pairs):
    the WAL retains one row copy per historical version and vacuum never
    touches it. Without vacuum Vr == Vh; the vacuum scenarios are the only
    rows where they differ and provide the contrast that separates the two
    coefficients.
    """
    s = rec["stats_after"]
    K = float(s["keys"])
    Vr = float(s["versions"])
    Vh = float(rec["ground_truth"]["versions_distinct_key_commit"])
    I = float(sum(s["index_entries"].values()))
    C = float(s["journal_records"])
    w = float(rec["scenario"]["text_width"])
    if model == "M0":
        return [1.0, K]
    if model == "M1":
        return [1.0, Vr]
    if model == "M2":
        return [1.0, Vr, Vh, K, I, C]
    if model == "M3":
        return [1.0, Vr, Vh, K, I, C, (Vr + Vh) * w]
    raise ValueError(model)


MODEL_PREDICTORS = {
    "M0": ["intercept", "live_keys_K"],
    "M1": ["intercept", "retained_versions_Vr"],
    "M2": ["intercept", "retained_versions_Vr", "historical_versions_Vh",
           "live_keys_K", "index_candidates_I", "wal_record_containers_C"],
    "M3": ["intercept", "retained_versions_Vr", "historical_versions_Vh",
           "live_keys_K", "index_candidates_I", "wal_record_containers_C",
           "row_text_bytes_(Vr+Vh)w"],
}


def fit_engine_models(records, phase="final"):
    """Fit M0..M3 on engine observations (one row per replicate)."""
    recs = [r for r in records if phase_mem(r, phase) is not None]
    base = mean_baseline_heap(records)
    y = [phase_mem(r, phase) - base for r in recs]
    out = {}
    for m in ("M0", "M1", "M2", "M3"):
        X = [design_row(r, m) for r in recs]
        try:
            fit = sx.ols(X, y)
        except ValueError:
            # A predictor lacks variation in this data set (e.g. all key
            # counts equal, or a single text width). Record the condition
            # explicitly instead of fabricating coefficients.
            out[m] = {
                "predictors": MODEL_PREDICTORS[m],
                "beta": None, "se_beta": None, "r2": None,
                "adj_r2": None, "rse_bytes": None, "dof": None,
                "n": len(recs),
                "status": "unidentifiable: predictor collinear with others "
                          "in the available data"}
            continue
        out[m] = {
            "predictors": MODEL_PREDICTORS[m],
            "beta": fit["beta"],
            "se_beta": fit["se_beta"],
            "r2": fit["r2"],
            "adj_r2": fit["adj_r2"],
            "rse_bytes": fit["rse"],
            "dof": fit["dof"],
            "n": fit["n"],
            "status": "fit"}
    return out, recs, y, base


def mean_baseline_heap(records):
    # baselines are emitted into a separate file; the loader passes them via
    # module-level state set by the driver.
    return _BASELINE_HEAP


_BASELINE_HEAP = 0.0


def set_baseline(h):
    global _BASELINE_HEAP
    _BASELINE_HEAP = h


def predict(fit_beta, x):
    return sum(b * xi for b, xi in zip(fit_beta, x))


def leave_one_axis_out(records, phase="final"):
    """Fit each model on all scenario axes but one, predict the held-out axis.

    This measures extrapolation error across workload *shape*, which plain
    in-sample R2 cannot show. Groups are the axis letter encoded in the run
    metadata by the driver (tidy stage adds 'axis').
    """
    recs = [r for r in records if phase_mem(r, phase) is not None and "axis" in r]
    base = mean_baseline_heap(records)
    groups = sorted({r["axis"] for r in recs})
    summary = {m: {"squared_errors": [], "abs_pct_errors": [], "held_out": []}
               for m in ("M0", "M1", "M2", "M3")}
    detail = []
    for g in groups:
        train = [r for r in recs if r["axis"] != g]
        test = [r for r in recs if r["axis"] == g]
        for m in ("M0", "M1", "M2", "M3"):
            Xtr = [design_row(r, m) for r in train]
            ytr = [phase_mem(r, phase) - base for r in train]
            try:
                fit = sx.ols(Xtr, ytr)
            except ValueError:
                # Predictor has no variation in the training fold (e.g. all
                # index variation sits in the held-out axis): record the fold
                # as unidentifiable rather than crashing or fabricating error.
                for r in test:
                    detail.append({"axis_held_out": g, "model": m,
                                   "scenario": r["scenario"]["scenario"],
                                   "replicate": r["replicate"],
                                   "observed_bytes_above_baseline":
                                       phase_mem(r, phase) - base,
                                   "predicted_bytes_above_baseline": None,
                                   "error_bytes": None})
                summary[m]["held_out"].append(g)
                continue
            for r in test:
                x = design_row(r, m)
                yhat = predict(fit["beta"], x)
                y = phase_mem(r, phase) - base
                err = y - yhat
                summary[m]["squared_errors"].append(err ** 2)
                if y > 0:
                    summary[m]["abs_pct_errors"].append(abs(err) / y * 100.0)
                detail.append({"axis_held_out": g, "model": m,
                               "scenario": r["scenario"]["scenario"],
                               "replicate": r["replicate"],
                               "observed_bytes_above_baseline": y,
                               "predicted_bytes_above_baseline": yhat,
                               "error_bytes": err})
    table = []
    for m, d in summary.items():
        if not d["squared_errors"]:
            table.append({
                "model": m, "loo_rmse_bytes": None,
                "loo_median_abs_pct": None, "loo_p95_abs_pct": None,
                "loo_max_abs_pct": None, "held_out_predictions": 0,
                "unidentifiable_folds": d["held_out"]})
            continue
        pcts = sorted(d["abs_pct_errors"])
        table.append({
            "model": m,
            "loo_rmse_bytes": sx.rmse(d["squared_errors"]),
            "loo_median_abs_pct": sx.median(d["abs_pct_errors"]),
            "loo_p95_abs_pct": pcts[min(len(pcts) - 1,
                                        int(0.95 * len(pcts)))],
            "loo_max_abs_pct": max(d["abs_pct_errors"]),
            "held_out_predictions": len(d["squared_errors"]),
            "unidentifiable_folds": d["held_out"],
        })
    return table, detail


def micro_slopes(micro_records):
    """Per-element marginal heap cost from fresh-process microbenchmarks.

    Each component gets an OLS line delta_heap = a + b*n; width variants for
    row_map give the per-text-byte term directly.
    """
    out = {}
    by_component = {}
    for r in micro_records:
        by_component.setdefault(r["component"], []).append(r)

    def line(rs, xkey):
        xs = [float(r[xkey]) for r in rs]
        ys = [float(r["delta"]["heap_alloc_bytes"]) for r in rs]
        if len(set(xs)) < 2:
            return {"intercept": ys[0] if ys else 0.0, "slope": 0.0,
                    "se_slope": None, "r2": None, "rse_bytes": None,
                    "n": len(rs), "note": "insufficient distinct x for slope"}
        fit = sx.ols([[1.0, x] for x in xs], ys)
        return {"intercept": fit["beta"][0], "slope": fit["beta"][1],
                "se_slope": fit["se_beta"][1], "r2": fit["r2"],
                "rse_bytes": fit["rse"], "n": len(rs)}

    w24 = [r for r in by_component["row_map"] if _width_of(r) == 24]
    if w24:
        out["row_map_w24"] = line(w24, "n")
    wide = [r for r in by_component["row_map"] if _width_of(r) != 24]
    if wide:
        out["row_map_wide"] = line(wide, "n")
    # text-byte coefficient: difference of slopes divided by width difference,
    # recomputed by a two-predictor fit on all row_map points.
    rm = by_component["row_map"]
    X = [[1.0, float(r["n"]), float(r["n"] * _width_of(r))] for r in rm]
    y = [float(r["delta"]["heap_alloc_bytes"]) for r in rm]
    try:
        fit = sx.ols(X, y)
        out["row_map_joint"] = {
            "predictors": ["intercept", "per_row", "per_row_text_byte"],
            "beta": fit["beta"], "se_beta": fit["se_beta"],
            "r2": fit["r2"], "rse_bytes": fit["rse"], "n": len(rm)}
    except ValueError:
        # Too few width variants for a joint fit: fall back to slope
        # difference when both width lines exist.
        per_byte = 0.0
        if "row_map_w24" in out and "row_map_wide" in out and wide:
            per_byte = ((out["row_map_wide"]["slope"]
                         - out["row_map_w24"]["slope"])
                        / max(1, _width_of(wide[0]) - 24))
        out["row_map_joint"] = {
            "predictors": ["intercept", "per_row", "per_row_text_byte"],
            "beta": [0.0, out.get("row_map_w24", {}).get("slope", 0.0),
                     per_byte],
            "se_beta": [None, None, None], "r2": None, "rse_bytes": None,
            "n": len(rm), "note": "fallback slope-difference estimate"}
    vs = by_component["version_store"]
    out["version_store_per_version"] = line(vs, "n")
    out["index_tree_per_entry"] = line(by_component["index_tree"], "n")
    # WAL cost split into two independently measured terms:
    #   * journal_empty_records: fixed record/frame container with zero rows
    #   * journal_records: one full-row record each; slope = container + row,
    #     so the marginal row cost is the difference of the two slopes.
    empty = line(by_component.get("journal_empty_records", []), "n")
    full = line(by_component.get("journal_records", []), "n")
    out["journal_empty_per_record"] = empty
    out["journal_full_per_record"] = full
    if isinstance(empty.get("slope"), (int, float)) and isinstance(full.get("slope"), (int, float)):
        out["journal_row_marginal"] = {
            "slope": full["slope"] - empty["slope"],
            "derivation": "journal_records slope minus empty-record slope",
            "full_record_slope": full["slope"],
            "empty_record_slope": empty["slope"],
            "n": full.get("n"), "r2": None}
    else:
        out["journal_row_marginal"] = {"slope": 0.0,
                                       "derivation": "insufficient points",
                                       "n": None, "r2": None}
    return out


def _width_of(r):
    # Micro records don't carry the width explicitly; recover it from notes
    # via the driver-enriched field when present, else default 24.
    return r.get("text_width", 24)


def micro_calibrated_predict(slopes, rec):
    """Predict engine heap above baseline using microbench slopes only.

    Heap profiles attribute live bytes to two full row copies:
      * version store Clone: one per *retained* version Vr, reclaimed by vacuum
      * WAL-retained Normalize row: one per *historical* version Vh, immune to
        vacuum because the journal is never truncated
    plus per-version index candidates and a small container per WAL record.
    Same-batch same-key overwrites leave one live row per distinct key-commit
    pair, exactly the ground-truth historical version count.
    """
    s = rec["stats_after"]
    Vr = s["versions"]
    Vh = rec["ground_truth"]["versions_distinct_key_commit"]
    I = sum(s["index_entries"].values())
    C = s["journal_records"]
    w = rec["scenario"]["text_width"]
    vs = slopes["version_store_per_version"]["slope"]
    wal_row = slopes["journal_row_marginal"]["slope"]
    wal_rec = slopes["journal_empty_per_record"].get("slope", 0.0) or 0.0
    ptb = slopes["row_map_joint"]["beta"][2]
    fixed = slopes["version_store_per_version"]["intercept"]
    row_cost = vs * Vr + wal_row * Vh + ptb * (Vr + Vh) * max(0, w - 24)
    i_cost = slopes["index_tree_per_entry"]["slope"] * I
    c_cost = wal_rec * C
    return fixed + row_cost + i_cost + c_cost


def evaluate_micro_calibrated(slopes, records, phase="final"):
    rows = []
    for r in records:
        if phase_mem(r, phase) is None:
            continue
        base = mean_baseline_heap(records)
        y = phase_mem(r, phase) - base
        yhat = micro_calibrated_predict(slopes, r)
        rows.append({"scenario": r["scenario"]["scenario"],
                     "replicate": r["replicate"],
                     "observed_bytes_above_baseline": y,
                     "predicted_bytes_above_baseline": yhat,
                     "error_bytes": y - yhat,
                     "abs_pct": abs(y - yhat) / y * 100 if y > 0 else None})
    rms = sx.rmse([d["error_bytes"] for d in rows])
    pcts = sorted(d["abs_pct"] for d in rows if d["abs_pct"] is not None)
    return {"loo_rmse_bytes": rms,
            "median_abs_pct": sx.median(pcts),
            "p95_abs_pct": pcts[min(len(pcts) - 1, int(0.95 * len(pcts)))],
            "max_abs_pct": max(pcts),
            "n": len(rows)}, rows


# ---------------------------------------------------------------- disk models

def disk_models(records):
    """Journal and image sizes. Journal is append-only: vacuum cannot shrink
    it, which is tested explicitly. Competing models:
      J0  J = a + b*mutations
      J1  J = a + b1*commits + b2*mutation_text_bytes
    """
    recs = [r for r in records if r["journal_file"]["exists"]]
    mut = [r["ground_truth"]["total_mutations"] for r in recs]
    commits = [r["stats_after"]["journal_records"] for r in recs]
    widths = [r["scenario"]["text_width"] for r in recs]
    y = [float(r["journal_file"]["bytes"]) for r in recs]
    f0 = sx.ols([[1.0, float(m)] for m in mut], y)
    f1 = sx.ols([[1.0, float(c), float(m * w)]
                 for c, m, w in zip(commits, mut, widths)], y)
    img = [r for r in recs if r["checkpoint_file"]["exists"]]
    yi = [float(r["checkpoint_file"]["bytes"]) for r in img]
    Vi = [float(r["stats_after"]["versions"]) for r in img]
    fi = sx.ols([[1.0, v] for v in Vi], yi) if img else None
    return {
        "journal_J0_mutations": {"beta": f0["beta"], "se_beta": f0["se_beta"],
                                 "r2": f0["r2"], "rse_bytes": f0["rse"]},
        "journal_J1_records_and_text": {"beta": f1["beta"],
                                        "se_beta": f1["se_beta"],
                                        "r2": f1["r2"], "rse_bytes": f1["rse"]},
        "image_per_version": ({"beta": fi["beta"], "se_beta": fi["se_beta"],
                               "r2": fi["r2"], "rse_bytes": fi["rse"]}
                              if fi else None),
    }


# ------------------------------------------------------------- uncertainty

def prediction_interval(fit_beta, fit_cov, rse, x_new, dof):
    """95% prediction interval for a fresh observation at x_new."""
    yhat = predict(fit_beta, x_new)
    se = sx.prediction_se({"p": len(x_new), "cov": fit_cov}, x_new, rse)
    tcrit = sx.t_critical(dof)
    return yhat - tcrit * se, yhat, yhat + tcrit * se
