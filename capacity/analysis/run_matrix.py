#!/usr/bin/env python3
"""Drive the LaminaDB capacity measurement matrix.

Every engine run is a separate OS process with an empty data directory, so
each heap observation starts from the same runtime baseline and WAL/image
forensics come from a known-empty directory. Output is one JSON object per
line (JSONL); nothing here computes derived quantities — that is the job of
the independent analysis stage.

Usage:
  run_matrix.py --harness ./harness --out ../data/raw --replicates 3 [--quick]
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

# Scenario matrix. Tunables are deliberately orthogonal so each competing
# model can be fit on one axis and challenged on the others.
# Fields: name, axis (for leave-one-axis-out validation), kwargs.
SCENARIOS = [
    # --- axis A: update pressure with identical current row count (1000 keys)
    dict(name="u02k_h010_b050", axis="A_pressure", keys=1000, updates=2000, hot=0.10, batch=50),
    dict(name="u05k_h010_b050", axis="A_pressure", keys=1000, updates=5000, hot=0.10, batch=50),
    dict(name="u10k_h010_b050", axis="A_pressure", keys=1000, updates=10000, hot=0.10, batch=50),
    # --- axis B: version-chain shape (same total mutations, skew varies)
    dict(name="u10k_h100_b050", axis="B_shape", keys=1000, updates=10000, hot=1.00, batch=50),
    dict(name="u10k_h050_b050", axis="B_shape", keys=1000, updates=10000, hot=0.50, batch=50),
    dict(name="u10k_h005_b050", axis="B_shape", keys=1000, updates=10000, hot=0.05, batch=50),
    # --- axis C: commit fanout (same writes, versions differ by batching)
    dict(name="u10k_h010_b001", axis="C_fanout", keys=1000, updates=10000, hot=0.10, batch=1),
    dict(name="u10k_h010_b010", axis="C_fanout", keys=1000, updates=10000, hot=0.10, batch=10),
    dict(name="u10k_h010_b250", axis="C_fanout", keys=1000, updates=10000, hot=0.10, batch=250),
    dict(name="u10k_h010_b500", axis="C_fanout", keys=1000, updates=10000, hot=0.10, batch=500),
    # --- axis D: payload width (per-version byte content)
    dict(name="u10k_h010_w064", axis="D_width", keys=1000, updates=10000, hot=0.10, batch=50, width=64),
    dict(name="u10k_h010_w128", axis="D_width", keys=1000, updates=10000, hot=0.10, batch=50, width=128),
    dict(name="u10k_h010_w256", axis="D_width", keys=1000, updates=10000, hot=0.10, batch=50, width=256),
    # --- axis E: tombstones and whole-chain removal
    dict(name="u10k_del10", axis="E_delete", keys=1000, updates=10000, hot=0.10, batch=50, delete=0.10),
    dict(name="u10k_del25", axis="E_delete", keys=1000, updates=10000, hot=0.10, batch=50, delete=0.25),
    # --- axis F: historical index candidates, early vs late build
    dict(name="u10k_idx1", axis="F_index", keys=1000, updates=10000, hot=0.10, batch=50, indexes=1),
    dict(name="u10k_idx2", axis="F_index", keys=1000, updates=10000, hot=0.10, batch=50, indexes=2),
    dict(name="u10k_idx1_late", axis="F_index", keys=1000, updates=10000, hot=0.10, batch=50,
         indexes=1, late=True),
    # Supplementary F/G overlap: index candidates actually rebuilt after vacuum.
    dict(name="u10k_idx1_vac", axis="F_index", keys=1000, updates=10000, hot=0.10, batch=50,
         indexes=1, vacuum=True),
    dict(name="u10k_idx1_pin_vac", axis="G_pin", keys=1000, updates=10000, hot=0.10, batch=50,
         indexes=1, pin=True, vacuum=True),
    # --- axis G: long-lived reader pinning the horizon
    dict(name="u10k_pin_novac", axis="G_pin", keys=1000, updates=10000, hot=0.10, batch=50, pin=True),
    dict(name="u10k_pin_vac", axis="G_pin", keys=1000, updates=10000, hot=0.10, batch=50,
         pin=True, vacuum=True),
    dict(name="u10k_vac_head", axis="G_pin", keys=1000, updates=10000, hot=0.10, batch=50, vacuum=True),
    dict(name="u10k_vac_mid", axis="G_pin", keys=1000, updates=10000, hot=0.10, batch=50,
         vacuum=True, horizon="mid"),
    # --- axis H: smaller/larger live row counts to expose the row coefficient
    dict(name="u05k_k0500", axis="H_rows", keys=500, updates=5000, hot=0.10, batch=50),
    dict(name="u05k_k2000", axis="H_rows", keys=2000, updates=5000, hot=0.10, batch=50),
]

QUICK_SCENARIOS = [
    dict(name="q_u02k", axis="A_pressure", keys=500, updates=2000, hot=0.10, batch=50),
    dict(name="q_u05k_skew", axis="B_shape", keys=500, updates=5000, hot=1.00, batch=50),
    dict(name="q_batch1", axis="C_fanout", keys=500, updates=5000, hot=0.10, batch=1),
    dict(name="q_vac", axis="G_pin", keys=500, updates=5000, hot=0.10, batch=50, vacuum=True),
    dict(name="q_pin", axis="G_pin", keys=500, updates=5000, hot=0.10, batch=50, pin=True, vacuum=True),
    dict(name="q_idx", axis="F_index", keys=500, updates=5000, hot=0.10, batch=50, indexes=1),
    dict(name="q_del", axis="E_delete", keys=500, updates=5000, hot=0.10, batch=50, delete=0.25, vacuum=True),
    dict(name="q_wide", axis="D_width", keys=500, updates=5000, hot=0.10, batch=50, width=256),
]

# Component microbenchmark sizes. Each row is one fresh process.
MICRO_RUNS = [
    ("row_map", dict(n=n, width=w))
    for n in (500, 1000, 2000, 4000, 8000, 16000, 32000)
    for w in (24, 128)
] + [
    ("version_store", dict(k=k, c=c))
    for (k, c) in ((250, 4), (500, 4), (1000, 4), (1000, 8), (1000, 16), (2000, 8))
] + [
    ("index_tree", dict(n=n))
    for n in (1000, 2000, 4000, 8000, 16000, 32000, 64000)
] + [
    ("journal_records", dict(n=n))
    for n in (500, 1000, 2000, 4000, 8000, 12000)
] + [
    # Zero-mutation records isolate the fixed WAL frame/container cost so the
    # per-row payload term can be separated (rows are priced via row_map).
    ("journal_empty_records", dict(n=n))
    for n in (2000, 8000, 20000, 40000)
]

MICRO_QUICK = [
    ("row_map", dict(n=1000, width=24)),
    ("row_map", dict(n=4000, width=128)),
    ("version_store", dict(k=1000, c=8)),
    ("index_tree", dict(n=8000)),
    ("journal_records", dict(n=4000)),
]


def engine_argv(harness, datadir, spec, seed):
    a = [harness, "engine", "-dir", datadir, "-scenario", spec["name"],
         "-keys", str(spec["keys"]), "-updates", str(spec["updates"]),
         "-hot-fraction", str(spec["hot"]), "-batch", str(spec["batch"]),
         "-text-width", str(spec.get("width", 24)),
         "-indexes", str(spec.get("indexes", 0)),
         "-delete-fraction", str(spec.get("delete", 0.0)),
         "-seed", str(seed)]
    if spec.get("late"):
        a.append("-late-index")
    if spec.get("pin"):
        a.append("-pin")
    if spec.get("vacuum"):
        a += ["-vacuum", "1"]
    if spec.get("horizon"):
        a += ["-vacuum-horizon", spec["horizon"]]
    return a


def micro_argv(harness, component, params):
    a = [harness, "micro", "-component", component]
    for k, v in params.items():
        flag = "-text-width" if k == "width" else "-" + k.replace("_", "-")
        a += [flag, str(v)]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--harness", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workdir", required=True,
                    help="scratch directory for per-run data directories")
    ap.add_argument("--replicates", type=int, default=3)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--only", help="comma-separated scenario names to run")
    ap.add_argument("--append", action="store_true",
                    help="append to existing JSONL instead of truncating")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.workdir, exist_ok=True)
    engine_path = os.path.join(args.out, "engine_runs.jsonl")
    micro_path = os.path.join(args.out, "microbench.jsonl")
    base_path = os.path.join(args.out, "baseline.jsonl")
    scenarios = QUICK_SCENARIOS if args.quick else SCENARIOS
    if args.only:
        wanted = {s.strip() for s in args.only.split(",") if s.strip()}
        scenarios = [s for s in scenarios if s["name"] in wanted]
        missing = wanted - {s["name"] for s in scenarios}
        if missing:
            raise SystemExit(f"unknown scenarios: {sorted(missing)}")
    micros = MICRO_QUICK if args.quick else MICRO_RUNS

    run_id = 0
    t0 = time.time()
    file_mode = "a" if args.append else "w"

    # Process baseline measured several times so analysis can estimate
    # measurement noise on the empty-runtime intercept.
    if not args.append:
        with open(base_path, "w") as bf:
            for rep in range(5):
                out = subprocess.run([args.harness, "baseline"], check=True,
                                     capture_output=True, text=True)
                rec = json.loads(out.stdout)
                rec["replicate"] = rep
                bf.write(json.dumps(rec, sort_keys=True) + "\n")

    with open(engine_path, file_mode) as ef, open(micro_path, file_mode) as mf:
        total = len(scenarios) * args.replicates + (0 if args.append else len(micros))
        if args.append:
            # Continue run numbering after the records already on disk.
            for path in (engine_path, micro_path):
                if os.path.exists(path):
                    with open(path) as f:
                        for line in f:
                            if line.strip():
                                run_id = max(run_id, json.loads(line).get("run_id", 0))
        for spec in scenarios:
            for rep in range(args.replicates):
                run_id += 1
                datadir = os.path.join(args.workdir, f"run_{run_id:04d}")
                os.makedirs(datadir, exist_ok=True)
                seed = 20261118 + rep * 7919
                argv = engine_argv(args.harness, datadir, spec, seed)
                proc = subprocess.run(argv, capture_output=True, text=True)
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    raise SystemExit(f"harness failed: {spec['name']} rep {rep}")
                rec = json.loads(proc.stdout)
                rec["replicate"] = rep
                rec["run_id"] = run_id
                rec["axis"] = spec["axis"]
                ef.write(json.dumps(rec, sort_keys=True) + "\n")
                ef.flush()
                shutil.rmtree(datadir, ignore_errors=True)
                elapsed = time.time() - t0
                print(f"[{run_id:3d}/{total}] {spec['name']} rep{rep} "
                      f"({elapsed:6.1f}s)", flush=True)

        if not args.append:
            for component, params in micros:
                run_id += 1
                argv = micro_argv(args.harness, component, params)
                proc = subprocess.run(argv, capture_output=True, text=True)
                if proc.returncode != 0:
                    sys.stderr.write(proc.stderr)
                    raise SystemExit(f"micro failed: {component} {params}")
                rec = json.loads(proc.stdout)
                rec["run_id"] = run_id
                rec["params"] = params
                if component == "row_map":
                    rec["text_width"] = params["width"]
                mf.write(json.dumps(rec, sort_keys=True) + "\n")
                mf.flush()
                print(f"[{run_id:3d}/{total}] micro {component} {params}",
                      flush=True)

    print(f"done in {time.time()-t0:.1f}s -> {args.out}")


if __name__ == "__main__":
    main()
