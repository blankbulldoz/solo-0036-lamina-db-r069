#!/usr/bin/env sh
# End-to-end reproduction of the LaminaDB capacity decision pack.
# Requires: Go 1.25+ on PATH (or set GO_BIN), Python 3.11+, no Python packages.
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
GO_BIN=${GO_BIN:-go}
REPLICATES=${REPLICATES:-3}
EXTRA=${QUICK:+--quick}

cd "$ROOT"

echo "== 1/5 build engine and measurement harness =="
CGO_ENABLED=0 "$GO_BIN" build -o bin/laminad ./cmd/laminad
CGO_ENABLED=0 "$GO_BIN" vet ./capacity/harness
CGO_ENABLED=0 "$GO_BIN" build -o capacity/tmp/harness ./capacity/harness

echo "== 2/5 measurement matrix (replicates=$REPLICATES $EXTRA) =="
rm -rf capacity/data/raw capacity/tmp/full
python3 -I capacity/analysis/run_matrix.py \
  --harness capacity/tmp/harness \
  --out capacity/data/raw \
  --workdir capacity/tmp/full \
  --replicates "$REPLICATES" $EXTRA

echo "== 3/5 independent ledger cross-checks and competing-model fits =="
rm -rf capacity/data/tidy
python3 -I capacity/analysis/analyze.py \
  --raw capacity/data/raw --tidy capacity/data/tidy

echo "== 4/5 render decision report =="
python3 -I capacity/analysis/make_report.py \
  --findings capacity/data/tidy/findings.json \
  --out capacity/CAPACITY_REPORT.md

echo "== 4b/5 full-rate heap profiles (two-row-copy attribution) =="
mkdir -p capacity/data/raw/profiles
for b in 1 50 500; do
  rm -rf capacity/tmp/hp_$b
  "$ROOT/capacity/tmp/harness" engine \
    -dir capacity/tmp/hp_$b -scenario hp$b \
    -keys 1000 -updates 10000 -hot-fraction 0.1 -batch $b \
    -profile capacity/data/raw/profiles/heap_batch$b.pprof >/dev/null
done
{
  for b in 1 50 500; do
    echo "===== batch=$b (keys=1000 updates=10000, inuse_space, full-rate) ====="
    CGO_ENABLED=0 "$GO_BIN" tool pprof -top -inuse_space -nodecount 10 \
      capacity/data/raw/profiles/heap_batch$b.pprof 2>/dev/null | sed -n '6,15p'
    echo
  done
} > capacity/data/tidy/heap_profile_attribution.txt

echo "== 5/5 calculator smoke test =="
python3 -I capacity/analysis/calculator.py \
  --findings capacity/data/tidy/findings.json \
  --keys 1000 --versions 12000 --journal-records 260 --budget-mb 128

echo "done. artifacts in capacity/data/{raw,tidy} and capacity/CAPACITY_REPORT.md"
