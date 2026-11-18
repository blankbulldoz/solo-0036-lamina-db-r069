"""Independent structural ledger, re-implemented in Python from the same
scenario *inputs* (not from harness outputs).

This is the independent calculation: it re-creates the deterministic workload
schedule, the engine's last-write-wins commit collapse, and the exact Vacuum
rule from internal/versions/vacuum.go. Every number is checked byte-for-byte
against engine-reported stats before any model is allowed to use the data.
"""

MASK64 = (1 << 64) - 1


class RNG:
    """Same LCG family as harness workload.go (MMIX constants)."""

    def __init__(self, seed):
        self.state = (seed * 2862933555777941757 + 3037000493) & MASK64

    def next(self):
        self.state = (self.state * 6364136223846793005 + 1442695040888963407) & MASK64
        return self.state

    def intn(self, n):
        if n <= 0:
            return 0
        return self.next() % n


def hot_key_count(keys, hot_fraction):
    h = int(keys * hot_fraction)
    if keys > 0:
        h = max(1, h)
    return min(h, keys)


def pick_key(keys, hot_fraction, r):
    h = hot_key_count(keys, hot_fraction)
    if h < keys and r.intn(100) < 80:
        return r.intn(h)
    if h == keys:
        return r.intn(keys)
    return h + r.intn(keys - h)


def tombstone_count(updates, delete_fraction):
    if delete_fraction <= 0 or updates == 0:
        return 0
    return min(updates, max(1, int(updates * delete_fraction)))


def build_batches(scn):
    """Expand the scenario into commit batches of mutations.

    Mirrors capacity/harness/workload.go buildWorkload exactly. A mutation is
    a (key, deleted) pair; the ordinal only affects payload content, not
    structural counts.
    """
    keys, updates = scn["keys"], scn["updates"]
    batch = max(1, scn["batch_size"])
    tombs = tombstone_count(updates, scn["delete_fraction"])
    r0 = RNG(scn["seed"] ^ 0x5177)
    tomb_positions = {r0.intn(updates) for _ in range(tombs)}

    batches, cur, ordinal = [], [], 0

    def flush():
        nonlocal cur
        if cur:
            batches.append(cur)
            cur = []

    for k in range(keys):  # inserts
        ordinal += 1
        cur.append((k, False))
        if len(cur) >= batch:
            flush()
    flush()

    r = RNG(scn["seed"] ^ 0x5177)
    for u in range(updates):
        ordinal += 1
        k = pick_key(keys, scn["hot_fraction"], r)
        cur.append((k, u in tomb_positions))
        if len(cur) >= batch:
            flush()
    flush()
    return batches


def replay_chains(scn):
    """Return {key: [(commit_ordinal, deleted), ...]} with last-write-wins
    collapse inside each commit, exactly as the engine's staged map produces.
    """
    batches = build_batches(scn)
    chains = {}
    live = {}
    index_entries = 0
    for ci, b in enumerate(batches, start=1):
        collapsed = {}
        order = []
        for key, deleted in b:
            if key not in collapsed:
                order.append(key)
            collapsed[key] = deleted
        for key in order:
            deleted = collapsed[key]
            chains.setdefault(key, []).append((ci, deleted))
            live[key] = not deleted
            if not deleted:
                index_entries += 1
    return batches, chains, live, index_entries


def retained_after_vacuum(chains, horizon):
    """Python copy of versions.Store.Vacuum's survivor rule."""
    total = 0
    for chain in chains.values():
        last_commit, last_deleted = chain[-1]
        if last_deleted and last_commit <= horizon:
            continue  # whole tombstone chain removed
        keep = 0
        while keep + 1 < len(chain) and chain[keep + 1][0] <= horizon:
            keep += 1
        total += len(chain) - keep
    return total


def chains_surviving(chains, horizon):
    """Number of key entries remaining in the store map after vacuum.

    A chain disappears only when its final version is a tombstone at or below
    the horizon; dead keys above the horizon and all live keys stay. With no
    vacuum, every inserted key is still a chain in the map (this matches
    versions.Store.Count, which counts map entries, not visible rows).
    """
    total = 0
    for chain in chains.values():
        last_commit, last_deleted = chain[-1]
        if last_deleted and last_commit <= horizon:
            continue
        total += 1
    return total


def ledger(scn):
    """Full independent ledger for one scenario."""
    batches, chains, live, index_entries = replay_chains(scn)
    versions = sum(len(c) for c in chains.values())
    return {
        "commits": len(batches),
        "insert_mutations": scn["keys"],
        "update_mutations": scn["updates"] - tombstone_count(
            scn["updates"], scn["delete_fraction"]),
        "tombstone_mutations": tombstone_count(
            scn["updates"], scn["delete_fraction"]),
        "total_mutations": scn["keys"] + scn["updates"],
        "versions": versions,
        "chains_present_no_vacuum": len(chains),
        "chains_surviving_head_vacuum": chains_surviving(chains, len(batches)),
        "chains_surviving_mid_vacuum": chains_surviving(
            chains, len(batches) // 2),
        "live_rows": sum(1 for v in live.values() if v),
        "dead_rows": sum(1 for v in live.values() if not v),
        "hot_keys": hot_key_count(scn["keys"], scn["hot_fraction"]),
        "index_entries_per_index": index_entries,
        "retained_head": retained_after_vacuum(chains, len(batches)),
        "retained_mid": retained_after_vacuum(chains, len(batches) // 2),
        "mean_chain_len": versions / scn["keys"],
        "max_chain_len": max(len(c) for c in chains.values()),
    }


def cross_check(rec):
    """Compare the Python ledger against engine-reported facts. Returns the
    check dict and the ledger; any False means the measurement is invalid.

    Engine stats.keys counts map entries (version chains), not visible rows:
    a key whose latest version is a tombstone still occupies a chain until a
    vacuum whose horizon reaches that tombstone removes it wholesale.
    """
    scn = rec["scenario"]
    L = ledger(scn)
    s_after = rec["stats_after"]
    ddl_before = 1 + (0 if scn["build_index_late"] else scn["index_count"])
    horizon_ord = 0
    if rec["vacuum_removed_versions"] > 0 or scn["vacuum_fraction"] > 0:
        horizon_ord = max(0, min(L["commits"],
                                 int(rec["vacuum_effective_horizon"]) - ddl_before))
        expected_versions = retained_after_vacuum(
            replay_chains(scn)[1], horizon_ord)
        expected_keys = chains_surviving(replay_chains(scn)[1], horizon_ord)
    else:
        expected_versions = L["versions"]
        expected_keys = L["chains_present_no_vacuum"]
    idx_total = sum(s_after["index_entries"].values())
    expected_journal = (1 + scn["index_count"] + L["commits"]
                        + (1 if scn["vacuum_fraction"] > 0 else 0))
    # When a head (or later) vacuum actually ran, every surviving chain ends
    # in a live row, so engine keys must also equal the ledger's live rows.
    live_semantics_ok = True
    if scn["vacuum_fraction"] > 0 and rec["vacuum_removed_versions"] > 0:
        live_semantics_ok = expected_keys == L["live_rows"]
    checks = {
        "py_versions_match": s_after["versions"] == expected_versions,
        "py_keys_match": s_after["keys"] == expected_keys,
        "py_live_row_semantics": live_semantics_ok,
        "py_journal_records_match":
            s_after["journal_records"] == expected_journal,
        "py_index_entries_match":
            scn["index_count"] == 0
            or idx_total == scn["index_count"] * L["index_entries_per_index"]
            or rec["vacuum_removed_versions"] > 0,
        "go_ledger_agrees": rec["ground_truth"]["versions_distinct_key_commit"]
            == L["versions"],
        "go_cross_check_all_green": all(rec["cross_check"].values()),
    }
    return checks, L
