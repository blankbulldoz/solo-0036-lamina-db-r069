package main

import (
	"lamina/internal/datum"
	"lamina/internal/index"
	"lamina/internal/journal"
	"lamina/internal/versions"
)

// MicroObservation captures one component-level cost point. The same n values
// are fit by the Python analysis into per-element marginal costs.
type MicroObservation struct {
	SchemaVersion int        `json:"schema_version"`
	Kind          string     `json:"kind"`
	StartedUTC    string     `json:"started_utc"`
	Component     string     `json:"component"`
	N             int        `json:"n"`
	MemEmpty      MemSnapshot `json:"mem_empty"`
	MemFull       MemSnapshot `json:"mem_full"`
	Delta         MemSnapshot `json:"delta"`
	Notes         string     `json:"notes"`
}

func newMicro(component string, n int, notes string) MicroObservation {
	forceGC()
	return MicroObservation{SchemaVersion: 1, Kind: "microbench", StartedUTC: utcNow(), Component: component, N: n, MemEmpty: snapshotMem(), Notes: notes}
}

func (m *MicroObservation) finish() MicroObservation {
	forceGC()
	m.MemFull = snapshotMem()
	m.Delta = memDelta(m.MemEmpty, m.MemFull)
	return *m
}

func memDelta(a, b MemSnapshot) MemSnapshot {
	return MemSnapshot{
		HeapAlloc:   subU(b.HeapAlloc, a.HeapAlloc),
		HeapInuse:   subU(b.HeapInuse, a.HeapInuse),
		HeapIdle:    subU(b.HeapIdle, a.HeapIdle),
		StackInuse:  subU(b.StackInuse, a.StackInuse),
		MSpanInuse:  subU(b.MSpanInuse, a.MSpanInuse),
		MCacheInuse: subU(b.MCacheInuse, a.MCacheInuse),
		GCPausedNS:  subU(b.GCPausedNS, a.GCPausedNS),
		NumGC:       b.NumGC - a.NumGC,
		Sys:         subU(b.Sys, a.Sys),
		RSS:         subU(b.RSS, a.RSS),
		NextGC:      subU(b.NextGC, a.NextGC),
	}
}

func subU(a, b uint64) uint64 {
	if a >= b {
		return a - b
	}
	return 0
}

// microRowMap measures the cost of n 4-column rows held in datum.Row maps.
// Mirrors how versions.Store keeps one map per retained version.
func microRowMap(n, textWidth int) MicroObservation {
	m := newMicro("row_map", n, "4-column rows; one text column of fixed width")
	rows := make([]datum.Row, n)
	for i := 0; i < n; i++ {
		rows[i] = datum.Row{
			"sample":    datum.Integer(int64(i)),
			"band":      datum.Value{Kind: "text", Text: bandText(textWidth, i, i)},
			"amplitude": datum.Integer(int64(7 * i)),
			"valid":     datum.Value{Kind: "bool", Bool: i%2 == 0},
		}
	}
	keep = rows
	return m.finish()
}

// microVersionChains measures versions.Store directly: k keys each with c
// versions, which is the exact structure Vacuum trims.
func microVersionChains(k, c int) MicroObservation {
	m := newMicro("version_store", k*c, "k keys x c versions per key")
	store := versions.New()
	store.Ensure("spectra")
	seq := uint64(1)
	for i := 0; i < k; i++ {
		key := datum.Integer(int64(i)).Key()
		for v := 0; v < c; v++ {
			store.Apply(seq, []versions.Mutation{{
				Relation: "spectra",
				Key:      key,
				Row: datum.Row{
					"sample":    datum.Integer(int64(i)),
					"band":      datum.Value{Kind: "text", Text: bandText(24, int(seq), i)},
					"amplitude": datum.Integer(int64(seq)),
					"valid":     datum.Value{Kind: "bool", Bool: true},
				},
			}})
			seq++
		}
	}
	keepStore = store
	return m.finish()
}

// microIndexTree measures the historical-candidate tree: one entry per
// non-deleted version, the structure whose size Vacuum rebuilds.
func microIndexTree(n int) MicroObservation {
	m := newMicro("index_tree", n, "n entries; unique value+key tokens")
	tree := index.New()
	for i := 0; i < n; i++ {
		tree.Add(datum.Integer(int64(i)), datum.Integer(int64(i % 97)).Key())
	}
	keepTree = tree
	return m.finish()
}

// microJournalEmptyRecords prices the fixed container cost of one WAL record
// (Record header + slice backing + frame bookkeeping) with zero mutations,
// so the per-row payload cost can be separated by difference.
func microJournalEmptyRecords(n int) MicroObservation {
	m := newMicro("journal_empty_records", n, "n zero-mutation records (container only)")
	recs := make([]journal.Record, n)
	for i := 0; i < n; i++ {
		recs[i] = journal.Record{
			Format: 1, Sequence: uint64(i + 1), Kind: "commit",
			Writes: []versions.Mutation{},
		}
	}
	keepRecs = recs
	return m.finish()
}

// microJournalRecords prices n WAL records each carrying one full row. The
// engine keeps one Normalize-built row per surviving (commit,key) pair, and
// same-batch overwrites are collected before commit, so the live WAL row
// count equals the retained version count rather than the mutation count;
// that collapse is modelled in the analysis, not here.
func microJournalRecords(n int) MicroObservation {
	m := newMicro("journal_records", n, "n records with one full-row mutation each")
	recs := make([]journal.Record, n)
	for i := 0; i < n; i++ {
		recs[i] = journal.Record{
			Format:   1,
			Sequence: uint64(i + 1),
			Kind:     "commit",
			Writes: []versions.Mutation{{
				Relation: "spectra",
				Key:      datum.Integer(int64(i)).Key(),
				Row: datum.Row{
					"sample":    datum.Integer(int64(i)),
					"band":      datum.Value{Kind: "text", Text: bandText(24, i, i)},
					"amplitude": datum.Integer(int64(i)),
					"valid":     datum.Value{Kind: "bool", Bool: true},
				},
			}},
		}
	}
	keepRecs = recs
	return m.finish()
}

// Package-level sinks prevent the compiler from eliminating test allocations.
var (
	keep      any
	keepStore *versions.Store
	keepTree  *index.Tree
	keepRecs  []journal.Record
)
