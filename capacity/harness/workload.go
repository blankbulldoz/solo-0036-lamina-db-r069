package main

import (
	"fmt"
	"lamina/internal/catalog"
	"lamina/internal/datum"
	"lamina/internal/engine"
	"strconv"
)

// Scenario is the full set of tunable workload inputs. Every field is rendered
// into the raw observation so results can be re-derived without the binary.
type Scenario struct {
	Name           string  `json:"scenario"`
	Keys           int     `json:"keys"`
	Updates        int     `json:"updates"`
	HotFraction    float64 `json:"hot_fraction"`
	BatchSize      int     `json:"batch_size"`
	DeleteFraction float64 `json:"delete_fraction"`
	TextWidth      int     `json:"text_width"`
	IndexCount     int     `json:"index_count"`
	BuildIndexLate bool    `json:"build_index_late"`
	PinSnapshot    bool    `json:"pin_snapshot"`
	VacuumFraction float64 `json:"vacuum_fraction"` // fraction of updates at which vacuum fires; 0 disables
	VacuumHorizon  string  `json:"vacuum_horizon"`  // "head" or "pinned"
	Seed           int64   `json:"seed"`
}

// GroundTruth is the workload's independently counted structural ledger.
// The Python analysis recomputes the same numbers from the scenario; harness
// and Python must agree (cross-check field in the output).
type GroundTruth struct {
	Commits                int     `json:"commits_planned"`
	InsertMutations        int     `json:"insert_mutations"`
	UpdateMutations        int     `json:"update_mutations"`
	TombstoneMutations     int     `json:"tombstone_mutations"`
	TotalMutations         int     `json:"total_mutations"`
	Versions               int     `json:"versions_distinct_key_commit"`
	LiveRowsExpected       int     `json:"live_rows_expected"`
	ChainsTouched          int     `json:"chains_touched"`
	HotKeys                int     `json:"hot_keys"`
	ExpectedIndexEntries   int     `json:"expected_index_entries_full_history"`
	TextBytesPerRow        int     `json:"text_bytes_per_row"`
	HotShareActual         float64 `json:"hot_share_actual"`
	RetainedAfterHeadVacuum  int    `json:"versions_retained_after_head_vacuum"`
	RetainedAfterMidVacuum   int    `json:"versions_retained_after_mid_vacuum"`
	ChainsSurvivingHead      int    `json:"chains_surviving_head_vacuum"`
	ChainsSurvivingMid       int    `json:"chains_surviving_mid_vacuum"`
}

// relationSpec builds the same 4-column relation used by every scenario.
func relationSpec() catalog.Relation {
	return catalog.Relation{
		Name:    "spectra",
		Primary: "sample",
		Columns: []catalog.Column{
			{Name: "sample", Type: "int"},
			{Name: "band", Type: "text"},
			{Name: "amplitude", Type: "int", Nullable: true},
			{Name: "valid", Type: "bool"},
		},
	}
}

// rng is a tiny deterministic LCG so the workload is reproducible without Go
// map iteration randomness affecting key selection.
type rng struct{ state uint64 }

func newRNG(seed int64) *rng { return &rng{state: uint64(seed)*2862933555777941757 + 3037000493} }

func (r *rng) next() uint64 {
	r.state = r.state*6364136223846793005 + 1442695040888963407
	return r.state
}

func (r *rng) intn(n int) int {
	if n <= 0 {
		return 0
	}
	return int(r.next() % uint64(n))
}

func bandText(width, seq, key int) string {
	// Fixed-width, deterministic per (version, key) so string storage cost is
	// a clean knob. Width includes the prefix.
	prefix := "b"
	body := strconv.Itoa(seq*1_000_003 + key)
	if len(body) >= width {
		return prefix + body[:width-1]
	}
	out := make([]byte, 0, width)
	out = append(out, prefix...)
	for len(out)+len(body) < width {
		out = append(out, 'x')
	}
	out = append(out, body...)
	return string(out)
}

func (s Scenario) hotKeyCount() int {
	h := int(float64(s.Keys) * s.HotFraction)
	if h < 1 && s.Keys > 0 {
		h = 1
	}
	if h > s.Keys {
		h = s.Keys
	}
	return h
}

// pickKey returns the next mutation target key. Hot keys receive the first
// hotKeyCount() indices; cold traffic is uniform over the remaining range.
func (s Scenario) pickKey(r *rng) int {
	h := s.hotKeyCount()
	if h < s.Keys && r.intn(100) < 80 {
		return r.intn(h)
	}
	if h == s.Keys {
		return r.intn(s.Keys)
	}
	return h + r.intn(s.Keys-h)
}

// replayedVersion is one entry of a key's replayed version chain.
type replayedVersion struct {
	commit  int
	deleted bool
	indexed bool // non-deleted versions feed index candidates
}

// countLedger computes expected structural counts by replaying the exact
// batch schedule the engine will execute. Same-key repeats inside one commit
// collapse to one version, so raw mutation counts overstate versions: this is
// the core reason a row-count coefficient cannot size the version store.
func (s Scenario) countLedger() GroundTruth {
	batches, _ := s.buildWorkload()
	gt := GroundTruth{
		Commits:         len(batches),
		TotalMutations:  s.Keys + s.Updates,
		InsertMutations: s.Keys,
		ChainsTouched:   s.Keys,
		HotKeys:         s.hotKeyCount(),
		TextBytesPerRow: s.TextWidth,
		HotShareActual:  s.HotFraction,
	}
	tombMut := 0
	updMut := 0
	for _, b := range batches[ceilDiv(s.Keys, s.BatchSize):] {
		for _, m := range b {
			if m.deleted {
				tombMut++
			} else {
				updMut++
			}
		}
	}
	gt.TombstoneMutations = tombMut
	gt.UpdateMutations = updMut

	// Replay per key: one version per (key, commit) pair; track sequence
	// ordinals (1-based global commit sequence, relation DDL is seq 1 so the
	// first user commit is seq 2 — but only relative order matters here).
	type ver = replayedVersion
	chains := map[int][]ver{}
	live := map[int]bool{}
	indexEntries := 0
	for ci, batch := range batches {
		// The engine's staged writes are keyed by (table, key): the last
		// mutation for a key in the commit wins and produces exactly one
		// version. Collapse the batch the same way.
		collapsed := map[int]mutation{}
		order := make([]int, 0, len(batch))
		for _, m := range batch {
			if _, seen := collapsed[m.key]; !seen {
				order = append(order, m.key)
			}
			collapsed[m.key] = m
		}
		for _, k := range order {
			m := collapsed[k]
			chains[m.key] = append(chains[m.key], replayedVersion{commit: ci + 1, deleted: m.deleted, indexed: !m.deleted})
			live[m.key] = !m.deleted
			if !m.deleted {
				indexEntries++
			}
		}
	}
	for _, chain := range chains {
		gt.Versions += len(chain)
	}
	for _, alive := range live {
		if alive {
			gt.LiveRowsExpected++
		}
	}
	gt.ExpectedIndexEntries = indexEntries

	// Expected survivors under the engine's Vacuum rule (versions/vacuum.go):
	// per chain keep every version with seq > horizon plus the last version at
	// or below it; chains ending in a tombstone at/below horizon vanish whole.
	// Horizons here are user-commit ordinals (1..C); run.go maps real engine
	// sequences, which include DDL records, onto this scale.
	gt.RetainedAfterHeadVacuum = s.retainedAt(len(batches))
	gt.RetainedAfterMidVacuum = s.retainedAt(len(batches) / 2)
	gt.ChainsSurvivingHead = s.chainsAt(len(batches))
	gt.ChainsSurvivingMid = s.chainsAt(len(batches) / 2)
	return gt
}

// chainsAt returns how many key entries remain in the store map at the given
// horizon: tombstone-terminated chains at or below the horizon are deleted;
// every other chain (including dead keys above the horizon) stays.
func (s Scenario) chainsAt(horizonOrdinal int) int {
	batches, _ := s.buildWorkload()
	chains := map[int][]replayedVersion{}
	for ci, batch := range batches {
		collapsed := map[int]mutation{}
		order := make([]int, 0, len(batch))
		for _, m := range batch {
			if _, seen := collapsed[m.key]; !seen {
				order = append(order, m.key)
			}
			collapsed[m.key] = m
		}
		for _, k := range order {
			m := collapsed[k]
			chains[m.key] = append(chains[m.key], replayedVersion{commit: ci + 1, deleted: m.deleted})
		}
	}
	surviving := 0
	for _, chain := range chains {
		last := chain[len(chain)-1]
		if last.deleted && last.commit <= horizonOrdinal {
			continue
		}
		surviving++
	}
	return surviving
}

// retainedAt replays the workload and applies the engine's vacuum rule at the
// given user-commit ordinal horizon.
func (s Scenario) retainedAt(horizonOrdinal int) int {
	batches, _ := s.buildWorkload()
	chains := map[int][]replayedVersion{}
	for ci, batch := range batches {
		collapsed := map[int]mutation{}
		order := make([]int, 0, len(batch))
		for _, m := range batch {
			if _, seen := collapsed[m.key]; !seen {
				order = append(order, m.key)
			}
			collapsed[m.key] = m
		}
		for _, k := range order {
			m := collapsed[k]
			chains[m.key] = append(chains[m.key], replayedVersion{commit: ci + 1, deleted: m.deleted})
		}
	}
	return retainedAfterVacuum(chains, horizonOrdinal)
}

// retainedAfterVacuum mirrors versions.Store.Vacuum over the replayed chains.
// horizon is expressed in commit ordinal relative to the first user commit.
func retainedAfterVacuum(chains map[int][]replayedVersion, horizon int) int {
	total := 0
	for _, chain := range chains {
		last := chain[len(chain)-1]
		if last.deleted && last.commit <= horizon {
			continue // whole tombstone chain removed
		}
		keep := 0
		for keep+1 < len(chain) && chain[keep+1].commit <= horizon {
			keep++
		}
		total += len(chain) - keep
	}
	return total
}

func ceilDiv(a, b int) int {
	if b <= 0 {
		return a
	}
	return (a + b - 1) / b
}

// mutation is one planned write with the version ordinal it belongs to.
type mutation struct {
	key     int
	deleted bool
	seq     int // 1-based version ordinal across the whole workload
}

// tombstoneCount is the number of update-phase mutations scheduled as deletes.
func (s Scenario) tombstoneCount() int {
	if s.DeleteFraction <= 0 || s.Updates == 0 {
		return 0
	}
	t := int(float64(s.Updates) * s.DeleteFraction)
	if t < 1 {
		t = 1
	}
	if t > s.Updates {
		t = s.Updates
	}
	return t
}

// buildWorkload expands the scenario into ordered insert + update batches.
// Each batch becomes one transaction/commit. Returns batches of mutations and
// the ordinal at which each batch starts.
func (s Scenario) buildWorkload() ([][]mutation, []int) {
	r := newRNG(s.Seed ^ 0x5177)
	tombs := s.tombstoneCount()
	tombSet := map[int]bool{}
	for i := 0; i < tombs; i++ {
		tombSet[r.intn(s.Updates)] = true
	}
	var batches [][]mutation
	var batchOrdinals []int
	ordinal := 0
	// Inserts: one commit per batch_size keys.
	var cur []mutation
	curStart := 1
	flush := func() {
		if len(cur) == 0 {
			return
		}
		batches = append(batches, cur)
		batchOrdinals = append(batchOrdinals, curStart)
		cur = nil
	}
	for k := 0; k < s.Keys; k++ {
		if len(cur) == 0 {
			curStart = ordinal + 1
		}
		ordinal++
		cur = append(cur, mutation{key: k, seq: ordinal})
		if len(cur) >= s.BatchSize {
			flush()
		}
	}
	flush()
	// Updates.
	r2 := newRNG(s.Seed ^ 0x5177)
	for u := 0; u < s.Updates; u++ {
		if len(cur) == 0 {
			curStart = ordinal + 1
		}
		ordinal++
		k := s.pickKey(r2)
		cur = append(cur, mutation{key: k, deleted: tombSet[u], seq: ordinal})
		if len(cur) >= s.BatchSize {
			flush()
		}
	}
	flush()
	return batches, batchOrdinals
}

// openDB creates a fresh engine instance in dir and installs the relation and
// requested secondary indexes (indexes can alternatively be built late).
func openDB(dir string, lateIndex bool, indexCount int) (*engine.DB, error) {
	db, err := engine.Open(dir)
	if err != nil {
		return nil, err
	}
	if _, err = db.CreateRelation(relationSpec()); err != nil {
		return nil, err
	}
	if !lateIndex {
		for i := 0; i < indexCount; i++ {
			if err := createAmpIndex(db, i); err != nil {
				return nil, err
			}
		}
	}
	return db, nil
}

func createAmpIndex(db *engine.DB, ordinal int) error {
	name := "by_amp"
	if ordinal > 0 {
		name = fmt.Sprintf("by_amp_%d", ordinal)
	}
	_, err := db.CreateIndex(catalog.Index{Name: name, Relation: "spectra", Column: "amplitude", Unique: false})
	return err
}

// applyMutation stages one mutation inside tx.
func applyMutation(db *engine.DB, txID string, m mutation, s Scenario) error {
	if m.deleted {
		return db.Delete(txID, "spectra", datum.Integer(int64(m.key)))
	}
	row := datum.Row{
		"sample":    datum.Integer(int64(m.key)),
		"band":      datum.Value{Kind: "text", Text: bandText(s.TextWidth, m.seq, m.key)},
		"amplitude": datum.Integer(int64(m.seq*7 + m.key)),
		"valid":     datum.Value{Kind: "bool", Bool: m.seq%2 == 0},
	}
	return db.Put(txID, "spectra", row, false)
}
