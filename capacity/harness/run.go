package main

import (
	"lamina/internal/engine"
	"path/filepath"
)

// PhaseMem records a memory observation at a named workload phase.
type PhaseMem struct {
	Phase string     `json:"phase"`
	Mem   MemSnapshot `json:"mem"`
}

// RunObservation is one complete fresh-process scenario execution.
type RunObservation struct {
	SchemaVersion int      `json:"schema_version"`
	Kind          string   `json:"kind"` // "engine_run" | "microbench" | "baseline"
	Host          string   `json:"host"`
	GoVersion     string   `json:"go_version"`
	StartedUTC    string   `json:"started_utc"`
	DataDirFS     string   `json:"data_dir_filesystem"`
	Scenario      Scenario `json:"scenario"`
	GroundTruth   GroundTruth `json:"ground_truth"`
	// Engine-reported facts.
	StatsBefore     engine.Stats `json:"stats_before"`
	StatsAfter      engine.Stats `json:"stats_after"`
	StatsPostVacuum engine.Stats `json:"stats_post_vacuum"`
	// Vacuum facts (zero value when no vacuum).
	VacuumRequested uint64 `json:"vacuum_requested_horizon"`
	VacuumEffective uint64 `json:"vacuum_effective_horizon"`
	VacuumPinned    bool   `json:"vacuum_pinned"`
	VacuumRemoved   int    `json:"vacuum_removed_versions"`
	VacuumRetained  int    `json:"vacuum_retained_versions"`
	VacuumMS        float64 `json:"vacuum_elapsed_ms"`
	// Timing.
	LoadMS       float64 `json:"load_elapsed_ms"`
	CommitMS     float64 `json:"commit_total_ms"`
	CommitCount  int     `json:"commit_count_observed"`
	// Memory.
	BaselineMem MemSnapshot `json:"mem_baseline"`
	Phases      []PhaseMem  `json:"phases"`
	// Durable footprint.
	JournalFile    fileSize `json:"journal_file"`
	CheckpointFile fileSize `json:"checkpoint_file"`
	BackupBytes    int64    `json:"backup_image_bytes_post_vacuum"`
	// Cross-check flags (Python independently recomputes expected counts).
	CrossCheck struct {
		VersionsMatch      bool `json:"versions_match_ledger"`
		KeysMatch          bool `json:"keys_match_chains_expected"`
		IndexEntriesMatch  bool `json:"index_entries_match_ledger"`
		JournalRecordsMatch bool `json:"journal_records_match_commits_plus_ddl_plus_vacuum"`
	} `json:"cross_check"`
}

func runScenario(dir string, s Scenario) (RunObservation, error) {
	obs, db, err := runScenarioNoClose(dir, s)
	if err != nil {
		return obs, err
	}
	db.Close()
	return obs, nil
}

func runScenarioNoClose(dir string, s Scenario) (RunObservation, *engine.DB, error) {
	obs := RunObservation{SchemaVersion: 1, Kind: "engine_run", Scenario: s, GroundTruth: s.countLedger()}
	obs.DataDirFS = filesystemOf(dir)
	obs.StartedUTC = utcNow()
	obs.BaselineMem = snapshotMem()

	db, err := openDB(dir, s.BuildIndexLate, s.IndexCount)
	if err != nil {
		return obs, nil, err
	}

	// Optional long reader opened before the write load: it pins the horizon.
	var pinTxID string
	if s.PinSnapshot {
		tx, err := db.Begin(true)
		if err != nil {
			return obs, nil, err
		}
		pinTxID = tx.ID
	}

	batches, _ := s.buildWorkload()
	loadTimer := startTimer()
	commitAccum := 0.0
	commitCount := 0
	phases := []PhaseMem{}
	midpoint := len(batches) / 2
	for bi, batch := range batches {
		tx, err := db.Begin(false)
		if err != nil {
			return obs, nil, err
		}
		for _, m := range batch {
			if err := applyMutation(db, tx.ID, m, s); err != nil {
				return obs, nil, err
			}
		}
		ct := startTimer()
		if _, err := db.Commit(tx.ID); err != nil {
			return obs, nil, err
		}
		commitAccum += ct.elapsedMS()
		commitCount++
		if bi == midpoint {
			forceGC()
			phases = append(phases, PhaseMem{Phase: "mid_load", Mem: snapshotMem()})
		}
	}
	obs.LoadMS = loadTimer.elapsedMS()
	obs.CommitMS = commitAccum
	obs.CommitCount = commitCount

	// Late index build after all writes, when requested.
	if s.BuildIndexLate {
		for i := 0; i < s.IndexCount; i++ {
			if err := createAmpIndex(db, i); err != nil {
				return obs, nil, err
			}
		}
	}

	forceGC()
	obs.Phases = append(phases, PhaseMem{Phase: "post_load", Mem: snapshotMem()})

	statsBefore, _ := db.Stats()
	obs.StatsBefore = statsBefore

	// Vacuum. A pinned reader forces the effective horizon down to its snapshot.
	if s.VacuumFraction > 0 {
		requested := statsBefore.Sequence
		if s.VacuumHorizon == "mid" {
			requested = statsBefore.Sequence / 2
		}
		obs.VacuumRequested = requested
		vt := startTimer()
		res, err := db.Vacuum(requested)
		if err != nil {
			return obs, nil, err
		}
		obs.VacuumMS = vt.elapsedMS()
		obs.VacuumEffective = res.Horizon
		obs.VacuumPinned = res.Pinned
		obs.VacuumRemoved = res.Removed
		obs.VacuumRetained = res.Retained
		forceGC()
		obs.Phases = append(obs.Phases, PhaseMem{Phase: "post_vacuum", Mem: snapshotMem()})
	}

	// Release the long reader after the vacuum decision so its pin effect is
	// measured; a second vacuum then shows what reclamation was deferred.
	if pinTxID != "" {
		if err := db.Rollback(pinTxID); err != nil {
			return obs, nil, err
		}
	}

	statsAfter, _ := db.Stats()
	obs.StatsAfter = statsAfter
	forceGC()
	obs.Phases = append(obs.Phases, PhaseMem{Phase: "final", Mem: snapshotMem()})
	obs.StatsPostVacuum = statsAfter

	// Durable artifacts: journal grows monotonically (never truncated); an
	// image captures exactly the retained store plus catalog.
	obs.JournalFile = statFile(filepath.Join(dir, "journal.lmw"))
	if b, err := db.Backup(); err == nil {
		obs.BackupBytes = int64(len(b))
	} else {
		obs.BackupBytes = -1
	}
	if _, err := db.Checkpoint(); err == nil {
		obs.CheckpointFile = statFile(filepath.Join(dir, "checkpoint.lmi"))
	}

	// Cross-check engine facts against the independently counted ledger.
	ledger := s.countLedger()
	expectedVersions := ledger.Versions
	expectedKeys := s.Keys // every inserted key is a chain until vacuum removes it
	if obs.VacuumRemoved > 0 || s.VacuumFraction > 0 {
		// Map the engine's effective horizon (global sequence, including DDL)
		// onto the user-commit ordinal scale used by the ledger.
		ddlBeforeLoad := 1 // relation DDL always precedes user commits
		if !s.BuildIndexLate {
			ddlBeforeLoad += s.IndexCount
		}
		horizonOrdinal := int(obs.VacuumEffective) - ddlBeforeLoad
		if horizonOrdinal < 0 {
			horizonOrdinal = 0
		}
		if horizonOrdinal > ledger.Commits {
			horizonOrdinal = ledger.Commits
		}
		expectedVersions = s.retainedAt(horizonOrdinal)
		// stats.keys counts store-map chains; vacuum drops only tombstone
		// chains whose final version is at or below the horizon.
		expectedKeys = s.chainsAt(horizonOrdinal)
	}
	obs.CrossCheck.VersionsMatch = statsAfter.Versions == expectedVersions
	obs.CrossCheck.KeysMatch = statsAfter.Keys == expectedKeys
	idxTotal := 0
	for _, n := range statsAfter.IndexEntries {
		idxTotal += n
	}
	// Index holds one candidate per non-deleted retained version per index.
	expectedIndex := 0
	if s.IndexCount > 0 && obs.VacuumRemoved == 0 {
		expectedIndex = s.IndexCount * ledger.ExpectedIndexEntries
	}
	obs.CrossCheck.IndexEntriesMatch = s.IndexCount == 0 ||
		idxTotal == expectedIndex ||
		(obs.VacuumRemoved > 0 && idxTotal <= ledger.ExpectedIndexEntries)
	// Journal holds every durable record: relation DDL, one per index DDL, one
	// per user commit and the vacuum record. Vacuum never truncates it.
	expectedJournalRecords := 1 + s.IndexCount + commitCount + boolToInt(s.VacuumFraction > 0)
	obs.CrossCheck.JournalRecordsMatch = statsAfter.JournalRecords == expectedJournalRecords

	// Keep the engine open so the caller can optionally capture a heap
	// profile of the live object graph; Close happens in runEngine.
	return obs, db, nil
}

// profileEngineScenario runs a scenario and writes a heap profile of the
// live object graph at the final phase.
func profileEngineScenario(dir string, s Scenario, profilePath string) error {
	_, db, err := runScenarioNoClose(dir, s)
	if err != nil {
		return err
	}
	defer db.Close()
	return writeHeapProfile(profilePath)
}

func boolToInt(b bool) int {
	if b {
		return 1
	}
	return 0
}
