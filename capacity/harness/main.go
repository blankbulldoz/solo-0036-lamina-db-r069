package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"runtime"
	"strings"
	"time"
)

func utcNow() string { return time.Now().UTC().Format(time.RFC3339Nano) }

// filesystemOf reports the mounted filesystem backing dir, read from
// /proc/mounts, so reports state whether fsync hits overlay, tmpfs or disk.
func filesystemOf(dir string) string {
	data, err := os.ReadFile("/proc/mounts")
	if err != nil {
		return "unknown"
	}
	longest := ""
	fsType := "unknown"
	for _, line := range strings.Split(string(data), "\n") {
		fields := strings.Fields(line)
		if len(fields) < 3 {
			continue
		}
		mountPoint, fstype := fields[1], fields[2]
		if strings.HasPrefix(dir, mountPoint) && len(mountPoint) > len(longest) {
			longest = mountPoint
			fsType = fstype
		}
	}
	return fsType
}

func emit(v any) {
	enc := json.NewEncoder(os.Stdout)
	if err := enc.Encode(v); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
}

func fatal(err error) {
	if err == nil {
		return
	}
	fmt.Fprintln(os.Stderr, "harness error:", err)
	os.Exit(1)
}

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: harness <engine|micro|baseline> [flags]")
		os.Exit(2)
	}
	switch os.Args[1] {
	case "engine":
		runEngine(os.Args[2:])
	case "micro":
		runMicro(os.Args[2:])
	case "baseline":
		runBaseline()
	default:
		fmt.Fprintln(os.Stderr, "unknown subcommand:", os.Args[1])
		os.Exit(2)
	}
}

func runEngine(args []string) {
	fs := flag.NewFlagSet("engine", flag.ExitOnError)
	dir := fs.String("dir", "", "empty data directory (required)")
	name := fs.String("scenario", "unnamed", "scenario label")
	keys := fs.Int("keys", 0, "live primary keys inserted")
	updates := fs.Int("updates", 0, "update/tombstone mutations after insert")
	hot := fs.Float64("hot-fraction", 1.0, "fraction of keys receiving 80%% of updates")
	batch := fs.Int("batch", 50, "mutations per transaction/commit")
	del := fs.Float64("delete-fraction", 0, "fraction of mutations that are tombstones")
	width := fs.Int("text-width", 24, "band text width in bytes")
	indexes := fs.Int("indexes", 0, "number of non-unique amplitude indexes")
	lateIndex := fs.Bool("late-index", false, "build indexes after the full write load")
	pin := fs.Bool("pin", false, "hold a read-only transaction open across the load")
	vacFrac := fs.Float64("vacuum", 0, "vacuum once after this fraction of load completes (0 off, 1 end)")
	vacHorizon := fs.String("vacuum-horizon", "head", "head or mid")
	seed := fs.Int64("seed", 20261118, "deterministic workload seed")
	profile := fs.String("profile", "", "write an inuse heap profile to this path")
	fatal(fs.Parse(args))
	if *dir == "" {
		fatal(fmt.Errorf("-dir is required"))
	}
	if *batch < 1 {
		*batch = 1
	}
	s := Scenario{
		Name: *name, Keys: *keys, Updates: *updates, HotFraction: *hot,
		BatchSize: *batch, DeleteFraction: *del, TextWidth: *width,
		IndexCount: *indexes, BuildIndexLate: *lateIndex, PinSnapshot: *pin,
		VacuumFraction: *vacFrac, VacuumHorizon: *vacHorizon, Seed: *seed,
	}
	if *profile != "" {
		fatal(profileEngineScenario(*dir, s, *profile))
		fmt.Fprintf(os.Stderr, "heap profile written to %s\n", *profile)
		return
	}
	obs, err := runScenario(*dir, s)
	obs.Host, _ = os.Hostname()
	obs.GoVersion = runtime.Version()
	fatal(err)
	emit(obs)
}

func runMicro(args []string) {
	fs := flag.NewFlagSet("micro", flag.ExitOnError)
	component := fs.String("component", "", "row_map|version_store|index_tree|journal_records|journal_empty_records")
	n := fs.Int("n", 0, "primary size parameter")
	k := fs.Int("k", 0, "keys (version_store only)")
	c := fs.Int("c", 0, "chain length (version_store only)")
	width := fs.Int("text-width", 24, "text width for row_map/version_store")
	fatal(fs.Parse(args))
	var obs MicroObservation
	switch *component {
	case "row_map":
		obs = microRowMap(*n, *width)
	case "version_store":
		obs = microVersionChains(*k, *c)
	case "index_tree":
		obs = microIndexTree(*n)
	case "journal_records":
		obs = microJournalRecords(*n)
	case "journal_empty_records":
		obs = microJournalEmptyRecords(*n)
	default:
		fatal(fmt.Errorf("unknown component %q", *component))
	}
	emit(obs)
}

func runBaseline() {
	forceGC()
	emit(struct {
		SchemaVersion int        `json:"schema_version"`
		Kind          string     `json:"kind"`
		Host          string     `json:"host"`
		GoVersion     string     `json:"go_version"`
		StartedUTC    string     `json:"started_utc"`
		Mem           MemSnapshot `json:"mem"`
		NumCPU        int        `json:"num_cpu"`
		PageSize      int        `json:"page_size"`
	}{
		SchemaVersion: 1, Kind: "baseline", GoVersion: runtime.Version(),
		StartedUTC: utcNow(), Mem: snapshotMem(),
		NumCPU: runtime.NumCPU(), PageSize: os.Getpagesize(),
	})
}
