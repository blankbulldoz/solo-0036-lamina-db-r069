package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"strconv"
	"strings"
	"syscall"
	"time"
)

// MemSnapshot is a process-level memory observation. Heap fields come from
// runtime.MemStats (Go runtime accounting), RSS from /proc/self/statm.
type MemSnapshot struct {
	HeapAlloc    uint64 `json:"heap_alloc_bytes"`
	HeapInuse    uint64 `json:"heap_inuse_bytes"`
	HeapIdle     uint64 `json:"heap_idle_bytes"`
	StackInuse   uint64 `json:"stack_inuse_bytes"`
	MSpanInuse   uint64 `json:"mspan_inuse_bytes"`
	MCacheInuse  uint64 `json:"mcache_inuse_bytes"`
	GCPausedNS   uint64 `json:"gc_pause_total_ns"`
	NumGC        uint32 `json:"num_gc"`
	Sys          uint64 `json:"runtime_sys_bytes"`
	RSS          uint64 `json:"rss_bytes"`
	NextGC       uint64 `json:"next_gc_bytes"`
}

func readRSS() uint64 {
	data, err := os.ReadFile("/proc/self/statm")
	if err != nil {
		return 0
	}
	fields := strings.Fields(string(data))
	if len(fields) < 1 {
		return 0
	}
	pages, err := strconv.ParseUint(fields[0], 10, 64)
	if err != nil {
		return 0
	}
	return pages * uint64(os.Getpagesize())
}

// forceGC runs two GC cycles so that freed garbage is not counted as live heap.
func forceGC() {
	runtime.GC()
	runtime.GC()
}

func snapshotMem() MemSnapshot {
	var m runtime.MemStats
	runtime.ReadMemStats(&m)
	return MemSnapshot{
		HeapAlloc:   m.HeapAlloc,
		HeapInuse:   m.HeapInuse,
		HeapIdle:    m.HeapIdle,
		StackInuse:  m.StackInuse,
		MSpanInuse:  m.MSpanInuse,
		MCacheInuse: m.MCacheInuse,
		GCPausedNS:  m.PauseTotalNs,
		NumGC:       m.NumGC,
		Sys:         m.Sys,
		RSS:         readRSS(),
		NextGC:      m.NextGC,
	}
}

type timer struct {
	start time.Time
}

func startTimer() timer { return timer{time.Now()} }

func (t timer) elapsedMS() float64 {
	return float64(time.Since(t.start).Microseconds()) / 1000.0
}

// fileSize is the apparent size and allocated blocks of one file.
type fileSize struct {
	Path       string `json:"path"`
	Bytes      int64  `json:"bytes"`
	BlockBytes int64  `json:"block_bytes"`
	Exists     bool   `json:"exists"`
}

// writeHeapProfile writes an inuse-space heap profile for forensic byte
// attribution by package/call stack (runtime/pprof, stdlib only).
func writeHeapProfile(path string) error {
	if path == "" {
		return nil
	}
	forceGC()
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	return pprof.WriteHeapProfile(f)
}

func init() {
	// Full-rate heap accounting: the measured heaps are only a few MB, where
	// the default 512KB sampling would miss most allocations.
	runtime.MemProfileRate = 1
}


func statFile(path string) fileSize {
	fs := fileSize{Path: path}
	var st syscall.Stat_t
	if err := syscall.Stat(path, &st); err != nil {
		return fs
	}
	fs.Exists = true
	fs.Bytes = st.Size
	fs.BlockBytes = st.Blocks * 512
	return fs
}
