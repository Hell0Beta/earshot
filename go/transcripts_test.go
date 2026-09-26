package main

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// seedSessions creates n fake sessions (md + backend.log) in dir.
func seedSessions(t *testing.T, dir string, n int) {
	t.Helper()
	for i := 1; i <= n; i++ {
		stem := fmt.Sprintf("2026-09-%02d_%04d", (i%28)+1, i*7)
		write(t, filepath.Join(dir, stem+".md"), "# transcript\n")
		write(t, filepath.Join(dir, stem+backendLogSuffix), "log\n")
	}
}

func write(t *testing.T, path, body string) {
	t.Helper()
	if err := os.WriteFile(path, []byte(body), 0o644); err != nil {
		t.Fatalf("write %s: %v", path, err)
	}
}

func countSessions(t *testing.T, dir string) int {
	t.Helper()
	entries, err := os.ReadDir(dir)
	if err != nil {
		t.Fatalf("readdir: %v", err)
	}
	n := 0
	for _, e := range entries {
		if sessionStamp.MatchString(e.Name()) {
			n++
		}
	}
	return n
}

func TestPruneKeepsNewest(t *testing.T) {
	dir := t.TempDir()
	seedSessions(t, dir, 35)

	removed, err := pruneTranscripts(dir, 30)
	if err != nil {
		t.Fatalf("prune: %v", err)
	}
	if removed != 5 {
		t.Errorf("removed = %d, want 5", removed)
	}
	if got := countSessions(t, dir); got != 30 {
		t.Errorf("sessions left = %d, want 30", got)
	}
}

func TestPruneIsIdempotent(t *testing.T) {
	dir := t.TempDir()
	seedSessions(t, dir, 35)
	if _, err := pruneTranscripts(dir, 30); err != nil {
		t.Fatal(err)
	}
	removed, err := pruneTranscripts(dir, 30)
	if err != nil {
		t.Fatal(err)
	}
	if removed != 0 {
		t.Errorf("second prune removed %d, want 0", removed)
	}
}

func TestPruneDeletesPairedLogs(t *testing.T) {
	dir := t.TempDir()
	seedSessions(t, dir, 12)
	if _, err := pruneTranscripts(dir, 10); err != nil {
		t.Fatal(err)
	}
	entries, _ := os.ReadDir(dir)
	mds, logs := 0, 0
	for _, e := range entries {
		if sessionStamp.MatchString(e.Name()) {
			mds++
		}
		if strings.HasSuffix(e.Name(), backendLogSuffix) {
			logs++
		}
	}
	if mds != 10 || logs != 10 {
		t.Errorf("md=%d logs=%d, want 10/10 (logs must pair with transcripts)", mds, logs)
	}
}

func TestPruneLeavesForeignFilesAlone(t *testing.T) {
	dir := t.TempDir()
	seedSessions(t, dir, 40)
	write(t, filepath.Join(dir, ".gitkeep"), "")
	write(t, filepath.Join(dir, "notes.md"), "my notes")
	write(t, filepath.Join(dir, "important.md"), "keep me")
	write(t, filepath.Join(dir, "2026-13-99_9999.md"), "bogus stamp")

	if _, err := pruneTranscripts(dir, 30); err != nil {
		t.Fatal(err)
	}
	for _, f := range []string{".gitkeep", "notes.md", "important.md", "2026-13-99_9999.md"} {
		if _, err := os.Stat(filepath.Join(dir, f)); err != nil {
			t.Errorf("prune deleted foreign file %q", f)
		}
	}
	if got := countSessions(t, dir); got != 30 {
		t.Errorf("sessions = %d, want 30", got)
	}
}

func TestPruneEdgeCases(t *testing.T) {
	dir := t.TempDir()
	seedSessions(t, dir, 5)
	// keep larger than the corpus: nothing removed
	if n, err := pruneTranscripts(dir, 30); err != nil || n != 0 {
		t.Errorf("keep>count: removed=%d err=%v", n, err)
	}
	// keep 0 removes everything
	if n, err := pruneTranscripts(dir, 0); err != nil || n != 5 {
		t.Errorf("keep=0: removed=%d err=%v", n, err)
	}
	if got := countSessions(t, dir); got != 0 {
		t.Errorf("keep=0 left %d sessions", got)
	}
}

func TestPruneEmptyDir(t *testing.T) {
	if n, err := pruneTranscripts(t.TempDir(), 30); err != nil || n != 0 {
		t.Errorf("empty dir: removed=%d err=%v", n, err)
	}
}

func TestPruneOrphanLog(t *testing.T) {
	dir := t.TempDir()
	seedSessions(t, dir, 3)
	write(t, filepath.Join(dir, "2026-01-01_0001"+backendLogSuffix), "orphan\n")
	// The orphan is within the keep window, so it must survive.
	if _, err := pruneTranscripts(dir, 30); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "2026-01-01_0001"+backendLogSuffix)); err != nil {
		t.Errorf("orphan log inside keep window was deleted")
	}
	// With keep=0 the orphan goes too.
	if _, err := pruneTranscripts(dir, 0); err != nil {
		t.Fatal(err)
	}
	if _, err := os.Stat(filepath.Join(dir, "2026-01-01_0001"+backendLogSuffix)); err == nil {
		t.Errorf("orphan log survived keep=0")
	}
}

func TestPruneMissingDir(t *testing.T) {
	if _, err := pruneTranscripts(filepath.Join(t.TempDir(), "nope"), 30); err == nil {
		t.Errorf("expected an error for a missing directory")
	}
}
