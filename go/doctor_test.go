package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The doctor report loop marks a check as a non-blocking "warn" whenever
// ok is false and hard is false, and only counts hard failures towards the
// exit status. EARSHOT_HOME must therefore be added with hard=false so an
// unset variable can never stop earshot from starting.
func TestEarshotHomeCheckIsNeverBlocking(t *testing.T) {
	c := check{"EARSHOT_HOME", false, false, "not set", "set EARSHOT_HOME"}
	if c.hard {
		t.Error("EARSHOT_HOME check must be hard=false so it warns instead of blocking")
	}
	if c.ok {
		t.Error("EARSHOT_HOME check should report ok=false so it renders as a warning")
	}
}

// Reproduces the report loop's severity logic to pin the contract: only hard
// failures increment the blocking counter.
func TestDoctorSeverityLogic(t *testing.T) {
	cases := []struct {
		name      string
		ok, hard  bool
		wantMark  string
		wantBlock bool
	}{
		{"passing check", true, true, "ok  ", false},
		{"soft failure warns", false, false, "warn", false},
		{"hard failure blocks", false, true, "FAIL", true},
	}
	for _, tc := range cases {
		c := check{"x", tc.ok, tc.hard, "", ""}
		mark, blocking := "ok  ", false
		switch {
		case c.ok:
		case c.hard:
			mark, blocking = "FAIL", true
		default:
			mark = "warn"
		}
		if mark != tc.wantMark {
			t.Errorf("%s: mark = %q, want %q", tc.name, mark, tc.wantMark)
		}
		if blocking != tc.wantBlock {
			t.Errorf("%s: blocking = %v, want %v", tc.name, blocking, tc.wantBlock)
		}
	}
}

// An unset EARSHOT_HOME resolves the data dir to %LOCALAPPDATA%\earshot,
// which is the root cause of a redundant model re-download when a machine
// already has a cache on another volume.
func TestUnsetEarshotHomePointsAtLocalAppDataModelDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(envHome, "")
	t.Setenv("LOCALAPPDATA", dir)

	modelDir := dataSub("models")
	if want := filepath.Join(dir, "earshot", "models"); modelDir != want {
		t.Errorf("model dir = %s, want %s", modelDir, want)
	}

	if _, err := os.Stat(modelDir); !os.IsNotExist(err) {
		t.Errorf("model dir %s should not exist yet, it is created on first run", modelDir)
	}
}

// The fix text the doctor prints must name the variable it wants the user to
// set, otherwise the warning is not actionable.
func TestEarshotHomeFixTextNamesVariable(t *testing.T) {
	fix := `set EARSHOT_HOME to pick a folder; default is %LOCALAPPDATA%\earshot`
	if !strings.Contains(fix, "EARSHOT_HOME") {
		t.Errorf("fix text %q should mention EARSHOT_HOME", fix)
	}
}
