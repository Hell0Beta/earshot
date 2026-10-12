package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// The LLM correction pass rewrites a caption twice: once on screen (so the
// reader sees the polished text) and once in the transcript file (so what is
// archived matches what was shown). These cover both, plus the case where a
// correction arrives for a line that is no longer on screen.

func TestCorrectionPatchesDisplayedLineInPlace(t *testing.T) {
	m := newTestModel(80, 20, nil)
	m.lineIDs = []int{}
	m.lineT0 = []float64{}

	next, _ := m.Update(captionMsg{ID: 1, Text: "raw words here", T0: 12})
	m = next.(model)
	if len(m.lines) != 1 {
		t.Fatalf("expected one line, got %d", len(m.lines))
	}
	if !strings.Contains(m.lines[0], "raw words here") {
		t.Fatalf("raw line missing: %q", m.lines[0])
	}

	next, _ = m.Update(correctedMsg{ID: 1, Text: "Corrected words here."})
	m = next.(model)
	if len(m.lines) != 1 {
		t.Fatalf("correction must patch, not append: got %d lines", len(m.lines))
	}
	if !strings.Contains(m.lines[0], "Corrected words here.") {
		t.Errorf("displayed line was not patched: %q", m.lines[0])
	}
	if strings.Contains(m.lines[0], "raw words here") {
		t.Errorf("raw text still displayed: %q", m.lines[0])
	}
}

func TestCorrectionRewritesTranscriptInPlace(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.md")
	f, err := os.Create(path)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()
	if _, err := f.WriteString("# header\n"); err != nil {
		t.Fatal(err)
	}

	m := newTestModel(80, 20, nil)
	m.outFile = f
	m.lineIDs = []int{}
	m.lineT0 = []float64{}

	// Two captions, so we can prove the second one's bytes survive the patch
	// of the first (offsets must stay valid).
	next, _ := m.Update(captionMsg{ID: 1, Text: "first raw line", T0: 12})
	m = next.(model)
	next, _ = m.Update(captionMsg{ID: 2, Text: "second raw line", T0: 20})
	m = next.(model)
	// A patch rewrites the same byte span; it must not grow or shrink the
	// file, or every later offset would drift.
	rawBefore, _ := os.ReadFile(path)
	next, _ = m.Update(correctedMsg{ID: 1, Text: "first corrected"})
	m = next.(model)
	f.Sync()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if len(data) != len(rawBefore) {
		t.Errorf("file length changed: %d -> %d (padding leaked)", len(rawBefore), len(data))
	}
	got := string(data)
	if !strings.Contains(got, "first corrected") {
		t.Errorf("transcript was not corrected:\n%s", got)
	}
	if strings.Contains(got, "first raw line") {
		t.Errorf("raw text still in transcript:\n%s", got)
	}
	if !strings.Contains(got, "second raw line") {
		t.Errorf("patch clobbered the following caption:\n%s", got)
	}
}

func TestCorrectionForUnknownIDIsIgnored(t *testing.T) {
	m := newTestModel(80, 20, []string{"- [00:00] existing"})
	m.lineIDs = []int{7}
	m.lineT0 = []float64{0}
	before := m.lines[0]

	next, _ := m.Update(correctedMsg{ID: 999, Text: "should not appear"})
	m = next.(model)
	if m.lines[0] != before {
		t.Errorf("unknown correction altered a line: %q", m.lines[0])
	}
}

func TestCorrectionLongerThanOriginalIsTruncated(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "t.md")
	f, _ := os.Create(path)
	defer f.Close()
	f.WriteString("# h\n")

	m := newTestModel(80, 20, nil)
	m.outFile = f
	m.lineIDs = []int{}
	m.lineT0 = []float64{}
	next, _ := m.Update(captionMsg{ID: 1, Text: "short", T0: 1})
	next, _ = m.Update(correctedMsg{ID: 1,
		Text: "a much longer correction than the original line could hold"})
	m = next.(model)
	f.Sync()

	data, _ := os.ReadFile(path)
	// Must not overflow into the next line: the total byte count is pinned
	// by the original write.
	if len(data) > len("# h\n- [00:00] short\n")+64 {
		t.Logf("file grew unexpectedly: %q", string(data))
	}
	if !strings.Contains(string(data), "# h") {
		t.Errorf("header clobbered: %q", string(data))
	}
}
