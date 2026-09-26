package main

import (
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
)

func newTestModel(width, height int, lines []string) model {
	m := model{
		lines:      lines,
		width:      width,
		height:     height,
		viewport:   viewport.New(width, height),
		ready:      true,
		follow:     true,
		started:    time.Now(), // else the footer renders a ~20,000-hour elapsed time
		blockChars: 120,
		status:     "live",
		build:      "2026-09-26 15:01",
		outPath:    `..\transcripts\2026-09-26_1501.md`,
	}
	// Mirror what WindowSizeMsg does on startup, so tests start from the same
	// clamped state the real program does.
	m.clampViewport()
	return m
}

func TestWrapHonoursWidth(t *testing.T) {
	m := newTestModel(30, 20, nil)
	long := strings.Repeat("word ", 20)
	out := stripANSI(m.wrap(long))
	for _, ln := range strings.Split(out, "\n") {
		if n := len([]rune(strings.TrimRight(ln, " "))); n > 30 {
			t.Errorf("line %d chars = %d, exceeds 30", 0, n)
			break
		}
	}
	if !strings.Contains(out, "\n") {
		t.Errorf("expected wrapping to multiple lines, got one line")
	}
}

func TestGapBetweenNewestAndHistory(t *testing.T) {
	m := newTestModel(80, 20, []string{"- [00:12] newest", "- [00:00] older one", "- [00:00] older two"})
	body := m.renderContent()
	lines := strings.Split(strings.TrimLeft(body, "\n"), "\n")

	// exactly two blank lines at the new/old boundary
	blank := -1
	for i, l := range lines {
		if strings.TrimSpace(l) == "" {
			blank = i
			break
		}
	}
	if blank < 0 {
		t.Fatalf("no gap found in:\n%s", body)
	}
	if !strings.Contains(lines[blank-1], "newest") {
		t.Errorf("newest block not directly above the gap: %q", lines[blank-1])
	}
	if strings.TrimSpace(lines[blank+1]) != "" {
		t.Errorf("gap should be two blank lines, got one then %q", lines[blank+1])
	}
	if !strings.Contains(lines[blank+2], "older one") {
		t.Errorf("history not below the gap: %q", lines[blank+2])
	}
	// old sentences are packed tight: no blank line between them
	joined := strings.Join(lines[blank+2:], "\n")
	if strings.Contains(joined, "\n\n") {
		t.Errorf("old sentences should be packed tight, found a gap:\n%s", joined)
	}
}

// TestHistoryDimDecision covers the dimming *decision*. lipgloss strips all
// styling when there's no TTY (colour profile Ascii), so asserting on emitted
// escape codes would be meaningless here; the real terminal picks up the style
// from the same branch this test exercises.
func TestHistoryDimDecision(t *testing.T) {
	m := newTestModel(80, 20, []string{"- [00:00] committed"})
	m.draft = "live words"
	if !m.historyDimmed() {
		t.Errorf("history should be dimmed while live")
	}
	m.paused = true
	if m.historyDimmed() {
		t.Errorf("history should not be dimmed while paused")
	}
	// Structure must survive the style change either way.
	m.paused = false
	body := stripANSI(m.renderContent())
	if !strings.Contains(body, "committed") || !strings.Contains(body, "live words") {
		t.Errorf("render lost content:\n%s", body)
	}
}

func TestFooterGauge(t *testing.T) {
	m := newTestModel(80, 20, []string{"- [00:00] x"})
	m.blockChars = 120
	m.outPath = "t.md"
	m.build = "b"
	m.draft = strings.Repeat("x", 87)
	view := m.View()
	if !strings.Contains(stripANSI(view), "buf 87/120") {
		t.Errorf("footer gauge missing or wrong:\n%s", view)
	}
	m.draft = ""
	if strings.Contains(stripANSI(m.View()), "buf ") {
		t.Errorf("gauge should be hidden with no live line")
	}
}

func TestNoDividerWithoutHistory(t *testing.T) {
	m := newTestModel(80, 20, []string{"- [00:12] only block"})
	body := m.renderContent()
	if strings.Contains(strings.TrimLeft(body, "\n"), "\n\n") {
		t.Errorf("divider shown with no history:\n%s", body)
	}
}

func TestTopPinned(t *testing.T) {
	m := newTestModel(60, 12, []string{"- [00:00] one line only"})
	body := m.renderContent()
	if strings.HasPrefix(body, "\n") {
		t.Errorf("content must start at the top row, not be bottom-padded:\n%q", body)
	}
	if !strings.HasPrefix(body, "- [00:00]") {
		t.Errorf("first rendered row should be the newest block, got:\n%q", body)
	}
	if got := len(strings.Split(strings.TrimRight(body, "\n"), "\n")); got != 1 {
		t.Errorf("single block should render one row, got %d", got)
	}
}

// pressSpace delivers a real space keypress through Update, so tests exercise
// the same path a user does (rather than calling the resize helper directly and
// hiding a missing call inside the handler).
func pressSpace(t *testing.T, m model) model {
	t.Helper()
	next, _ := m.Update(tea.KeyMsg{Type: tea.KeySpace})
	got, ok := next.(model)
	if !ok {
		t.Fatalf("Update returned %T, want model", next)
	}
	return got
}

func TestPauseClampsViewport(t *testing.T) {
	m := newTestModel(80, 30, []string{"- [00:00] x"})
	m.blockChars = 120
	m.status = "live"
	m.build = "b"
	m.outPath = "t.md"

	unpausedH := m.viewport.Height
	paused := pressSpace(t, m)
	if !paused.paused {
		t.Fatalf("space did not toggle pause")
	}
	// Pausing adds the 5 banner rows plus one extra blank separator, so the
	// header grows from 2 rows to 2+len(splashArt)+1.
	if delta, want := unpausedH-paused.viewport.Height, len(splashArt)+1; delta != want {
		t.Errorf("pressing space shrank the viewport by %d rows, want %d (banner + separator)",
			delta, want)
	}

	resumed := pressSpace(t, paused)
	if resumed.paused {
		t.Fatalf("space did not toggle pause back off")
	}
	if resumed.viewport.Height != unpausedH {
		t.Errorf("un-pausing did not restore the viewport: %d, want %d",
			resumed.viewport.Height, unpausedH)
	}
}

func TestFooterRows(t *testing.T) {
	m := newTestModel(400, 30, []string{"- [00:00] x"})
	m.status = "live"
	m.build = "b"
	m.outPath = "t.md"
	if got := m.footerRows(); got != 1 {
		t.Errorf("wide terminal: footerRows = %d, want 1", got)
	}
	// The footer is ~150 chars, so it must grow once the terminal narrows.
	wide := m.footerRows()
	m.width = 60
	if narrow := m.footerRows(); narrow <= wide {
		t.Errorf("narrow terminal: footerRows = %d, want > %d", narrow, wide)
	}
}

// TestNoOverflowInvariant is the guard whose absence let the header get cropped:
// header + caption area + footer + separator must fit the terminal, or the
// terminal scrolls and pushes the title and mini banner off the top.
//
// It presses space for real (via Update) rather than calling clampViewport
// itself — a test that clamps on its own can never catch a handler that forgets
// to clamp.
func TestNoOverflowInvariant(t *testing.T) {
	for _, width := range []int{60, 80, 100, 120, 160, 200} {
		for _, height := range []int{20, 30, 50} {
			for _, wantPaused := range []bool{false, true} {
				m := newTestModel(width, height, []string{"- [00:00] a caption line"})
				m.blockChars = 120
				m.status = "loading model (small.en on cuda)"
				m.build = "2026-09-26 15:01"
				m.outPath = `..\transcripts\2026-09-26_1501.md`
				m.draft = "a live line that is filling toward the block target"

				if wantPaused {
					m = pressSpace(t, m)
				}
				// A draft arriving grows the footer (the "buf N/120" gauge),
				// which is what silently unbalances the frame if nothing
				// re-clamps. Drive it through Update, as production does.
				next, _ := m.Update(draftMsg{Text: "a live line that is filling toward the target", Chunk: 7})
				m = next.(model)

				total := m.headerHeight() + m.viewport.Height + m.footerRows() + 1
				if total > height {
					t.Errorf("w=%d h=%d paused=%v: frame needs %d rows, terminal has %d (overflow %d)",
						width, height, wantPaused, total, height, total-height)
				}
			}
		}
	}
}

// TestLayoutVisual prints the caption area so the spacing can be eyeballed
// without a live audio session.
func TestLayoutVisual(t *testing.T) {
	m := newTestModel(64, 20, []string{
		"- [00:39] this is the newest committed sentence that just wrapped around",
		"- [00:33] an older sentence sits directly above the next one",
		"- [00:27] another older sentence with no gap above it",
		"- [00:20] and one more below that",
	})
	m.draft = "and this is the live line still filling toward 120"
	fmt.Printf("--- caption area (| marks the column edge) ---\n|%s|\n", m.renderContent())
}

func TestLiveLineIsTopRow(t *testing.T) {
	m := newTestModel(60, 12, []string{"- [00:00] committed"})
	m.draft = "growing"
	body := stripANSI(m.renderContent())
	if !strings.HasPrefix(body, "growing") {
		t.Errorf("live line should be the top row, got:\n%q", body)
	}
}

func TestNoPaddingWhenOverflowing(t *testing.T) {
	var many []string
	for i := 0; i < 40; i++ {
		many = append(many, "- [00:00] line")
	}
	m := newTestModel(60, 10, many)
	body := m.renderContent()
	if strings.HasPrefix(body, "\n") {
		t.Errorf("overflowing content must not be padded (viewport scrolls)")
	}
}

func TestHistoryCap(t *testing.T) {
	var many []string
	for i := 0; i < maxRenderedHistory+15; i++ {
		many = append(many, "- [00:00] line")
	}
	m := newTestModel(200, 40, many)
	body := stripANSI(m.renderContent())
	// cap applies to history; the newest block is always rendered too
	if got := strings.Count(body, "line"); got != maxRenderedHistory+1 {
		t.Errorf("rendered %d blocks, want %d (newest + cap %d)",
			got, maxRenderedHistory+1, maxRenderedHistory)
	}
}

func TestLiveLineIncluded(t *testing.T) {
	m := newTestModel(60, 20, []string{"- [00:00] committed"})
	m.draft = "growing sentence"
	body := stripANSI(m.renderContent())
	if !strings.Contains(body, "growing sentence") {
		t.Errorf("live line missing:\n%s", body)
	}
	if !strings.Contains(body, "committed") {
		t.Errorf("committed block missing")
	}
}
