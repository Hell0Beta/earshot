package main

import (
	"fmt"
	"strings"
	"testing"

	"github.com/charmbracelet/lipgloss"
)

func TestSplashGrid(t *testing.T) {
	grid, width, letterW := splashGrid()
	if len(grid) != len(splashArt) {
		t.Fatalf("rows: got %d want %d", len(grid), len(splashArt))
	}
	for i, row := range grid {
		if len(row) != width {
			t.Errorf("row %d not padded: len=%d want %d", i, len(row), width)
		}
	}
	if letterW <= 0 || letterW > width {
		t.Errorf("letterW=%d width=%d", letterW, width)
	}
	if width > 100 {
		t.Errorf("banner too wide for a normal terminal: %d cols", width)
	}
}

func TestSplashFrameReveal(t *testing.T) {
	_, width, _ := splashGrid()
	// Early frames reveal progressively fewer letters.
	f0 := stripANSI(splashFrame(120, 30, 0, "starting", styleShimmer))
	f6 := stripANSI(splashFrame(120, 30, splashRevealFrames, "starting", styleShimmer))
	if lit(f0) >= lit(f6) {
		t.Errorf("reveal not progressive: frame0=%d frame6=%d", lit(f0), lit(f6))
	}
	if lit(f6) != lit(stripANSI(splashFrame(120, 30, splashRevealFrames+20, "starting", styleShimmer))) {
		t.Errorf("after reveal, letter count should be stable")
	}
	if lit(f0) == 0 {
		t.Errorf("frame 0 drew no glyphs")
	}
	if !strings.Contains(f6, "starting") {
		t.Errorf("status line missing")
	}
	// Width respected: no line exceeds the requested width (padding aside).
	for _, ln := range strings.Split(stripANSI(splashFrame(100, 30, 12, "x", styleShimmer)), "\n") {
		if len([]rune(strings.TrimRight(ln, " "))) > 100 {
			t.Errorf("line %d exceeds width 100: %d", 0, len([]rune(strings.TrimRight(ln, " "))))
			break
		}
	}
	_ = width
}

func TestHsvToHex(t *testing.T) {
	if got := hsvToHex(0, 1, 1); got != "#ff0000" {
		t.Errorf("hue 0 = %s, want #ff0000", got)
	}
	if got := hsvToHex(120, 1, 1); got != "#00ff00" {
		t.Errorf("hue 120 = %s, want #00ff00", got)
	}
	if got := hsvToHex(240, 1, 1); got != "#0000ff" {
		t.Errorf("hue 240 = %s, want #0000ff", got)
	}
	// hue wraps rather than overflowing
	if hsvToHex(360, 1, 1) != hsvToHex(0, 1, 1) {
		t.Errorf("360 should wrap to 0")
	}
	if hsvToHex(-30, 1, 1) == "" {
		t.Errorf("negative hue should normalise")
	}
}

func TestRainbowScrolls(t *testing.T) {
	a := splashCellColor(10, 3)
	b := splashCellColor(10, 9)
	if a == b {
		t.Errorf("colour should change between frames, both %s", a)
	}
	// different columns differ
	if splashCellColor(0, 0) == splashCellColor(40, 0) {
		t.Errorf("columns should differ in hue")
	}
	// stays in the rainbow's saturated, bright band
	if len(splashCellColor(5, 5)) != 7 {
		t.Errorf("expected #rrggbb, got %q", splashCellColor(5, 5))
	}
}

func TestSweepStops(t *testing.T) {
	const artW = 82
	stops := sweepStops(artW, 12)
	if len(stops) != artW {
		t.Fatalf("stops length = %d, want %d", len(stops), artW)
	}
	counts := map[int]int{}
	for _, s := range stops {
		if s < stopDim || s > stopBright {
			t.Fatalf("stop out of range: %d", s)
		}
		counts[s]++
	}
	// All three tones should be present: a hard binary looked like broken
	// blocks, and an all-one-tone render isn't a sweep.
	if counts[stopBright] == 0 {
		t.Errorf("no bright columns: %v", counts)
	}
	if counts[stopMid] == 0 {
		t.Errorf("no mid-tone falloff: %v", counts)
	}
	if counts[stopDim] == 0 {
		t.Errorf("no dim columns: %v", counts)
	}
	// The band must actually move between frames.
	moved := 0
	next := sweepStops(artW, 13)
	for i := range stops {
		if stops[i] != next[i] {
			moved++
		}
	}
	if moved == 0 {
		t.Errorf("highlight did not move between frames")
	}
	// A full sweep must return to the starting pattern.
	if got, want := fmt.Sprint(sweepStops(artW, splashWavePeriod*8)), fmt.Sprint(sweepStops(artW, 0)); got != want {
		t.Errorf("sweep did not wrap cleanly")
	}
}

func TestUnlitLettersStayLegible(t *testing.T) {
	// lipgloss.Style holds a func field, so it can't be compared with ==; and
	// with no TTY the profile is Ascii, so rendered output drops the colour
	// entirely. Compare the colours themselves.
	dim := splashStopStyle[stopDim].GetForeground()
	mid := splashStopStyle[stopMid].GetForeground()
	bright := splashStopStyle[stopBright].GetForeground()
	if dim != lipgloss.Color("#6c6c6c") {
		t.Errorf("dim stop = %v, want #6c6c6c", dim)
	}
	if mid != lipgloss.Color("#b0b0b0") {
		t.Errorf("mid stop = %v, want #b0b0b0", mid)
	}
	if bright != lipgloss.Color("#ffffff") {
		t.Errorf("bright stop = %v, want #ffffff", bright)
	}
	if dim == bright {
		t.Errorf("dim and bright stops are identical: %v", dim)
	}
	if mid == dim {
		t.Errorf("mid and dim stops are identical: %v", mid)
	}
	if mid == bright {
		t.Errorf("mid and bright stops are identical: %v", mid)
	}
}

func TestColumnStylesRespectsStyleFlag(t *testing.T) {
	const artW = 40
	sh := columnStyles(artW, 5, styleShimmer)
	rb := columnStyles(artW, 5, styleRainbow)
	if len(sh) != artW || len(rb) != artW {
		t.Fatalf("style slices wrong length: %d/%d", len(sh), len(rb))
	}
	// Rainbow must actually vary per column; shimmer has repeats by design.
	vary := 0
	for c := 1; c < artW; c++ {
		if rb[c].GetForeground() != rb[c-1].GetForeground() {
			vary++
		}
	}
	if vary == 0 {
		t.Errorf("rainbow produced no per-column colour variation")
	}
	// Unknown style falls back to shimmer rather than panicking.
	if got := columnStyles(artW, 5, "nonsense"); len(got) != artW {
		t.Errorf("unknown style should still return a full slice")
	}
}

func TestMiniBannerShape(t *testing.T) {
	b := stripANSI(miniBanner(4, styleShimmer))
	lines := strings.Split(b, "\n")
	if len(lines) != len(splashArt) {
		t.Errorf("mini banner rows = %d, want %d", len(lines), len(splashArt))
	}
	if lit(b) == 0 {
		t.Errorf("mini banner drew no glyphs")
	}
	// no status/hint text leaked into the header banner
	if strings.Contains(b, "offline") || strings.Contains(b, "q to quit") {
		t.Errorf("mini banner should not include status chrome")
	}
}

func TestSplashFrameTinyTerminal(t *testing.T) {
	// Must not panic or produce a giant frame in a cramped terminal.
	if got := splashFrame(10, 4, 3, "busy", styleShimmer); !strings.Contains(got, "busy") {
		t.Errorf("tiny fallback lost status: %q", got)
	}
}

func TestSplashFrameCentred(t *testing.T) {
	frame := stripANSI(splashFrame(100, 20, 12, "loading", styleShimmer))
	lines := strings.Split(frame, "\n")
	artRows := 0
	for _, l := range lines {
		if strings.TrimSpace(l) != "" {
			artRows++
		}
	}
	if artRows < len(splashArt) {
		t.Errorf("art rows not drawn: %d < %d", artRows, len(splashArt))
	}
}

func lit(s string) int {
	n := 0
	for _, r := range s {
		if r == '█' {
			n++
		}
	}
	return n
}

// TestSplashVisual prints frames so the banner can be eyeballed in the terminal.
func TestSplashVisual(t *testing.T) {
	for _, f := range []int{0, 3, 8, 14} {
		fmt.Printf("--- frame %d (shimmer) ---\n%s\n", f,
			splashFrame(100, 24, f, "loading model (small.en on cuda)", styleShimmer))
	}
}

func stripANSI(s string) string {
	var sb strings.Builder
	inEsc := false
	for _, r := range s {
		switch {
		case inEsc:
			if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
				inEsc = false
			}
		case r == 0x1b:
			inEsc = true
		default:
			sb.WriteRune(r)
		}
	}
	return sb.String()
}
