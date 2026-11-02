package main

import (
	"fmt"
	"math"
	"strings"

	"github.com/charmbracelet/lipgloss"
)

// Splash animation tuning.
const (
	splashRevealFrames = 7   // letter-by-letter reveal before the rainbow leads
	splashWavePeriod   = 24  // frames per full hue sweep
	splashWaveWidth    = 6   // half-width of the bright band, in cells
	splashStatusLines  = 2   // status + hint under the banner
	splashHuePerCol    = 3.0 // degrees of hue per column across the banner
	splashHuePerFrame  = 4.0 // degrees of hue per frame (40 deg/s at 10fps)
	splashSat          = 0.85
	splashVal          = 0.95
)

// Splash styles.
const (
	styleShimmer = "shimmer"
	styleRainbow = "rainbow"
)

// Luminance stops for the sweep. A hard bright/dim binary made unlit letters
// read as holes and the band edge cut through the letterforms, so the sweep
// falls off through a mid tone instead.
const (
	stopDim    = 0
	stopMid    = 1
	stopBright = 2
)

var (
	splashStatus = lipgloss.NewStyle().Foreground(lipgloss.Color("12"))
	splashHint   = lipgloss.NewStyle().Faint(true).Foreground(lipgloss.Color("8"))
)

// splashShimmerColor is the wordmark blue (cyan -> indigo across the banner)
// modulated by the travelling highlight: unlit letters sit in deep blue, the
// band lifts them toward white, and in-between columns get a mid tone. Same
// sweep as before, same blue as the header.
func splashShimmerColor(col, artW, stop int) string {
	t := 0.0
	if artW > 1 {
		t = float64(col) / float64(artW-1)
	}
	rgb := [3]float64{
		float64(wordmarkFrom[0]) + (float64(wordmarkTo[0])-float64(wordmarkFrom[0]))*t,
		float64(wordmarkFrom[1]) + (float64(wordmarkTo[1])-float64(wordmarkFrom[1]))*t,
		float64(wordmarkFrom[2]) + (float64(wordmarkTo[2])-float64(wordmarkFrom[2]))*t,
	}
	switch stop {
	case stopBright:
		rgb = mixRGB(rgb, [3]float64{255, 255, 255}, 0.62)
	case stopMid:
		rgb = mixRGB(rgb, [3]float64{255, 255, 255}, 0.18)
	default:
		rgb = mixRGB(rgb, [3]float64{0, 0, 0}, 0.45)
	}
	return rgbHex(rgb)
}

func mixRGB(a, b [3]float64, t float64) [3]float64 {
	return [3]float64{
		a[0] + (b[0]-a[0])*t,
		a[1] + (b[1]-a[1])*t,
		a[2] + (b[2]-a[2])*t,
	}
}

func rgbHex(c [3]float64) string {
	return fmt.Sprintf("#%02x%02x%02x",
		int(math.Round(c[0])), int(math.Round(c[1])), int(math.Round(c[2])))
}

// sweepStops returns the luminance stop per column (stopDim/stopMid/stopBright)
// for the travelling highlight. Split out from the renderer so it is
// unit-testable — lipgloss strips styling when there's no TTY, so the emitted
// escape codes can't be asserted on.
func sweepStops(artW, frame int) []int {
	span := artW + 2*splashWaveWidth
	stops := make([]int, artW)
	if span <= 0 {
		return stops
	}
	head := (frame*span/splashWavePeriod)%span - splashWaveWidth
	for c := 0; c < artW; c++ {
		d := c - head
		if d < 0 {
			d = -d
		}
		switch {
		case d <= splashWaveWidth/2:
			stops[c] = stopBright
		case d <= splashWaveWidth+2:
			stops[c] = stopMid
		default:
			stops[c] = stopDim
		}
	}
	return stops
}

// columnStyles returns the per-column style for the chosen animation.
func columnStyles(artW, frame int, style string) []lipgloss.Style {
	if style == styleRainbow {
		out := make([]lipgloss.Style, artW)
		for c := range out {
			out[c] = lipgloss.NewStyle().Foreground(lipgloss.Color(splashCellColor(c, frame)))
		}
		return out
	}
	stops := sweepStops(artW, frame)
	out := make([]lipgloss.Style, artW)
	for c := range out {
		out[c] = lipgloss.NewStyle().
			Foreground(lipgloss.Color(splashShimmerColor(c, artW, stops[c])))
	}
	return out
}

// hsvToHex converts HSV (h in degrees, s/v in 0..1) to a #rrggbb string.
// Pure function so the rainbow is unit-testable.
func hsvToHex(h, s, v float64) string {
	for h < 0 {
		h += 360
	}
	h = math.Mod(h, 360)
	c := v * s
	x := c * (1 - math.Abs(math.Mod(h/60, 2)-1))
	m := v - c
	var r, g, b float64
	switch {
	case h < 60:
		r, g, b = c, x, 0
	case h < 120:
		r, g, b = x, c, 0
	case h < 180:
		r, g, b = 0, c, x
	case h < 240:
		r, g, b = 0, x, c
	case h < 300:
		r, g, b = x, 0, c
	default:
		r, g, b = c, 0, x
	}
	return fmt.Sprintf("#%02x%02x%02x",
		int(math.Round((r+m)*255)),
		int(math.Round((g+m)*255)),
		int(math.Round((b+m)*255)))
}

// splashCellColor is the scrolling rainbow: hue advances across the banner by
// column and every frame, so the whole spectrum flows left.
func splashCellColor(col, frame int) string {
	return hsvToHex(float64(col)*splashHuePerCol+float64(frame)*splashHuePerFrame, splashSat, splashVal)
}

// splashGrid returns the art as a rune grid padded to a uniform width, plus the
// width of one letter cell (glyphs are half-block pairs, so a letter is a run of
// non-space cells).
func splashGrid() (grid [][]rune, width, letterW int) {
	width = splashArtWidth
	if width == 0 {
		for _, row := range splashArt {
			if n := len([]rune(row)); n > width {
				width = n
			}
		}
	}
	letterW = 0
	for _, r := range splashArt[0] {
		if r == ' ' {
			break
		}
		letterW++
	}
	if letterW == 0 {
		letterW = width
	}
	grid = make([][]rune, len(splashArt))
	for i, row := range splashArt {
		cells := []rune(row)
		for len(cells) < width {
			cells = append(cells, ' ')
		}
		grid[i] = cells[:width]
	}
	return grid, width, letterW
}

// paintRow colours one row of art from column `from` onward. Glyph cells take
// the per-column style; spaces stay plain so letterforms stay crisp.
func paintRow(row []rune, from, frame int, style string) string {
	styles := columnStyles(len(row), frame, style)
	var sb strings.Builder
	for c := from; c < len(row); c++ {
		ch := row[c]
		if ch == ' ' {
			sb.WriteRune(' ')
			continue
		}
		sb.WriteString(styles[c].Render(string(ch)))
	}
	return sb.String()
}

// splashFrame renders the full-screen loading banner, centred in width x height.
// frame drives the letter reveal (first frames) and then the chosen animation.
func splashFrame(width, height, frame int, status, style string) string {
	if width < 24 || height < 8 {
		return splashHint.Render(status)
	}
	grid, artW, letterW := splashGrid()
	nLetters := (artW + letterW - 1) / letterW

	shown := nLetters
	if frame < splashRevealFrames {
		shown = frame + 1
	}
	revealCols := shown * letterW
	if revealCols > artW {
		revealCols = artW
	}
	hidePad := (artW - revealCols) / 2

	leftPad := (width - artW) / 2
	if leftPad < 0 {
		leftPad = 0
	}
	indent := strings.Repeat(" ", leftPad)
	hidden := strings.Repeat(" ", leftPad+hidePad)

	blockH := len(grid) + splashStatusLines
	topPad := (height - blockH) / 2
	if topPad < 0 {
		topPad = 0
	}

	var out strings.Builder
	out.WriteString(strings.Repeat("\n", topPad))
	for _, row := range grid {
		out.WriteString(hidden)
		out.WriteString(paintRow(row, hidePad, frame, style))
		out.WriteString("\n")
	}
	out.WriteString("\n")
	out.WriteString(indent + splashStatus.Render(status) + "\n")
	out.WriteString(indent + splashHint.Render("local · offline · on-device whisper — q to quit"))
	return out.String()
}

// miniBanner is the paused-state header banner: the same art and animation, no
// status or hint, so it can sit above the captions without hiding them.
func miniBanner(frame int, style string) string {
	grid, _, _ := splashGrid()
	indent := strings.Repeat(" ", splashMiniIndent)
	var out strings.Builder
	for _, row := range grid {
		out.WriteString(indent)
		out.WriteString(paintRow(row, 0, frame, style))
		out.WriteString("\n")
	}
	return strings.TrimRight(out.String(), "\n")
}

// splashMiniIndent left-aligns the mini banner so it lines up with the title.
const splashMiniIndent = 0
