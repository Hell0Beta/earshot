// Earshot TUI: spawns the Python backend, renders live captions.
//
//	space      pause/resume (UI-only: display freezes, backend keeps running)
//	up/dn/pgup/pgdn  scroll (auto-follow resumes on End)
//	q / ctrl+c quit, kill backend, save transcript
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"time"

	"github.com/charmbracelet/bubbles/textarea"
	"github.com/charmbracelet/bubbles/viewport"
	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
	"github.com/muesli/reflow/wordwrap"
)

// ---- events from the backend reader goroutine ----

type captionMsg struct {
	ID   int
	Text string
	T0   float64
}

// correctedMsg patches an already-committed caption in place when the
// backend's LLM correction pass returns a polished version.
type correctedMsg struct {
	ID   int
	Text string
}

type draftMsg struct {
	Text  string
	Chunk int
}

type backendStatusMsg struct{ Msg string }

// polishedMsg carries a whole-document readability pass back from the LLM.
// It replaces the pane's text; unlike correctedMsg it has no block id.
type polishedMsg struct{ Text string }

// resetMsg confirms the backend dropped its pending sentence after `x`.
type resetMsg struct{}

// flushedMsg means the backend committed the words that were still waiting on
// a second hypothesis. Only then is the document complete enough to edit.
type flushedMsg struct{}

type backendDoneMsg struct{ Err error }

type tickMsg time.Time

type backendEvent struct {
	Type string  `json:"type"`
	ID   int     `json:"id"`
	Text string  `json:"text"`
	Msg  string  `json:"msg"`
	T0   float64 `json:"t0"`
	T1   float64 `json:"t1"`
	N    int     `json:"n"`
}

// ---- styles ----

var (
	liveStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("10"))
	pauseStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("11"))
	errStyle      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("9"))
	liveLineStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("15"))
	docStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("15"))
	rawStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("252"))
	tailStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("245"))
	draftStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("14"))
	histStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("250"))
	stampStyle    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
	keyStyle      = lipgloss.NewStyle().Foreground(lipgloss.Color("14"))
	ruleStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("238"))
	dimmedStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("242"))
	chipStyle     = lipgloss.NewStyle().Foreground(lipgloss.Color("252")).Background(lipgloss.Color("236"))
	chipActive    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("15")).Background(lipgloss.Color("12"))
	chipFlash     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("16")).Background(lipgloss.Color("14"))
	chipCopied    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("16")).Background(lipgloss.Color("34"))
	chipOnDark    = lipgloss.NewStyle().Foreground(lipgloss.Color("240"))
)

// spinnerFrames animates the polish indicator. The UI ticks at 10fps, so this
// reads as a calm rotation rather than a strobe.
var spinnerFrames = []string{"⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"}

var keyHintRE = regexp.MustCompile(`\[[^\]]+\]`)

func spinnerGlyph(frame int) string {
	return spinnerFrames[(frame/2)%len(spinnerFrames)]
}

// The wordmark ramp, shared with the splash and the paused hero so all three
// wear the same blue.
var (
	wordmarkFrom = [3]int{0x5A, 0xC8, 0xFA} // cyan
	wordmarkTo   = [3]int{0x4A, 0x6C, 0xF7} // indigo
)

// gradientText paints each rune along a straight RGB ramp.
func gradientText(s string, from, to [3]int) string {
	runes := []rune(s)
	if len(runes) == 0 {
		return ""
	}
	var b strings.Builder
	for i, ch := range runes {
		t := 0.0
		if len(runes) > 1 {
			t = float64(i) / float64(len(runes)-1)
		}
		c := [3]int{
			from[0] + int(float64(to[0]-from[0])*t),
			from[1] + int(float64(to[1]-from[1])*t),
			from[2] + int(float64(to[2]-from[2])*t),
		}
		st := lipgloss.NewStyle().Bold(true).
			Foreground(lipgloss.Color(fmt.Sprintf("#%02x%02x%02x", c[0], c[1], c[2])))
		b.WriteString(st.Render(string(ch)))
	}
	return b.String()
}

// maxLineRunes is a safety cap only — the backend already blocks to
// block_chars (default 300), and this must stay above it or every committed
// line and live draft gets truncated mid-sentence.
const maxLineRunes = 320

// ---- model ----

// archiveEntry locates one committed caption inside the transcript file so a
// correction can rewrite its bytes in place. Captions are written in file
// order (oldest first) while the on-screen stack is newest-first, so the two
// are tracked separately.
type archiveEntry struct {
	id    int
	pos   int64
	bytes int
	t0    float64
	text  string // committed (or corrected) text, in file order
}

// mode is the top-bar toggle: raw live captions, or a dictation pane whose
// text is polished for readability and can be copied or edited.
type mode int

const (
	modeLive mode = iota
	modeV2T
)

// polishDebounce is how long transcription must be quiet before a new
// readability pass is requested. Polishing mid-sentence would rewrite text the
// speaker has not finished saying, and the model would guess the ending.
const polishDebounce = 1200 * time.Millisecond

// polishTimeout is how long an unanswered request may hold the pane before it
// is abandoned. A live 0.5B model answers a prompt-sized document in a second
// or two; anything near this means the worker is gone.
const polishTimeout = 20 * time.Second

type model struct {
	viewport    viewport.Model
	lines       []string // newest first (stack)
	draft       string   // in-progress sentence, shown dimmed on top
	paused      bool
	follow      bool // pinned to the top (newest)
	ready       bool
	status      string
	backendUp   bool
	nCaptions   int
	chunk       int    // last backend chunk index: proves the loop is alive
	frame       int    // animation frame for the loading splash
	blockChars  int    // sentence block target, for the footer gauge
	splashStyle string // styleShimmer (default) or styleRainbow
	width       int
	height      int
	started     time.Time
	build       string // exe build stamp (own mtime): "am I running the new one?"
	outFile     *os.File
	outPath     string
	outPos      int64
	lineIDs     []int          // block id per on-screen line (newest-first)
	lineT0      []float64      // timestamp per on-screen line
	archived    []archiveEntry // the voice-to-text document only
	fileIndex   []archiveEntry // every caption in the .md, for in-place rewrites

	// ---- voice-to-text ----
	mode        mode
	sawV2T      bool      // the document is worth saving as .txt
	hasPolish   bool      // an LLM model was found, so polish will work
	backendIn   io.Writer // JSON commands to the backend (nil = unavailable)
	polished    string    // latest readability pass; "" until one lands
	polishing   bool
	polishWait  bool      // document changed since the last request
	polishAt    time.Time // when the document last changed
	polishSent  time.Time // when the in-flight request was written
	polishReqAt int       // len(archived) when the in-flight polish was sent
	polishedAt  int       // len(archived) the polished text actually covers
	docEpoch    int       // bumped on clear, so stale polish results are dropped
	polishEpoch int       // docEpoch of the in-flight polish
	docBase     string    // edited text that replaces everything before it
	docBaseAt   int       // len(archived) the edit supersedes
	editing     bool
	flushing    bool      // a flush the editor asked for is in flight
	flushAt     time.Time // when that flush was sent
	editor      textarea.Model
	copiedChars int
	copiedAt    time.Time
	modeAt      time.Time // last mode switch, for the switch flash
}

func (m model) Init() tea.Cmd { return tickCmd() }

func tickCmd() tea.Cmd {
	// 10fps: fast enough for the shimmer sweep, cheap enough to idle at.
	return tea.Tick(100*time.Millisecond, func(t time.Time) tea.Msg { return tickMsg(t) })
}

func (m model) Update(msg tea.Msg) (tea.Model, tea.Cmd) {
	switch msg := msg.(type) {
	case tea.WindowSizeMsg:
		m.width, m.height = msg.Width, msg.Height
		if !m.ready {
			m.viewport = viewport.New(msg.Width, msg.Height-4)
			m.ready = true
		} else {
			m.viewport.Width = msg.Width
		}
		m.clampViewport()
		if m.editing {
			m.editor.SetWidth(msg.Width)
			m.editor.SetHeight(m.viewport.Height)
		}
		m.refresh()
		return m, nil

	case captionMsg:
		if m.paused {
			return m, nil // UI-only pause: freeze display, stop saving
		}
		m.draft = ""
		// File keeps the full chronological record; the screen is a
		// newest-first stack (backend already blocks + chains to 200).
		fmt.Fprintf(m.outFile, "- [%s] %s\n", fmtClock(msg.T0), msg.Text)
		// Record the byte span this caption occupies so a later correction can
		// rewrite it in place. Captions are written oldest-first, so the span
		// is captured *after* the write above.
		entry := archiveEntry{id: msg.ID, t0: msg.T0, text: msg.Text}
		if end, err := m.outFile.Seek(0, io.SeekCurrent); err == nil {
			entry.bytes = int(end - m.outPos)
			entry.pos = m.outPos
			m.outPos = end
		}
		m.fileIndex = append(m.fileIndex, entry)

		// Each mode owns its text. A final only joins the pane that was
		// active when it was spoken, so switching never carries speech across
		// from the captions into the dictation document.
		if m.mode == modeV2T {
			// While the editor is open, or while the flush the user asked for
			// is in flight, these words are already in the document/editor.
			// Appending them again is what printed the same sentence twice.
			if !m.editing && !m.flushing {
				m.archived = append(m.archived, entry)
				// The document grew: ask for a fresh readability pass once the
				// speaker pauses. Editing suppresses it so the request never
				// races the user's typing.
				m.polishWait = true
				m.polishAt = time.Now()
			}
		} else {
			m.nCaptions++
			line := fmt.Sprintf("- [%s] %s", fmtClock(msg.T0), capRunes(msg.Text, maxLineRunes))
			m.lines = append([]string{line}, m.lines...)
			m.lineIDs = append([]int{msg.ID}, m.lineIDs...)
			m.lineT0 = append([]float64{msg.T0}, m.lineT0...)
		}
		m.refresh()
		return m, nil

	case correctedMsg:
		if m.paused {
			return m, nil
		}
		// Patch the displayed line if it is still on screen; corrections for
		// lines that scrolled into history are ignored rather than popping
		// them back into view.
		for i, id := range m.lineIDs {
			if id == msg.ID {
				m.lines[i] = fmt.Sprintf("- [%s] %s",
					fmtClock(m.lineT0[i]), capRunes(msg.Text, maxLineRunes))
				break
			}
		}
		// The archive stores the corrected text: what the reader sees is what
		// is kept. Rewrite this block's bytes in place, padding with spaces
		// so subsequent offsets stay valid.
		for i := range m.fileIndex {
			a := &m.fileIndex[i]
			if a.id != msg.ID {
				continue
			}
			a.text = msg.Text
			if a.bytes > 0 {
				old := a.bytes
				newText := fmt.Sprintf("- [%s] %s", fmtClock(a.t0), msg.Text)
				if len(newText) > old {
					newText = newText[:old]
				}
				padded := newText + strings.Repeat(" ", old-len(newText))
				m.outFile.WriteAt([]byte(padded), a.pos)
			}
			break
		}
		// The document keeps its own copy; it needs no file write.
		for i := range m.archived {
			if m.archived[i].id == msg.ID {
				m.archived[i].text = msg.Text
				break
			}
		}
		m.refresh()
		return m, nil

	case polishedMsg:
		// Clear the in-flight flag even while paused: otherwise the pane would
		// stay stuck on "polishing…" until the timeout expired.
		m.polishing = false
		// Cleared while the model was working: drop the stale result.
		if m.docEpoch != m.polishEpoch {
			return m, nil
		}
		if m.paused {
			return m, nil
		}
		if strings.TrimSpace(msg.Text) != "" {
			m.polished = msg.Text
			m.polishedAt = m.polishReqAt
		}
		// Speech landed while the model was working: the pane is already
		// stale. Let the debounce pick it up again rather than looping here.
		m.refresh()
		return m, nil

	case draftMsg:
		if m.paused {
			return m, nil
		}
		m.draft = msg.Text
		m.chunk = msg.Chunk
		m.refresh()
		return m, nil

	case resetMsg:
		// Confirmation from the backend that the pending sentence is gone.
		// Anything it sent before this is already on screen and was cleared
		// by the keypress; this closes the gap for messages still in flight.
		m.draft = ""
		m.refresh()
		return m, nil

	case flushedMsg:
		// The backend committed the pending sentence. If the editor is open,
		// it already holds those words; do not re-seed it under the user.
		m.flushing = false
		m.draft = ""
		m.refresh()
		return m, nil

	case backendStatusMsg:
		m.status = msg.Msg
		if msg.Msg == "live" {
			m.backendUp = true
		}
		return m, nil

	case backendDoneMsg:
		m.backendUp = false
		if msg.Err != nil {
			m.status = "backend exited: " + msg.Err.Error() + " — press q"
		} else {
			m.status = "backend exited — press q"
		}
		return m, nil

	case tickMsg:
		m.frame++
		m.maybePolish()
		m.maybeDropStuckFlush()
		return m, tickCmd()

	case tea.MouseMsg:
		if m.editing || !m.ready {
			return m, nil
		}
		if msg.Action == tea.MouseActionPress && msg.Button == tea.MouseButtonLeft && msg.Y == 0 {
			h := m.headerHits()
			switch {
			case msg.X >= h.copyStart && msg.X < h.copyEnd:
				m.copyDocument()
			case msg.X >= h.toggleStart && msg.X < h.toggleEnd:
				if msg.X < h.toggleMid {
					m.setMode(modeLive)
				} else {
					m.setMode(modeV2T)
				}
			}
			m.refresh()
		}
		return m, nil

	case tea.KeyMsg:
		// Editing owns the keyboard: `q` must type a q, not quit.
		if m.editing {
			switch msg.String() {
			case "esc":
				m.endEdit()
				m.refresh()
				return m, nil
			case "ctrl+c":
				return m, tea.Quit
			}
			var cmd tea.Cmd
			m.editor, cmd = m.editor.Update(msg)
			return m, cmd
		}
		switch msg.String() {
		case " ":
			m.paused = !m.paused
			// Clamp on BOTH transitions: pausing grows the header by the
			// mini banner's rows, and without resizing the viewport the
			// frame exceeds the terminal, which scrolls the header (and the
			// banner) off the top. refresh() clamps too — this makes the
			// intent explicit at the transition, and refresh() is the safety
			// net for every other layout change.
			m.clampViewport()
			m.refresh()
			return m, nil
		case "tab":
			if m.mode == modeV2T {
				m.setMode(modeLive)
			} else {
				m.setMode(modeV2T)
			}
			m.refresh()
			return m, nil
		case "c":
			m.copyDocument()
			m.refresh()
			return m, nil
		case "e":
			m.requestEdit()
			m.refresh()
			return m, nil
		case "x":
			m.clearDocument()
			return m, nil
		case "q", "ctrl+c":
			return m, tea.Quit
		case "home":
			m.follow = true
			m.viewport.GotoTop()
			return m, nil
		case "end":
			m.follow = false
			m.viewport.GotoBottom()
			return m, nil
		}
		// scrolling keys go to the viewport; pinned to top until the user
		// scrolls down into history.
		var cmd tea.Cmd
		m.viewport, cmd = m.viewport.Update(msg)
		m.follow = m.viewport.AtTop()
		return m, cmd
	}
	return m, nil
}

// clearDocument wipes the on-screen transcript and asks the backend to forget
// the pending sentence too, so cleared words cannot resurface in the next
// update. The transcript FILE is left alone: clearing the pane is a reading
// convenience, not a way to destroy the record.
func (m *model) clearDocument() {
	m.draft = ""
	m.docEpoch++
	m.flushing = false
	if m.mode == modeV2T {
		m.archived = nil
		m.polished = ""
		m.polishing = false
		m.polishWait = false
		m.polishAt = time.Time{}
		m.polishSent = time.Time{}
		m.polishReqAt = 0
		m.polishedAt = 0
		m.docBase = ""
		m.docBaseAt = 0
	} else {
		m.lines = nil
		m.lineIDs = nil
		m.lineT0 = nil
		m.nCaptions = 0
	}
	m.status = "transcript cleared"
	// The backend still holds the half-heard sentence; drop it so the words
	// that were just cleared cannot arrive a moment later.
	m.sendBackend(map[string]any{"cmd": "reset"})
	m.refresh()
}

// ---- mode, polish, copy and edit ----

// setMode switches the pane. It never restarts the backend: the backend swaps
// the capture device on request, so switching costs no audio and loses no
// words. Dictation listens to the microphone; captions to the speakers.
func (m *model) setMode(next mode) {
	if m.mode == next {
		return
	}
	m.mode = next
	m.modeAt = time.Now()
	m.draft = ""
	if next == modeV2T {
		m.sawV2T = true
		m.status = "voice to text"
		if !m.hasPolish {
			m.status = "voice to text (no polish model — raw text only)"
		}
		if len(m.docText()) > 0 {
			m.polishWait = true
			m.polishAt = time.Now()
		}
		m.sendBackend(map[string]any{"cmd": "source", "kind": "mic"})
	} else {
		m.status = "live captions"
		m.sendBackend(map[string]any{"cmd": "source", "kind": "loopback"})
	}
}

// maybePolish sends a readability request once speech has been quiet for the
// debounce window. Called from the animation tick; the request itself is a
// non-blocking pipe write.
// maybeDropStuckFlush clears the flush guard if the backend never answered.
// Without it a lost reply would silently stop later speech from joining the
// dictation document.
func (m *model) maybeDropStuckFlush() {
	if m.flushing && time.Since(m.flushAt) > 2*time.Second {
		m.flushing = false
	}
}

func (m *model) maybePolish() {
	if m.mode != modeV2T || m.editing || !m.polishWait {
		return
	}
	// If the model died or the request was lost, do not stay stuck on
	// "polishing…": the backend always answers, but a crashed worker cannot.
	if m.polishing {
		if time.Since(m.polishSent) > polishTimeout {
			m.polishing = false
		} else {
			return
		}
	}
	if time.Since(m.polishAt) < polishDebounce {
		return
	}
	doc := m.docText()
	if doc == "" {
		return
	}
	m.polishEpoch = m.docEpoch
	if !m.sendBackend(map[string]any{"cmd": "polish", "text": doc}) {
		return
	}
	m.polishing = true
	m.polishSent = time.Now()
	m.polishReqAt = len(m.archived)
	m.polishWait = false
	m.refresh()
}

// sendBackend writes one JSON command line to the backend. Never blocks the UI:
// the pipe is small and the backend reads continuously.
func (m *model) sendBackend(v any) bool {
	if m.backendIn == nil {
		return false
	}
	b, err := json.Marshal(v)
	if err != nil {
		return false
	}
	_, err = fmt.Fprintf(m.backendIn, "%s\n", b)
	return err == nil
}

// docText is the dictation document in speaking order: the text the user
// edited (if any), then every caption committed since that edit.
func (m model) docText() string {
	var b strings.Builder
	b.WriteString(strings.TrimSpace(m.docBase))
	start := m.docBaseAt
	if start > len(m.archived) {
		start = len(m.archived)
	}
	for _, a := range m.archived[start:] {
		t := strings.TrimSpace(a.text)
		if t == "" {
			continue
		}
		if b.Len() > 0 {
			b.WriteString(" ")
		}
		b.WriteString(t)
	}
	return strings.TrimSpace(b.String())
}

// tailText is the speech committed after the current polish was requested. It
// has had no readability pass yet, so the pane shows it dim rather than
// pretending it is not there.
func (m model) tailText() string {
	if m.polished == "" {
		return ""
	}
	start := m.polishedAt
	if start > len(m.archived) {
		start = len(m.archived)
	}
	var words []string
	for _, a := range m.archived[start:] {
		if t := strings.TrimSpace(a.text); t != "" {
			words = append(words, t)
		}
	}
	return strings.Join(words, " ")
}

// bestText is the most readable complete text available: the polish, plus any
// speech that landed while it was being generated. Copying only the polish
// would silently drop the last thing the speaker said.
func (m model) bestText() string {
	if m.polished == "" {
		return m.docText()
	}
	tail := m.tailText()
	if tail == "" {
		return m.polished
	}
	return strings.TrimSpace(m.polished + " " + tail)
}

// liveText is the live-captions pane as plain running text: oldest first,
// without the timestamps the display adds.
func (m model) liveText() string {
	var out []string
	for i := len(m.lines) - 1; i >= 0; i-- {
		l := m.lines[i]
		if j := strings.Index(l, "] "); j >= 0 {
			l = l[j+2:]
		}
		if t := strings.TrimSpace(l); t != "" {
			out = append(out, t)
		}
	}
	return strings.Join(out, " ")
}

// editSeed is what the editor starts with. The flush should already have
// committed the live sentence, but if it has not, the words on screen must
// still reach the editor: an edit may add text, never silently lose it.
func (m model) editSeed() string {
	text := strings.TrimSpace(m.bestText())
	if m.mode == modeLive {
		text = strings.TrimSpace(m.liveText())
	}
	d := strings.TrimSpace(m.draft)
	switch {
	case text == "":
		return d
	case d == "":
		return text
	default:
		return text + " " + d
	}
}

// copyDocument puts the best available text on the clipboard. It copies
// immediately rather than waiting for the model: a slow polish must never make
// the copy button feel broken.
func (m *model) copyDocument() {
	if m.mode == modeLive {
		m.copyLive()
		return
	}
	text := strings.TrimSpace(m.bestText())
	src := "polished"
	if m.polished == "" {
		src = "raw"
	} else if strings.TrimSpace(m.tailText()) != "" {
		src = "polished + new speech"
	}
	// The live sentence is on screen, so it should be in the clipboard too.
	if d := strings.TrimSpace(m.draft); d != "" {
		if text == "" {
			text = d
		} else {
			text += " " + d
		}
		if src == "polished" {
			src = "polished + live speech"
		} else {
			src = "raw + live speech"
		}
	}
	if text == "" {
		m.status = "nothing to copy yet"
		return
	}
	if err := clipboardWrite(text); err != nil {
		m.status = "copy failed: " + err.Error()
		return
	}
	m.copiedChars = len([]rune(text))
	m.copiedAt = time.Now()
	m.status = fmt.Sprintf("copied %d chars (%s)", m.copiedChars, src)
	// The copied text is behind the document: refresh in the background.
	if m.mode == modeV2T && len(m.archived) > m.polishedAt {
		m.polishWait = true
		m.polishAt = time.Now()
	}
}

// copyLive copies the captions pane (plus the in-progress line).
func (m *model) copyLive() {
	text := strings.TrimSpace(m.liveText())
	if d := strings.TrimSpace(m.draft); d != "" {
		if text == "" {
			text = d
		} else {
			text += " " + d
		}
	}
	if text == "" {
		m.status = "nothing to copy yet"
		return
	}
	if err := clipboardWrite(text); err != nil {
		m.status = "copy failed: " + err.Error()
		return
	}
	m.copiedChars = len([]rune(text))
	m.copiedAt = time.Now()
	m.status = fmt.Sprintf("copied %d chars (live captions)", m.copiedChars)
}

// requestEdit opens the text editor, first asking the backend to commit the
// sentence that is still waiting on a second hypothesis. Without that flush
// the pane would show words the editor silently drops.
func (m *model) requestEdit() {
	if m.editing {
		return
	}
	// Ask the backend to commit the sentence that has not agreed with a
	// second hypothesis yet, so the transcript FILE is complete. The editor
	// does not wait for that round trip: it opens now, seeded from the words
	// already on screen. Waiting for the flush is what made `e` appear to
	// clear the text whenever the reply was slow or empty.
	// Always flush, even when the draft looks empty: the assembler may still
	// be holding a short sentence that has not filled a block yet.
	if m.sendBackend(map[string]any{"cmd": "flush"}) {
		m.flushing = true
		m.flushAt = time.Now()
	}
	m.beginEdit()
}

// beginEdit swaps the pane for a text area holding the current text. Speech
// during editing is still written to the transcript on disk, but it is not
// merged into the document: the user's text wins.
func (m *model) beginEdit() {
	if !m.ready {
		return
	}
	ta := textarea.New()
	ta.Prompt = ""
	ta.ShowLineNumbers = false
	ta.CharLimit = 0
	ta.Placeholder = "speak, or type here…"
	ta.SetWidth(m.viewport.Width)
	ta.SetHeight(m.viewport.Height)
	ta.FocusedStyle.Text = docStyle
	ta.FocusedStyle.Placeholder = dimmedStyle
	ta.FocusedStyle.CursorLine = lipgloss.NewStyle().Background(lipgloss.Color("237"))
	ta.FocusedStyle.Prompt = keyStyle
	ta.BlurredStyle.Text = rawStyle
	ta.SetValue(m.editSeed())
	ta.Focus()
	m.editor = ta
	m.editing = true
	m.draft = ""
	m.polishWait = false
	m.status = "editing — esc to finish"
}

// endEdit makes the typed text the new document base. Everything committed
// before this point is superseded; later speech appends after it.
func (m *model) endEdit() {
	text := strings.TrimSpace(m.editor.Value())
	m.editing = false
	m.editor.Blur()
	m.draft = ""
	if m.mode == modeLive {
		// The edited captions become one block anchored at the newest line's
		// timestamp; the old per-line times no longer describe the text.
		t0 := 0.0
		if len(m.lineT0) > 0 {
			t0 = m.lineT0[0]
		}
		if text == "" {
			m.lines = nil
		} else {
			m.lines = []string{fmt.Sprintf("- [%s] %s", fmtClock(t0), text)}
		}
		m.lineIDs = nil
		m.lineT0 = nil
		m.status = "edited"
		return
	}
	m.docBase = text
	m.docBaseAt = len(m.archived)
	m.polished = ""
	m.polishedAt = 0
	m.polishWait = text != ""
	m.polishAt = time.Now()
	m.status = "edited"
}

// finalText is what gets saved as <stamp>.txt on quit.
func (m model) finalText() string {
	if m.editing {
		return strings.TrimSpace(m.editor.Value())
	}
	return m.bestText()
}

// headerLeftPlain is the header text without styles, for width computation.
func (m model) headerLeftPlain() string {
	state := "● LIVE"
	if m.paused {
		state = "❚❚ PAUSED"
	} else if !m.backendUp {
		state = "○ backend down"
	}
	return "earshot  " + state
}

// modeLabels shrink on narrow terminals so the switch never crowds the title
// off the row.
func (m model) modeLabels() (string, string) {
	if m.width > 0 && m.width < 70 {
		return "Live", "Voice"
	}
	return "Live Captions", "Voice to Text"
}

// switchRendered draws the segmented mode control, with the active side lit.
func (m model) switchRendered() string {
	live, v2t := m.modeLabels()
	off := chipStyle.Padding(0, 1)
	l, r := off.Render(live), off.Render(v2t)
	on := chipActive.Padding(0, 1)
	if !m.modeAt.IsZero() && time.Since(m.modeAt) < 280*time.Millisecond {
		on = chipFlash.Padding(0, 1)
	}
	if m.mode == modeLive {
		l = on.Render(live)
	} else {
		r = on.Render(v2t)
	}
	return chipOnDark.Render("\u2502") + l + chipOnDark.Render("\u2502") + r + chipOnDark.Render("\u2502")
}

// liveState is the connection pill: a gently pulsing green dot while live,
// amber while paused, red when the backend is gone.
func (m model) liveState() string {
	if m.paused {
		return pauseStyle.Render("\u275a\u275a PAUSED")
	}
	if !m.backendUp {
		return errStyle.Render("\u25cb backend down")
	}
	dot := "10"
	if (m.frame/10)%2 == 1 {
		dot = "34"
	}
	return lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color(dot)).
		Render("\u25cf") + liveStyle.Render(" LIVE")
}

// copyChip is the top-right action. It turns green and says "Copied" for a
// moment so the copy is confirmed where the eye already is.
func (m model) copyChip() string {
	label := "Copy"
	st := chipStyle.Padding(0, 1)
	if !m.copiedAt.IsZero() && time.Since(m.copiedAt) < 1200*time.Millisecond {
		label = "Copied"
		st = chipCopied.Padding(0, 1)
	}
	return st.Render(label)
}

// headerRule separates the header from the text: one dim row, no layout cost
// beyond the row the blank line already used.
func (m model) headerRule() string {
	if m.width <= 0 {
		return ""
	}
	return ruleStyle.Render(strings.Repeat("\u2500", m.width))
}

// headerLine is the top row: identity and state on the left, actions on the
// right. Right alignment is computed, not padded by hand, so it stays correct
// while the state label (LIVE / PAUSED / backend down) changes width.
func (m model) headerLine() string {
	state := liveStyle.Render("● LIVE")
	if m.paused {
		state = pauseStyle.Render("❚❚ PAUSED")
	} else if !m.backendUp {
		state = errStyle.Render("○ backend down")
	}
	left := gradientText("earshot", wordmarkFrom, wordmarkTo) + "  " + state
	lw := lipgloss.Width(m.headerLeftPlain())
	sw := m.switchRendered()
	copyChip := m.copyChip()
	copyW, swW := lipgloss.Width(copyChip), lipgloss.Width(sw)

	if m.width > 0 && lw+copyW+2+swW < m.width {
		pad := m.width - lw - copyW - 2 - swW
		return left + strings.Repeat(" ", pad) + copyChip + "  " + sw
	}
	if m.width > 0 && lw+swW < m.width {
		pad := m.width - lw - swW
		return left + strings.Repeat(" ", pad) + sw
	}
	return left
}

// headerHits maps the rendered header back to clickable column ranges. It
// mirrors headerLine's fit logic exactly, so a click can never land on a chip
// that was dropped to make room.
type headerHits struct {
	copyStart, copyEnd                int
	toggleStart, toggleMid, toggleEnd int
}

func (m model) headerHits() headerHits {
	var h headerHits
	if !m.ready || m.width <= 0 {
		return h
	}
	lw := lipgloss.Width(m.headerLeftPlain())
	swW := lipgloss.Width(m.switchRendered())
	copyW := lipgloss.Width(m.copyChip())

	if lw+copyW+2+swW < m.width {
		start := m.width - copyW - 2 - swW
		h.copyStart, h.copyEnd = start, start+copyW
		t := start + copyW + 2
		h.toggleStart, h.toggleEnd = t, t+swW
		h.toggleMid = t + swW/2
		return h
	}
	if lw+swW < m.width {
		t := m.width - swW
		h.toggleStart, h.toggleEnd = t, t+swW
		h.toggleMid = t + swW/2
	}
	return h
}

// maxRenderedHistory caps how many older blocks are wrapped and drawn. Wrapping
// is O(history x width) and runs on every draft event, so an unbounded stack
// would eventually make the UI feel sluggish. It scales with block size: at
// block_chars=300 a 40-block stack is ~12k chars re-wrapped per event, so the
// count is cut to keep the same screen-filling depth at a fraction of the cost.
const maxRenderedHistory = 24

// wrap hard-wraps text to the terminal width, preserving blank lines.
func (m model) wrap(s string) string {
	if m.width <= 0 {
		return s
	}
	return strings.ReplaceAll(wordwrap.String(s, m.width), "\n", "\n")
}

// renderContent builds the caption area: live line, newest block, a two-line
// gap, then history packed tight — wrapped to the terminal width, pinned to the
// top. The live line and the newest committed line share a style so the pair
// reads as one block mid-sentence; older lines are dimmed so the in-progress
// line reads as the hero. Pausing restores full brightness for reading history.
func (m model) renderContent() string {
	var parts []string
	if m.draft != "" {
		for _, ln := range strings.Split(m.wrap(capRunes(m.draft, maxLineRunes)), "\n") {
			parts = append(parts, liveLineStyle.Render(ln+"▌"))
		}
	}
	style := m.historyStyle()
	if len(m.lines) > 0 {
		// The newest committed line shares the live line's style so the pair
		// reads as one block while a sentence builds; the ▌ cursor is what
		// tells them apart. Older lines fall back to the history style.
		for _, ln := range strings.Split(m.wrap(m.lines[0]), "\n") {
			parts = append(parts, m.renderCaptionLine(ln, m.newestStyle()))
		}
		if len(m.lines) > 1 {
			// The gap sits between the new sentence and the block of old
			// ones; old sentences themselves are packed tight.
			parts = append(parts, "", "")
			hist := m.lines[1:]
			if len(hist) > maxRenderedHistory {
				hist = hist[len(hist)-maxRenderedHistory:]
			}
			for _, l := range hist {
				for _, ln := range strings.Split(m.wrap(l), "\n") {
					parts = append(parts, m.renderCaptionLine(ln, style))
				}
			}
		}
	}
	return strings.Join(parts, "\n")
}

// renderCaptionLine styles the clock separately from the words, so the reading
// text carries the emphasis and the timestamp recedes.
func (m model) renderCaptionLine(line string, st lipgloss.Style) string {
	if strings.HasPrefix(line, "- [") {
		if i := strings.Index(line, "] "); i > 0 {
			return stampStyle.Render(line[:i+1]) + " " + st.Render(line[i+2:])
		}
	}
	return st.Render(line)
}

// historyDimmed reports whether committed lines are dimmed. The live
// in-progress line is the hero while capturing; pausing hands the screen back
// to full-brightness history for reading.
func (m model) historyDimmed() bool { return !m.paused }

// historyStyle is the style applied to committed lines beyond the newest.
func (m model) historyStyle() lipgloss.Style {
	if m.historyDimmed() {
		return dimmedStyle
	}
	return histStyle
}

// newestStyle is the style for the most recently committed line. It matches
// the live line so the newest sentence and the one being generated read as a
// single block; the cursor is what distinguishes them.
//
// Deliberately independent of `paused`: when paused the history style already
// renders plain, so the highlight is a live-only cue and pausing still hands
// the screen back for reading. Also independent of whether a draft is
// present, so the display does not flicker between styles during a pause in
// speech.
func (m model) newestStyle() lipgloss.Style { return liveLineStyle }

// headerHeight is the number of rows reserved above the caption area: title +
// rule + blank, plus the 5-row mini banner while paused.
func (m model) headerHeight() int {
	if m.paused {
		return 3 + len(splashArt)
	}
	return 3
}

// clampViewport resizes the caption viewport so the whole frame fits: header +
// caption area + footer (+1 blank separator) must never exceed the terminal
// height, or the terminal scrolls and the header gets cropped.
func (m *model) clampViewport() {
	if !m.ready {
		return
	}
	h := m.height - m.headerHeight() - m.footerRows() - 1
	if h < 1 {
		h = 1
	}
	m.viewport.Height = h
}

func (m model) View() string {
	if !m.ready {
		return "starting…"
	}
	if !m.backendUp && len(m.lines) == 0 {
		return splashFrame(m.width, m.height, m.frame, m.splashStatus(), m.splashStyle)
	}

	var head strings.Builder
	head.WriteString(m.headerLine())
	head.WriteString("\n")
	head.WriteString(m.headerRule())
	if m.paused {
		head.WriteString("\n")
		head.WriteString(miniBanner(m.frame, m.splashStyle))
	}
	head.WriteString("\n\n")

	body := m.viewport.View()
	if m.editing {
		body = m.editor.View()
	}
	return head.String() + body + "\n" + m.footerRendered()
}

// footerText is the status line, unstyled, so its wrapped width can be measured.
func (m model) footerText() string {
	elapsed := time.Since(m.started).Round(time.Second)
	if m.editing {
		return fmt.Sprintf("%s  editing — [esc] done  [ctrl+c] quit  %s  %s",
			elapsed, m.status, filepath.Base(m.outPath))
	}
	// Live buffer gauge: makes the commit gate legible ("87/120") so a
	// sentence that hasn't landed yet reads as filling, not as stalled.
	gauge := ""
	if n := len([]rune(m.draft)); n > 0 {
		gauge = fmt.Sprintf("  buf %d/%d", n, m.blockChars)
	}
	hints := "[tab] voice  [c] copy  [e] edit  [x] clear  [space] pause  [q] quit"
	what := fmt.Sprintf("%d captions, newest first", m.nCaptions)
	if m.mode == modeV2T {
		hints = "[c] copy  [e] edit  [x] clear  [tab] live  [space] pause  [q] quit"
		what = "voice to text"
		if m.polishing {
			what += ", " + spinnerGlyph(m.frame) + " polishing"
		}
	}
	return fmt.Sprintf("%s  %s  chunk %d%s  %s  %s  %s  (build %s)",
		elapsed, what, m.chunk, gauge, m.status, hints,
		filepath.Base(m.outPath), m.build)
}

// footerRendered colours the key hints without changing the text itself, so
// footerRows can keep measuring the plain string.
func (m model) footerRendered() string {
	s := keyHintRE.ReplaceAllStringFunc(m.footerText(), func(tok string) string {
		return keyStyle.Render(tok)
	})
	// The transcript name is metadata; dim it so the status and keys lead.
	if name := filepath.Base(m.outPath); name != "." && name != "" {
		s = strings.Replace(s, "  "+name, "  "+stampStyle.Render(name), 1)
	}
	return s
}

// footerRows is how many terminal rows the footer occupies once wrapped. The
// footer is ~150 chars, so on a narrow terminal it takes 2-3 rows; reserving a
// fixed count is what pushed the header off the top of the screen.
func (m model) footerRows() int {
	if m.width <= 0 {
		return 1
	}
	n := len([]rune(m.footerText()))
	return 1 + n/m.width
}

// splashStatus is the line under the loading banner.
func (m model) splashStatus() string {
	if m.status == "" || m.status == "starting backend…" {
		return "starting backend…"
	}
	return m.status
}

// ---- helpers ----

func fmtClock(sec float64) string {
	s := int(sec)
	return fmt.Sprintf("%02d:%02d", s/60, s%60)
}

// buildStamp answers "am I running the new one?" — the exe's own mtime,
// no build plumbing needed.
func buildStamp() string {
	exe, err := os.Executable()
	if err != nil {
		return "unknown"
	}
	st, err := os.Stat(exe)
	if err != nil {
		return "unknown"
	}
	return st.ModTime().Format("2006-01-02 15:04")
}

// renderV2T builds the dictation pane: the polished document, with the
// in-progress sentence below it in dim text so the speaker can see they are
// being heard. There are no timestamps here; this text is meant to be copied.
func styledLines(st lipgloss.Style, s string) []string {
	lines := strings.Split(s, "\n")
	for i := range lines {
		lines[i] = st.Render(lines[i])
	}
	return lines
}

func (m model) renderV2T() string {
	var parts []string
	// The document is always what matters; a polish in flight is a footer
	// indicator, never a reason to blank the text the speaker just dictated.
	switch {
	case strings.TrimSpace(m.polished) != "":
		parts = append(parts, styledLines(docStyle, m.wrap(m.polished))...)
	case strings.TrimSpace(m.docText()) != "":
		parts = append(parts, styledLines(rawStyle, m.wrap(m.docText()))...)
	case m.polishing || m.polishWait:
		parts = append(parts, dimmedStyle.Render(
			spinnerGlyph(m.frame)+" listening\u2026"))
	default:
		parts = append(parts, dimmedStyle.Render(
			"listening \u2014 speak, then press c to copy, e to edit"))
	}
	// Words committed after the polish was requested have no punctuation yet.
	// Dim them rather than hide them: the speaker needs to see they landed.
	if tail := m.tailText(); tail != "" {
		parts = append(parts, "")
		parts = append(parts, styledLines(tailStyle, m.wrap(tail))...)
	}
	if m.draft != "" {
		parts = append(parts, "")
		parts = append(parts, styledLines(draftStyle, m.wrap(m.draft+"\u258c"))...)
	}
	return strings.Join(parts, "\n")
}

// refresh re-renders the caption area into the viewport, honouring the
// follow (pinned-to-top) state.
//
// It also re-clamps: the footer grows and shrinks as the live "buf N/120"
// gauge appears and disappears, so a viewport sized for the previous footer
// length would overflow the frame and push the header off the top.
func (m *model) refresh() {
	m.clampViewport()
	if m.mode == modeV2T && !m.editing {
		m.viewport.SetContent(m.renderV2T())
	} else {
		m.viewport.SetContent(m.renderContent())
	}
	if m.follow {
		m.viewport.GotoTop()
	}
}

// capRunes hard-caps s at n runes.
func capRunes(s string, n int) string {
	r := []rune(s)
	if len(r) > n {
		return string(r[:n])
	}
	return s
}

// ---- backend plumbing ----

func runBackendReader(prog *tea.Program, stdout *bufio.Reader) {
	sc := bufio.NewScanner(stdout)
	sc.Buffer(make([]byte, 1024*1024), 1024*1024)
	for sc.Scan() {
		var ev backendEvent
		line := strings.TrimSpace(sc.Text())
		if line == "" {
			continue
		}
		if err := json.Unmarshal([]byte(line), &ev); err != nil {
			continue
		}
		switch ev.Type {
		case "final":
			prog.Send(captionMsg{ID: ev.ID, Text: ev.Text, T0: ev.T0})
		case "corrected":
			prog.Send(correctedMsg{ID: ev.ID, Text: ev.Text})
		case "polished":
			prog.Send(polishedMsg{Text: ev.Text})
		case "draft":
			prog.Send(draftMsg{Text: ev.Text, Chunk: ev.N})
		case "flushed":
			prog.Send(flushedMsg{})
		case "reset":
			prog.Send(resetMsg{})
		case "status":
			prog.Send(backendStatusMsg{Msg: ev.Msg})
		case "error":
			prog.Send(backendStatusMsg{Msg: "backend error: " + ev.Msg})
		}
	}
	prog.Send(backendDoneMsg{Err: sc.Err()})
}

// correctionNote records in the transcript header whether an LLM was involved,
// and what it was allowed to rewrite, so a transcript can be read knowing which
// text it holds.
func correctionNote(model string, sentences bool) string {
	if model == "" {
		return ", no LLM"
	}
	if sentences {
		return ", LLM + sentence repair (" + filepath.Base(model) + ")"
	}
	return ", voice-to-text polish (" + filepath.Base(model) + ")"
}

func main() {
	// Subcommands are dispatched before flag parsing: they don't take the
	// live-capture flags, and `earshot setup` must work on a machine where
	// nothing else resolves yet.
	if len(os.Args) > 1 {
		switch os.Args[1] {
		case "setup":
			os.Exit(runSetup(os.Args[2:]))
		case "doctor", "dr":
			os.Exit(runDoctor(os.Args[2:]))
		case "version", "--version", "-v":
			fmt.Println("earshot", version)
			os.Exit(0)
		case "help", "--help", "-h":
			usage()
			os.Exit(0)
		}
	}

	pythonBin := flag.String("python", "", "python with backend deps (auto)")
	backend := flag.String("backend", "", "backend script (auto)")
	modelName := flag.String("model", "small.en", "whisper model")
	modelDir := flag.String("model-dir", "", "model cache dir (default <data>/models)")
	cudaDLLs := flag.String("cuda-dll-dir", "", "CUDA DLL dir prepended to PATH (auto)")
	hfCache := flag.String("hf-cache", "", "HF hub cache dir (default <data>/models/hf-hub)")
	deviceFlag := flag.String("device", "auto", "inference device: auto, cuda or cpu")
	computeFlag := flag.String("compute-type", "", "ctranslate2 compute type (auto)")
	stepSecs := flag.Float64("step-secs", 0.3, "minimum audio per cycle; the effective step adapts to measured inference cost")
	stepHeadroom := flag.Float64("step-headroom", 1.25, "multiplier on the median pass time used to size each cycle")
	maxStepSecs := flag.Float64("max-step-secs", 2.5, "hard ceiling on the adaptive step, so one slow pass cannot freeze captions")
	tempFallback := flag.String("temp-fallback", "off", "temperature fallback: off (default), short or full. A segment failing Whisper's confidence checks is otherwise retried at up to six temperatures, which freezes the display")
	saveAudio := flag.Bool("save-audio", true, "save the captured audio for the newest few sessions, so a stall can be replayed")
	maxBacklog := flag.Float64("max-backlog", 1.5, "seconds of audio allowed to queue before the oldest is dropped")
	maxBufSecs := flag.Float64("max-buf-secs", 12, "rolling buffer cap before the unstable tail is flushed")
	pauseSecs := flag.Float64("pause-secs", 0.2, "trailing silence that counts as a pause and commits the tail")
	beamSize := flag.Int("beam-size", 1, "beam width (1 = greedy: cheaper on call-quality hot spans; see docs/latency-measurements.md)")
	blockChars := flag.Int("block-chars", 200, "sentence block size")
	chainChars := flag.Int("chain-chars", 15, "trailing runes chained into next line")
	correctModel := flag.String("correct-model", "", "path to a GGUF model; each committed sentence is polished by an LLM")
	correctThreads := flag.Int("correct-threads", 0, "threads for the correction model (0 = auto)")
	correctSentences := flag.Bool("correct-sentences", false, "also repair each committed sentence as it lands (voice-to-text polishes the whole document instead)")
	v2t := flag.Bool("v2t", false, "start in voice-to-text mode: polished, copyable dictation")
	outDir := flag.String("out-dir", "", "transcript output dir (default <data>/transcripts)")
	keep := flag.Int("keep", 30, "max transcript sessions to retain (0 = keep none)")
	noPrune := flag.Bool("no-prune", false, "never delete old transcripts")
	splashStyle := flag.String("splash-style", styleShimmer,
		"loading/pause banner animation: shimmer (default) or rainbow")
	flag.Parse()

	py, err := resolvePython(*pythonBin)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	be, err := resolveBackend(*backend)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	cuda := resolveCUDADir(*cudaDLLs)
	models := *modelDir
	if models == "" {
		models = dataSub("models")
	}
	cache := *hfCache
	if cache == "" {
		cache = dataSub("models", "hf-hub")
	}
	transcripts := *outDir
	if transcripts == "" {
		transcripts = dataSub("transcripts")
	}
	device, compute, note := deviceFor(*deviceFlag, *computeFlag, cuda.Path != "" || cudaDLLsReady())
	if note != "" {
		fmt.Fprintln(os.Stderr, "note: "+note)
	}

	if err := os.MkdirAll(transcripts, 0o755); err != nil {
		fmt.Fprintln(os.Stderr, "transcripts dir:", err)
		os.Exit(1)
	}
	// Retention: prune before opening this session so the directory is
	// self-healing even if a previous run was killed.
	if !*noPrune {
		if n, err := pruneTranscripts(*outDir, *keep); err != nil {
			fmt.Fprintf(os.Stderr, "prune: %v\n", err)
		} else if n > 0 {
			fmt.Printf("pruned %d old transcript(s), keeping %d\n", n, *keep)
		}
		if n, err := pruneAudio(*outDir, audioKeep); err != nil {
			fmt.Fprintf(os.Stderr, "prune audio: %v\n", err)
		} else if n > 0 {
			fmt.Printf("pruned %d old recording(s), keeping %d\n", n, audioKeep)
		}
	}
	if *stepSecs <= 0 {
		fmt.Fprintf(os.Stderr, "--step-secs must be positive (got %.2f)\n", *stepSecs)
		os.Exit(2)
	}
	if *maxBufSecs < *stepSecs {
		fmt.Fprintf(os.Stderr, "--max-buf-secs must be >= --step-secs (got %.2f < %.2f)\n", *maxBufSecs, *stepSecs)
		os.Exit(2)
	}
	if *maxStepSecs < *stepSecs {
		fmt.Fprintf(os.Stderr, "--max-step-secs must be >= --step-secs (got %.2f < %.2f)\n", *maxStepSecs, *stepSecs)
		os.Exit(2)
	}

	stamp := time.Now().Format("2006-01-02_1504")
	outPath := filepath.Join(transcripts, stamp+".md")
	f, err := os.Create(outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "create transcript:", err)
		os.Exit(1)
	}
	build := buildStamp()
	// Voice-to-text needs an LLM, and the mode toggle is a runtime control, so
	// the model is resolved up front: switching into v2t mid-session must not
	// find itself without one. Per-sentence repair stays opt-in.
	polishModel := *correctModel
	if polishModel == "" {
		polishModel = findPolishModel(models)
	}
	// Captions are appended after the header; corrections rewrite byte spans
	// in place, so seed the running offset past the header.
	headerPos, _ := f.Seek(0, io.SeekCurrent)
	fmt.Fprintf(f, "# Earshot %s (build %s)\n\nmodel: %s (%s/%s), step %.2fs, buffer cap %.1fs, beam %d, temp-fallback %s, block %d+%d%s\n\n",
		time.Now().Format("2006-01-02 15:04"), build, *modelName, device, compute,
		*stepSecs, *maxBufSecs, *beamSize, *tempFallback, *blockChars, *chainChars, correctionNote(polishModel, *correctSentences))

	// The backend opens one capture device; the TUI switches it over stdin
	// whenever the mode changes. Dictation must hear the microphone, never the
	// speakers' loopback.
	inputSource := "loopback"
	if *v2t {
		inputSource = "mic"
	}

	cmd := exec.Command(py.Path, "-u", be.Path, "--live",
		"--model", *modelName, "--model-dir", models,
		"--device", device, "--compute-type", compute,
		"--input", inputSource,
		"--step-secs", fmt.Sprint(*stepSecs),
		"--step-headroom", fmt.Sprint(*stepHeadroom),
		"--max-step-secs", fmt.Sprint(*maxStepSecs),
		"--max-backlog", fmt.Sprint(*maxBacklog),
		"--max-buf-secs", fmt.Sprint(*maxBufSecs),
		"--pause-secs", fmt.Sprint(*pauseSecs),
		"--beam-size", fmt.Sprint(*beamSize),
		"--block-chars", fmt.Sprint(*blockChars),
		"--chain-chars", fmt.Sprint(*chainChars),
		"--latency-csv", filepath.Join(transcripts, stamp+latencyCSVSuffix),
		"--temp-fallback", *tempFallback)
	if *saveAudio {
		cmd.Args = append(cmd.Args, "--dump-audio",
			filepath.Join(transcripts, stamp+audioSuffix))
	}
	if polishModel != "" {
		cmd.Args = append(cmd.Args, "--correct-model", polishModel,
			"--correct-threads", fmt.Sprint(*correctThreads))
		if *correctSentences {
			cmd.Args = append(cmd.Args, "--correct-sentences")
		}
	}
	env := os.Environ()
	// Only touch PATH when we actually have a directory to add; if the CUDA
	// DLLs are already on PATH, leave it alone.
	if cuda.Path != "" {
		env = append(env, "PATH="+cuda.Path+string(os.PathListSeparator)+os.Getenv("PATH"))
	}
	env = append(env, "HF_HUB_CACHE="+cache)
	// Resampling needs no BLAS threads; pinning to 1 avoids OpenBLAS spinning
	// up a 12-thread pool per backend (oversubscribes the 6-core/12-thread CPU).
	env = append(env, "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1")
	cmd.Env = env
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		fmt.Fprintln(os.Stderr, "stdout pipe:", err)
		os.Exit(1)
	}
	stdin, err := cmd.StdinPipe()
	if err != nil {
		fmt.Fprintln(os.Stderr, "stdin pipe:", err)
		os.Exit(1)
	}
	logF, err := os.Create(filepath.Join(transcripts, stamp+".backend.log"))
	if err != nil {
		fmt.Fprintln(os.Stderr, "backend log:", err)
		os.Exit(1)
	}
	defer logF.Close()
	cmd.Stderr = logF
	if err := cmd.Start(); err != nil {
		fmt.Fprintln(os.Stderr, "start backend:", err)
		os.Exit(1)
	}

	m := model{
		lines:       []string{},
		follow:      true,
		status:      "starting backend…",
		started:     time.Now(),
		build:       build,
		blockChars:  *blockChars,
		outPos:      headerPos,
		splashStyle: *splashStyle,
		outFile:     f,
		outPath:     outPath,
		backendIn:   stdin,
		hasPolish:   polishModel != "",
	}
	if *v2t {
		m.mode = modeV2T
		m.sawV2T = true
		m.status = "voice to text"
		if polishModel == "" {
			m.status = "voice to text (no polish model — raw text only)"
		}
	}
	prog := tea.NewProgram(m, tea.WithAltScreen(), tea.WithMouseCellMotion())
	go runBackendReader(prog, bufio.NewReader(stdout))
	finalModel, err := prog.Run()
	if err != nil {
		fmt.Fprintln(os.Stderr, "tui:", err)
	}

	// Graceful stop: ask the backend to flush its remainder ("quit" on stdin),
	// fall back to kill after 2s.
	fmt.Fprintln(stdin, "quit")
	done := make(chan struct{})
	go func() { cmd.Wait(); close(done) }()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		if cmd.Process != nil {
			cmd.Process.Kill()
		}
		<-done
	}
	nSaved := 0
	if fm, ok := finalModel.(model); ok {
		nSaved = len(fm.lines)
		// The dictation text is the point of v2t, so it gets its own plain
		// file: no timestamps, nothing to strip before pasting. Only sessions
		// that actually used the mode produce one.
		if fm.sawV2T {
			if txt := fm.finalText(); strings.TrimSpace(txt) != "" {
				txtPath := filepath.Join(transcripts, stamp+textSuffix)
				if err := os.WriteFile(txtPath, []byte(txt+"\n"), 0o644); err != nil {
					fmt.Fprintf(os.Stderr, "save text: %v\n", err)
				} else {
					fmt.Printf("saved text to %s\n", txtPath)
				}
			}
		}
	}
	fmt.Fprintf(f, "\n(session ended %s, %d captions)\n",
		time.Now().Format("15:04"), nSaved)
	f.Close()

	// ...and again on exit, so long sessions don't accumulate past the cap.
	if !*noPrune {
		if n, err := pruneTranscripts(*outDir, *keep); err != nil {
			fmt.Fprintf(os.Stderr, "prune: %v\n", err)
		} else if n > 0 {
			fmt.Printf("pruned %d old transcript(s), keeping %d\n", n, *keep)
		}
		// Audio is pruned *after* the session ends, so this session's own
		// recording is eligible (keep = 5 means the 6th-oldest goes).
		if n, err := pruneAudio(*outDir, audioKeep); err != nil {
			fmt.Fprintf(os.Stderr, "prune audio: %v\n", err)
		} else if n > 0 {
			fmt.Printf("pruned %d old recording(s), keeping %d\n", n, audioKeep)
		}
	}
	fmt.Printf("saved %d captions to %s\n", nSaved, outPath)
}
