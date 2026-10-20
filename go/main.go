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
	"strings"
	"time"

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
	titleStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("12"))
	statusStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
	liveStyle     = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("10"))
	pauseStyle    = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("11"))
	errStyle      = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("9"))
	liveLineStyle = lipgloss.NewStyle().Bold(true).Foreground(lipgloss.Color("15"))
	dimmedStyle   = lipgloss.NewStyle().Foreground(lipgloss.Color("8"))
)

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
}

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
	lineIDs     []int     // block id per on-screen line (newest-first)
	lineT0      []float64 // timestamp per on-screen line
	archived    []archiveEntry
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
		m.refresh()
		return m, nil

	case captionMsg:
		if m.paused {
			return m, nil // UI-only pause: freeze display, stop saving
		}
		m.nCaptions++
		m.draft = ""
		// File keeps the full chronological record; the screen is a
		// newest-first stack (backend already blocks + chains to 200).
		fmt.Fprintf(m.outFile, "- [%s] %s\n", fmtClock(msg.T0), msg.Text)
		line := fmt.Sprintf("- [%s] %s", fmtClock(msg.T0), capRunes(msg.Text, maxLineRunes))
		m.lines = append([]string{line}, m.lines...)
		m.lineIDs = append([]int{msg.ID}, m.lineIDs...)
		m.lineT0 = append([]float64{msg.T0}, m.lineT0...)
		// Record the byte span this caption occupies so a later correction can
		// rewrite it in place. Captions are written oldest-first, so the span
		// is captured *after* the write above.
		if end, err := m.outFile.Seek(0, io.SeekCurrent); err == nil {
			entry := archiveEntry{id: msg.ID, t0: msg.T0}
			entry.bytes = int(end - m.outPos)
			entry.pos = m.outPos
			m.archived = append(m.archived, entry)
			m.outPos = end
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
		for _, a := range m.archived {
			if a.id != msg.ID {
				continue
			}
			old := a.bytes
			newText := fmt.Sprintf("- [%s] %s", fmtClock(a.t0), msg.Text)
			if len(newText) > old {
				newText = newText[:old]
			}
			padded := newText + strings.Repeat(" ", old-len(newText))
			m.outFile.WriteAt([]byte(padded), a.pos)
			break
		}
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
		return m, tickCmd()

	case tea.KeyMsg:
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
			parts = append(parts, m.newestStyle().Render(ln))
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
					parts = append(parts, style.Render(ln))
				}
			}
		}
	}
	return strings.Join(parts, "\n")
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
	return lipgloss.NewStyle()
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
// blank, plus the 5-row mini banner while paused.
func (m model) headerHeight() int {
	if m.paused {
		return 2 + len(splashArt) + 1
	}
	return 2
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
	state := liveStyle.Render("● LIVE")
	if m.paused {
		state = pauseStyle.Render("❚❚ PAUSED")
	} else if !m.backendUp {
		state = errStyle.Render("○ backend down")
	}
	title := titleStyle.Render("earshot") + "  " + state

	var head strings.Builder
	head.WriteString(title)
	if m.paused {
		head.WriteString("\n")
		head.WriteString(miniBanner(m.frame, m.splashStyle))
	}
	head.WriteString("\n\n")

	body := m.viewport.View()
	return head.String() + body + "\n" + statusStyle.Render(m.footerText())
}

// footerText is the status line, unstyled, so its wrapped width can be measured.
func (m model) footerText() string {
	elapsed := time.Since(m.started).Round(time.Second)
	// Live buffer gauge: makes the commit gate legible ("87/120") so a
	// sentence that hasn't landed yet reads as filling, not as stalled.
	gauge := ""
	if n := len([]rune(m.draft)); n > 0 {
		gauge = fmt.Sprintf("  buf %d/%d", n, m.blockChars)
	}
	return fmt.Sprintf("%s  %d captions, newest first  chunk %d%s  %s  [space] pause  [q] quit  %s  (build %s)",
		elapsed, m.nCaptions, m.chunk, gauge, m.status, m.outPath, m.build)
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

// refresh re-renders the caption area into the viewport, honouring the
// follow (pinned-to-top) state.
//
// It also re-clamps: the footer grows and shrinks as the live "buf N/120"
// gauge appears and disappears, so a viewport sized for the previous footer
// length would overflow the frame and push the header off the top.
func (m *model) refresh() {
	m.clampViewport()
	m.viewport.SetContent(m.renderContent())
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
		case "draft":
			prog.Send(draftMsg{Text: ev.Text, Chunk: ev.N})
		case "status":
			prog.Send(backendStatusMsg{Msg: ev.Msg})
		case "error":
			prog.Send(backendStatusMsg{Msg: "backend error: " + ev.Msg})
		}
	}
	prog.Send(backendDoneMsg{Err: sc.Err()})
}

// correctionNote records in the transcript header whether the LLM pass ran,
// so a transcript can be read knowing which text it holds.
func correctionNote(model string) string {
	if model == "" {
		return ", correction off"
	}
	return ", correction on (" + filepath.Base(model) + ")"
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
	beamSize := flag.Int("beam-size", 5, "beam width (1 = greedy, cannot recover a bad early token)")
	blockChars := flag.Int("block-chars", 200, "sentence block size")
	chainChars := flag.Int("chain-chars", 15, "trailing runes chained into next line")
	correctModel := flag.String("correct-model", "", "path to a GGUF model; each committed sentence is polished by an LLM")
	correctThreads := flag.Int("correct-threads", 0, "threads for the correction model (0 = auto)")
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
	// Captions are appended after the header; corrections rewrite byte spans
	// in place, so seed the running offset past the header.
	headerPos, _ := f.Seek(0, io.SeekCurrent)
	fmt.Fprintf(f, "# Earshot %s (build %s)\n\nmodel: %s (%s/%s), step %.2fs, buffer cap %.1fs, beam %d, temp-fallback %s, block %d+%d%s\n\n",
		time.Now().Format("2006-01-02 15:04"), build, *modelName, device, compute,
		*stepSecs, *maxBufSecs, *beamSize, *tempFallback, *blockChars, *chainChars, correctionNote(*correctModel))

	cmd := exec.Command(py.Path, "-u", be.Path, "--live",
		"--model", *modelName, "--model-dir", models,
		"--device", device, "--compute-type", compute,
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
	if *correctModel != "" {
		cmd.Args = append(cmd.Args, "--correct-model", *correctModel,
			"--correct-threads", fmt.Sprint(*correctThreads))
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
	}
	prog := tea.NewProgram(m, tea.WithAltScreen())
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
