package main

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"testing"
	"time"

	tea "github.com/charmbracelet/bubbletea"
	"github.com/charmbracelet/lipgloss"
)

func runeKey(r rune) tea.KeyMsg {
	return tea.KeyMsg{Type: tea.KeyRunes, Runes: []rune{r}}
}

// stubClipboard replaces the clipboard for one test and restores it after.
func stubClipboard(t *testing.T, sink *string) {
	t.Helper()
	old := clipboardWrite
	clipboardWrite = func(s string) error {
		*sink = s
		return nil
	}
	t.Cleanup(func() { clipboardWrite = old })
}

func TestTabTogglesModeWithoutRestarting(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.archived = []archiveEntry{{id: 1, text: "spoken words"}}

	next, _ := m.Update(tea.KeyMsg{Type: tea.KeyTab})
	m = next.(model)
	if m.mode != modeV2T {
		t.Fatalf("mode = %v, want v2t", m.mode)
	}
	body := stripANSI(m.viewport.View())
	if strings.Contains(body, "- [") {
		t.Errorf("v2t pane still shows timestamps: %q", body)
	}
	if !strings.Contains(body, "spoken words") {
		t.Errorf("v2t pane lost the text: %q", body)
	}

	next, _ = m.Update(tea.KeyMsg{Type: tea.KeyTab})
	if got := next.(model).mode; got != modeLive {
		t.Fatalf("second tab: mode = %v, want live", got)
	}
}

func TestHeaderFillsTheRowAndCarriesBothControls(t *testing.T) {
	m := newTestModel(100, 20, nil)
	head := stripANSI(m.headerLine())
	if w := lipgloss.Width(head); w != 100 {
		t.Errorf("header width = %d, want 100", w)
	}
	for _, want := range []string{"earshot", "Live Captions", "Voice to Text", "Copy"} {
		if !strings.Contains(head, want) {
			t.Errorf("header missing %q: %q", want, head)
		}
	}
}

func TestHeaderHitsMatchRenderedLayout(t *testing.T) {
	m := newTestModel(100, 20, nil)
	h := m.headerHits()
	head := []rune(stripANSI(m.headerLine()))
	if h.copyStart <= 0 || h.copyEnd > len(head) {
		t.Fatalf("copy range %d..%d outside %d", h.copyStart, h.copyEnd, len(head))
	}
	if got := strings.TrimSpace(string(head[h.copyStart:h.copyEnd])); got != "Copy" {
		t.Errorf("copy hit region = %q", got)
	}
	if h.toggleMid <= h.toggleStart || h.toggleMid >= h.toggleEnd {
		t.Errorf("bad toggle midpoint %d in %d..%d", h.toggleMid, h.toggleStart, h.toggleEnd)
	}
	if h.toggleEnd > len(head) {
		t.Errorf("toggle end %d past row %d", h.toggleEnd, len(head))
	}
}

func TestNarrowHeaderStillFits(t *testing.T) {
	m := newTestModel(40, 20, nil)
	head := stripANSI(m.headerLine())
	if w := lipgloss.Width(head); w > 40 {
		t.Errorf("header width = %d, overflows 40: %q", w, head)
	}
	if !strings.Contains(head, "Voice") {
		t.Errorf("narrow header dropped the mode switch: %q", head)
	}
}

func TestMouseClickOnSwitchChangesMode(t *testing.T) {
	m := newTestModel(100, 20, nil)
	h := m.headerHits()
	click := tea.MouseMsg{X: h.toggleStart, Y: 0,
		Action: tea.MouseActionPress, Button: tea.MouseButtonLeft}
	next, _ := m.Update(click)
	if got := next.(model).mode; got != modeLive {
		t.Errorf("left half should pick live, got %v", got)
	}

	m = newTestModel(100, 20, nil)
	h = m.headerHits()
	click.X = h.toggleEnd - 1
	next, _ = m.Update(click)
	if got := next.(model).mode; got != modeV2T {
		t.Errorf("right half should pick v2t, got %v", got)
	}
}

func TestCopyPrefersPolishedText(t *testing.T) {
	var got string
	stubClipboard(t, &got)

	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "raw words"}}
	m.polished = "Polished words."
	m.polishedAt = 1
	m.copyDocument()

	if got != "Polished words." {
		t.Errorf("copied %q, want the polished text", got)
	}
	if !strings.Contains(m.status, "copied") {
		t.Errorf("status = %q, want a copy confirmation", m.status)
	}
}

func TestCopyFallsBackToRawText(t *testing.T) {
	var got string
	stubClipboard(t, &got)

	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "raw words"}}
	m.copyDocument()

	if got != "raw words" {
		t.Errorf("copied %q, want raw text", got)
	}
	if !strings.Contains(m.status, "raw") {
		t.Errorf("status = %q, want it to say the copy was raw", m.status)
	}
}

func TestCopyWithNothingSaysSo(t *testing.T) {
	called := false
	old := clipboardWrite
	clipboardWrite = func(string) error { called = true; return nil }
	t.Cleanup(func() { clipboardWrite = old })

	m := newTestModel(100, 20, nil)
	m.copyDocument()
	if called {
		t.Error("clipboard was written with no text")
	}
	if !strings.Contains(m.status, "nothing") {
		t.Errorf("status = %q", m.status)
	}
}

func TestPolishWaitsForAQuietDocument(t *testing.T) {
	var buf bytes.Buffer
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.backendIn = &buf
	m.archived = []archiveEntry{{id: 1, text: "hello there"}}
	m.polishWait = true
	m.polishAt = time.Now()

	m.maybePolish()
	if buf.Len() != 0 {
		t.Errorf("polish requested mid-sentence: %q", buf.String())
	}

	m.polishAt = time.Now().Add(-2 * polishDebounce)
	m.maybePolish()
	if !strings.Contains(buf.String(), `"cmd":"polish"`) {
		t.Errorf("no polish request: %q", buf.String())
	}
	if !strings.Contains(buf.String(), "hello there") {
		t.Errorf("request lost the document: %q", buf.String())
	}
	if !m.polishing {
		t.Error("polishing flag not set")
	}
	if m.polishWait {
		t.Error("polishWait should clear once a request is in flight")
	}
}

func TestEditModeOwnsTheKeyboard(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.polished = "hello"

	next, _ := m.Update(runeKey('e'))
	m = next.(model)
	if !m.editing {
		t.Fatal("e did not enter edit mode")
	}

	// `q` must type, not quit.
	next, _ = m.Update(runeKey('q'))
	m = next.(model)
	if !m.editing {
		t.Fatal("typing q left edit mode")
	}
	if !strings.Contains(m.editor.Value(), "q") {
		t.Errorf("typed q was dropped: %q", m.editor.Value())
	}

	next, _ = m.Update(tea.KeyMsg{Type: tea.KeyEsc})
	m = next.(model)
	if m.editing {
		t.Fatal("esc did not leave edit mode")
	}
	if m.docBase != strings.TrimSpace(m.editor.Value()) {
		t.Errorf("docBase = %q, want the typed text", m.docBase)
	}
	if m.polished != "" {
		t.Error("edited text must invalidate the old polish")
	}
}

func TestEditedTextSupersedesEarlierSpeech(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.archived = []archiveEntry{
		{id: 1, text: "first"},
		{id: 2, text: "second"},
	}
	if got := m.docText(); got != "first second" {
		t.Fatalf("docText = %q", got)
	}

	m.docBase = "rewritten"
	m.docBaseAt = 2
	if got := m.docText(); got != "rewritten" {
		t.Errorf("after edit docText = %q, want %q", got, "rewritten")
	}

	// Speech after the edit appends; speech during it is superseded.
	m.archived = append(m.archived, archiveEntry{id: 3, text: "third"})
	if got := m.docText(); got != "rewritten third" {
		t.Errorf("docText = %q, want %q", got, "rewritten third")
	}
}

func TestFinalTextPrefersPolished(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.archived = []archiveEntry{{id: 1, text: "raw"}}
	if got := m.finalText(); got != "raw" {
		t.Errorf("raw finalText = %q", got)
	}
	m.polished = "Polished."
	m.polishedAt = 1
	if got := m.finalText(); got != "Polished." {
		t.Errorf("finalText = %q, want polished", got)
	}
}

// A polish covers only the speech that existed when it was requested. Copying
// the polish alone would silently drop the last thing the speaker said.
func TestCopyAppendsSpeechThePolishMissed(t *testing.T) {
	var got string
	stubClipboard(t, &got)

	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{
		{id: 1, text: "first sentence"},
		{id: 2, text: "second sentence"},
	}
	m.polished = "First sentence."
	m.polishedAt = 1

	m.copyDocument()
	if got != "First sentence. second sentence" {
		t.Errorf("copied %q, want the polish plus the uncovered tail", got)
	}
	if !strings.Contains(m.status, "new speech") {
		t.Errorf("status = %q, want it to flag the unpolished tail", m.status)
	}
}

func TestUnpolishedTailIsDimmedNotHidden(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{
		{id: 1, text: "polished bit"},
		{id: 2, text: "brand new words"},
	}
	m.polished = "Polished bit."
	m.polishedAt = 1

	body := stripANSI(m.renderV2T())
	if !strings.Contains(body, "Polished bit.") {
		t.Errorf("polished text missing: %q", body)
	}
	if !strings.Contains(body, "brand new words") {
		t.Errorf("unpolished tail hidden: %q", body)
	}
}

func TestPolishedResponseMarksCoverage(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "a"}, {id: 2, text: "b"}}
	m.polishing = true
	m.polishReqAt = 2

	next, _ := m.Update(polishedMsg{Text: "A. B."})
	m = next.(model)
	if m.polished != "A. B." {
		t.Errorf("polished = %q", m.polished)
	}
	if m.polishing {
		t.Error("still marked polishing after a response")
	}
	if m.polishedAt != 2 {
		t.Errorf("coverage = %d, want 2", m.polishedAt)
	}
}

func TestEmptyPolishDoesNotBlankThePane(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.polished = "keep me"
	next, _ := m.Update(polishedMsg{Text: "   "})
	if got := next.(model).polished; got != "keep me" {
		t.Errorf("polished = %q, want the previous text", got)
	}
}

func TestPauseDoesNotStrandThePolishFlag(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.paused = true
	m.polishing = true

	next, _ := m.Update(polishedMsg{Text: "Polished."})
	if next.(model).polishing {
		t.Error("a response while paused must clear the in-flight flag")
	}
}

func TestModeSwitchAsksTheBackendForTheMatchingSource(t *testing.T) {
	cases := []struct {
		name string
		from mode
		to   mode
		want string
	}{
		{"dictation listens to the mic", modeLive, modeV2T, "mic"},
		{"captions listen to the loopback", modeV2T, modeLive, "loopback"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			m := newTestModel(100, 20, nil)
			m.mode = tc.from
			pr, pw := io.Pipe()
			defer pw.Close()
			m.backendIn = pw

			got := make(chan string, 1)
			go func() {
				buf := make([]byte, 256)
				n, _ := pr.Read(buf)
				got <- string(buf[:n])
			}()

			m.setMode(tc.to)

			select {
			case line := <-got:
				var cmd map[string]string
				if err := json.Unmarshal([]byte(strings.TrimSpace(line)), &cmd); err != nil {
					t.Fatalf("backend received %q: %v", line, err)
				}
				if cmd["cmd"] != "source" || cmd["kind"] != tc.want {
					t.Errorf("sent %v, want source=%s", cmd, tc.want)
				}
			case <-time.After(2 * time.Second):
				t.Fatal("no source command reached the backend")
			}
		})
	}
}

func TestEditOpensAtOnceWithEverythingOnScreen(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.draft = "half a thought"
	m.archived = []archiveEntry{{id: 1, text: "committed words"}}
	var sent strings.Builder
	m.backendIn = &sent

	next, _ := m.Update(runeKey('e'))
	m = next.(model)

	if !m.editing {
		t.Fatal("editor should open without waiting for the flush")
	}
	if !strings.Contains(sent.String(), `"cmd":"flush"`) {
		t.Fatalf("the pending sentence was not sent for commit: %q", sent.String())
	}
	got := m.editor.Value()
	if !strings.Contains(got, "committed words") || !strings.Contains(got, "half a thought") {
		t.Fatalf("editor seeded %q, want the whole visible text", got)
	}
}

func TestEditOpensImmediatelyWhenNothingIsPending(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "already committed"}}

	next, _ := m.Update(runeKey('e'))
	m = next.(model)
	if !m.editing {
		t.Fatal("editor should open without waiting for a flush")
	}
}

func TestCopyIncludesTheLiveSentence(t *testing.T) {
	var got string
	old := clipboardWrite
	clipboardWrite = func(s string) error { got = s; return nil }
	t.Cleanup(func() { clipboardWrite = old })

	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "committed words"}}
	m.draft = "still speaking"

	m.copyDocument()
	if !strings.Contains(got, "still speaking") {
		t.Errorf("copy dropped the live sentence: %q", got)
	}
	if !strings.Contains(m.status, "live speech") {
		t.Errorf("status = %q, want it to mention live speech", m.status)
	}
}

func TestPolishingNeverHidesTheDocument(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "the document"}}
	m.polishing = true

	view := stripANSI(m.renderV2T())
	if !strings.Contains(view, "the document") {
		t.Errorf("polishing blanked the pane: %q", view)
	}
}

func TestHeaderRuleMatchesTheWidth(t *testing.T) {
	for _, w := range []int{40, 80, 120} {
		m := newTestModel(w, 20, nil)
		if got := lipgloss.Width(m.headerRule()); got != w {
			t.Errorf("width %d: rule is %d cells", w, got)
		}
	}
}

func TestFlushReplyDoesNotReseedAnOpenEditor(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "committed words"}}
	m.draft = "pending words"
	m.backendIn = io.Discard

	m.requestEdit()
	m.editor.SetValue("my own edit")

	next, _ := m.Update(flushedMsg{})
	if got := next.(model).editor.Value(); got != "my own edit" {
		t.Fatalf("the flush reply replaced the user's text with %q", got)
	}
}

func TestClearOnlyWipesTheActiveModesText(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "document"}}
	m.lines = []string{"- [00:00] caption"}
	m.polished = "Polished document."
	m.polishedAt = 1
	m.docBase = "edited text"
	m.docBaseAt = 1

	m.clearDocument()

	if got := strings.TrimSpace(m.docText()); got != "" {
		t.Fatalf("v2t document still has text: %q", got)
	}
	if m.polished != "" || m.docBase != "" || m.polishedAt != 0 || m.docBaseAt != 0 {
		t.Fatalf("clear left the v2t document behind: %+v", m)
	}
	if len(m.lines) != 1 {
		t.Fatalf("clearing dictation wiped the captions: %v", m.lines)
	}

	live := newTestModel(100, 20, nil)
	live.mode = modeLive
	live.lines = []string{"- [00:00] caption"}
	live.lineIDs = []int{1}
	live.archived = []archiveEntry{{id: 1, text: "document"}}
	live.clearDocument()
	if len(live.lines) != 0 || live.nCaptions != 0 {
		t.Fatalf("clearing captions left text behind: %v", live.lines)
	}
	if live.docText() == "" {
		t.Fatal("clearing captions wiped the dictation document")
	}
}

func TestClearAsksTheBackendToForgetThePendingSentence(t *testing.T) {
	m := newTestModel(100, 20, nil)
	var buf strings.Builder
	m.backendIn = &buf

	m.clearDocument()

	if !strings.Contains(buf.String(), `"cmd":"reset"`) {
		t.Fatalf("clear did not tell the backend: %q", buf.String())
	}
}

func TestStalePolishCannotReturnAfterClear(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.hasPolish = true
	m.polishEpoch = m.docEpoch

	m.clearDocument()

	next, _ := m.Update(polishedMsg{Text: "words the user already cleared"})
	if got := next.(model).polished; got != "" {
		t.Fatalf("clear was undone by an in-flight polish: %q", got)
	}
}

func TestEditorSeedsFromTheTranscript(t *testing.T) {
	m := newTestModel(100, 20, []string{"- [00:05] hello there"})
	m.archived = []archiveEntry{{id: 1, text: "hello there"}}
	m.mode = modeV2T
	m.backendIn = io.Discard

	m.requestEdit()
	got := m
	if !got.editing {
		t.Fatal("editor did not open")
	}
	if v := got.editor.Value(); v != "hello there" {
		t.Fatalf("editor seeded %q, want the transcript", v)
	}
	if plain := stripANSI(got.View()); !strings.Contains(plain, "hello there") {
		t.Fatalf("editor view does not show the text: %s", plain)
	}
}

func TestModesKeepTheirOwnText(t *testing.T) {
	m := newTestModel(100, 20, nil)

	next, _ := m.Update(captionMsg{ID: 1, Text: "live words", T0: 1})
	m = next.(model)
	if len(m.lines) != 1 || len(m.archived) != 0 {
		t.Fatalf("live caption did not stay in its mode: %d lines, %d doc entries",
			len(m.lines), len(m.archived))
	}

	m.setMode(modeV2T)
	if got := m.docText(); got != "" {
		t.Fatalf("voice to text inherited live captions: %q", got)
	}

	next, _ = m.Update(captionMsg{ID: 2, Text: "dictated words", T0: 2})
	m = next.(model)
	if got := m.docText(); got != "dictated words" {
		t.Fatalf("document = %q", got)
	}
	if len(m.lines) != 1 {
		t.Fatalf("dictation added to the captions: %v", m.lines)
	}

	m.setMode(modeLive)
	if got := m.liveText(); got != "live words" {
		t.Fatalf("captions = %q, want only the caption-mode speech", got)
	}
}

func TestEditKeepsTheSentenceThatHasNotCommittedYet(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.backendIn = nil // no flush can be sent, so the draft is all there is
	m.draft = "the half sentence"

	m.requestEdit()

	if !m.editing {
		t.Fatal("editor did not open")
	}
	if got := m.editor.Value(); got != "the half sentence" {
		t.Fatalf("editor seeded %q, want the live sentence", got)
	}
}

func TestCopyInLiveModeCopiesTheCaptions(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeLive
	m.lines = []string{"- [00:02] second line", "- [00:01] first line"}

	var copied string
	restore := clipboardWrite
	clipboardWrite = func(s string) error { copied = s; return nil }
	defer func() { clipboardWrite = restore }()

	m.copyDocument()

	if copied != "first line second line" {
		t.Fatalf("clipboard = %q", copied)
	}
}

func TestEditingLiveCaptionsKeepsAndReplacesTheirText(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeLive
	m.lines = []string{"- [00:02] hello", "- [00:01] world"} // newest first
	m.lineT0 = []float64{2, 1}

	m.requestEdit()
	if got := m.editor.Value(); got != "world hello" {
		t.Fatalf("editor seeded %q, want the captions in speaking order", got)
	}
	m.editor.SetValue("edited text")
	m.endEdit()

	if got := m.liveText(); got != "edited text" {
		t.Fatalf("captions after edit = %q", got)
	}
	if m.docBase != "" {
		t.Fatalf("a live edit leaked into the dictation document: %q", m.docBase)
	}
}

func TestLateFlushSentenceDoesNotDuplicateTheEditedText(t *testing.T) {
	m := newTestModel(100, 20, nil)
	m.mode = modeV2T
	m.archived = []archiveEntry{{id: 1, text: "first sentence"}}
	m.draft = "second sentence"
	m.backendIn = io.Discard

	m.requestEdit()
	m.endEdit() // the user finishes before the flush reply lands

	next, _ := m.Update(captionMsg{ID: 2, Text: "second sentence", T0: 2})
	m = next.(model)

	if got := m.docText(); strings.Count(got, "second sentence") != 1 {
		t.Fatalf("document repeats the flushed sentence: %q", got)
	}
}
