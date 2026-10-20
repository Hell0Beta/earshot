package main

import (
	"fmt"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"strings"
)

// sessionStamp matches the transcript filename format this app writes:
// 2026-09-26_1434.md  (and its paired 2026-09-26_1434.backend.log)
// Nothing else in the directory is ever considered, so notes, .gitkeep and any
// unrelated files are safe.
var sessionStamp = regexp.MustCompile(`^\d{4}-\d{2}-\d{2}_\d{4}\.md$`)

const backendLogSuffix = ".backend.log"

// latencyCSVSuffix pairs with a session like the backend log does. It is
// pruned alongside the .md so measurement files cannot accumulate forever.
const latencyCSVSuffix = ".latency.csv"

// audioSuffix pairs with a session too, but audio is by far the largest
// artifact, so it gets its own (much smaller) retention: only the newest few
// sessions are kept, so a stall can still be replayed.
const audioSuffix = ".audio.wav"

// audioKeep is how many sessions' audio survive. Deliberately tiny.
const audioKeep = 5

// pruneAudio keeps the newest `keep` session recordings and deletes older
// ones. Audio is pruned far more aggressively than transcripts: it is the
// biggest file by an order of magnitude and its only job is to let the newest
// few sessions be replayed. Only names matching the app's stamp format count,
// and existence of a paired .md is *not* required -- otherwise audio whose
// transcript was already pruned would leak forever.
func pruneAudio(dir string, keep int) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, fmt.Errorf("read audio dir: %w", err)
	}

	var stems []string
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		if !strings.HasSuffix(name, audioSuffix) {
			continue
		}
		stem := strings.TrimSuffix(name, audioSuffix)
		if sessionStamp.MatchString(stem + ".md") {
			stems = append(stems, stem)
		}
	}

	// Lexicographic order == chronological for this stamp format.
	sort.Sort(sort.Reverse(sort.StringSlice(stems)))

	removed := 0
	for i, stem := range stems {
		if i < keep {
			continue
		}
		if err := os.Remove(filepath.Join(dir, stem+audioSuffix)); err == nil {
			removed++
		} else if !os.IsNotExist(err) {
			return removed, fmt.Errorf("remove %s%s: %w", stem, audioSuffix, err)
		}
	}
	return removed, nil
}

// pruneTranscripts keeps the newest `keep` sessions (a .md plus its .backend.log)
// and deletes older ones. Returns how many sessions were removed.
// Only files matching the app's own stamp format are touched.
func pruneTranscripts(dir string, keep int) (int, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return 0, fmt.Errorf("read transcripts dir: %w", err)
	}

	var sessions []string // stems, e.g. "2026-09-26_1434"
	orphanLogs := map[string]bool{}
	for _, e := range entries {
		if e.IsDir() {
			continue
		}
		name := e.Name()
		switch {
		case sessionStamp.MatchString(name):
			sessions = append(sessions, strings.TrimSuffix(name, ".md"))
		case strings.HasSuffix(name, backendLogSuffix):
			stem := strings.TrimSuffix(name, backendLogSuffix)
			if sessionStamp.MatchString(stem + ".md") {
				orphanLogs[stem] = true
			}
		case strings.HasSuffix(name, latencyCSVSuffix):
			stem := strings.TrimSuffix(name, latencyCSVSuffix)
			if sessionStamp.MatchString(stem + ".md") {
				orphanLogs[stem] = true
			}
		}
	}

	// Lexicographic order == chronological for this stamp format.
	sort.Sort(sort.Reverse(sort.StringSlice(sessions)))

	removed := 0
	for i, stem := range sessions {
		if i < keep {
			continue
		}
		gone := false
		for _, f := range []string{stem + ".md", stem + backendLogSuffix, stem + latencyCSVSuffix} {
			if err := os.Remove(filepath.Join(dir, f)); err == nil {
				gone = true
			} else if !os.IsNotExist(err) {
				return removed, fmt.Errorf("remove %s: %w", f, err)
			}
		}
		if gone {
			removed++
		}
		delete(orphanLogs, stem)
	}

	// Logs with no surviving .md, beyond the cap, are pruned too.
	if len(orphanLogs) > 0 {
		extra := make([]string, 0, len(orphanLogs))
		for stem := range orphanLogs {
			extra = append(extra, stem)
		}
		sort.Sort(sort.Reverse(sort.StringSlice(extra)))
		for i, stem := range extra {
			if i < keep {
				continue
			}
			for _, suffix := range []string{backendLogSuffix, latencyCSVSuffix} {
				if err := os.Remove(filepath.Join(dir, stem+suffix)); err == nil {
					removed++
				}
			}
		}
	}
	return removed, nil
}
