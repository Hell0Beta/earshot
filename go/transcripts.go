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
		for _, f := range []string{stem + ".md", stem + backendLogSuffix} {
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
			if err := os.Remove(filepath.Join(dir, stem+backendLogSuffix)); err == nil {
				removed++
			}
		}
	}
	return removed, nil
}
