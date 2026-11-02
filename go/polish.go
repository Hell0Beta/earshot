package main

import (
	"os"
	"path/filepath"
	"strings"
)

// findPolishModel looks for the LLM used by voice-to-text. The model is a
// separate, much smaller download than Whisper, so its absence is not an
// error: v2t still works, it just cannot polish. Picking the first GGUF in
// lexicographic order keeps the choice deterministic across runs.
func findPolishModel(modelsDir string) string {
	entries, err := os.ReadDir(filepath.Join(modelsDir, "llm"))
	if err != nil {
		return ""
	}
	best := ""
	for _, e := range entries {
		if e.IsDir() || !strings.HasSuffix(strings.ToLower(e.Name()), ".gguf") {
			continue
		}
		p := filepath.Join(modelsDir, "llm", e.Name())
		if best == "" || p < best {
			best = p
		}
	}
	return best
}
