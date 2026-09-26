package main

import (
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
)

// Path resolution. Nothing here may assume a particular working directory:
// the binary is launched from a .bat shim, from npm, or from a source
// checkout, and relative defaults (the original ..\.venv, ..\backend,
// D:\earshot\models) resolved against the CWD and broke all but one of them.
//
// Every chain is ordered most-specific-first and every resolver returns how it
// resolved, so `earshot doctor` can explain the choice.

const (
	envHome     = "EARSHOT_HOME"
	envPython   = "EARSHOT_PYTHON"
	envCUDADLLs = "EARSHOT_CUDA_DLL_DIR"
)

// resolved records a chosen path and the rule that chose it.
type resolved struct {
	Path   string
	Source string
}

func (r resolved) String() string {
	if r.Source == "" {
		return r.Path
	}
	return r.Path + "  (" + r.Source + ")"
}

// dataDir is the writable root for models, transcripts and the fallback venv.
func dataDir() resolved {
	if v := os.Getenv(envHome); v != "" {
		return resolved{os.ExpandEnv(v), envHome}
	}
	if v := os.Getenv("LOCALAPPDATA"); v != "" {
		return resolved{filepath.Join(v, "earshot"), "LOCALAPPDATA"}
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return resolved{filepath.Join(".", "earshot"), "cwd (no LOCALAPPDATA, no home)"}
	}
	return resolved{filepath.Join(home, ".earshot"), "home"}
}

func dataSub(parts ...string) string {
	return filepath.Join(append([]string{dataDir().Path}, parts...)...)
}

// exeDirOverride lets tests point resolution at a synthetic tree instead of the
// real install location.
var exeDirOverride string

// exeDir is the directory holding the running binary.
func exeDir() (string, error) {
	if exeDirOverride != "" {
		return exeDirOverride, nil
	}
	exe, err := os.Executable()
	if err != nil {
		return "", err
	}
	if abs, err := filepath.Abs(exe); err == nil {
		exe = abs
	}
	return filepath.Dir(exe), nil
}

// isSourceCheckout reports whether the binary sits inside a real source tree,
// as opposed to an installed package. In an npm install the exe lives at
// <pkg>/vendor/earshot.exe, so "<exe>/../backend" also exists — the giveaway is
// a go.mod next to the exe or a .git in either directory. Getting this wrong
// would make setup look for a venv that will never be there.
func isSourceCheckout() bool {
	dir, err := exeDir()
	if err != nil {
		return false
	}
	if _, err := os.Stat(filepath.Join(dir, "..", "backend", "captions.py")); err != nil {
		return false
	}
	for _, p := range []string{
		filepath.Join(dir, "go.mod"),
		filepath.Join(dir, ".git"),
		filepath.Join(dir, "..", ".git"),
		filepath.Join(dir, "..", "go.mod"),
	} {
		if _, err := os.Stat(p); err == nil {
			return true
		}
	}
	return false
}

func fileExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && !st.IsDir()
}

// dirExists is for installed packages, which are directories (fileExists
// deliberately rejects those, so the two can't be confused).
func dirExists(p string) bool {
	st, err := os.Stat(p)
	return err == nil && st.IsDir()
}

// resolvePython finds an interpreter that has the backend's dependencies.
func resolvePython(flagVal string) (resolved, error) {
	type cand struct {
		path   string
		source string
	}
	dir, _ := exeDir()
	var cands []cand
	if flagVal != "" {
		cands = append(cands, cand{flagVal, "--python"})
	}
	if v := os.Getenv(envPython); v != "" {
		cands = append(cands, cand{v, envPython})
	}
	if isSourceCheckout() {
		// Prefer the repo venv: no second ~1GB copy.
		cands = append(cands, cand{
			filepath.Join(dir, "..", ".venv", "Scripts", "python.exe"),
			"source checkout .venv",
		})
	}
	cands = append(cands,
		cand{filepath.Join(dataDir().Path, "venv", "Scripts", "python.exe"), "data dir venv"},
	)
	for _, c := range cands {
		if fileExists(c.path) {
			return resolved{c.path, c.source}, nil
		}
	}
	// Last resort: whatever python is on PATH. Don't import-probe it here
	// (that costs seconds on every launch) — `earshot doctor` is the
	// thorough diagnostic and the backend reports missing imports clearly.
	if p, err := exec.LookPath("python"); err == nil {
		return resolved{p, "PATH"}, nil
	}
	return resolved{}, fmt.Errorf(
		"no Python with earshot's dependencies found.\n"+
			"  Looked for: --python, %s, the source checkout .venv, %s, and python on PATH\n"+
			"  Fix: run `earshot setup`",
		envPython, filepath.Join(dataDir().Path, "venv", "Scripts", "python.exe"))
}

// resolveBackend locates captions.py across the npm and source layouts.
func resolveBackend(flagVal string) (resolved, error) {
	type cand struct {
		path   string
		source string
	}
	dir, _ := exeDir()
	var cands []cand
	if flagVal != "" {
		cands = append(cands, cand{flagVal, "--backend"})
	}
	cands = append(cands,
		cand{filepath.Join(dir, "backend", "captions.py"), "flat layout (exe beside backend/)"},
		cand{filepath.Join(dir, "..", "backend", "captions.py"), "npm install or source checkout"},
		cand{dataSub("backend", "captions.py"), "data dir"},
	)
	for _, c := range cands {
		if fileExists(c.path) {
			return resolved{c.path, c.source}, nil
		}
	}
	return resolved{}, fmt.Errorf(
		"backend script captions.py not found.\n  Looked next to the exe, in ../backend, and in %s\n"+
			"  Fix: reinstall the package, or pass --backend <path>", dataDir().Path)
}

// pathHasDLL reports whether a DLL is already reachable via PATH. exec.LookPath
// won't find DLLs (PATHEXT only covers executables), so scan PATH directly.
func pathHasDLL(name string) bool {
	for _, dir := range filepath.SplitList(os.Getenv("Path")) {
		if dir == "" {
			continue
		}
		if fileExists(filepath.Join(dir, name)) {
			return true
		}
	}
	return false
}

// cudaDLLsReady reports whether cuBLAS and cuDNN are already importable.
func cudaDLLsReady() bool {
	return pathHasDLL("cublas64_12.dll") && pathHasDLL("cudnn64_9.dll")
}

// resolveCUDADir returns a directory to prepend to PATH, or "" to leave PATH
// alone (the common case: the DLLs are already on PATH).
func resolveCUDADir(flagVal string) resolved {
	if flagVal != "" {
		return resolved{flagVal, "--cuda-dll-dir"}
	}
	if v := os.Getenv(envCUDADLLs); v != "" {
		return resolved{v, envCUDADLLs}
	}
	if cudaDLLsReady() {
		return resolved{"", "already on PATH"}
	}
	return resolved{"", "not found"}
}

// deviceFor picks the inference device, falling back to CPU when the CUDA
// libraries are missing so the tool still works on a machine without them.
func deviceFor(flagDevice, flagCompute string, cudaReady bool) (device, compute, note string) {
	device, compute = flagDevice, flagCompute
	if device == "" || device == "auto" {
		if cudaReady {
			device, compute = "cuda", "float16"
		} else {
			device, compute = "cpu", "int8"
			note = "CUDA libraries not found - running on CPU (slower; " +
				"try --model base.en, or see `earshot doctor`)"
		}
	}
	if compute == "" {
		if device == "cpu" {
			compute = "int8"
		} else {
			compute = "float16"
		}
	}
	return device, compute, note
}
