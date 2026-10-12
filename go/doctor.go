package main

import (
	"flag"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
)

// version is stamped at build time from package.json by scripts/prepack.js
// (-X main.version). It is a var, not a const, so the linker can override it.
// Keeping a hand-written copy of the version in two places guaranteed drift:
// the published 1.2.0 tarball reported "earshot 0.2.0" because this line was
// never updated. A plain `go build` yields "dev", which is the honest answer
// for an unstamped build.
var version = "dev"

func usage() {
	fmt.Print(`earshot - live, local captions for whatever your speakers are playing

USAGE
  earshot                 capture and show live captions
  earshot setup           create the Python environment and install backend deps
  earshot doctor          diagnose the setup (run this if something looks wrong)
  earshot version         print the version

KEYS (while capturing)
  space   pause / resume the display (the backend keeps running)
  q       quit, flush the last sentence and save the transcript
  home    jump to the newest caption      end   jump to the oldest
  pgup/pgdn, up/down   scroll the history

COMMON FLAGS
  --model NAME        whisper model (default small.en; base.en is faster on CPU)
  --device auto|cuda|cpu
  --step-secs N      minimum audio per cycle; the step adapts to your
                     machine's measured inference cost (default 0.3)
  --step-headroom N  multiplier on the p90 pass time; raise if captions lag
  --max-backlog N    seconds of audio allowed to queue before old audio is
                     dropped, which bounds caption lag (default 1.5)
  --beam-size N       beam width (default 5)
  --block-chars N     sentence block size (default 200)
  --correct-model F   GGUF model; polish each sentence with an LLM (off)
  --out-dir DIR       transcripts (default <data dir>/transcripts)
  --keep N            transcript sessions to retain (default 30)
  --splash-style shimmer|rainbow

ENVIRONMENT
  EARSHOT_HOME            data dir (default %LOCALAPPDATA%\earshot)
  EARSHOT_PYTHON          python to use
  EARSHOT_CUDA_DLL_DIR    folder holding cuBLAS/cuDNN

Run earshot --help for the full flag list, or see the README.
`)
}

// ---- doctor ----

type check struct {
	name string
	ok   bool
	hard bool // a failure here stops earshot from working at all
	info string
	fix  string
}

func runDoctor(args []string) int {
	fs := flag.NewFlagSet("doctor", flag.ExitOnError)
	model := fs.String("model", "small.en", "model to look for in the cache")
	_ = fs.Parse(args)

	fmt.Println("earshot doctor")
	fmt.Println()

	var checks []check

	// 1. data dir
	dd := dataDir()
	writable := false
	if err := os.MkdirAll(dd.Path, 0o755); err == nil {
		probe := filepath.Join(dd.Path, ".write-probe")
		if err := os.WriteFile(probe, []byte("x"), 0o644); err == nil {
			os.Remove(probe)
			writable = true
		}
	}
	checks = append(checks, check{"data dir", writable, true, dd.String(),
		"set EARSHOT_HOME to a writable folder"})

	// 1b. EARSHOT_HOME is advisory, never blocking. Unset means the data dir
	// falls back to %LOCALAPPDATA%\earshot, so a machine that keeps its model
	// cache on another volume silently re-downloads ~460 MB on first run.
	if strings.TrimSpace(os.Getenv(envHome)) == "" {
		checks = append(checks, check{"EARSHOT_HOME", false, false,
			"not set - models and transcripts use " + dd.Path,
			`set EARSHOT_HOME to pick a folder; default is %LOCALAPPDATA%\earshot`})
	}

	// 2. python
	py, pyErr := resolvePython("")
	if pyErr != nil {
		checks = append(checks, check{"python", false, true, pyErr.Error(),
			"run `earshot setup`"})
	} else {
		checks = append(checks, check{"python", true, true, py.String(), ""})
	}

	// 3. backend script
	be, beErr := resolveBackend("")
	if beErr != nil {
		checks = append(checks, check{"backend script", false, true, beErr.Error(),
			"reinstall the package, or pass --backend <path>"})
	} else {
		checks = append(checks, check{"backend script", true, true, be.String(), ""})
	}

	// 4. python dependencies
	if pyErr == nil {
		has, detail := pythonHasDeps(py.Path)
		checks = append(checks, check{"backend dependencies", has, true, detail,
			"run `earshot setup`"})
	}

	// 5. model cache
	models := dataSub("models")
	cached := findCachedModel(models, *model)
	info := "not downloaded yet (downloads on first run)"
	if cached != "" {
		info = cached
	}
	checks = append(checks, check{"model cache", true, false, info, ""})

	// 6. CUDA libraries
	cudaOK := cudaDLLsReady()
	cudaInfo := "not on PATH"
	if cudaOK {
		cudaInfo = "cuBLAS + cuDNN found on PATH"
	} else if r := resolveCUDADir(""); r.Path != "" {
		cudaInfo = r.String()
	}
	checks = append(checks, check{"CUDA libraries", cudaOK, false, cudaInfo,
		"GPU is optional - earshot falls back to CPU. For GPU: install the\n" +
			"      cuBLAS/cuDNN 12 pack and add it to PATH, or set EARSHOT_CUDA_DLL_DIR"})

	// 7. GPU visible to ctranslate2
	if pyErr == nil && cudaOK {
		n, err := ctranslate2DeviceCount(py.Path)
		info := fmt.Sprintf("%d CUDA device(s) visible", n)
		if err != nil {
			info = "could not query: " + err.Error()
		}
		checks = append(checks, check{"GPU visible to ctranslate2", err == nil && n > 0, false, info,
			"if this says 0 devices, another process may be holding the driver"})
	}

	// 8. loopback capture device
	if pyErr == nil && beErr == nil {
		devices, err := listLoopbackDevices(py.Path, be.Path)
		info := "no WASAPI loopback device found"
		if err == nil && devices > 0 {
			info = fmt.Sprintf("%d loopback device(s) available", devices)
		}
		checks = append(checks, check{"audio capture device", err == nil && devices > 0, true, info,
			"plug in headphones/speakers, or pass --device-index"})
	}

	// report
	failed := 0
	for _, c := range checks {
		mark := "ok  "
		severity := false
		switch {
		case c.ok:
		case c.hard:
			mark, severity = "FAIL", true
		default:
			mark = "warn"
		}
		if severity {
			failed++
		}
		fmt.Printf("  [%s] %-26s %s\n", mark, c.name, c.info)
		if !c.ok && c.fix != "" {
			fmt.Printf("         fix: %s\n", c.fix)
		}
	}
	fmt.Println()
	if failed > 0 {
		fmt.Printf("%d blocking problem(s). earshot will not start until they're fixed.\n", failed)
		return 1
	}
	fmt.Println("All blocking checks passed. Run `earshot` to start captioning.")
	return 0
}

// pythonHasDeps checks for the backend's imports without paying a full
// interpreter start on the common (venv) path.
func pythonHasDeps(python string) (bool, string) {
	if isVenvPython(python) {
		root := filepath.Dir(filepath.Dir(python))
		sp := filepath.Join(root, "Lib", "site-packages")
		ok := dirExists(filepath.Join(sp, "faster_whisper")) &&
			dirExists(filepath.Join(sp, "pyaudiowpatch"))
		return ok, sp
	}
	// A PATH python: probe it directly (slower, but this is a diagnostic).
	cmd := exec.Command(python, "-c", "import faster_whisper, pyaudiowpatch")
	if out, err := cmd.CombinedOutput(); err != nil {
		return false, strings.TrimSpace(string(out))
	}
	return true, "imports verified"
}

func isVenvPython(python string) bool {
	root := filepath.Dir(filepath.Dir(python))
	return strings.EqualFold(filepath.Base(root), ".venv") ||
		strings.EqualFold(filepath.Base(filepath.Dir(root)), "earshot")
}

func findCachedModel(dir, name string) string {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return ""
	}
	want := "models--Systran--faster-whisper-" + name
	for _, e := range entries {
		if e.Name() == want {
			return name + " cached in " + dir
		}
	}
	return ""
}

func ctranslate2DeviceCount(python string) (int, error) {
	out, err := exec.Command(python, "-c",
		"import ctranslate2; print(ctranslate2.get_cuda_device_count())").Output()
	if err != nil {
		return 0, err
	}
	var n int
	if _, err := fmt.Sscanf(strings.TrimSpace(string(out)), "%d", &n); err != nil {
		return 0, err
	}
	return n, nil
}

func listLoopbackDevices(python, backend string) (int, error) {
	out, err := exec.Command(python, backend, "--list-devices").CombinedOutput()
	if err != nil {
		return 0, fmt.Errorf("%s", strings.TrimSpace(string(out)))
	}
	return strings.Count(string(out), "LOOPBACK"), nil
}

// ---- setup ----

func runSetup(args []string) int {
	fs := flag.NewFlagSet("setup", flag.ExitOnError)
	venvPath := fs.String("venv", "", "venv to create (default: reuse a source checkout's .venv)")
	_ = fs.Parse(args)

	fmt.Println("earshot setup")
	fmt.Println()

	be, err := resolveBackend("")
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		return 1
	}
	reqs := filepath.Join(filepath.Dir(be.Path), "requirements.txt")
	if !fileExists(reqs) {
		fmt.Fprintf(os.Stderr, "requirements.txt not found next to %s\n", be.Path)
		return 1
	}

	// Prefer an existing source-checkout venv so we never duplicate ~1GB.
	target := *venvPath
	source := "--venv"
	if target == "" {
		dir, _ := exeDir()
		checkout := filepath.Join(dir, "..", ".venv")
		if isSourceCheckout() && fileExists(filepath.Join(checkout, "Scripts", "python.exe")) {
			target, source = checkout, "source checkout .venv"
		} else {
			target, source = filepath.Join(dataDir().Path, "venv"), "data dir"
		}
	}
	fmt.Printf("  python environment : %s  (%s)\n", target, source)
	fmt.Printf("  requirements       : %s\n", reqs)
	fmt.Println()

	pyBin := filepath.Join(target, "Scripts", "python.exe")
	if !fileExists(pyBin) {
		fmt.Println("  creating venv (a few seconds)...")
		base, err := exec.LookPath("python")
		if err != nil {
			fmt.Fprintln(os.Stderr, "  no python on PATH to build a venv from")
			return 1
		}
		if out, err := exec.Command(base, "-m", "venv", target).CombinedOutput(); err != nil {
			fmt.Fprintf(os.Stderr, "  venv creation failed: %v\n%s\n", err, out)
			return 1
		}
	} else {
		fmt.Println("  reusing the existing venv")
	}

	steps := [][]string{
		{"-m", "pip", "install", "--upgrade", "pip"},
		{"-m", "pip", "install", "-r", reqs},
	}
	for _, s := range steps {
		fmt.Printf("  %s\n", strings.Join(s, " "))
		cmd := exec.Command(pyBin, s...)
		cmd.Stdout, cmd.Stderr = os.Stdout, os.Stderr
		if err := cmd.Run(); err != nil {
			fmt.Fprintf(os.Stderr, "\n  step failed: %v\n", err)
			return 1
		}
	}

	fmt.Println()
	if cudaDLLsReady() {
		fmt.Println("  CUDA libraries     : found on PATH (GPU will be used)")
	} else {
		fmt.Println("  CUDA libraries     : not found - earshot will run on CPU.")
		fmt.Println("                      For GPU: install the cuBLAS/cuDNN 12 (CUDA 12) pack,")
		fmt.Println("                      add it to PATH, or set EARSHOT_CUDA_DLL_DIR.")
	}
	fmt.Println("  model              : downloads on first run (~460 MB for small.en)")
	fmt.Println()
	fmt.Println("Setup complete. Next:")
	fmt.Println("  earshot doctor     verify everything resolves")
	fmt.Println("  earshot            start captioning")
	return 0
}
