package main

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func touch(t *testing.T, p string) {
	t.Helper()
	if err := os.MkdirAll(filepath.Dir(p), 0o755); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(p, []byte("x"), 0o644); err != nil {
		t.Fatal(err)
	}
}

func TestDataDirPrefersEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(envHome, dir)
	if got := dataDir(); got.Path != dir || got.Source != envHome {
		t.Errorf("dataDir = %v, want %s from %s", got, dir, envHome)
	}
}

func TestDataDirFallsBackToLocalAppData(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(envHome, "")
	t.Setenv("LOCALAPPDATA", dir)
	got := dataDir()
	if got.Path != filepath.Join(dir, "earshot") {
		t.Errorf("dataDir = %s, want %s", got.Path, filepath.Join(dir, "earshot"))
	}
	if got.Source != "LOCALAPPDATA" {
		t.Errorf("source = %s, want LOCALAPPDATA", got.Source)
	}
}

func TestDataSubJoinsUnderDataDir(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(envHome, dir)
	if got, want := dataSub("models"), filepath.Join(dir, "models"); got != want {
		t.Errorf("dataSub = %s, want %s", got, want)
	}
}

func TestResolveBackendFlatLayout(t *testing.T) {
	root := t.TempDir()
	exeDirOverride = filepath.Join(root, "bin")
	defer func() { exeDirOverride = "" }()
	touch(t, filepath.Join(root, "bin", "backend", "captions.py"))

	got, err := resolveBackend("")
	if err != nil {
		t.Fatalf("resolveBackend: %v", err)
	}
	if !strings.Contains(got.Source, "flat layout") {
		t.Errorf("source = %q, want the flat layout", got.Source)
	}
}

// The npm install shape: exe in vendor/, backend one level up. This must
// resolve, and must NOT be mistaken for a source checkout (which would send
// setup looking for a .venv that does not exist).
func TestResolveBackendNpmInstallLayout(t *testing.T) {
	root := t.TempDir()
	exeDirOverride = filepath.Join(root, "vendor")
	defer func() { exeDirOverride = "" }()
	touch(t, filepath.Join(root, "vendor", "earshot.exe"))
	touch(t, filepath.Join(root, "backend", "captions.py"))

	got, err := resolveBackend("")
	if err != nil {
		t.Fatalf("resolveBackend: %v", err)
	}
	if !strings.Contains(got.Source, "npm install") {
		t.Errorf("source = %q, want the npm layout", got.Source)
	}
	if isSourceCheckout() {
		t.Errorf("an npm install must not be treated as a source checkout")
	}
}

func TestResolveBackendSourceCheckoutLayout(t *testing.T) {
	root := t.TempDir()
	exeDirOverride = filepath.Join(root, "go")
	defer func() { exeDirOverride = "" }()
	touch(t, filepath.Join(root, "go", "earshot.exe"))
	touch(t, filepath.Join(root, "backend", "captions.py"))

	got, err := resolveBackend("")
	if err != nil {
		t.Fatalf("resolveBackend: %v", err)
	}
	if !strings.Contains(got.Source, "source checkout") {
		t.Errorf("source = %q, want the source checkout", got.Source)
	}
}

func TestResolveBackendPrefersFlag(t *testing.T) {
	root := t.TempDir()
	exeDirOverride = filepath.Join(root, "bin")
	defer func() { exeDirOverride = "" }()
	explicit := filepath.Join(root, "custom", "captions.py")
	touch(t, explicit)
	touch(t, filepath.Join(root, "bin", "backend", "captions.py"))

	got, err := resolveBackend(explicit)
	if err != nil {
		t.Fatalf("resolveBackend: %v", err)
	}
	if got.Path != explicit || got.Source != "--backend" {
		t.Errorf("got %v, want the explicit --backend path", got)
	}
}

func TestResolveBackendNotFoundIsActionable(t *testing.T) {
	exeDirOverride = t.TempDir()
	defer func() { exeDirOverride = "" }()
	t.Setenv(envHome, t.TempDir())
	_, err := resolveBackend("")
	if err == nil {
		t.Fatal("expected an error when no backend exists")
	}
	for _, want := range []string{"captions.py", "reinstall", "--backend"} {
		if !strings.Contains(err.Error(), want) {
			t.Errorf("error should mention %q, got: %v", want, err)
		}
	}
}

func TestResolvePythonPrefersSourceCheckoutVenv(t *testing.T) {
	root := t.TempDir()
	exeDirOverride = filepath.Join(root, "go")
	defer func() { exeDirOverride = "" }()
	touch(t, filepath.Join(root, "go", "earshot.exe"))
	touch(t, filepath.Join(root, "go", "go.mod")) // marks it a real source tree
	touch(t, filepath.Join(root, "backend", "captions.py"))
	checkoutPy := filepath.Join(root, ".venv", "Scripts", "python.exe")
	touch(t, checkoutPy)
	dataHome := t.TempDir()
	touch(t, filepath.Join(dataHome, "venv", "Scripts", "python.exe"))
	t.Setenv(envHome, dataHome)
	t.Setenv(envPython, "")

	if !isSourceCheckout() {
		t.Fatalf("fixture should look like a source checkout")
	}
	got, err := resolvePython("")
	if err != nil {
		t.Fatalf("resolvePython: %v", err)
	}
	if got.Path != checkoutPy {
		t.Errorf("picked %s, want the source checkout venv %s", got.Path, checkoutPy)
	}
	if !strings.Contains(got.Source, "source checkout") {
		t.Errorf("source = %q", got.Source)
	}
}

func TestResolvePythonPrefersEnv(t *testing.T) {
	exeDirOverride = t.TempDir()
	defer func() { exeDirOverride = "" }()
	custom := filepath.Join(t.TempDir(), "py", "python.exe")
	touch(t, custom)
	t.Setenv(envPython, custom)

	got, err := resolvePython("")
	if err != nil {
		t.Fatalf("resolvePython: %v", err)
	}
	if got.Path != custom || got.Source != envPython {
		t.Errorf("got %v, want %s from %s", got, custom, envPython)
	}
}

func TestResolveCUDADirLeavesPathAloneWhenPresent(t *testing.T) {
	dir := t.TempDir()
	touch(t, filepath.Join(dir, "cublas64_12.dll"))
	touch(t, filepath.Join(dir, "cudnn64_9.dll"))
	t.Setenv("Path", dir)
	t.Setenv(envCUDADLLs, "")

	if got := resolveCUDADir(""); got.Path != "" {
		t.Errorf("path = %q, want empty (already on PATH)", got.Path)
	}
	if got := resolveCUDADir(""); got.Source != "already on PATH" {
		t.Errorf("source = %q, want 'already on PATH'", got.Source)
	}
}

func TestResolveCUDADirReportsMissing(t *testing.T) {
	t.Setenv("Path", t.TempDir())
	t.Setenv(envCUDADLLs, "")
	got := resolveCUDADir("")
	if got.Path != "" {
		t.Errorf("path = %q, want empty when the DLLs are absent", got.Path)
	}
	if got.Source != "not found" {
		t.Errorf("source = %q, want 'not found'", got.Source)
	}
}

func TestResolveCUDADirPrefersEnv(t *testing.T) {
	t.Setenv(envCUDADLLs, `D:\some\cuda\bin`)
	if got := resolveCUDADir(""); got.Path != `D:\some\cuda\bin` {
		t.Errorf("path = %q", got.Path)
	}
}

func TestDeviceForFallsBackToCPU(t *testing.T) {
	device, compute, note := deviceFor("auto", "", false)
	if device != "cpu" || compute != "int8" {
		t.Errorf("got %s/%s, want cpu/int8 when CUDA is missing", device, compute)
	}
	if note == "" {
		t.Errorf("expected a note explaining the CPU fallback")
	}
	if !strings.Contains(note, "doctor") {
		t.Errorf("note should point at doctor, got %q", note)
	}
}

func TestDeviceForPrefersCUDAWhenReady(t *testing.T) {
	device, compute, note := deviceFor("auto", "", true)
	if device != "cuda" || compute != "float16" {
		t.Errorf("got %s/%s, want cuda/float16", device, compute)
	}
	if note != "" {
		t.Errorf("no note expected on the GPU path, got %q", note)
	}
}

func TestDeviceForRespectsExplicitFlags(t *testing.T) {
	device, compute, _ := deviceFor("cpu", "float32", true)
	if device != "cpu" || compute != "float32" {
		t.Errorf("explicit flags ignored: got %s/%s", device, compute)
	}
}

func TestDirExistsRejectsFiles(t *testing.T) {
	dir := t.TempDir()
	file := filepath.Join(dir, "f.txt")
	touch(t, file)
	if !dirExists(dir) {
		t.Errorf("dirExists should accept a directory")
	}
	if dirExists(file) {
		t.Errorf("dirExists must reject a plain file")
	}
}

// A batch shim's `set VAR= ` leaves a single space in the variable, which
// os.Getenv reports as non-empty. Treating that as a real path sends an empty
// directory to MkdirAll and yields a "data dir FAIL" naming no folder, so
// blank values must fall through to the default instead.
func TestDataDirIgnoresBlankEnv(t *testing.T) {
	for _, blank := range []string{"", " ", "\t", "  \t "} {
		local := t.TempDir()
		t.Setenv(envHome, blank)
		t.Setenv("LOCALAPPDATA", local)

		got := dataDir()
		if got.Source != "LOCALAPPDATA" {
			t.Errorf("dataDir with EARSHOT_HOME=%q = %v, want LOCALAPPDATA fallback", blank, got)
		}
		if want := filepath.Join(local, "earshot"); got.Path != want {
			t.Errorf("dataDir with EARSHOT_HOME=%q = %q, want %q", blank, got.Path, want)
		}
	}
}

// Surrounding whitespace on a genuine value is noise, not part of the path.
func TestDataDirTrimsEnv(t *testing.T) {
	dir := t.TempDir()
	t.Setenv(envHome, "  "+dir+"\t")
	if got := dataDir(); got.Path != dir || got.Source != envHome {
		t.Errorf("dataDir = %v, want %s from %s", got, dir, envHome)
	}
}
