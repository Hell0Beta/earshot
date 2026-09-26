// Prepack: clean Python bytecode, then build the Windows binary that ships in
// the npm tarball.
//
// The version is injected from package.json via -ldflags -X so package.json is
// the single source of truth. Hand-maintaining a second copy in the Go source
// is what let the published 1.2.0 report "earshot 0.2.0".
//
// Kept as a script rather than an inline npm command because the ldflags value
// needs nested quotes that do not survive JSON -> cmd -> go build intact.

const { execFileSync } = require("child_process");
const fs = require("fs");
const path = require("path");

const root = path.join(__dirname, "..");
const pkg = JSON.parse(fs.readFileSync(path.join(root, "package.json"), "utf8"));

// Stale __pycache__ would otherwise be packed into the tarball.
const pycache = path.join(root, "backend", "__pycache__");
if (fs.existsSync(pycache)) {
  fs.rmSync(pycache, { recursive: true, force: true });
  console.log("prepack: removed backend/__pycache__");
}

const out = path.join(root, "vendor", "earshot.exe");
fs.mkdirSync(path.dirname(out), { recursive: true });

const ldflags = `-s -w -X main.version=${pkg.version}`;
console.log(`prepack: building ${path.relative(root, out)} (version ${pkg.version})`);

// No shell: execFile with an argv array keeps "-s -w -X main.version=x" as one
// argument. shell:true re-splits it through cmd and go sees a bare -w.
execFileSync(
  "go",
  ["build", "-trimpath", "-ldflags", ldflags, "-o", out, "."],
  { cwd: path.join(root, "go"), stdio: "inherit" }
);

const { size } = fs.statSync(out);
console.log(`prepack: done, ${(size / 1048576).toFixed(1)} MB`);
