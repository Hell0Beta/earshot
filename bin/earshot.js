#!/usr/bin/env node
"use strict";

// earshot launcher.
//
// The real program is a native Windows binary that ships in vendor/. This
// wrapper exists only to hand the process over to it with stdio attached
// (Bubble Tea needs a real console) and to forward the exit code.

const path = require("path");
const { spawnSync } = require("child_process");

const exe = path.join(__dirname, "..", "vendor", "earshot.exe");

const res = spawnSync(exe, process.argv.slice(2), { stdio: "inherit" });

if (res.error) {
  if (res.error.code === "ENOENT") {
    console.error(
      "earshot: the native binary is missing from the package.\n" +
        "  expected: " + exe + "\n" +
        "  This usually means the package was published without running `npm run prepack`.\n" +
        "  Reinstall, or build it yourself:  cd go && go build -o ../vendor/earshot.exe ."
    );
    process.exit(1);
  }
  console.error("earshot: " + res.error.message);
  process.exit(1);
}

process.exit(res.status === null ? 1 : res.status);
