package main

import "github.com/atotto/clipboard"

// clipboardWrite puts text on the system clipboard.
//
// `atotto/clipboard` is already in the tree as a dependency of the textarea
// component, and on Windows it calls the Win32 clipboard APIs with UTF-16
// directly. That matters here: going through `clip.exe` or PowerShell stdin
// would decode the text with the console code page and mangle the curly quotes
// and dashes an LLM pass produces constantly. It also means no process spawn,
// so a copy is instant. Indirection through a variable so tests can stub it.
var clipboardWrite = clipboard.WriteAll
