package main

import (
	"strings"
	"testing"
)

func TestFmtClock(t *testing.T) {
	if got := fmtClock(83.0); got != "01:23" {
		t.Errorf("fmtClock(83) = %q, want 01:23", got)
	}
	if got := fmtClock(5.7); got != "00:05" {
		t.Errorf("fmtClock(5.7) = %q, want 00:05", got)
	}
}

func TestCapRunes(t *testing.T) {
	if got := capRunes("hello", 200); got != "hello" {
		t.Errorf("short: got %q", got)
	}
	long := strings.Repeat("ab ", 100) // 300 runes
	got := capRunes(long, 200)
	if len([]rune(got)) != 200 {
		t.Errorf("cap: len=%d", len([]rune(got)))
	}
	// Multibyte-safe: never splits a rune.
	uni := strings.Repeat("héllo ", 50)
	got = capRunes(uni, 200)
	if len([]rune(got)) != 200 {
		t.Errorf("unicode cap: len=%d", len([]rune(got)))
	}
}

func TestRenderContent(t *testing.T) {
	m := model{lines: []string{"- [00:01] committed line"}}
	if got := m.renderContent(); got != "- [00:01] committed line" {
		t.Errorf("no draft: got %q", got)
	}
	m.draft = "growing words"
	got := m.renderContent()
	if !strings.Contains(got, "growing words") || !strings.Contains(got, "▌") {
		t.Errorf("draft line: got %q", got)
	}
	if !strings.Contains(got, "committed line") {
		t.Errorf("draft must sit above history: got %q", got)
	}
}
