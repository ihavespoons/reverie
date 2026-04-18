package cluster

import (
	"math"
	"testing"
)

func almostEqual(a, b float32, eps float64) bool {
	return math.Abs(float64(a)-float64(b)) < eps
}

func TestUpdateCentroid_N1_Midpoint(t *testing.T) {
	old := []float32{1, 0}
	vec := []float32{0, 1}
	got := UpdateCentroid(old, 1, vec)

	if len(got) != 2 {
		t.Fatalf("expected len 2, got %d", len(got))
	}
	// (1*1 + 0) / 2 = 0.5
	if !almostEqual(got[0], 0.5, 1e-6) {
		t.Errorf("got[0] = %f, want 0.5", got[0])
	}
	// (0*1 + 1) / 2 = 0.5
	if !almostEqual(got[1], 0.5, 1e-6) {
		t.Errorf("got[1] = %f, want 0.5", got[1])
	}
}

func TestUpdateCentroid_N9_BarelyMoves(t *testing.T) {
	old := []float32{1, 0, 0}
	vec := []float32{0, 1, 0}
	got := UpdateCentroid(old, 9, vec)

	if len(got) != 3 {
		t.Fatalf("expected len 3, got %d", len(got))
	}
	// (1*9 + 0) / 10 = 0.9
	if !almostEqual(got[0], 0.9, 1e-6) {
		t.Errorf("got[0] = %f, want 0.9", got[0])
	}
	// (0*9 + 1) / 10 = 0.1
	if !almostEqual(got[1], 0.1, 1e-6) {
		t.Errorf("got[1] = %f, want 0.1", got[1])
	}
	// (0*9 + 0) / 10 = 0
	if !almostEqual(got[2], 0.0, 1e-6) {
		t.Errorf("got[2] = %f, want 0.0", got[2])
	}
}

func TestUpdateCentroid_MismatchedLengths_ReturnsVec(t *testing.T) {
	old := []float32{1, 0}
	vec := []float32{0, 1, 0}
	got := UpdateCentroid(old, 5, vec)

	if len(got) != 3 {
		t.Fatalf("expected len 3, got %d", len(got))
	}
	for i, want := range vec {
		if got[i] != want {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}

func TestUpdateCentroid_NilOld_ReturnsVec(t *testing.T) {
	vec := []float32{0.5, 0.5, 0.5}
	got := UpdateCentroid(nil, 0, vec)

	if len(got) != 3 {
		t.Fatalf("expected len 3, got %d", len(got))
	}
	for i, want := range vec {
		if got[i] != want {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want)
		}
	}

	// Verify it's a copy, not the same slice.
	got[0] = 999
	if vec[0] == 999 {
		t.Error("UpdateCentroid returned the input slice, not a copy")
	}
}

func TestUpdateCentroid_EmptyOld_ReturnsVec(t *testing.T) {
	old := []float32{}
	vec := []float32{1, 2, 3}
	got := UpdateCentroid(old, 0, vec)

	if len(got) != 3 {
		t.Fatalf("expected len 3, got %d", len(got))
	}
	for i, want := range vec {
		if got[i] != want {
			t.Errorf("got[%d] = %f, want %f", i, got[i], want)
		}
	}
}
