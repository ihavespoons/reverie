package vecmath

import (
	"math"
	"testing"
)

func TestCosine(t *testing.T) {
	tests := []struct {
		name string
		a, b []float32
		want float32
		tol  float32
	}{
		{
			name: "identical vectors",
			a:    []float32{1, 2, 3},
			b:    []float32{1, 2, 3},
			want: 1.0,
			tol:  1e-6,
		},
		{
			name: "orthogonal vectors",
			a:    []float32{1, 0, 0},
			b:    []float32{0, 1, 0},
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "anti-parallel vectors",
			a:    []float32{1, 0, 0},
			b:    []float32{-1, 0, 0},
			want: -1.0,
			tol:  1e-6,
		},
		{
			name: "zero vector a",
			a:    []float32{0, 0, 0},
			b:    []float32{1, 2, 3},
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "zero vector b",
			a:    []float32{1, 2, 3},
			b:    []float32{0, 0, 0},
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "both zero vectors",
			a:    []float32{0, 0, 0},
			b:    []float32{0, 0, 0},
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "mismatched lengths",
			a:    []float32{1, 2},
			b:    []float32{1, 2, 3},
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "empty vectors",
			a:    []float32{},
			b:    []float32{},
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "nil vectors",
			a:    nil,
			b:    nil,
			want: 0.0,
			tol:  1e-6,
		},
		{
			name: "known cosine similarity",
			a:    []float32{1, 2, 3},
			b:    []float32{4, 5, 6},
			// cos = (4+10+18) / (sqrt(14) * sqrt(77)) = 32 / sqrt(1078)
			want: 32.0 / float32(math.Sqrt(1078)),
			tol:  1e-5,
		},
		{
			name: "scaled vectors are identical direction",
			a:    []float32{1, 2, 3},
			b:    []float32{2, 4, 6},
			want: 1.0,
			tol:  1e-6,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Cosine(tt.a, tt.b)
			if diff := got - tt.want; diff < -tt.tol || diff > tt.tol {
				t.Errorf("Cosine(%v, %v) = %v, want %v (tol %v)", tt.a, tt.b, got, tt.want, tt.tol)
			}
		})
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		name string
		a    []float32
		// checkUnit: if true, verify the result has unit magnitude.
		checkUnit bool
		// checkZero: if true, verify the result is all zeros.
		checkZero bool
	}{
		{
			name:      "standard vector becomes unit",
			a:         []float32{3, 4},
			checkUnit: true,
		},
		{
			name:      "unit vector stays unit",
			a:         []float32{1, 0, 0},
			checkUnit: true,
		},
		{
			name:      "zero vector stays zero",
			a:         []float32{0, 0, 0},
			checkZero: true,
		},
		{
			name:      "negative components",
			a:         []float32{-3, 4, 0},
			checkUnit: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Normalize(tt.a)
			if len(result) != len(tt.a) {
				t.Fatalf("Normalize(%v) returned len %d, want %d", tt.a, len(result), len(tt.a))
			}

			// Verify input was not mutated.
			origCopy := make([]float32, len(tt.a))
			copy(origCopy, tt.a)
			_ = Normalize(tt.a)
			for i := range tt.a {
				if tt.a[i] != origCopy[i] {
					t.Errorf("Normalize mutated input at index %d: got %v, want %v", i, tt.a[i], origCopy[i])
				}
			}

			if tt.checkUnit {
				var mag float64
				for _, v := range result {
					mag += float64(v) * float64(v)
				}
				mag = math.Sqrt(mag)
				if diff := mag - 1.0; diff < -1e-5 || diff > 1e-5 {
					t.Errorf("Normalize(%v) magnitude = %v, want 1.0", tt.a, mag)
				}
			}

			if tt.checkZero {
				for i, v := range result {
					if v != 0 {
						t.Errorf("Normalize(%v)[%d] = %v, want 0", tt.a, i, v)
					}
				}
			}
		})
	}

	// Edge case: empty slice.
	t.Run("empty slice", func(t *testing.T) {
		result := Normalize([]float32{})
		if len(result) != 0 {
			t.Errorf("Normalize(empty) returned len %d, want 0", len(result))
		}
	})

	// Edge case: nil slice.
	t.Run("nil slice", func(t *testing.T) {
		result := Normalize(nil)
		if len(result) != 0 {
			t.Errorf("Normalize(nil) returned len %d, want 0", len(result))
		}
	})
}
