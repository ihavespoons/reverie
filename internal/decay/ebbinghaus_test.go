package decay

import (
	"math"
	"testing"
)

func TestRetention(t *testing.T) {
	tests := []struct {
		name        string
		turnsSince  int
		utility     float64
		frequency   float64
		temperature float64
		wantApprox  float64
		tolerance   float64 // absolute tolerance for comparison
	}{
		{
			name:        "newly accessed cluster is fully retained",
			turnsSince:  0,
			utility:     0.5,
			frequency:   0.5,
			temperature: 10.0,
			// S = (0.5 + 0.5 + 0.01) * 10 = 10.1
			// R = exp(-0 / 10.1) = exp(0) = 1.0
			wantApprox: 1.0,
			tolerance:  1e-10,
		},
		{
			name:        "moderate decay after 10 turns",
			turnsSince:  10,
			utility:     0.5,
			frequency:   0.5,
			temperature: 10.0,
			// S = 10.1, R = exp(-10/10.1) ≈ 0.3725
			wantApprox: math.Exp(-10.0 / 10.1),
			tolerance:  1e-10,
		},
		{
			name:        "cold uninteracted cluster decays fast",
			turnsSince:  100,
			utility:     0.0,
			frequency:   0.0,
			temperature: 10.0,
			// S = (0 + 0 + 0.01) * 10 = 0.1
			// R = exp(-100/0.1) = exp(-1000) ≈ 0
			wantApprox: 0.0,
			tolerance:  1e-100, // effectively zero
		},
		{
			name:        "temperature zero returns zero without panic",
			turnsSince:  5,
			utility:     0.5,
			frequency:   0.5,
			temperature: 0.0,
			// S = (0.5 + 0.5 + 0.01) * 0 = 0 → guard returns 0
			wantApprox: 0.0,
			tolerance:  1e-10,
		},
		{
			name:        "new cluster no access max retention",
			turnsSince:  0,
			utility:     0.0,
			frequency:   0.0,
			temperature: 10.0,
			// S = (0 + 0 + 0.01) * 10 = 0.1
			// R = exp(-0/0.1) = exp(0) = 1.0
			wantApprox: 1.0,
			tolerance:  1e-10,
		},
		{
			name:        "negative temperature returns zero",
			turnsSince:  5,
			utility:     0.5,
			frequency:   0.5,
			temperature: -5.0,
			// S = (1.01) * (-5) = -5.05 → guard returns 0
			wantApprox: 0.0,
			tolerance:  1e-10,
		},
		{
			name:        "high utility and frequency slow decay",
			turnsSince:  10,
			utility:     1.0,
			frequency:   1.0,
			temperature: 10.0,
			// S = (1 + 1 + 0.01) * 10 = 20.1
			// R = exp(-10/20.1) ≈ 0.6084
			wantApprox: math.Exp(-10.0 / 20.1),
			tolerance:  1e-10,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Retention(tt.turnsSince, tt.utility, tt.frequency, tt.temperature)
			if math.Abs(got-tt.wantApprox) > tt.tolerance {
				t.Errorf("Retention(%d, %g, %g, %g) = %g, want ≈ %g (tolerance %g)",
					tt.turnsSince, tt.utility, tt.frequency, tt.temperature,
					got, tt.wantApprox, tt.tolerance)
			}
		})
	}
}

func TestStability(t *testing.T) {
	tests := []struct {
		name        string
		utility     float64
		frequency   float64
		temperature float64
		want        float64
	}{
		{
			name:        "cold start defaults",
			utility:     0.5,
			frequency:   0.5,
			temperature: 10.0,
			want:        10.1, // (0.5 + 0.5 + 0.01) * 10
		},
		{
			name:        "zero utility and frequency",
			utility:     0.0,
			frequency:   0.0,
			temperature: 10.0,
			want:        0.1, // (0 + 0 + 0.01) * 10
		},
		{
			name:        "max utility and frequency",
			utility:     1.0,
			frequency:   1.0,
			temperature: 10.0,
			want:        20.1, // (1 + 1 + 0.01) * 10
		},
		{
			name:        "zero temperature",
			utility:     0.5,
			frequency:   0.5,
			temperature: 0.0,
			want:        0.0, // (1.01) * 0 = 0
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Stability(tt.utility, tt.frequency, tt.temperature)
			if math.Abs(got-tt.want) > 1e-10 {
				t.Errorf("Stability(%g, %g, %g) = %g, want %g",
					tt.utility, tt.frequency, tt.temperature, got, tt.want)
			}
		})
	}
}
