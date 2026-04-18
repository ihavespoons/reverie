package decay

import (
	"math"
	"testing"

	"personal/reverie/internal/memory"
)

func TestNewDecayer(t *testing.T) {
	t.Run("respects configured temperature and threshold", func(t *testing.T) {
		d := NewDecayer(10.0, 0.3)
		if d.Temperature() != 10.0 {
			t.Errorf("Temperature() = %g, want 10.0", d.Temperature())
		}
		if d.Threshold() != 0.3 {
			t.Errorf("Threshold() = %g, want 0.3", d.Threshold())
		}
	})

	t.Run("zero temperature falls back to default", func(t *testing.T) {
		d := NewDecayer(0, 0.3)
		if d.Temperature() != DefaultTemperature {
			t.Errorf("Temperature() = %g, want %g (DefaultTemperature)", d.Temperature(), DefaultTemperature)
		}
	})

	t.Run("negative temperature falls back to default", func(t *testing.T) {
		d := NewDecayer(-5.0, 0.3)
		if d.Temperature() != DefaultTemperature {
			t.Errorf("Temperature() = %g, want %g (DefaultTemperature)", d.Temperature(), DefaultTemperature)
		}
	})
}

func TestDecayerRetention(t *testing.T) {
	d := NewDecayer(10.0, 0.3)

	tests := []struct {
		name       string
		cluster    memory.ClusterNode
		wantApprox float64
		tolerance  float64
	}{
		{
			name: "delegates correctly for fresh cluster",
			cluster: memory.ClusterNode{
				TurnsSince: 0,
				Utility:    0.5,
				Frequency:  0.5,
			},
			wantApprox: 1.0,
			tolerance:  1e-10,
		},
		{
			name: "delegates correctly for aged cluster",
			cluster: memory.ClusterNode{
				TurnsSince: 10,
				Utility:    0.5,
				Frequency:  0.5,
			},
			// S = 10.1, R = exp(-10/10.1) ≈ 0.3725
			wantApprox: math.Exp(-10.0 / 10.1),
			tolerance:  1e-10,
		},
		{
			name: "matches package-level function",
			cluster: memory.ClusterNode{
				TurnsSince: 20,
				Utility:    0.8,
				Frequency:  0.3,
			},
			wantApprox: Retention(20, 0.8, 0.3, 10.0),
			tolerance:  1e-15,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := d.Retention(tt.cluster)
			if math.Abs(got-tt.wantApprox) > tt.tolerance {
				t.Errorf("Decayer.Retention(%+v) = %g, want ≈ %g", tt.cluster, got, tt.wantApprox)
			}
		})
	}
}

func TestDecayerGateC(t *testing.T) {
	d := NewDecayer(10.0, 0.3)

	tests := []struct {
		name    string
		cluster memory.ClusterNode
		want    bool
	}{
		{
			name: "full retention passes gate",
			cluster: memory.ClusterNode{
				TurnsSince: 0,
				Utility:    0.5,
				Frequency:  0.5,
			},
			// R = 1.0 > 0.3 → true
			want: true,
		},
		{
			name: "half retention above threshold passes",
			cluster: memory.ClusterNode{
				TurnsSince: 5,
				Utility:    0.5,
				Frequency:  0.5,
			},
			// S = 10.1, R = exp(-5/10.1) ≈ 0.6097 > 0.3 → true
			want: true,
		},
		{
			name: "very low retention fails gate",
			cluster: memory.ClusterNode{
				TurnsSince: 100,
				Utility:    0.0,
				Frequency:  0.0,
			},
			// S = 0.1, R = exp(-100/0.1) = exp(-1000) ≈ 0 → false
			want: false,
		},
		{
			name: "retention near threshold boundary",
			cluster: memory.ClusterNode{
				TurnsSince: 12,
				Utility:    0.5,
				Frequency:  0.5,
			},
			// S = 10.1, R = exp(-12/10.1) ≈ 0.3057 > 0.3 → true (barely)
			want: Retention(12, 0.5, 0.5, 10.0) > 0.3,
		},
		{
			name: "retention just below threshold",
			cluster: memory.ClusterNode{
				TurnsSince: 13,
				Utility:    0.5,
				Frequency:  0.5,
			},
			// S = 10.1, R = exp(-13/10.1) ≈ 0.2762 < 0.3 → false
			want: Retention(13, 0.5, 0.5, 10.0) > 0.3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := d.GateC(tt.cluster)
			if got != tt.want {
				r := d.Retention(tt.cluster)
				t.Errorf("GateC(%+v) = %v, want %v (retention=%g, threshold=%g)",
					tt.cluster, got, tt.want, r, d.Threshold())
			}
		})
	}
}
