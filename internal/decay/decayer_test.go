package decay

import (
	"math"
	"testing"

	"github.com/diffsec/reverie/internal/memory"
	"github.com/diffsec/reverie/pkg/ebbinghaus"
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
		if d.Temperature() != ebbinghaus.DefaultTemperature {
			t.Errorf("Temperature() = %g, want %g (ebbinghaus.DefaultTemperature)", d.Temperature(), ebbinghaus.DefaultTemperature)
		}
	})

	t.Run("negative temperature falls back to default", func(t *testing.T) {
		d := NewDecayer(-5.0, 0.3)
		if d.Temperature() != ebbinghaus.DefaultTemperature {
			t.Errorf("Temperature() = %g, want %g (ebbinghaus.DefaultTemperature)", d.Temperature(), ebbinghaus.DefaultTemperature)
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
			wantApprox: ebbinghaus.Retention(20, 0.8, 0.3, 10.0),
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

func TestDecayer_RetentionFromState(t *testing.T) {
	d := NewDecayer(10.0, 0.3)

	t.Run("zero turns_since yields full retention", func(t *testing.T) {
		got := d.RetentionFromState(0.5, 0.5, 0)
		if math.Abs(got-1.0) > 1e-12 {
			t.Errorf("RetentionFromState(0.5, 0.5, 0) = %g, want 1.0", got)
		}
	})

	t.Run("strictly decreasing as turnsSince grows", func(t *testing.T) {
		prev := math.Inf(1)
		for n := 0; n <= 25; n++ {
			cur := d.RetentionFromState(0.4, 0.6, n)
			if n > 0 && cur >= prev {
				t.Errorf("retention not strictly decreasing at n=%d: prev=%g cur=%g", n, prev, cur)
			}
			prev = cur
		}
	})

	t.Run("matches Retention(ClusterNode)", func(t *testing.T) {
		c := memory.ClusterNode{TurnsSince: 13, Utility: 0.7, Frequency: 0.2}
		viaCluster := d.Retention(c)
		viaScalar := d.RetentionFromState(0.7, 0.2, 13)
		if math.Abs(viaCluster-viaScalar) > 1e-15 {
			t.Errorf("Retention(c)=%g RetentionFromState(...)=%g; want equal", viaCluster, viaScalar)
		}
	})
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
			want: ebbinghaus.Retention(12, 0.5, 0.5, 10.0) > 0.3,
		},
		{
			name: "retention just below threshold",
			cluster: memory.ClusterNode{
				TurnsSince: 13,
				Utility:    0.5,
				Frequency:  0.5,
			},
			// S = 10.1, R = exp(-13/10.1) ≈ 0.2762 < 0.3 → false
			want: ebbinghaus.Retention(13, 0.5, 0.5, 10.0) > 0.3,
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
