// Package decay implements the Ebbinghaus retention curve and Gate C filter
// for the reverie memory system. Retention is computed on demand from cluster
// state (utility, frequency, turns-since-access) and a global temperature
// parameter. No state is held; all functions are pure.
package decay

import "math"

const (
	// DefaultTemperature is the global temperature T used when no
	// configuration value is provided (or the configured value is <= 0).
	// Higher values slow decay; lower values accelerate it.
	DefaultTemperature = 10.0

	// Epsilon is a small additive constant that prevents stability from
	// reaching zero when both utility and frequency are zero. Without it,
	// a brand-new cluster with U=F=0 would have stability=0, making the
	// retention formula undefined (division by zero).
	Epsilon = 0.01
)

// Stability computes the stability term S_t(c) = (U_t(c) + F_t(c) + epsilon) * T.
// This is the denominator scaling factor in the Ebbinghaus retention curve.
// Exported separately for debuggability and testing.
func Stability(utility, frequency, temperature float64) float64 {
	return (utility + frequency + Epsilon) * temperature
}

// Retention computes R_t(c) = exp(-n_t(c) / S_t(c))
// where S_t(c) = (U_t(c) + F_t(c) + epsilon) * T.
//
// Parameters:
//   - turnsSince: n_t(c) — the number of turns since the cluster was last accessed.
//   - utility:    U_t(c) — the cluster's utility score, typically in [0,1].
//   - frequency:  F_t(c) — the cluster's frequency score, typically in [0,1].
//   - temperature: T     — the global decay temperature (higher = slower decay).
//
// Returns 0 if stability is non-positive (guards against divide-by-zero
// when temperature <= 0). Returns a value in (0, 1] otherwise.
func Retention(turnsSince int, utility, frequency, temperature float64) float64 {
	stability := Stability(utility, frequency, temperature)
	if stability <= 0 {
		return 0
	}
	return math.Exp(-float64(turnsSince) / stability)
}
