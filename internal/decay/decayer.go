// Package decay wraps the pure Ebbinghaus formula (pkg/ebbinghaus) with
// cluster-aware helpers and the configured temperature/threshold. Callers
// that need the raw formula (no ClusterNode, no config) should import
// pkg/ebbinghaus directly.
package decay

import (
	"github.com/diffsec/reverie/internal/memory"
	"github.com/diffsec/reverie/pkg/ebbinghaus"
)

// Decayer evaluates Gate C (Ebbinghaus retention) on individual clusters.
// It has no state of its own — it reads from the passed ClusterNode.
type Decayer interface {
	// Retention returns R_t(c) for the cluster. Pure function of cluster state + config.
	Retention(c memory.ClusterNode) float64
	// RetentionFromState computes Ebbinghaus retention from the raw decay
	// inputs (utility, frequency, turns since last access). Useful for
	// callers that do not model their decayed objects as ClusterNode
	// (e.g. entities). Same formula as Retention; both delegate to
	// pkg/ebbinghaus so the math has one source of truth.
	RetentionFromState(utility, frequency float64, turnsSince int) float64
	// GateC returns true if the cluster's retention exceeds the configured threshold.
	GateC(c memory.ClusterNode) bool
	// Temperature returns the configured T value (exposed for observability).
	Temperature() float64
	// Threshold returns the configured retention threshold (exposed for observability).
	Threshold() float64
}

// NewDecayer constructs a Decayer with the given temperature and retention threshold.
// If temperature <= 0, falls back to ebbinghaus.DefaultTemperature.
func NewDecayer(temperature, retentionThreshold float64) Decayer {
	if temperature <= 0 {
		temperature = ebbinghaus.DefaultTemperature
	}
	return &decayer{
		temperature:        temperature,
		retentionThreshold: retentionThreshold,
	}
}

// decayer is the concrete, unexported implementation of Decayer.
type decayer struct {
	temperature        float64
	retentionThreshold float64
}

// Retention returns R_t(c) for the given cluster. Thin wrapper around
// RetentionFromState so the Ebbinghaus formula has a single source of
// truth (Retention/RetentionFromState/GateC all share one math path).
func (d *decayer) Retention(c memory.ClusterNode) float64 {
	return d.RetentionFromState(c.Utility, c.Frequency, c.TurnsSince)
}

// RetentionFromState computes R_t from raw scalar state. Used by callers
// that operate on entities (or any other decayed object that is not a
// ClusterNode) — same formula, just without packing into a ClusterNode
// first. Delegates to pkg/ebbinghaus.
func (d *decayer) RetentionFromState(utility, frequency float64, turnsSince int) float64 {
	return ebbinghaus.Retention(turnsSince, utility, frequency, d.temperature)
}

// GateC returns true if the cluster's Ebbinghaus retention score exceeds
// the configured retention threshold. A cluster that passes Gate C is
// considered "remembered" and eligible for inclusion in the working memory.
func (d *decayer) GateC(c memory.ClusterNode) bool {
	return d.Retention(c) > d.retentionThreshold
}

// Temperature returns the configured decay temperature T.
func (d *decayer) Temperature() float64 {
	return d.temperature
}

// Threshold returns the configured retention threshold used by GateC.
func (d *decayer) Threshold() float64 {
	return d.retentionThreshold
}
