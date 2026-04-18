package decay

import "personal/reverie/internal/memory"

// Decayer evaluates Gate C (Ebbinghaus retention) on individual clusters.
// It has no state of its own — it reads from the passed ClusterNode.
type Decayer interface {
	// Retention returns R_t(c) for the cluster. Pure function of cluster state + config.
	Retention(c memory.ClusterNode) float64
	// GateC returns true if the cluster's retention exceeds the configured threshold.
	GateC(c memory.ClusterNode) bool
	// Temperature returns the configured T value (exposed for observability).
	Temperature() float64
	// Threshold returns the configured retention threshold (exposed for observability).
	Threshold() float64
}

// NewDecayer constructs a Decayer with the given temperature and retention threshold.
// If temperature <= 0, falls back to DefaultTemperature.
func NewDecayer(temperature, retentionThreshold float64) Decayer {
	if temperature <= 0 {
		temperature = DefaultTemperature
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

// Retention returns R_t(c) for the given cluster, delegating to the
// package-level Retention function with the configured temperature.
func (d *decayer) Retention(c memory.ClusterNode) float64 {
	return Retention(c.TurnsSince, c.Utility, c.Frequency, d.temperature)
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
