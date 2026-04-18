package cluster

// UpdateCentroid computes the running mean of the old centroid with a new vec.
//
//	new_centroid[i] = (oldCentroid[i] * itemCount + vec[i]) / (itemCount + 1)
//
// If oldCentroid is nil or has different length than vec, returns a copy of vec
// (caller's data was malformed but we recover gracefully).
func UpdateCentroid(oldCentroid []float32, itemCount int, vec []float32) []float32 {
	if len(oldCentroid) != len(vec) || len(oldCentroid) == 0 {
		return append([]float32(nil), vec...)
	}
	result := make([]float32, len(vec))
	n := float32(itemCount)
	for i := range vec {
		result[i] = (oldCentroid[i]*n + vec[i]) / (n + 1)
	}
	return result
}
