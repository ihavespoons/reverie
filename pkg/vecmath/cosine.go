// Package vecmath provides pure-Go vector math utilities for similarity
// search. Used by reverie's brute-force cosine search over memory embeddings.
package vecmath

import "math"

// Cosine computes the cosine similarity between vectors a and b.
// Returns a value in [-1, 1] where 1 means identical direction, 0 means
// orthogonal, and -1 means opposite direction.
//
// Returns 0 if either vector is empty, if they have mismatched lengths,
// or if either vector has zero magnitude.
func Cosine(a, b []float32) float32 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		ai := float64(a[i])
		bi := float64(b[i])
		dot += ai * bi
		normA += ai * ai
		normB += bi * bi
	}

	if normA == 0 || normB == 0 {
		return 0
	}

	return float32(dot / math.Sqrt(normA*normB))
}

// Normalize returns a new unit vector in the same direction as a.
// Returns a zero-length slice if a is empty. Returns a zero vector (same
// length) if a has zero magnitude. The input slice is never mutated.
func Normalize(a []float32) []float32 {
	if len(a) == 0 {
		return []float32{}
	}

	var norm float64
	for _, v := range a {
		norm += float64(v) * float64(v)
	}

	result := make([]float32, len(a))
	if norm == 0 {
		return result
	}

	norm = math.Sqrt(norm)
	for i, v := range a {
		result[i] = float32(float64(v) / norm)
	}
	return result
}
