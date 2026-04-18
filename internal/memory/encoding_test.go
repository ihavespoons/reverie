package memory

import (
	"math"
	"testing"
)

func TestEncodeDecodeVector(t *testing.T) {
	tests := []struct {
		name string
		vec  []float32
	}{
		{
			name: "standard vector",
			vec:  []float32{1.0, 2.0, 3.0, -4.5, 0.001},
		},
		{
			name: "single element",
			vec:  []float32{42.0},
		},
		{
			name: "zeros",
			vec:  []float32{0, 0, 0},
		},
		{
			name: "special float values",
			vec:  []float32{math.MaxFloat32, math.SmallestNonzeroFloat32, -math.MaxFloat32},
		},
		{
			name: "negative values",
			vec:  []float32{-1.5, -2.5, -3.5},
		},
		{
			name: "high-dimensional (1024)",
			vec:  make1024Vec(),
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			encoded := EncodeVector(tt.vec)
			if len(encoded) != len(tt.vec)*4 {
				t.Fatalf("EncodeVector: got %d bytes, want %d", len(encoded), len(tt.vec)*4)
			}

			decoded := DecodeVector(encoded)
			if len(decoded) != len(tt.vec) {
				t.Fatalf("DecodeVector: got %d elements, want %d", len(decoded), len(tt.vec))
			}

			for i := range tt.vec {
				if decoded[i] != tt.vec[i] {
					t.Errorf("round-trip mismatch at index %d: got %v, want %v", i, decoded[i], tt.vec[i])
				}
			}
		})
	}
}

func TestEncodeVectorNil(t *testing.T) {
	result := EncodeVector(nil)
	if result != nil {
		t.Errorf("EncodeVector(nil) = %v, want nil", result)
	}
}

func TestEncodeVectorEmpty(t *testing.T) {
	result := EncodeVector([]float32{})
	if result != nil {
		t.Errorf("EncodeVector(empty) = %v, want nil", result)
	}
}

func TestDecodeVectorNil(t *testing.T) {
	result := DecodeVector(nil)
	if result != nil {
		t.Errorf("DecodeVector(nil) = %v, want nil", result)
	}
}

func TestDecodeVectorEmpty(t *testing.T) {
	result := DecodeVector([]byte{})
	if result != nil {
		t.Errorf("DecodeVector(empty) = %v, want nil", result)
	}
}

func TestDecodeVectorTrailingBytes(t *testing.T) {
	// 5 bytes: 1 full float32 + 1 trailing byte ignored.
	vec := []float32{1.5}
	encoded := EncodeVector(vec)
	padded := append(encoded, 0xFF) // trailing byte
	decoded := DecodeVector(padded)
	if len(decoded) != 1 {
		t.Fatalf("DecodeVector with trailing byte: got %d elements, want 1", len(decoded))
	}
	if decoded[0] != 1.5 {
		t.Errorf("DecodeVector with trailing byte: got %v, want 1.5", decoded[0])
	}
}

func make1024Vec() []float32 {
	v := make([]float32, 1024)
	for i := range v {
		v[i] = float32(i) * 0.001
	}
	return v
}
