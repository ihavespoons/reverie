package memory

import (
	"encoding/binary"
	"math"
)

// EncodeVector encodes a float32 vector as a little-endian byte slice for
// storage as a SQLite BLOB. Each float32 is encoded as 4 bytes.
// Returns nil for nil or empty input.
func EncodeVector(v []float32) []byte {
	if len(v) == 0 {
		return nil
	}
	buf := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(f))
	}
	return buf
}

// DecodeVector decodes a little-endian byte slice back into a float32 vector.
// Returns nil for nil or empty input. If the byte slice length is not a
// multiple of 4, trailing bytes are silently ignored.
func DecodeVector(b []byte) []float32 {
	if len(b) == 0 {
		return nil
	}
	n := len(b) / 4
	v := make([]float32, n)
	for i := range n {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}
