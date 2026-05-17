package data

import (
	"encoding/binary"
	"math"
	"os"
	"testing"

	"github.com/evinicius16/fraud-detection-knn-go/internal/vectorize"
)

func TestLoadReferencesJSON(t *testing.T) {
	ds, err := LoadReferencesJSON("../../testdata/references_small.json")
	if err != nil {
		t.Fatalf("LoadReferencesJSON failed: %v", err)
	}

	if ds.Count != 10 {
		t.Errorf("Expected 10 vectors, got %d", ds.Count)
	}

	if len(ds.Vectors) != 10*vectorize.VectorDim {
		t.Errorf("Expected %d float32 values, got %d", 10*vectorize.VectorDim, len(ds.Vectors))
	}

	if len(ds.Labels) != 10 {
		t.Errorf("Expected 10 labels, got %d", len(ds.Labels))
	}

	// First 5 should be legit (false), last 5 should be fraud (true)
	for i := 0; i < 5; i++ {
		if ds.Labels[i] {
			t.Errorf("Label[%d] should be legit (false), got true", i)
		}
	}
	for i := 5; i < 10; i++ {
		if !ds.Labels[i] {
			t.Errorf("Label[%d] should be fraud (true), got false", i)
		}
	}
}

func TestLoadReferencesBinary(t *testing.T) {
	// Create a small binary file for testing
	tmpFile := createTestBinaryFile(t)
	defer os.Remove(tmpFile)

	ds, err := LoadReferencesBinary(tmpFile)
	if err != nil {
		t.Fatalf("LoadReferencesBinary failed: %v", err)
	}

	if ds.Count != 3 {
		t.Errorf("Expected 3 vectors, got %d", ds.Count)
	}

	// Verify first vector values
	expected := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5, 0.3, 0.1}
	for i, exp := range expected {
		if math.Abs(float64(ds.Vectors[i]-exp)) > 0.0001 {
			t.Errorf("Vector[0][%d] = %f, want %f", i, ds.Vectors[i], exp)
		}
	}

	// Verify labels
	if ds.Labels[0] != false {
		t.Error("Label[0] should be legit")
	}
	if ds.Labels[1] != true {
		t.Error("Label[1] should be fraud")
	}
	if ds.Labels[2] != false {
		t.Error("Label[2] should be legit")
	}
}

func TestLoadReferencesMmap(t *testing.T) {
	tmpFile := createTestBinaryFile(t)
	defer os.Remove(tmpFile)

	ds, err := LoadReferencesMmap(tmpFile)
	if err != nil {
		t.Fatalf("LoadReferencesMmap failed: %v", err)
	}

	if ds.Count != 3 {
		t.Errorf("Expected 3 vectors, got %d", ds.Count)
	}

	// Verify first vector values
	expected := []float32{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5, 0.3, 0.1}
	for i, exp := range expected {
		if math.Abs(float64(ds.Vectors[i]-exp)) > 0.0001 {
			t.Errorf("Vector[0][%d] = %f, want %f", i, ds.Vectors[i], exp)
		}
	}

	// Verify labels
	if ds.Labels[0] != false {
		t.Error("Label[0] should be legit")
	}
	if ds.Labels[1] != true {
		t.Error("Label[1] should be fraud")
	}
	if ds.Labels[2] != false {
		t.Error("Label[2] should be legit")
	}
}

func TestLoadMCCRisk(t *testing.T) {
	risk, err := LoadMCCRisk("../../data/mcc_risk.json")
	if err != nil {
		t.Fatalf("LoadMCCRisk failed: %v", err)
	}

	if len(risk) != 10 {
		t.Errorf("Expected 10 MCC entries, got %d", len(risk))
	}

	if risk["5411"] != 0.15 {
		t.Errorf("MCC 5411 risk = %f, want 0.15", risk["5411"])
	}

	if risk["7995"] != 0.85 {
		t.Errorf("MCC 7995 risk = %f, want 0.85", risk["7995"])
	}
}

func TestLoadNormConstants(t *testing.T) {
	norm, err := LoadNormConstants("../../data/normalization.json")
	if err != nil {
		t.Fatalf("LoadNormConstants failed: %v", err)
	}

	if norm.MaxAmount != 10000 {
		t.Errorf("MaxAmount = %f, want 10000", norm.MaxAmount)
	}
	if norm.MaxInstallments != 12 {
		t.Errorf("MaxInstallments = %f, want 12", norm.MaxInstallments)
	}
	if norm.MaxKm != 1000 {
		t.Errorf("MaxKm = %f, want 1000", norm.MaxKm)
	}
}

// createTestBinaryFile creates a temporary binary file with 3 test vectors.
func createTestBinaryFile(t *testing.T) string {
	t.Helper()

	f, err := os.CreateTemp("", "test-refs-*.bin")
	if err != nil {
		t.Fatalf("create temp file: %v", err)
	}

	count := uint32(3)
	dim := vectorize.VectorDim

	// Write header
	header := make([]byte, 4)
	binary.LittleEndian.PutUint32(header, count)
	f.Write(header)

	// Write 3 vectors
	vectors := [][]float32{
		{0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 0.0, 0.5, 0.3, 0.1},
		{0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 1.0, 0.5, 0.7, 0.9},
		{0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0.0, 1.0, 0.0, 0.5, 0.5},
	}

	buf := make([]byte, 4)
	for _, vec := range vectors {
		if len(vec) != dim {
			t.Fatalf("test vector has %d dims, want %d", len(vec), dim)
		}
		for _, v := range vec {
			binary.LittleEndian.PutUint32(buf, math.Float32bits(v))
			f.Write(buf)
		}
	}

	// Write labels: legit, fraud, legit
	f.Write([]byte{0, 1, 0})

	name := f.Name()
	f.Close()
	return name
}
