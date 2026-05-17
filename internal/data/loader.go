package data

import (
	"compress/gzip"
	"encoding/json"
	"fmt"
	"os"

	"github.com/evinicius16/fraud-detection-knn-go/internal/search"
	"github.com/evinicius16/fraud-detection-knn-go/internal/vectorize"
)

// ReferenceEntry represents a single entry in references.json.
type ReferenceEntry struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

// LoadReferences loads the reference dataset from a gzipped JSON file.
func LoadReferences(path string) (*search.Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open references file: %w", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		return nil, fmt.Errorf("gzip reader: %w", err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)

	// Expect opening bracket
	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("read opening token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, fmt.Errorf("expected '[', got %v", tok)
	}

	// Pre-allocate for ~3M entries
	const estimatedCount = 3_000_000
	dim := vectorize.VectorDim

	vectors := make([]float32, 0, estimatedCount*dim)
	labels := make([]bool, 0, estimatedCount)

	var entry ReferenceEntry
	count := 0

	for dec.More() {
		if err := dec.Decode(&entry); err != nil {
			return nil, fmt.Errorf("decode entry %d: %w", count, err)
		}

		if len(entry.Vector) != dim {
			return nil, fmt.Errorf("entry %d: expected %d dimensions, got %d", count, dim, len(entry.Vector))
		}

		for _, v := range entry.Vector {
			vectors = append(vectors, float32(v))
		}
		labels = append(labels, entry.Label == "fraud")
		count++
	}

	// Read closing bracket
	_, _ = dec.Token()

	fmt.Printf("[data] Loaded %d reference vectors\n", count)

	return &search.Dataset{
		Vectors: vectors,
		Labels:  labels,
		Count:   count,
	}, nil
}

// LoadReferencesJSON loads from an uncompressed JSON file (for testing).
func LoadReferencesJSON(path string) (*search.Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open references file: %w", err)
	}
	defer f.Close()

	dec := json.NewDecoder(f)

	tok, err := dec.Token()
	if err != nil {
		return nil, fmt.Errorf("read opening token: %w", err)
	}
	if delim, ok := tok.(json.Delim); !ok || delim != '[' {
		return nil, fmt.Errorf("expected '[', got %v", tok)
	}

	dim := vectorize.VectorDim
	vectors := make([]float32, 0, 1000*dim)
	labels := make([]bool, 0, 1000)

	var entry ReferenceEntry
	count := 0

	for dec.More() {
		if err := dec.Decode(&entry); err != nil {
			return nil, fmt.Errorf("decode entry %d: %w", count, err)
		}

		if len(entry.Vector) != dim {
			return nil, fmt.Errorf("entry %d: expected %d dimensions, got %d", count, dim, len(entry.Vector))
		}

		for _, v := range entry.Vector {
			vectors = append(vectors, float32(v))
		}
		labels = append(labels, entry.Label == "fraud")
		count++
	}

	_, _ = dec.Token()

	return &search.Dataset{
		Vectors: vectors,
		Labels:  labels,
		Count:   count,
	}, nil
}

// LoadMCCRisk loads the MCC risk map from a JSON file.
func LoadMCCRisk(path string) (vectorize.MCCRisk, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open mcc_risk file: %w", err)
	}
	defer f.Close()

	var risk vectorize.MCCRisk
	if err := json.NewDecoder(f).Decode(&risk); err != nil {
		return nil, fmt.Errorf("decode mcc_risk: %w", err)
	}
	return risk, nil
}

// LoadNormConstants loads normalization constants from a JSON file.
func LoadNormConstants(path string) (vectorize.NormConstants, error) {
	f, err := os.Open(path)
	if err != nil {
		return vectorize.NormConstants{}, fmt.Errorf("open normalization file: %w", err)
	}
	defer f.Close()

	var norm vectorize.NormConstants
	if err := json.NewDecoder(f).Decode(&norm); err != nil {
		return vectorize.NormConstants{}, fmt.Errorf("decode normalization: %w", err)
	}
	return norm, nil
}
