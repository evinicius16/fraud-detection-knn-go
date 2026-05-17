package search

import (
	"math"
	"testing"

	"github.com/evinicius16/fraud-detection-knn-go/internal/vectorize"
)

func makeTestDataset() *Dataset {
	// 10 vectors from testdata/references_small.json
	vectors := []float32{
		// 0: legit
		0.01, 0.0833, 0.05, 0.8261, 0.1667, -1, -1, 0.0432, 0.25, 0, 1, 0, 0.2, 0.0416,
		// 1: legit
		0.02, 0.0833, 0.06, 0.8261, 0.1667, -1, -1, 0.0450, 0.25, 0, 1, 0, 0.2, 0.0420,
		// 2: legit
		0.015, 0.0833, 0.055, 0.8261, 0.3333, -1, -1, 0.0440, 0.20, 0, 1, 0, 0.15, 0.0400,
		// 3: legit
		0.03, 0.1667, 0.04, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.0060,
		// 4: legit
		0.025, 0.0833, 0.07, 0.8261, 0.1667, -1, -1, 0.0500, 0.30, 0, 1, 0, 0.2, 0.0450,
		// 5: fraud
		0.5796, 0.9167, 1.0, 0.0435, 0, 0.0056, 0.4394, 0.4598, 0.4, 1, 0, 1, 0.85, 0.0032,
		// 6: fraud
		0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055,
		// 7: fraud
		0.8500, 0.7500, 0.95, 0.1304, 0.6667, -1, -1, 0.8800, 0.9, 1, 0, 1, 0.80, 0.0040,
		// 8: fraud
		0.7200, 0.9167, 0.88, 0.0870, 0.5000, 0.0100, 0.5000, 0.7500, 0.85, 1, 0, 1, 0.85, 0.0028,
		// 9: fraud
		0.9000, 1.0, 0.99, 0.1739, 0.8333, -1, -1, 0.9100, 0.95, 0, 1, 1, 0.75, 0.0050,
	}

	labels := []bool{
		false, false, false, false, false, // legit (0-4)
		true, true, true, true, true, // fraud (5-9)
	}

	return &Dataset{
		Vectors: vectors,
		Labels:  labels,
		Count:   10,
	}
}

func TestBruteForceKNN_LegitQuery(t *testing.T) {
	ds := makeTestDataset()

	// Query similar to legit vectors (low amount, normal hour, no fraud indicators)
	query := make([]float32, vectorize.VectorDim)
	query[0] = 0.02   // amount
	query[1] = 0.0833 // installments
	query[2] = 0.05   // amount_vs_avg
	query[3] = 0.8261 // hour_of_day
	query[4] = 0.1667 // day_of_week
	query[5] = -1     // minutes_since_last_tx (null)
	query[6] = -1     // km_from_last_tx (null)
	query[7] = 0.04   // km_from_home
	query[8] = 0.25   // tx_count_24h
	query[9] = 0      // is_online
	query[10] = 1     // card_present
	query[11] = 0     // unknown_merchant
	query[12] = 0.2   // mcc_risk
	query[13] = 0.04  // merchant_avg_amount

	fraudCount := ds.BruteForceKNN(query)

	// Should find mostly legit neighbors (0 or very few frauds)
	if fraudCount > 2 {
		t.Errorf("Expected mostly legit neighbors for legit-like query, got %d frauds out of 5", fraudCount)
	}
}

func TestBruteForceKNN_FraudQuery(t *testing.T) {
	ds := makeTestDataset()

	// Query similar to fraud vectors (high amount, unusual patterns)
	query := make([]float32, vectorize.VectorDim)
	query[0] = 0.9    // amount (high)
	query[1] = 0.9    // installments (high)
	query[2] = 1.0    // amount_vs_avg (way above average)
	query[3] = 0.2    // hour_of_day (early morning)
	query[4] = 0.8    // day_of_week
	query[5] = -1     // minutes_since_last_tx
	query[6] = -1     // km_from_last_tx
	query[7] = 0.9    // km_from_home (far)
	query[8] = 0.95   // tx_count_24h (many)
	query[9] = 0      // is_online
	query[10] = 1     // card_present
	query[11] = 1     // unknown_merchant
	query[12] = 0.75  // mcc_risk (high)
	query[13] = 0.005 // merchant_avg_amount

	fraudCount := ds.BruteForceKNN(query)

	// Should find mostly fraud neighbors (3+ out of 5)
	if fraudCount < 3 {
		t.Errorf("Expected mostly fraud neighbors for fraud-like query, got %d frauds out of 5", fraudCount)
	}
}

func TestComputeFraudScore(t *testing.T) {
	tests := []struct {
		fraudCount int
		expected   float64
	}{
		{0, 0.0},
		{1, 0.2},
		{2, 0.4},
		{3, 0.6},
		{4, 0.8},
		{5, 1.0},
	}

	for _, tt := range tests {
		score := ComputeFraudScore(tt.fraudCount)
		if math.Abs(score-tt.expected) > 0.001 {
			t.Errorf("ComputeFraudScore(%d) = %f, want %f", tt.fraudCount, score, tt.expected)
		}
	}
}

func TestIsApproved(t *testing.T) {
	tests := []struct {
		score    float64
		approved bool
	}{
		{0.0, true},
		{0.2, true},
		{0.4, true},
		{0.59, true},
		{0.6, false},  // threshold is strict <
		{0.8, false},
		{1.0, false},
	}

	for _, tt := range tests {
		result := IsApproved(tt.score)
		if result != tt.approved {
			t.Errorf("IsApproved(%f) = %v, want %v", tt.score, result, tt.approved)
		}
	}
}

func TestEuclideanDistSq(t *testing.T) {
	a := []float32{1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}
	b := []float32{0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}

	dist := euclideanDistSq(a, b)
	if math.Abs(float64(dist)-1.0) > 0.0001 {
		t.Errorf("euclideanDistSq = %f, want 1.0", dist)
	}

	// Same vector should have distance 0
	dist = euclideanDistSq(a, a)
	if dist != 0.0 {
		t.Errorf("euclideanDistSq(a, a) = %f, want 0.0", dist)
	}
}

func TestBruteForceKNN_ExactMatch(t *testing.T) {
	ds := makeTestDataset()

	// Use exact copy of vector[6] (fraud) as query
	query := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}

	fraudCount := ds.BruteForceKNN(query)

	// The exact match (fraud) + 4 nearest should be mostly fraud
	if fraudCount < 4 {
		t.Errorf("Expected at least 4 fraud neighbors for exact fraud vector match, got %d", fraudCount)
	}
}

func BenchmarkBruteForceKNN(b *testing.B) {
	ds := makeTestDataset()
	query := []float32{0.5, 0.5, 0.5, 0.5, 0.5, -1, -1, 0.5, 0.5, 0, 1, 0, 0.5, 0.5}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ds.BruteForceKNN(query)
	}
}
