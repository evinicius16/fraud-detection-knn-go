package integration

import (
	"encoding/json"
	"math"
	"os"
	"testing"

	"github.com/evinicius16/fraud-detection-knn-go/internal/data"
	"github.com/evinicius16/fraud-detection-knn-go/internal/model"
	"github.com/evinicius16/fraud-detection-knn-go/internal/search"
	"github.com/evinicius16/fraud-detection-knn-go/internal/vectorize"
)

// TestEndToEnd_LegitTransaction tests the full pipeline with the legit example from docs.
func TestEndToEnd_LegitTransaction(t *testing.T) {
	ds := loadTestDataset(t)
	vec := vectorize.NewVectorizer(vectorize.DefaultNormConstants(), vectorize.DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-1329056812",
		Transaction: model.Transaction{
			Amount:      41.12,
			Installments: 2,
			RequestedAt: "2026-03-11T18:45:53Z",
		},
		Customer: model.Customer{
			AvgAmount:      82.24,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-003", "MERC-016"},
		},
		Merchant: model.Merchant{
			ID:        "MERC-016",
			MCC:       "5411",
			AvgAmount: 60.25,
		},
		Terminal: model.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  29.23,
		},
		LastTransaction: nil,
	}

	query := make([]float32, vectorize.VectorDim)
	vec.Vectorize(req, query)

	fraudCount := ds.BruteForceKNN(query)
	fraudScore := search.ComputeFraudScore(fraudCount)
	approved := search.IsApproved(fraudScore)

	// This should be approved (legit transaction)
	if !approved {
		t.Errorf("Expected legit transaction to be approved, got fraud_score=%f", fraudScore)
	}
	t.Logf("Legit transaction: fraud_score=%f, approved=%v", fraudScore, approved)
}

// TestEndToEnd_FraudTransaction tests the full pipeline with the fraud example from docs.
func TestEndToEnd_FraudTransaction(t *testing.T) {
	ds := loadTestDataset(t)
	vec := vectorize.NewVectorizer(vectorize.DefaultNormConstants(), vectorize.DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-3330991687",
		Transaction: model.Transaction{
			Amount:      9505.97,
			Installments: 10,
			RequestedAt: "2026-03-14T05:15:12Z",
		},
		Customer: model.Customer{
			AvgAmount:      81.28,
			TxCount24h:     20,
			KnownMerchants: []string{"MERC-008", "MERC-007", "MERC-005"},
		},
		Merchant: model.Merchant{
			ID:        "MERC-068",
			MCC:       "7802",
			AvgAmount: 54.86,
		},
		Terminal: model.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  952.27,
		},
		LastTransaction: nil,
	}

	query := make([]float32, vectorize.VectorDim)
	vec.Vectorize(req, query)

	fraudCount := ds.BruteForceKNN(query)
	fraudScore := search.ComputeFraudScore(fraudCount)
	approved := search.IsApproved(fraudScore)

	// This should NOT be approved (fraud transaction)
	if approved {
		t.Errorf("Expected fraud transaction to be denied, got fraud_score=%f", fraudScore)
	}
	t.Logf("Fraud transaction: fraud_score=%f, approved=%v", fraudScore, approved)
}

// TestEndToEnd_WithLastTransaction tests a transaction with last_transaction present.
func TestEndToEnd_WithLastTransaction(t *testing.T) {
	ds := loadTestDataset(t)
	vec := vectorize.NewVectorizer(vectorize.DefaultNormConstants(), vectorize.DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-100",
		Transaction: model.Transaction{
			Amount:      384.88,
			Installments: 3,
			RequestedAt: "2026-03-11T20:23:35Z",
		},
		Customer: model.Customer{
			AvgAmount:      769.76,
			TxCount24h:     3,
			KnownMerchants: []string{"MERC-009", "MERC-001"},
		},
		Merchant: model.Merchant{
			ID:        "MERC-001",
			MCC:       "5912",
			AvgAmount: 298.95,
		},
		Terminal: model.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  13.709,
		},
		LastTransaction: &model.LastTransaction{
			Timestamp:     "2026-03-11T14:58:35Z",
			KmFromCurrent: 18.863,
		},
	}

	query := make([]float32, vectorize.VectorDim)
	vec.Vectorize(req, query)

	// Verify indices 5 and 6 are NOT -1
	if query[5] == -1.0 || query[6] == -1.0 {
		t.Error("Expected indices 5 and 6 to NOT be -1 when last_transaction is present")
	}

	fraudCount := ds.BruteForceKNN(query)
	fraudScore := search.ComputeFraudScore(fraudCount)
	approved := search.IsApproved(fraudScore)

	t.Logf("Transaction with last_tx: fraud_score=%f, approved=%v, vector[5]=%f, vector[6]=%f",
		fraudScore, approved, query[5], query[6])
}

// TestEndToEnd_ResponseFormat verifies the response JSON format.
func TestEndToEnd_ResponseFormat(t *testing.T) {
	resp := model.FraudResponse{
		Approved:   false,
		FraudScore: 1.0,
	}

	b, err := json.Marshal(resp)
	if err != nil {
		t.Fatalf("Failed to marshal response: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(b, &parsed); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if _, ok := parsed["approved"]; !ok {
		t.Error("Response missing 'approved' field")
	}
	if _, ok := parsed["fraud_score"]; !ok {
		t.Error("Response missing 'fraud_score' field")
	}

	// Verify fraud_score is a number
	score, ok := parsed["fraud_score"].(float64)
	if !ok {
		t.Error("fraud_score is not a number")
	}
	if score < 0 || score > 1 {
		t.Errorf("fraud_score %f is out of range [0, 1]", score)
	}
}

// TestEndToEnd_ScoreValues verifies all possible fraud_score values.
func TestEndToEnd_ScoreValues(t *testing.T) {
	// fraud_score can only be 0.0, 0.2, 0.4, 0.6, 0.8, or 1.0
	validScores := map[float64]bool{
		0.0: true, 0.2: true, 0.4: true, 0.6: true, 0.8: true, 1.0: true,
	}

	for fraudCount := 0; fraudCount <= 5; fraudCount++ {
		score := search.ComputeFraudScore(fraudCount)
		if !validScores[score] {
			t.Errorf("ComputeFraudScore(%d) = %f, not a valid score", fraudCount, score)
		}
	}
}

// TestEndToEnd_ApprovalThreshold verifies the 0.6 threshold behavior.
func TestEndToEnd_ApprovalThreshold(t *testing.T) {
	// 0 frauds → 0.0 → approved
	// 1 fraud  → 0.2 → approved
	// 2 frauds → 0.4 → approved
	// 3 frauds → 0.6 → NOT approved (>= 0.6)
	// 4 frauds → 0.8 → NOT approved
	// 5 frauds → 1.0 → NOT approved

	tests := []struct {
		fraudCount int
		approved   bool
	}{
		{0, true},
		{1, true},
		{2, true},
		{3, false},
		{4, false},
		{5, false},
	}

	for _, tt := range tests {
		score := search.ComputeFraudScore(tt.fraudCount)
		approved := search.IsApproved(score)
		if approved != tt.approved {
			t.Errorf("fraudCount=%d → score=%f → approved=%v, want %v",
				tt.fraudCount, score, approved, tt.approved)
		}
	}
}

// TestEndToEnd_VectorDimensions verifies all 14 dimensions are populated.
func TestEndToEnd_VectorDimensions(t *testing.T) {
	vec := vectorize.NewVectorizer(vectorize.DefaultNormConstants(), vectorize.DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-dim-test",
		Transaction: model.Transaction{
			Amount:      500,
			Installments: 6,
			RequestedAt: "2026-06-15T14:30:00Z",
		},
		Customer: model.Customer{
			AvgAmount:      1000,
			TxCount24h:     5,
			KnownMerchants: []string{"MERC-001"},
		},
		Merchant: model.Merchant{
			ID:        "MERC-002",
			MCC:       "5812",
			AvgAmount: 300,
		},
		Terminal: model.Terminal{
			IsOnline:    true,
			CardPresent: false,
			KmFromHome:  50,
		},
		LastTransaction: &model.LastTransaction{
			Timestamp:     "2026-06-15T12:00:00Z",
			KmFromCurrent: 25,
		},
	}

	query := make([]float32, vectorize.VectorDim)
	vec.Vectorize(req, query)

	// Verify no NaN or Inf
	for i, v := range query {
		if math.IsNaN(float64(v)) || math.IsInf(float64(v), 0) {
			t.Errorf("dim[%d] is NaN or Inf: %f", i, v)
		}
	}

	// Verify dimensions 0-4, 7-13 are in [0, 1] range
	for _, idx := range []int{0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13} {
		if query[idx] < 0 || query[idx] > 1 {
			t.Errorf("dim[%d] = %f, expected in [0, 1]", idx, query[idx])
		}
	}

	// Dimensions 5 and 6 should be in [0, 1] when last_transaction is present
	if query[5] < 0 || query[5] > 1 {
		t.Errorf("dim[5] = %f, expected in [0, 1] when last_transaction present", query[5])
	}
	if query[6] < 0 || query[6] > 1 {
		t.Errorf("dim[6] = %f, expected in [0, 1] when last_transaction present", query[6])
	}
}

func loadTestDataset(t *testing.T) *search.Dataset {
	t.Helper()

	// Try relative paths from different test locations
	paths := []string{
		"../../testdata/references_small.json",
		"testdata/references_small.json",
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			ds, err := data.LoadReferencesJSON(p)
			if err != nil {
				t.Fatalf("Failed to load test dataset from %s: %v", p, err)
			}
			return ds
		}
	}

	t.Fatal("Could not find testdata/references_small.json")
	return nil
}
