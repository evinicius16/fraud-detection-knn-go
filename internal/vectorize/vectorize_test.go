package vectorize

import (
	"math"
	"testing"

	"github.com/evinicius16/fraud-detection-knn-go/internal/model"
)

func TestClamp(t *testing.T) {
	tests := []struct {
		input    float32
		expected float32
	}{
		{-0.5, 0.0},
		{0.0, 0.0},
		{0.5, 0.5},
		{1.0, 1.0},
		{1.5, 1.0},
		{100.0, 1.0},
	}

	for _, tt := range tests {
		result := clamp(tt.input)
		if result != tt.expected {
			t.Errorf("clamp(%f) = %f, want %f", tt.input, result, tt.expected)
		}
	}
}

func TestVectorize_LegitTransaction(t *testing.T) {
	// Example from REGRAS_DE_DETECCAO.md - legit transaction
	vec := NewVectorizer(DefaultNormConstants(), DefaultMCCRisk())

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

	dst := make([]float32, VectorDim)
	vec.Vectorize(req, dst)

	// Expected from docs: [0.0041, 0.1667, 0.05, 0.7826, 0.3333, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006]
	expected := []float32{0.0041, 0.1667, 0.05, 0.8152, 0.1667, -1, -1, 0.0292, 0.15, 0, 1, 0, 0.15, 0.006}
	// Note: hour_of_day = 18/23 = 0.7826, day_of_week: 2026-03-11 is Wednesday = 2/6 = 0.3333
	// Let me recalculate: 2026-03-11 is a Wednesday. In our system: mon=0, tue=1, wed=2 → 2/6 = 0.3333
	// hour = 18, 18/23 = 0.7826...

	// Adjust expected based on actual calculation
	expected[3] = float32(18) / 23.0 // 0.7826...
	expected[4] = float32(2) / 6.0   // Wednesday = 2, 2/6 = 0.3333...

	tolerance := float32(0.01)
	for i, exp := range expected {
		if math.Abs(float64(dst[i]-exp)) > float64(tolerance) {
			t.Errorf("dim[%d] = %f, want ~%f (diff: %f)", i, dst[i], exp, dst[i]-exp)
		}
	}

	// Verify specific sentinel values
	if dst[5] != -1.0 {
		t.Errorf("dim[5] should be -1 for null last_transaction, got %f", dst[5])
	}
	if dst[6] != -1.0 {
		t.Errorf("dim[6] should be -1 for null last_transaction, got %f", dst[6])
	}
}

func TestVectorize_FraudTransaction(t *testing.T) {
	// Example from REGRAS_DE_DETECCAO.md - fraud transaction
	vec := NewVectorizer(DefaultNormConstants(), DefaultMCCRisk())

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

	dst := make([]float32, VectorDim)
	vec.Vectorize(req, dst)

	// Expected: [0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055]
	expected := []float32{0.9506, 0.8333, 1.0, 0.2174, 0.8333, -1, -1, 0.9523, 1.0, 0, 1, 1, 0.75, 0.0055}

	// Recalculate:
	// amount: 9505.97/10000 = 0.950597
	// installments: 10/12 = 0.8333
	// amount_vs_avg: (9505.97/81.28)/10 = 116.95.../10 = 11.69... → clamp → 1.0
	// hour: 5/23 = 0.2174
	// day: 2026-03-14 is Saturday. sat=5, 5/6 = 0.8333
	// last_tx: null → -1, -1
	// km_from_home: 952.27/1000 = 0.95227
	// tx_count_24h: 20/20 = 1.0
	// is_online: false → 0
	// card_present: true → 1
	// unknown_merchant: MERC-068 not in [MERC-008, MERC-007, MERC-005] → 1
	// mcc_risk: 7802 → 0.75
	// merchant_avg: 54.86/10000 = 0.005486

	expected[3] = float32(5) / 23.0  // 0.2174
	expected[4] = float32(5) / 6.0   // Saturday = 5, 5/6 = 0.8333

	tolerance := float32(0.01)
	for i, exp := range expected {
		if math.Abs(float64(dst[i]-exp)) > float64(tolerance) {
			t.Errorf("dim[%d] = %f, want ~%f (diff: %f)", i, dst[i], exp, dst[i]-exp)
		}
	}
}

func TestVectorize_WithLastTransaction(t *testing.T) {
	vec := NewVectorizer(DefaultNormConstants(), DefaultMCCRisk())

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

	dst := make([]float32, VectorDim)
	vec.Vectorize(req, dst)

	// minutes_since_last_tx: 20:23:35 - 14:58:35 = 5h25m = 325 minutes
	// 325/1440 = 0.2257
	expectedMinutes := float32(325.0 / 1440.0)
	tolerance := float32(0.01)

	if math.Abs(float64(dst[5]-expectedMinutes)) > float64(tolerance) {
		t.Errorf("dim[5] (minutes_since_last_tx) = %f, want ~%f", dst[5], expectedMinutes)
	}

	// km_from_last_tx: 18.863/1000 = 0.018863
	expectedKm := float32(18.863 / 1000.0)
	if math.Abs(float64(dst[6]-expectedKm)) > float64(tolerance) {
		t.Errorf("dim[6] (km_from_last_tx) = %f, want ~%f", dst[6], expectedKm)
	}

	// Verify NOT -1
	if dst[5] == -1.0 {
		t.Error("dim[5] should NOT be -1 when last_transaction is present")
	}
	if dst[6] == -1.0 {
		t.Error("dim[6] should NOT be -1 when last_transaction is present")
	}
}

func TestVectorize_UnknownMCC(t *testing.T) {
	vec := NewVectorizer(DefaultNormConstants(), DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-200",
		Transaction: model.Transaction{
			Amount:      100,
			Installments: 1,
			RequestedAt: "2026-01-01T12:00:00Z",
		},
		Customer: model.Customer{
			AvgAmount:      200,
			TxCount24h:     1,
			KnownMerchants: []string{},
		},
		Merchant: model.Merchant{
			ID:        "MERC-999",
			MCC:       "9999", // Not in mcc_risk.json
			AvgAmount: 150,
		},
		Terminal: model.Terminal{
			IsOnline:    true,
			CardPresent: false,
			KmFromHome:  0,
		},
		LastTransaction: nil,
	}

	dst := make([]float32, VectorDim)
	vec.Vectorize(req, dst)

	// Unknown MCC should default to 0.5
	if dst[12] != 0.5 {
		t.Errorf("dim[12] (mcc_risk) = %f, want 0.5 for unknown MCC", dst[12])
	}

	// is_online = true → 1
	if dst[9] != 1.0 {
		t.Errorf("dim[9] (is_online) = %f, want 1.0", dst[9])
	}

	// card_present = false → 0
	if dst[10] != 0.0 {
		t.Errorf("dim[10] (card_present) = %f, want 0.0", dst[10])
	}

	// unknown_merchant: MERC-999 not in [] → 1
	if dst[11] != 1.0 {
		t.Errorf("dim[11] (unknown_merchant) = %f, want 1.0", dst[11])
	}
}

func TestVectorize_AmountOverflow(t *testing.T) {
	vec := NewVectorizer(DefaultNormConstants(), DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-300",
		Transaction: model.Transaction{
			Amount:      15000, // Over max_amount (10000)
			Installments: 24,   // Over max_installments (12)
			RequestedAt: "2026-06-01T23:59:59Z",
		},
		Customer: model.Customer{
			AvgAmount:      100,
			TxCount24h:     50, // Over max (20)
			KnownMerchants: []string{"MERC-001"},
		},
		Merchant: model.Merchant{
			ID:        "MERC-001",
			MCC:       "5411",
			AvgAmount: 20000, // Over max (10000)
		},
		Terminal: model.Terminal{
			IsOnline:    false,
			CardPresent: true,
			KmFromHome:  5000, // Over max (1000)
		},
		LastTransaction: nil,
	}

	dst := make([]float32, VectorDim)
	vec.Vectorize(req, dst)

	// All overflowing values should be clamped to 1.0
	if dst[0] != 1.0 {
		t.Errorf("dim[0] (amount) = %f, want 1.0 (clamped)", dst[0])
	}
	if dst[1] != 1.0 {
		t.Errorf("dim[1] (installments) = %f, want 1.0 (clamped)", dst[1])
	}
	if dst[7] != 1.0 {
		t.Errorf("dim[7] (km_from_home) = %f, want 1.0 (clamped)", dst[7])
	}
	if dst[8] != 1.0 {
		t.Errorf("dim[8] (tx_count_24h) = %f, want 1.0 (clamped)", dst[8])
	}
	if dst[13] != 1.0 {
		t.Errorf("dim[13] (merchant_avg_amount) = %f, want 1.0 (clamped)", dst[13])
	}
}
