package integration

import (
	"os"
	"testing"

	jsoniter "github.com/json-iterator/go"

	"github.com/evinicius16/fraud-detection-knn-go/internal/data"
	"github.com/evinicius16/fraud-detection-knn-go/internal/model"
	"github.com/evinicius16/fraud-detection-knn-go/internal/search"
	"github.com/evinicius16/fraud-detection-knn-go/internal/vectorize"
)

var jsonFast = jsoniter.ConfigFastest

// BenchmarkFullPipeline benchmarks the entire hot path:
// JSON parse → vectorize → KNN search → score computation
func BenchmarkFullPipeline(b *testing.B) {
	ds := loadBenchDataset(b)
	vec := vectorize.NewVectorizer(vectorize.DefaultNormConstants(), vectorize.DefaultMCCRisk())

	payload := []byte(`{
		"id": "tx-bench-001",
		"transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
		"customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
		"merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 13.71},
		"last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.86}
	}`)

	query := make([]float32, vectorize.VectorDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var req model.FraudRequest
		jsonFast.Unmarshal(payload, &req)
		vec.Vectorize(&req, query)
		fraudCount := ds.BruteForceKNN(query)
		_ = search.ComputeFraudScore(fraudCount)
	}
}

// BenchmarkJSONParse benchmarks only the JSON parsing step.
func BenchmarkJSONParse(b *testing.B) {
	payload := []byte(`{
		"id": "tx-bench-001",
		"transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
		"customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
		"merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
		"terminal": {"is_online": false, "card_present": true, "km_from_home": 13.71},
		"last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.86}
	}`)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		var req model.FraudRequest
		jsonFast.Unmarshal(payload, &req)
	}
}

// BenchmarkVectorize benchmarks only the vectorization step.
func BenchmarkVectorize(b *testing.B) {
	vec := vectorize.NewVectorizer(vectorize.DefaultNormConstants(), vectorize.DefaultMCCRisk())

	req := &model.FraudRequest{
		ID: "tx-bench",
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
			KmFromHome:  13.71,
		},
		LastTransaction: &model.LastTransaction{
			Timestamp:     "2026-03-11T14:58:35Z",
			KmFromCurrent: 18.86,
		},
	}

	query := make([]float32, vectorize.VectorDim)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		vec.Vectorize(req, query)
	}
}

// BenchmarkKNNSearch benchmarks only the KNN search step.
func BenchmarkKNNSearch(b *testing.B) {
	ds := loadBenchDataset(b)
	query := []float32{0.04, 0.17, 0.05, 0.78, 0.33, -1, -1, 0.03, 0.15, 0, 1, 0, 0.15, 0.006}

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		ds.BruteForceKNN(query)
	}
}

func loadBenchDataset(b *testing.B) *search.Dataset {
	b.Helper()

	paths := []string{
		"../../testdata/references_small.json",
	}

	for _, p := range paths {
		if _, err := os.Stat(p); err == nil {
			ds, err := data.LoadReferencesJSON(p)
			if err != nil {
				b.Fatalf("Failed to load: %v", err)
			}
			return ds
		}
	}

	b.Fatal("Could not find testdata")
	return nil
}
