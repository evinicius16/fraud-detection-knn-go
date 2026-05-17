package vectorize

import (
	"time"

	"github.com/evinicius16/fraud-detection-knn-go/internal/model"
)

const VectorDim = 14

// Normalization constants (from normalization.json)
type NormConstants struct {
	MaxAmount           float64 `json:"max_amount"`
	MaxInstallments     float64 `json:"max_installments"`
	AmountVsAvgRatio    float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes          float64 `json:"max_minutes"`
	MaxKm               float64 `json:"max_km"`
	MaxTxCount24h       float64 `json:"max_tx_count_24h"`
	MaxMerchantAvgAmount float64 `json:"max_merchant_avg_amount"`
}

// DefaultNormConstants returns the standard normalization constants.
func DefaultNormConstants() NormConstants {
	return NormConstants{
		MaxAmount:            10000,
		MaxInstallments:      12,
		AmountVsAvgRatio:     10,
		MaxMinutes:           1440,
		MaxKm:                1000,
		MaxTxCount24h:        20,
		MaxMerchantAvgAmount: 10000,
	}
}

// MCCRisk maps MCC codes to risk scores.
type MCCRisk map[string]float64

// DefaultMCCRisk returns the standard MCC risk map.
func DefaultMCCRisk() MCCRisk {
	return MCCRisk{
		"5411": 0.15,
		"5812": 0.30,
		"5912": 0.20,
		"5944": 0.45,
		"7801": 0.80,
		"7802": 0.75,
		"7995": 0.85,
		"4511": 0.35,
		"5311": 0.25,
		"5999": 0.50,
	}
}

// Vectorizer transforms a FraudRequest into a 14-dimensional vector.
type Vectorizer struct {
	Norm    NormConstants
	MCCRisk MCCRisk
}

// NewVectorizer creates a Vectorizer with the given constants.
func NewVectorizer(norm NormConstants, mccRisk MCCRisk) *Vectorizer {
	return &Vectorizer{
		Norm:    norm,
		MCCRisk: mccRisk,
	}
}

// Vectorize transforms a request into a 14-dimensional float32 vector.
// The output is written into the provided dst slice (must have len >= 14).
func (v *Vectorizer) Vectorize(req *model.FraudRequest, dst []float32) {
	norm := &v.Norm

	// 0: amount
	dst[0] = clamp(float32(req.Transaction.Amount / norm.MaxAmount))

	// 1: installments
	dst[1] = clamp(float32(float64(req.Transaction.Installments) / norm.MaxInstallments))

	// 2: amount_vs_avg
	if req.Customer.AvgAmount > 0 {
		dst[2] = clamp(float32((req.Transaction.Amount / req.Customer.AvgAmount) / norm.AmountVsAvgRatio))
	} else {
		dst[2] = clamp(float32(req.Transaction.Amount / norm.AmountVsAvgRatio))
	}

	// Parse requested_at for hour and day_of_week
	t, err := time.Parse(time.RFC3339, req.Transaction.RequestedAt)
	if err != nil {
		// fallback: try without timezone
		t, _ = time.Parse("2006-01-02T15:04:05Z", req.Transaction.RequestedAt)
	}

	// 3: hour_of_day (0-23 UTC, divided by 23)
	dst[3] = float32(t.Hour()) / 23.0

	// 4: day_of_week (mon=0, sun=6, divided by 6)
	dow := int(t.Weekday()) // Sunday=0, Monday=1, ..., Saturday=6
	// Convert to mon=0, sun=6
	if dow == 0 {
		dow = 6 // Sunday
	} else {
		dow = dow - 1 // Monday=0, Tuesday=1, ..., Saturday=5
	}
	dst[4] = float32(dow) / 6.0

	// 5: minutes_since_last_tx
	// 6: km_from_last_tx
	if req.LastTransaction == nil {
		dst[5] = -1.0
		dst[6] = -1.0
	} else {
		lastT, err := time.Parse(time.RFC3339, req.LastTransaction.Timestamp)
		if err != nil {
			lastT, _ = time.Parse("2006-01-02T15:04:05Z", req.LastTransaction.Timestamp)
		}
		minutes := t.Sub(lastT).Minutes()
		dst[5] = clamp(float32(minutes / norm.MaxMinutes))
		dst[6] = clamp(float32(req.LastTransaction.KmFromCurrent / norm.MaxKm))
	}

	// 7: km_from_home
	dst[7] = clamp(float32(req.Terminal.KmFromHome / norm.MaxKm))

	// 8: tx_count_24h
	dst[8] = clamp(float32(float64(req.Customer.TxCount24h) / norm.MaxTxCount24h))

	// 9: is_online
	if req.Terminal.IsOnline {
		dst[9] = 1.0
	} else {
		dst[9] = 0.0
	}

	// 10: card_present
	if req.Terminal.CardPresent {
		dst[10] = 1.0
	} else {
		dst[10] = 0.0
	}

	// 11: unknown_merchant (1 if merchant NOT in known_merchants)
	known := false
	for _, m := range req.Customer.KnownMerchants {
		if m == req.Merchant.ID {
			known = true
			break
		}
	}
	if known {
		dst[11] = 0.0
	} else {
		dst[11] = 1.0
	}

	// 12: mcc_risk
	risk, ok := v.MCCRisk[req.Merchant.MCC]
	if !ok {
		dst[12] = 0.5 // default
	} else {
		dst[12] = float32(risk)
	}

	// 13: merchant_avg_amount
	dst[13] = clamp(float32(req.Merchant.AvgAmount / norm.MaxMerchantAvgAmount))
}

func clamp(x float32) float32 {
	if x < 0.0 {
		return 0.0
	}
	if x > 1.0 {
		return 1.0
	}
	return x
}
