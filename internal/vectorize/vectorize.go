package vectorize

import (
	"github.com/evinicius16/fraud-detection-knn-go/internal/model"
)

const VectorDim = 14

// Normalization constants (from normalization.json)
type NormConstants struct {
	MaxAmount            float64 `json:"max_amount"`
	MaxInstallments      float64 `json:"max_installments"`
	AmountVsAvgRatio     float64 `json:"amount_vs_avg_ratio"`
	MaxMinutes           float64 `json:"max_minutes"`
	MaxKm                float64 `json:"max_km"`
	MaxTxCount24h        float64 `json:"max_tx_count_24h"`
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
// Hot path: zero allocations, no time.Parse — uses manual ISO8601 parsing.
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

	// Parse requested_at without allocations — format: "2006-01-02T15:04:05Z" or with offset
	// We only need hour (chars 11-12) and the full unix timestamp for diff.
	reqTS := req.Transaction.RequestedAt
	hour, reqUnix := parseTimestamp(reqTS)

	// 3: hour_of_day (0-23 UTC, divided by 23)
	dst[3] = float32(hour) / 23.0

	// 4: day_of_week — derive from unix epoch (days since Thu 1970-01-01, weekday=4)
	// epoch day: unix / 86400; weekday: (epochDay + 4) % 7 where Sun=0
	epochDay := reqUnix / 86400
	if reqUnix < 0 {
		epochDay-- // floor division for negative
	}
	// Go weekday: Sun=0, Mon=1 ... Sat=6
	goWeekday := int((epochDay + 4) % 7)
	if goWeekday < 0 {
		goWeekday += 7
	}
	// Convert to mon=0, sun=6
	dow := goWeekday - 1
	if dow < 0 {
		dow = 6
	}
	dst[4] = float32(dow) / 6.0

	// 5: minutes_since_last_tx
	// 6: km_from_last_tx
	if req.LastTransaction == nil {
		dst[5] = -1.0
		dst[6] = -1.0
	} else {
		_, lastUnix := parseTimestamp(req.LastTransaction.Timestamp)
		diffSec := reqUnix - lastUnix
		minutes := float64(diffSec) / 60.0
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

// parseTimestamp parses an ISO8601/RFC3339 timestamp string without allocations.
// Returns (hour int, unixSeconds int64).
// Supports formats: "2006-01-02T15:04:05Z" and "2006-01-02T15:04:05+HH:MM"
func parseTimestamp(s string) (hour int, unix int64) {
	if len(s) < 19 {
		return 0, 0
	}

	year := int(s[0]-'0')*1000 + int(s[1]-'0')*100 + int(s[2]-'0')*10 + int(s[3]-'0')
	month := int(s[5]-'0')*10 + int(s[6]-'0')
	day := int(s[8]-'0')*10 + int(s[9]-'0')
	hour = int(s[11]-'0')*10 + int(s[12]-'0')
	minute := int(s[14]-'0')*10 + int(s[15]-'0')
	second := int(s[17]-'0')*10 + int(s[18]-'0')

	// Parse timezone offset in seconds
	tzOffsetSec := 0
	if len(s) > 19 {
		c := s[19]
		if c == '+' || c == '-' {
			if len(s) >= 25 {
				tzH := int(s[20]-'0')*10 + int(s[21]-'0')
				tzM := int(s[23]-'0')*10 + int(s[24]-'0')
				tzOffsetSec = tzH*3600 + tzM*60
				if c == '-' {
					tzOffsetSec = -tzOffsetSec
				}
			}
		}
		// 'Z' means UTC, offset = 0
	}

	// Compute unix timestamp using civil-time → epoch algorithm
	unix = civilToUnix(year, month, day, hour, minute, second) - int64(tzOffsetSec)
	return hour, unix
}

// civilToUnix converts a UTC civil time to Unix seconds.
// Uses the algorithm from http://howardhinnant.github.io/date_algorithms.html
func civilToUnix(year, month, day, hour, minute, second int) int64 {
	if month <= 2 {
		year--
	}
	era := year / 400
	if year < 0 {
		era = (year - 399) / 400
	}
	yoe := year - era*400                           // [0, 399]
	doy := (153*(month+9)%12+2)/5 + day - 1         // [0, 365]
	doe := yoe*365 + yoe/4 - yoe/100 + doy          // [0, 146096]
	days := int64(era)*146097 + int64(doe) - 719468 // days since epoch
	return days*86400 + int64(hour)*3600 + int64(minute)*60 + int64(second)
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
