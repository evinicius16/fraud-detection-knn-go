package main

import (
	"fmt"
	"log"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"

	jsoniter "github.com/json-iterator/go"

	"github.com/evinicius16/fraud-detection-knn-go/internal/data"
	"github.com/evinicius16/fraud-detection-knn-go/internal/model"
	"github.com/evinicius16/fraud-detection-knn-go/internal/search"
	"github.com/evinicius16/fraud-detection-knn-go/internal/vectorize"
	"github.com/valyala/fasthttp"
)

var json = jsoniter.ConfigFastest

var (
	hybridIndex *search.HybridIndex
	vec         *vectorize.Vectorizer
	queryPool   sync.Pool
)

// Pre-computed responses
var responses [6][]byte

func init() {
	scores := [6]float64{0.0, 0.2, 0.4, 0.6, 0.8, 1.0}
	for i, score := range scores {
		approved := score < 0.6
		responses[i] = []byte(`{"approved":` + strconv.FormatBool(approved) + `,"fraud_score":` + strconv.FormatFloat(score, 'f', 1, 64) + `}`)
	}
}

func main() {
	start := time.Now()

	indexPath := getEnv("IVF_INDEX_PATH", "/app/data/ivf_index.bin")
	mccPath := getEnv("MCC_RISK_PATH", "/app/data/mcc_risk.json")
	normPath := getEnv("NORMALIZATION_PATH", "/app/data/normalization.json")
	port := getEnv("PORT", "8080")

	// Load config
	norm, err := data.LoadNormConstants(normPath)
	if err != nil {
		log.Fatalf("norm: %v", err)
	}
	mccRisk, err := data.LoadMCCRisk(mccPath)
	if err != nil {
		log.Fatalf("mcc: %v", err)
	}

	// Load hybrid index
	fmt.Printf("[startup] Loading hybrid index from %s...\n", indexPath)
	hybridIndex, err = data.LoadHybridIndexMmap(indexPath)
	if err != nil {
		log.Fatalf("index: %v", err)
	}

	vec = vectorize.NewVectorizer(norm, mccRisk)

	loadTime := time.Since(start)
	fmt.Printf("[startup] Loaded in %v (%d vectors)\n", loadTime, hybridIndex.TotalVectors)

	// Warmup
	fmt.Println("[startup] Warming up (3000 queries)...")
	warmupStart := time.Now()
	warmup()
	fmt.Printf("[startup] Warmup done in %v\n", time.Since(warmupStart))

	queryPool = sync.Pool{
		New: func() interface{} {
			v := make([]float32, vectorize.VectorDim)
			return &v
		},
	}

	fmt.Printf("[startup] Listening on :%s (total: %v)\n", port, time.Since(start))

	server := &fasthttp.Server{
		Handler:                      requestHandler,
		Name:                         "rinha",
		DisableHeaderNamesNormalizing: true,
		NoDefaultContentType:         true,
	}

	if err := server.ListenAndServe(":" + port); err != nil {
		log.Fatalf("server: %v", err)
	}
}

func warmup() {
	rng := rand.New(rand.NewSource(42))
	query := make([]float32, vectorize.VectorDim)

	// Generate realistic queries by sampling from known partition ranges
	for w := 0; w < 3000; w++ {
		// Random continuous values in [0, 1]
		for d := 0; d < vectorize.VectorDim; d++ {
			query[d] = rng.Float32()
		}
		// Set discrete dims to random binary values
		if rng.Intn(2) == 0 {
			query[9] = 1.0
		} else {
			query[9] = 0.0
		}
		if rng.Intn(2) == 0 {
			query[10] = 1.0
		} else {
			query[10] = 0.0
		}
		if rng.Intn(2) == 0 {
			query[11] = 1.0
		} else {
			query[11] = 0.0
		}
		// Sentinel dims
		if rng.Intn(4) == 0 {
			query[5] = -1.0
			query[6] = -1.0
		} else {
			query[5] = rng.Float32()
			query[6] = rng.Float32()
		}

		hybridIndex.FindFraudCount(query)
	}
}

func requestHandler(ctx *fasthttp.RequestCtx) {
	path := ctx.Path()
	if len(path) == 6 && path[1] == 'r' {
		ctx.SetStatusCode(200)
		ctx.SetBodyString("OK")
		return
	}
	if len(path) == 12 && path[1] == 'f' {
		handleFraudScore(ctx)
		return
	}
	ctx.SetStatusCode(404)
}

func handleFraudScore(ctx *fasthttp.RequestCtx) {
	var req model.FraudRequest
	if err := json.Unmarshal(ctx.PostBody(), &req); err != nil {
		ctx.SetContentType("application/json")
		ctx.SetStatusCode(200)
		ctx.SetBody(responses[0])
		return
	}

	queryPtr := queryPool.Get().(*[]float32)
	query := *queryPtr
	defer queryPool.Put(queryPtr)

	vec.Vectorize(&req, query)
	fraudCount := hybridIndex.FindFraudCount(query)

	ctx.SetContentType("application/json")
	ctx.SetStatusCode(200)
	ctx.SetBody(responses[fraudCount])
}

func getEnv(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
