package search

import (
	"math"
)

const K = 5

// ContDim is the number of continuous dimensions stored per vector.
// The 5 discrete dims (is_online, card_present, unknown_merchant, sentinel_5, sentinel_6)
// are used for partition routing and excluded from distance computation.
const ContDim = 9

// Partition holds a mini-IVF index for one of the 32 bit-partitions.
type Partition struct {
	Key            uint8
	NumVectors     int
	NumClusters    int
	Centroids      []float32 // [NumClusters * ContDim]
	ClusterOffsets []int     // [NumClusters + 1]
	BboxMin        []int16   // [NumClusters * ContDim]
	BboxMax        []int16   // [NumClusters * ContDim]
	Vectors        []int16   // [NumVectors * ContDim]
	Labels         []bool    // [NumVectors]
}

// HybridIndex holds all partitions keyed by their 5-bit partition key.
type HybridIndex struct {
	Partitions   [32]*Partition // indexed by partition key (0-31)
	ContDimOrder []int          // which original dims map to positions 0..8
	TotalVectors int
}

// FindFraudCount routes the query to the correct partition and searches.
// query is the full 14-dim vector in original order.
func (idx *HybridIndex) FindFraudCount(query []float32) int {
	// 1. Compute partition key from discrete dims (O(1) routing!)
	key := computeQueryKey(query)

	part := idx.Partitions[key]
	if part == nil || part.NumVectors == 0 {
		// Fallback: try nearest partition (flip least significant bit)
		for flip := uint8(1); flip < 32; flip++ {
			alt := key ^ flip
			if idx.Partitions[alt] != nil && idx.Partitions[alt].NumVectors > 0 {
				part = idx.Partitions[alt]
				break
			}
		}
		if part == nil {
			return 0
		}
	}

	// 2. Extract continuous dims from query, quantize to int16
	var q [ContDim]int16
	var qFloat [ContDim]float32
	for d := 0; d < ContDim; d++ {
		val := query[idx.ContDimOrder[d]]
		qFloat[d] = val
		q[d] = int16(val * 10000)
	}

	// 3. Find best cluster
	bestCluster := 0
	var bestDist float32 = math.MaxFloat32
	for c := 0; c < part.NumClusters; c++ {
		off := c * ContDim
		var dist float32
		for d := 0; d < ContDim; d++ {
			diff := qFloat[d] - part.Centroids[off+d]
			dist += diff * diff
		}
		if dist < bestDist {
			bestDist = dist
			bestCluster = c
		}
	}

	// 4. Scan best cluster, build top-5
	var d0, d1, d2, d3, d4 int64 = math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64
	var i0, i1, i2, i3, i4 int = -1, -1, -1, -1, -1

	start := part.ClusterOffsets[bestCluster]
	end := part.ClusterOffsets[bestCluster+1]

	for vi := start; vi < end; vi++ {
		dist := distSq9(&q, part.Vectors, vi*ContDim)
		if dist < d4 {
			insertTop5(dist, vi, &d0, &d1, &d2, &d3, &d4, &i0, &i1, &i2, &i3, &i4)
		}
	}

	// 5. Scan other clusters with bbox pruning
	for c := 0; c < part.NumClusters; c++ {
		if c == bestCluster {
			continue
		}

		lb := bboxLB9(&q, part.BboxMin, part.BboxMax, c*ContDim)
		if lb >= d4 {
			continue
		}

		start = part.ClusterOffsets[c]
		end = part.ClusterOffsets[c+1]

		for vi := start; vi < end; vi++ {
			dist := distSq9EarlyExit(&q, part.Vectors, vi*ContDim, d4)
			if dist < d4 {
				insertTop5(dist, vi, &d0, &d1, &d2, &d3, &d4, &i0, &i1, &i2, &i3, &i4)
			}
		}
	}

	// 6. Count frauds
	fraudCount := 0
	if i0 >= 0 && part.Labels[i0] {
		fraudCount++
	}
	if i1 >= 0 && part.Labels[i1] {
		fraudCount++
	}
	if i2 >= 0 && part.Labels[i2] {
		fraudCount++
	}
	if i3 >= 0 && part.Labels[i3] {
		fraudCount++
	}
	if i4 >= 0 && part.Labels[i4] {
		fraudCount++
	}
	return fraudCount
}

func computeQueryKey(query []float32) uint8 {
	var key uint8
	if query[9] > 0.5 { // is_online
		key |= 1
	}
	if query[10] > 0.5 { // card_present
		key |= 2
	}
	if query[11] > 0.5 { // unknown_merchant
		key |= 4
	}
	if query[5] < 0 { // sentinel_5 (minutes_since_last == -1)
		key |= 8
	}
	if query[6] < 0 { // sentinel_6 (km_from_last == -1)
		key |= 16
	}
	return key
}

// distSq9 computes squared distance for 9 continuous dimensions.
func distSq9(q *[ContDim]int16, vectors []int16, off int) int64 {
	var sum int64
	var diff int64
	diff = int64(q[0]) - int64(vectors[off])
	sum += diff * diff
	diff = int64(q[1]) - int64(vectors[off+1])
	sum += diff * diff
	diff = int64(q[2]) - int64(vectors[off+2])
	sum += diff * diff
	diff = int64(q[3]) - int64(vectors[off+3])
	sum += diff * diff
	diff = int64(q[4]) - int64(vectors[off+4])
	sum += diff * diff
	diff = int64(q[5]) - int64(vectors[off+5])
	sum += diff * diff
	diff = int64(q[6]) - int64(vectors[off+6])
	sum += diff * diff
	diff = int64(q[7]) - int64(vectors[off+7])
	sum += diff * diff
	diff = int64(q[8]) - int64(vectors[off+8])
	sum += diff * diff
	return sum
}

// distSq9EarlyExit with aggressive early exit every 2 dims.
func distSq9EarlyExit(q *[ContDim]int16, vectors []int16, off int, worst int64) int64 {
	var sum int64
	var diff int64

	// Dims 0-1 (highest variance)
	diff = int64(q[0]) - int64(vectors[off])
	sum += diff * diff
	diff = int64(q[1]) - int64(vectors[off+1])
	sum += diff * diff
	if sum > worst {
		return math.MaxInt64
	}

	// Dims 2-3
	diff = int64(q[2]) - int64(vectors[off+2])
	sum += diff * diff
	diff = int64(q[3]) - int64(vectors[off+3])
	sum += diff * diff
	if sum > worst {
		return math.MaxInt64
	}

	// Dims 4-5
	diff = int64(q[4]) - int64(vectors[off+4])
	sum += diff * diff
	diff = int64(q[5]) - int64(vectors[off+5])
	sum += diff * diff
	if sum > worst {
		return math.MaxInt64
	}

	// Dims 6-7-8
	diff = int64(q[6]) - int64(vectors[off+6])
	sum += diff * diff
	diff = int64(q[7]) - int64(vectors[off+7])
	sum += diff * diff
	diff = int64(q[8]) - int64(vectors[off+8])
	sum += diff * diff

	return sum
}

func bboxLB9(q *[ContDim]int16, bboxMin, bboxMax []int16, off int) int64 {
	var sum int64
	for d := 0; d < ContDim; d++ {
		qd := q[d]
		minVal := bboxMin[off+d]
		maxVal := bboxMax[off+d]
		if qd < minVal {
			diff := int64(minVal) - int64(qd)
			sum += diff * diff
		} else if qd > maxVal {
			diff := int64(qd) - int64(maxVal)
			sum += diff * diff
		}
	}
	return sum
}

func insertTop5(dist int64, idx int, d0, d1, d2, d3, d4 *int64, i0, i1, i2, i3, i4 *int) {
	if dist < *d0 {
		*d4 = *d3; *i4 = *i3
		*d3 = *d2; *i3 = *i2
		*d2 = *d1; *i2 = *i1
		*d1 = *d0; *i1 = *i0
		*d0 = dist; *i0 = idx
	} else if dist < *d1 {
		*d4 = *d3; *i4 = *i3
		*d3 = *d2; *i3 = *i2
		*d2 = *d1; *i2 = *i1
		*d1 = dist; *i1 = idx
	} else if dist < *d2 {
		*d4 = *d3; *i4 = *i3
		*d3 = *d2; *i3 = *i2
		*d2 = dist; *i2 = idx
	} else if dist < *d3 {
		*d4 = *d3; *i4 = *i3
		*d3 = dist; *i3 = idx
	} else {
		*d4 = dist; *i4 = idx
	}
}

// ComputeFraudScore calculates the fraud score from the KNN result.
func ComputeFraudScore(fraudCount int) float64 {
	return math.Round(float64(fraudCount)/float64(K)*100) / 100
}

// IsApproved determines if the transaction should be approved.
func IsApproved(fraudScore float64) bool {
	return fraudScore < 0.6
}

// Dataset kept for backward compat with tests.
type Dataset struct {
	Vectors []float32
	Labels  []bool
	Count   int
}
