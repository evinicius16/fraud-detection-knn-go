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
	// 1. Compute partition key from discrete dims (O(1) routing)
	key := computeQueryKey(query)

	part := idx.Partitions[key]
	if part == nil || part.NumVectors == 0 {
		// Fallback: try nearest partition by flipping bits one at a time
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

	// 3. Find best 2 clusters (probing 2 clusters improves recall significantly)
	best1, best2 := findTop2Clusters(part, qFloat)

	// 4. Scan best cluster first, build top-5
	var d0, d1, d2, d3, d4 int64 = math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64, math.MaxInt64
	var i0, i1, i2, i3, i4 int = -1, -1, -1, -1, -1

	scanCluster(part, &q, best1, &d0, &d1, &d2, &d3, &d4, &i0, &i1, &i2, &i3, &i4)

	// 5. Scan second best cluster
	if best2 >= 0 {
		scanCluster(part, &q, best2, &d0, &d1, &d2, &d3, &d4, &i0, &i1, &i2, &i3, &i4)
	}

	// 6. Scan remaining clusters with bbox pruning
	for c := 0; c < part.NumClusters; c++ {
		if c == best1 || c == best2 {
			continue
		}

		lb := bboxLB9(&q, part.BboxMin, part.BboxMax, c*ContDim)
		if lb >= d4 {
			continue
		}

		start := part.ClusterOffsets[c]
		end := part.ClusterOffsets[c+1]
		for vi := start; vi < end; vi++ {
			dist := distSq9EarlyExit(&q, part.Vectors, vi*ContDim, d4)
			if dist < d4 {
				insertTop5(dist, vi, &d0, &d1, &d2, &d3, &d4, &i0, &i1, &i2, &i3, &i4)
			}
		}
	}

	// 7. Count frauds among top-5
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

// findTop2Clusters returns the indices of the 2 closest centroids.
// Returns (best, second) — second is -1 if there's only 1 cluster.
func findTop2Clusters(part *Partition, qFloat [ContDim]float32) (int, int) {
	best1, best2 := 0, -1
	var dist1 float32 = math.MaxFloat32
	var dist2 float32 = math.MaxFloat32

	for c := 0; c < part.NumClusters; c++ {
		off := c * ContDim
		var dist float32
		for d := 0; d < ContDim; d++ {
			diff := qFloat[d] - part.Centroids[off+d]
			dist += diff * diff
		}
		if dist < dist1 {
			dist2 = dist1
			best2 = best1
			dist1 = dist
			best1 = c
		} else if dist < dist2 {
			dist2 = dist
			best2 = c
		}
	}
	return best1, best2
}

// scanCluster scans all vectors in cluster c and updates the top-5 heap.
func scanCluster(part *Partition, q *[ContDim]int16, c int,
	d0, d1, d2, d3, d4 *int64, i0, i1, i2, i3, i4 *int) {
	start := part.ClusterOffsets[c]
	end := part.ClusterOffsets[c+1]
	for vi := start; vi < end; vi++ {
		dist := distSq9(q, part.Vectors, vi*ContDim)
		if dist < *d4 {
			insertTop5(dist, vi, d0, d1, d2, d3, d4, i0, i1, i2, i3, i4)
		}
	}
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

// distSq9 computes squared distance for 9 continuous dimensions (fully unrolled).
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

	diff = int64(q[0]) - int64(vectors[off])
	sum += diff * diff
	diff = int64(q[1]) - int64(vectors[off+1])
	sum += diff * diff
	if sum >= worst {
		return math.MaxInt64
	}

	diff = int64(q[2]) - int64(vectors[off+2])
	sum += diff * diff
	diff = int64(q[3]) - int64(vectors[off+3])
	sum += diff * diff
	if sum >= worst {
		return math.MaxInt64
	}

	diff = int64(q[4]) - int64(vectors[off+4])
	sum += diff * diff
	diff = int64(q[5]) - int64(vectors[off+5])
	sum += diff * diff
	if sum >= worst {
		return math.MaxInt64
	}

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
		*d4 = *d3
		*i4 = *i3
		*d3 = *d2
		*i3 = *i2
		*d2 = *d1
		*i2 = *i1
		*d1 = *d0
		*i1 = *i0
		*d0 = dist
		*i0 = idx
	} else if dist < *d1 {
		*d4 = *d3
		*i4 = *i3
		*d3 = *d2
		*i3 = *i2
		*d2 = *d1
		*i2 = *i1
		*d1 = dist
		*i1 = idx
	} else if dist < *d2 {
		*d4 = *d3
		*i4 = *i3
		*d3 = *d2
		*i3 = *i2
		*d2 = dist
		*i2 = idx
	} else if dist < *d3 {
		*d4 = *d3
		*i4 = *i3
		*d3 = dist
		*i3 = idx
	} else {
		*d4 = dist
		*i4 = idx
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

// BruteForceKNN performs exact KNN search over the dataset (used in tests).
func (ds *Dataset) BruteForceKNN(query []float32) int {
	dim := len(query)

	// top-K using a max-heap of size K (worst distance at index 0)
	topDist := [K]float32{math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32, math.MaxFloat32}
	topIdx := [K]int{-1, -1, -1, -1, -1}
	filled := 0

	for i := 0; i < ds.Count; i++ {
		d := euclideanDistSq(query, ds.Vectors[i*dim:(i+1)*dim])

		if filled < K {
			// Find position to insert (keep sorted descending so [0] is worst)
			pos := filled
			for pos > 0 && d > topDist[pos-1] {
				topDist[pos] = topDist[pos-1]
				topIdx[pos] = topIdx[pos-1]
				pos--
			}
			topDist[pos] = d
			topIdx[pos] = i
			filled++
		} else if d < topDist[0] {
			// Replace worst
			topDist[0] = d
			topIdx[0] = i
			// Bubble down to maintain descending order
			for j := 0; j+1 < K && topDist[j] < topDist[j+1]; j++ {
				topDist[j], topDist[j+1] = topDist[j+1], topDist[j]
				topIdx[j], topIdx[j+1] = topIdx[j+1], topIdx[j]
			}
		}
	}

	fraudCount := 0
	for k := 0; k < filled; k++ {
		if topIdx[k] >= 0 && ds.Labels[topIdx[k]] {
			fraudCount++
		}
	}
	return fraudCount
}

// euclideanDistSq computes squared Euclidean distance between two float32 slices.
func euclideanDistSq(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return sum
}
