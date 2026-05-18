// preprocess builds a Hybrid Partition Index from references.json.gz.
//
// Strategy:
// Level 1: Partition by 5 discrete dimensions (is_online, card_present,
//
//	unknown_merchant, sentinel_5, sentinel_6) → up to 32 partitions.
//	Routing is O(1) — just compute the partition key from the query bits.
//
// Level 2: Within each partition, run mini k-means (K up to 128) to create
//
//	sub-clusters with bounding boxes for lower-bound pruning.
//	More clusters = smaller clusters = faster scan per query.
//
// Dimensions are reordered: continuous dims sorted by variance (high first)
// come before the discrete dims (which are used for partitioning, not distance).
// Since discrete dims are used for partitioning, vectors within a partition all
// share the same discrete values → those dims contribute 0 to intra-partition
// distances. We EXCLUDE them from the stored vectors entirely (9 continuous dims
// only), saving memory and compute.
//
// Binary format:
//
//	Header:
//	  4 bytes: numPartitions (uint32 LE)
//	  4 bytes: totalVectors (uint32 LE)
//	  4 bytes: numContinuousDims (uint32 LE) = 9
//	  9 * 4 bytes: continuousDimOrder (uint32 LE) — which original dims, in order
//	Per partition (numPartitions times):
//	  4 bytes: partitionKey (uint32 LE) — the 5-bit key
//	  4 bytes: numVectors in this partition (uint32 LE)
//	  4 bytes: numClusters in this partition (uint32 LE)
//	  numClusters * 9 * 4 bytes: centroids (float32 LE)
//	  (numClusters + 1) * 4 bytes: clusterOffsets (uint32 LE, relative to partition start)
//	  numClusters * 9 * 2 bytes: bboxMin (int16 LE)
//	  numClusters * 9 * 2 bytes: bboxMax (int16 LE)
//	  numVectors * 9 * 2 bytes: vectors (int16 LE, ordered by cluster)
//	  numVectors bytes: labels
package main

import (
	"bufio"
	"compress/gzip"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"sort"
)

const (
	origDim           = 14
	contDim           = 9 // continuous dimensions (excluding 5 discrete)
	scale             = 10000
	clustersPerPart   = 128 // more clusters → smaller clusters → faster scan
	kmeansIters       = 20  // more iterations → better cluster quality
	minVecsPerCluster = 50  // target: ~50 vectors per cluster on average
)

// Discrete dimension indices (in original 14-dim space)
var discreteDims = [5]int{9, 10, 11, 5, 6}

// Continuous dimension indices (the other 9)
var continuousDims = [9]int{0, 1, 2, 3, 4, 7, 8, 12, 13}

type entry struct {
	Vector []float64 `json:"vector"`
	Label  string    `json:"label"`
}

func main() {
	if len(os.Args) < 3 {
		log.Fatal("Usage: preprocess <input.json.gz> <output.bin>")
	}

	inputPath := os.Args[1]
	outputPath := os.Args[2]

	fmt.Printf("[build] Reading %s...\n", inputPath)

	vectors, labels, count := readVectors(inputPath)
	fmt.Printf("[build] Total: %d vectors\n", count)

	fmt.Println("[build] Computing partition keys...")
	partitionMap := make(map[uint8][]int)
	for i := 0; i < count; i++ {
		key := computePartitionKey(vectors, i)
		partitionMap[key] = append(partitionMap[key], i)
	}
	fmt.Printf("[build] Found %d unique partitions\n", len(partitionMap))

	fmt.Println("[build] Computing continuous dim variance ordering...")
	contDimOrder := computeContinuousDimOrder(vectors, count)
	fmt.Printf("[build] Continuous dim order (high variance first): %v\n", contDimOrder)

	fmt.Println("[build] Building per-partition IVF indices...")

	var sortedKeys []uint8
	for k := range partitionMap {
		sortedKeys = append(sortedKeys, k)
	}
	sort.Slice(sortedKeys, func(a, b int) bool { return sortedKeys[a] < sortedKeys[b] })

	partitions := make([]partitionData, 0, len(sortedKeys))

	for _, key := range sortedKeys {
		indices := partitionMap[key]
		n := len(indices)

		// Extract continuous dims in variance order
		contVecs := make([]float32, n*contDim)
		partLabels := make([]bool, n)

		for vi, origIdx := range indices {
			off := origIdx * origDim
			for d := 0; d < contDim; d++ {
				contVecs[vi*contDim+d] = float32(vectors[off+contDimOrder[d]])
			}
			partLabels[vi] = labels[origIdx]
		}

		// Target ~minVecsPerCluster vectors per cluster, capped at clustersPerPart
		numClusters := n / minVecsPerCluster
		if numClusters < 1 {
			numClusters = 1
		}
		if numClusters > clustersPerPart {
			numClusters = clustersPerPart
		}

		centroids := make([]float32, numClusters*contDim)
		assignments := make([]int, n)

		if numClusters == 1 {
			for vi := 0; vi < n; vi++ {
				for d := 0; d < contDim; d++ {
					centroids[d] += contVecs[vi*contDim+d]
				}
			}
			for d := 0; d < contDim; d++ {
				centroids[d] /= float32(n)
			}
		} else {
			kmeanspp(contVecs, n, centroids, assignments, numClusters)
		}

		// Reorder by cluster and quantize
		clusterSizes := make([]int, numClusters)
		for vi := 0; vi < n; vi++ {
			clusterSizes[assignments[vi]]++
		}

		clusterOffsets := make([]int, numClusters+1)
		for c := 1; c <= numClusters; c++ {
			clusterOffsets[c] = clusterOffsets[c-1] + clusterSizes[c-1]
		}

		orderedVecs := make([]int16, n*contDim)
		orderedLbls := make([]byte, n)
		writePos := make([]int, numClusters)
		copy(writePos, clusterOffsets[:numClusters])

		for vi := 0; vi < n; vi++ {
			c := assignments[vi]
			destIdx := writePos[c]
			writePos[c]++
			for d := 0; d < contDim; d++ {
				orderedVecs[destIdx*contDim+d] = int16(contVecs[vi*contDim+d] * scale)
			}
			if partLabels[vi] {
				orderedLbls[destIdx] = 1
			}
		}

		// Compute bounding boxes
		bboxMin := make([]int16, numClusters*contDim)
		bboxMax := make([]int16, numClusters*contDim)
		for i := range bboxMin {
			bboxMin[i] = math.MaxInt16
			bboxMax[i] = math.MinInt16
		}

		for c := 0; c < numClusters; c++ {
			cStart := clusterOffsets[c]
			cEnd := clusterOffsets[c+1]
			bOff := c * contDim
			for idx := cStart; idx < cEnd; idx++ {
				vOff := idx * contDim
				for d := 0; d < contDim; d++ {
					val := orderedVecs[vOff+d]
					if val < bboxMin[bOff+d] {
						bboxMin[bOff+d] = val
					}
					if val > bboxMax[bOff+d] {
						bboxMax[bOff+d] = val
					}
				}
			}
		}

		partitions = append(partitions, partitionData{
			key:            key,
			numVectors:     n,
			numClusters:    numClusters,
			centroids:      centroids,
			clusterOffsets: clusterOffsets,
			bboxMin:        bboxMin,
			bboxMax:        bboxMax,
			orderedVectors: orderedVecs,
			orderedLabels:  orderedLbls,
		})

		fmt.Printf("[build]   partition %02d: %6d vectors, %3d clusters (~%d/cluster)\n",
			key, n, numClusters, n/numClusters)
	}

	fmt.Printf("[build] Writing %s...\n", outputPath)
	writeBinary(outputPath, partitions, contDimOrder, count)
}

func computePartitionKey(vectors []float64, idx int) uint8 {
	off := idx * origDim
	var key uint8
	if vectors[off+9] > 0.5 {
		key |= 1
	}
	if vectors[off+10] > 0.5 {
		key |= 2
	}
	if vectors[off+11] > 0.5 {
		key |= 4
	}
	if vectors[off+5] < 0 {
		key |= 8
	}
	if vectors[off+6] < 0 {
		key |= 16
	}
	return key
}

func computeContinuousDimOrder(vectors []float64, count int) []int {
	means := make([]float64, contDim)
	for i := 0; i < count; i++ {
		off := i * origDim
		for d := 0; d < contDim; d++ {
			means[d] += vectors[off+continuousDims[d]]
		}
	}
	for d := 0; d < contDim; d++ {
		means[d] /= float64(count)
	}

	variances := make([]float64, contDim)
	for i := 0; i < count; i++ {
		off := i * origDim
		for d := 0; d < contDim; d++ {
			diff := vectors[off+continuousDims[d]] - means[d]
			variances[d] += diff * diff
		}
	}

	indices := make([]int, contDim)
	for d := 0; d < contDim; d++ {
		indices[d] = d
	}
	sort.Slice(indices, func(a, b int) bool {
		return variances[indices[a]] > variances[indices[b]]
	})

	result := make([]int, contDim)
	for d := 0; d < contDim; d++ {
		result[d] = continuousDims[indices[d]]
	}
	return result
}

// kmeanspp runs k-means with k-means++ initialization for better cluster quality.
// Better init → fewer iterations needed → more balanced clusters.
func kmeanspp(vectors []float32, count int, centroids []float32, assignments []int, k int) {
	rng := rand.New(rand.NewSource(42))

	// k-means++ init: pick first centroid randomly, then pick each subsequent
	// centroid with probability proportional to squared distance from nearest centroid.
	first := rng.Intn(count)
	copy(centroids[0:contDim], vectors[first*contDim:(first+1)*contDim])

	minDist := make([]float32, count)
	for i := range minDist {
		minDist[i] = math.MaxFloat32
	}

	for c := 1; c < k; c++ {
		// Update min distances to nearest chosen centroid
		prevCOff := (c - 1) * contDim
		var totalDist float64
		for i := 0; i < count; i++ {
			vOff := i * contDim
			var d float32
			for dim := 0; dim < contDim; dim++ {
				diff := vectors[vOff+dim] - centroids[prevCOff+dim]
				d += diff * diff
			}
			if d < minDist[i] {
				minDist[i] = d
			}
			totalDist += float64(minDist[i])
		}

		// Sample next centroid proportional to minDist^2
		target := rng.Float64() * totalDist
		var cumul float64
		chosen := count - 1
		for i := 0; i < count; i++ {
			cumul += float64(minDist[i])
			if cumul >= target {
				chosen = i
				break
			}
		}
		copy(centroids[c*contDim:(c+1)*contDim], vectors[chosen*contDim:(chosen+1)*contDim])
	}

	// k-means iterations
	for iter := 0; iter < kmeansIters; iter++ {
		changed := 0

		// Assign each vector to nearest centroid
		for i := 0; i < count; i++ {
			bestC := 0
			var bestDist float32 = math.MaxFloat32
			vOff := i * contDim
			for c := 0; c < k; c++ {
				cOff := c * contDim
				var dist float32
				for d := 0; d < contDim; d++ {
					diff := vectors[vOff+d] - centroids[cOff+d]
					dist += diff * diff
				}
				if dist < bestDist {
					bestDist = dist
					bestC = c
				}
			}
			if assignments[i] != bestC {
				assignments[i] = bestC
				changed++
			}
		}

		if changed == 0 {
			break // converged
		}

		// Update centroids
		sums := make([]float64, k*contDim)
		counts := make([]int, k)
		for i := 0; i < count; i++ {
			c := assignments[i]
			counts[c]++
			vOff := i * contDim
			cOff := c * contDim
			for d := 0; d < contDim; d++ {
				sums[cOff+d] += float64(vectors[vOff+d])
			}
		}
		for c := 0; c < k; c++ {
			if counts[c] > 0 {
				cOff := c * contDim
				for d := 0; d < contDim; d++ {
					centroids[cOff+d] = float32(sums[cOff+d] / float64(counts[c]))
				}
			}
		}
	}
}

func readVectors(path string) ([]float64, []bool, int) {
	f, err := os.Open(path)
	if err != nil {
		log.Fatalf("open: %v", err)
	}
	defer f.Close()

	gz, err := gzip.NewReader(f)
	if err != nil {
		log.Fatalf("gzip: %v", err)
	}
	defer gz.Close()

	dec := json.NewDecoder(gz)
	tok, _ := dec.Token()
	if d, ok := tok.(json.Delim); !ok || d != '[' {
		log.Fatal("expected '['")
	}

	const est = 3_000_000
	vectors := make([]float64, 0, est*origDim)
	labels := make([]bool, 0, est)
	count := 0

	var e entry
	for dec.More() {
		if err := dec.Decode(&e); err != nil {
			log.Fatalf("decode %d: %v", count, err)
		}
		vectors = append(vectors, e.Vector...)
		labels = append(labels, e.Label == "fraud")
		count++
		if count%500_000 == 0 {
			fmt.Printf("[build] %d entries read...\n", count)
		}
	}

	return vectors, labels, count
}

func writeBinary(path string, partitions []partitionData, contDimOrder []int, totalVectors int) {
	f, err := os.Create(path)
	if err != nil {
		log.Fatalf("create: %v", err)
	}
	defer f.Close()

	w := bufio.NewWriterSize(f, 1<<20) // 1MB write buffer
	buf4 := make([]byte, 4)
	buf2 := make([]byte, 2)

	// Header
	binary.LittleEndian.PutUint32(buf4, uint32(len(partitions)))
	w.Write(buf4)
	binary.LittleEndian.PutUint32(buf4, uint32(totalVectors))
	w.Write(buf4)
	binary.LittleEndian.PutUint32(buf4, uint32(contDim))
	w.Write(buf4)

	for d := 0; d < contDim; d++ {
		binary.LittleEndian.PutUint32(buf4, uint32(contDimOrder[d]))
		w.Write(buf4)
	}

	for _, p := range partitions {
		binary.LittleEndian.PutUint32(buf4, uint32(p.key))
		w.Write(buf4)
		binary.LittleEndian.PutUint32(buf4, uint32(p.numVectors))
		w.Write(buf4)
		binary.LittleEndian.PutUint32(buf4, uint32(p.numClusters))
		w.Write(buf4)

		for i := 0; i < p.numClusters*contDim; i++ {
			binary.LittleEndian.PutUint32(buf4, math.Float32bits(p.centroids[i]))
			w.Write(buf4)
		}

		for i := 0; i <= p.numClusters; i++ {
			binary.LittleEndian.PutUint32(buf4, uint32(p.clusterOffsets[i]))
			w.Write(buf4)
		}

		for i := 0; i < p.numClusters*contDim; i++ {
			binary.LittleEndian.PutUint16(buf2, uint16(p.bboxMin[i]))
			w.Write(buf2)
		}

		for i := 0; i < p.numClusters*contDim; i++ {
			binary.LittleEndian.PutUint16(buf2, uint16(p.bboxMax[i]))
			w.Write(buf2)
		}

		for i := 0; i < p.numVectors*contDim; i++ {
			binary.LittleEndian.PutUint16(buf2, uint16(p.orderedVectors[i]))
			w.Write(buf2)
		}

		w.Write(p.orderedLabels)
	}

	w.Flush()

	fi, _ := f.Stat()
	fmt.Printf("[build] Done! %s (%.1f MB)\n", path, float64(fi.Size())/(1024*1024))
}

type partitionData struct {
	key            uint8
	numVectors     int
	numClusters    int
	centroids      []float32
	clusterOffsets []int
	bboxMin        []int16
	bboxMax        []int16
	orderedVectors []int16
	orderedLabels  []byte
}
