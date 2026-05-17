package data

import (
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"syscall"
	"unsafe"

	"github.com/evinicius16/fraud-detection-knn-go/internal/search"
)

// LoadHybridIndexMmap loads the hybrid partition index via mmap.
func LoadHybridIndexMmap(path string) (*search.HybridIndex, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open index: %w", err)
	}
	defer f.Close()

	fi, err := f.Stat()
	if err != nil {
		return nil, fmt.Errorf("stat: %w", err)
	}

	size := int(fi.Size())
	data, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	offset := 0

	// Header
	numPartitions := int(binary.LittleEndian.Uint32(data[offset:]))
	offset += 4
	totalVectors := int(binary.LittleEndian.Uint32(data[offset:]))
	offset += 4
	contDim := int(binary.LittleEndian.Uint32(data[offset:]))
	offset += 4

	if contDim != search.ContDim {
		return nil, fmt.Errorf("contDim mismatch: got %d, expected %d", contDim, search.ContDim)
	}

	// Continuous dim order
	contDimOrder := make([]int, contDim)
	for d := 0; d < contDim; d++ {
		contDimOrder[d] = int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
	}

	fmt.Printf("[data] Hybrid index: %d partitions, %d total vectors\n", numPartitions, totalVectors)

	idx := &search.HybridIndex{
		ContDimOrder: contDimOrder,
		TotalVectors: totalVectors,
	}

	// Per partition
	for p := 0; p < numPartitions; p++ {
		key := uint8(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
		numVectors := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4
		numClusters := int(binary.LittleEndian.Uint32(data[offset:]))
		offset += 4

		// Centroids
		centroidsSize := numClusters * contDim
		centroidsBytes := data[offset : offset+centroidsSize*4]
		centroids := unsafe.Slice((*float32)(unsafe.Pointer(&centroidsBytes[0])), centroidsSize)
		offset += centroidsSize * 4

		// Cluster offsets
		clusterOffsets := make([]int, numClusters+1)
		for i := 0; i <= numClusters; i++ {
			clusterOffsets[i] = int(binary.LittleEndian.Uint32(data[offset:]))
			offset += 4
		}

		// BBox min
		bboxSize := numClusters * contDim
		bboxMinBytes := data[offset : offset+bboxSize*2]
		bboxMin := unsafe.Slice((*int16)(unsafe.Pointer(&bboxMinBytes[0])), bboxSize)
		offset += bboxSize * 2

		// BBox max
		bboxMaxBytes := data[offset : offset+bboxSize*2]
		bboxMax := unsafe.Slice((*int16)(unsafe.Pointer(&bboxMaxBytes[0])), bboxSize)
		offset += bboxSize * 2

		// Vectors
		vectorsSize := numVectors * contDim
		vectorsBytes := data[offset : offset+vectorsSize*2]
		vectors := unsafe.Slice((*int16)(unsafe.Pointer(&vectorsBytes[0])), vectorsSize)
		offset += vectorsSize * 2

		// Labels
		labelsBytes := data[offset : offset+numVectors]
		labels := make([]bool, numVectors)
		for i := 0; i < numVectors; i++ {
			labels[i] = labelsBytes[i] == 1
		}
		offset += numVectors

		part := &search.Partition{
			Key:            key,
			NumVectors:     numVectors,
			NumClusters:    numClusters,
			Centroids:      centroids,
			ClusterOffsets: clusterOffsets,
			BboxMin:        bboxMin,
			BboxMax:        bboxMax,
			Vectors:        vectors,
			Labels:         labels,
		}

		if key < 32 {
			idx.Partitions[key] = part
		}
	}

	fmt.Printf("[data] Hybrid index loaded (%.1f MB mapped)\n", float64(size)/(1024*1024))
	return idx, nil
}

// LoadReferencesMmap kept for backward compat.
func LoadReferencesMmap(path string) (*search.Dataset, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open: %w", err)
	}
	defer f.Close()

	fi, _ := f.Stat()
	size := int(fi.Size())
	mmapData, err := syscall.Mmap(int(f.Fd()), 0, size, syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap: %w", err)
	}

	count := int(binary.LittleEndian.Uint32(mmapData[0:4]))
	dim := 14
	vectorsBytes := mmapData[4 : 4+count*dim*4]
	vectors := unsafe.Slice((*float32)(unsafe.Pointer(&vectorsBytes[0])), count*dim)

	labelsBytes := mmapData[4+count*dim*4:]
	labels := make([]bool, count)
	for i := 0; i < count; i++ {
		labels[i] = labelsBytes[i] == 1
	}

	return &search.Dataset{Vectors: vectors, Labels: labels, Count: count}, nil
}

func LoadReferencesBinary(path string) (*search.Dataset, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read: %w", err)
	}
	count := int(binary.LittleEndian.Uint32(raw[0:4]))
	dim := 14
	vectors := make([]float32, count*dim)
	offset := 4
	for i := range vectors {
		vectors[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[offset:]))
		offset += 4
	}
	labels := make([]bool, count)
	for i := 0; i < count; i++ {
		labels[i] = raw[offset+i] == 1
	}
	return &search.Dataset{Vectors: vectors, Labels: labels, Count: count}, nil
}
