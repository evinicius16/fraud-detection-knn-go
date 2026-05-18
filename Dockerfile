# ============================================================
# Stage 1: Build the server and preprocessor binaries
# ============================================================
FROM golang:1.22-alpine AS builder

WORKDIR /build

COPY go.mod go.sum ./
RUN go mod download

COPY . .

# Build server — optimize for speed on amd64, strip debug info
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -gcflags="-B" -o /app/server ./cmd/server

# Build preprocessor
RUN CGO_ENABLED=0 GOOS=linux GOARCH=amd64 \
    go build -ldflags="-s -w" -o /app/preprocess ./cmd/preprocess

# ============================================================
# Stage 2: Build IVF index (JSON.gz → k-means + quantize + bbox)
# ============================================================
FROM alpine:3.19 AS preprocessor

WORKDIR /app

COPY --from=builder /app/preprocess .

# Download dataset
ADD https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz /app/data/references.json.gz

# Build IVF index: hybrid partition + mini-IVF per partition
RUN ./preprocess /app/data/references.json.gz /app/data/ivf_index.bin

# ============================================================
# Stage 3: Final minimal runtime image
# ============================================================
FROM alpine:3.19

WORKDIR /app

# Copy server binary
COPY --from=builder /app/server .

# Copy IVF index
COPY --from=preprocessor /app/data/ivf_index.bin /app/data/ivf_index.bin

# Copy small config files
COPY data/mcc_risk.json /app/data/mcc_risk.json
COPY data/normalization.json /app/data/normalization.json

EXPOSE 8080

CMD ["./server"]
