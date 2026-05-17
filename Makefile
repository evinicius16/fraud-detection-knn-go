.PHONY: test bench build run docker-up docker-down smoke-test load-test clean download-data

# Download the official reference dataset
download-data:
	@echo "Downloading references.json.gz (~16 MB)..."
	curl -L -o data/references.json.gz \
		https://github.com/zanfranceschi/rinha-de-backend-2026/raw/main/resources/references.json.gz
	@echo "Done!"

# Run all unit and integration tests
test:
	go test ./... -v -count=1

# Run benchmarks
bench:
	go test ./internal/search/ -bench=. -benchmem -count=3

# Build the server binary
build:
	CGO_ENABLED=0 go build -ldflags="-s -w" -o bin/server ./cmd/server

# Run locally (requires data files)
run: build
	REFERENCES_PATH=./data/references.json.gz \
	MCC_RISK_PATH=./data/mcc_risk.json \
	NORMALIZATION_PATH=./data/normalization.json \
	PORT=8080 \
	./bin/server

# Docker compose up
docker-up:
	docker compose up --build -d
	@echo "Waiting for API to be ready..."
	@for i in $$(seq 1 60); do \
		if curl -s http://localhost:9999/ready > /dev/null 2>&1; then \
			echo "API is ready!"; \
			break; \
		fi; \
		sleep 1; \
	done

# Docker compose down
docker-down:
	docker compose down

# Run smoke tests against running instance
smoke-test:
	./test/smoke_test.sh http://localhost:9999

# Run k6 load test (requires k6 installed)
load-test:
	k6 run test/loadtest.js

# Clean build artifacts
clean:
	rm -rf bin/
	docker compose down --rmi local --volumes 2>/dev/null || true
