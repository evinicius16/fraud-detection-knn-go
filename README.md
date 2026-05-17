# Fraud Detection KNN — Go

Solução em Go para a [Rinha de Backend 2026](https://github.com/zanfranceschi/rinha-de-backend-2026): detecção de fraude em transações de cartão usando busca vetorial KNN-5 sobre 3 milhões de vetores de referência.

## Algoritmo: Hybrid Partition + IVF

A busca é exata (mesmo resultado que brute-force) mas varre tipicamente <5% do dataset.

```
Query chega
  → Calcula partition key (5 bits discretos, O(1))
  → Roteia para 1 das 32 partições (~94K vetores)
  → Mini-IVF dentro da partição (K=32 clusters)
    → Scan do cluster mais próximo (~3K vetores)
    → Bbox lower-bound pruning nos outros 31 clusters
    → Early exit a cada 2 dimensões
  → Retorna top-5 → fraud_score → approved/denied
```

### Por que funciona

5 das 14 dimensões são discretas (0/1 ou -1 sentinela):
- `is_online`, `card_present`, `unknown_merchant` → binários
- `minutes_since_last_tx`, `km_from_last_tx` → -1 quando `last_transaction == null`

Vetores na mesma partição compartilham os mesmos valores nessas dims, então elas contribuem **zero** pra distância intra-partição. Eliminamos essas 5 dims do cálculo e armazenamos apenas 9 dimensões contínuas por vetor.

**Resultado:** 36% menos compute por comparação + roteamento O(1) + dataset 32x menor por partição.

### Otimizações adicionais

| Técnica | Impacto |
|---------|---------|
| Particionamento por bits discretos | Roteamento O(1), elimina 5 dims do cálculo |
| Mini-IVF por partição (K=32) | Scan de ~3K vetores vs ~94K |
| Bbox lower-bound pruning | Descarta clusters sem varrer nenhum vetor |
| Dimensões reordenadas por variância | Early exit dispara mais cedo |
| Quantização int16 (×10000) | Metade da memória, cache-friendly |
| mmap compartilhado | 2 instâncias, 1 cópia física dos dados |
| Respostas pré-computadas em bytes | Zero alocação por response |
| fasthttp + jsoniter | Menos overhead que net/http + encoding/json |
| sync.Pool para vetores de query | Zero GC pressure no hot path |

## Arquitetura

```
Nginx (porta 9999, round-robin)
  ├── API #1 (Go, porta 8080)
  └── API #2 (Go, porta 8080)
       └── mmap compartilhado (ivf_index.bin, ~87 MB)
```

## Recursos (dentro do limite da rinha)

| Serviço | CPU | Memória |
|---------|-----|---------|
| Nginx | 0.05 | 10 MB |
| API #1 | 0.475 | 170 MB |
| API #2 | 0.475 | 170 MB |
| **Total** | **1.0** | **350 MB** |

## Como rodar

```bash
# Subir com Docker (build inclui download do dataset + pré-processamento)
docker compose up --build -d

# Esperar /ready
until curl -sf http://localhost:9999/ready > /dev/null; do sleep 1; done

# Teste rápido
curl -X POST http://localhost:9999/fraud-score \
  -H "Content-Type: application/json" \
  -d '{"id":"tx-1","transaction":{"amount":100,"installments":1,"requested_at":"2026-03-11T20:00:00Z"},"customer":{"avg_amount":500,"tx_count_24h":2,"known_merchants":["MERC-001"]},"merchant":{"id":"MERC-001","mcc":"5411","avg_amount":200},"terminal":{"is_online":false,"card_present":true,"km_from_home":5},"last_transaction":{"timestamp":"2026-03-11T18:00:00Z","km_from_current":10}}'

# Load test com k6 (requer k6 instalado)
k6 run test/loadtest.js
```

## Estrutura

```
├── cmd/
│   ├── server/main.go          # HTTP server + warmup + startup
│   └── preprocess/main.go      # Build do índice híbrido (roda no Docker build)
├── internal/
│   ├── model/model.go          # Structs do payload
│   ├── vectorize/              # Normalização → vetor 14D
│   ├── search/                 # Hybrid Partition + IVF search
│   ├── data/                   # Loaders: mmap, JSON, binário
│   └── integration/            # Testes end-to-end
├── data/                       # mcc_risk.json, normalization.json
├── test/                       # k6 load test + smoke test
├── nginx.conf
├── Dockerfile                  # Multi-stage: build → preprocess → runtime
└── docker-compose.yml
```

## Pré-requisitos

- Docker + Docker Compose
- k6 (para load test)
- Go 1.22+ (para desenvolvimento local)
