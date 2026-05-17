#!/bin/bash
# Smoke test - validates the API is working correctly after docker-compose up.
# Usage: ./test/smoke_test.sh [base_url]

BASE_URL="${1:-http://localhost:9999}"
PASS=0
FAIL=0

echo "=== Rinha de Backend 2026 - Smoke Test ==="
echo "Base URL: $BASE_URL"
echo ""

# Helper function
assert_status() {
    local desc="$1"
    local expected="$2"
    local actual="$3"
    if [ "$actual" = "$expected" ]; then
        echo "✅ PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL: $desc (expected $expected, got $actual)"
        FAIL=$((FAIL + 1))
    fi
}

assert_contains() {
    local desc="$1"
    local expected="$2"
    local body="$3"
    if echo "$body" | grep -q "$expected"; then
        echo "✅ PASS: $desc"
        PASS=$((PASS + 1))
    else
        echo "❌ FAIL: $desc (expected to contain '$expected', got: $body)"
        FAIL=$((FAIL + 1))
    fi
}

# Test 1: GET /ready
echo "--- Test 1: GET /ready ---"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" "$BASE_URL/ready")
assert_status "GET /ready returns 2xx" "200" "$STATUS"
echo ""

# Test 2: POST /fraud-score - Legit transaction
echo "--- Test 2: Legit transaction ---"
BODY=$(curl -s -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-smoke-001",
    "transaction": {"amount": 41.12, "installments": 2, "requested_at": "2026-03-11T18:45:53Z"},
    "customer": {"avg_amount": 82.24, "tx_count_24h": 3, "known_merchants": ["MERC-003", "MERC-016"]},
    "merchant": {"id": "MERC-016", "mcc": "5411", "avg_amount": 60.25},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 29.23},
    "last_transaction": null
  }')
assert_contains "Response has 'approved' field" '"approved"' "$BODY"
assert_contains "Response has 'fraud_score' field" '"fraud_score"' "$BODY"
echo "  Response: $BODY"
echo ""

# Test 3: POST /fraud-score - Fraud transaction
echo "--- Test 3: Fraud transaction ---"
BODY=$(curl -s -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-smoke-002",
    "transaction": {"amount": 9505.97, "installments": 10, "requested_at": "2026-03-14T05:15:12Z"},
    "customer": {"avg_amount": 81.28, "tx_count_24h": 20, "known_merchants": ["MERC-008", "MERC-007", "MERC-005"]},
    "merchant": {"id": "MERC-068", "mcc": "7802", "avg_amount": 54.86},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 952.27},
    "last_transaction": null
  }')
assert_contains "Response has 'approved' field" '"approved"' "$BODY"
assert_contains "Response has 'fraud_score' field" '"fraud_score"' "$BODY"
echo "  Response: $BODY"
echo ""

# Test 4: POST /fraud-score - With last_transaction
echo "--- Test 4: Transaction with last_transaction ---"
BODY=$(curl -s -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-smoke-003",
    "transaction": {"amount": 384.88, "installments": 3, "requested_at": "2026-03-11T20:23:35Z"},
    "customer": {"avg_amount": 769.76, "tx_count_24h": 3, "known_merchants": ["MERC-009", "MERC-001"]},
    "merchant": {"id": "MERC-001", "mcc": "5912", "avg_amount": 298.95},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 13.71},
    "last_transaction": {"timestamp": "2026-03-11T14:58:35Z", "km_from_current": 18.86}
  }')
assert_contains "Response has 'approved' field" '"approved"' "$BODY"
assert_contains "Response has 'fraud_score' field" '"fraud_score"' "$BODY"
echo "  Response: $BODY"
echo ""

# Test 5: POST /fraud-score - Unknown MCC
echo "--- Test 5: Unknown MCC ---"
BODY=$(curl -s -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-smoke-004",
    "transaction": {"amount": 100, "installments": 1, "requested_at": "2026-01-01T12:00:00Z"},
    "customer": {"avg_amount": 200, "tx_count_24h": 1, "known_merchants": []},
    "merchant": {"id": "MERC-999", "mcc": "9999", "avg_amount": 150},
    "terminal": {"is_online": true, "card_present": false, "km_from_home": 0},
    "last_transaction": null
  }')
assert_contains "Response has 'approved' field" '"approved"' "$BODY"
assert_contains "Response has 'fraud_score' field" '"fraud_score"' "$BODY"
echo "  Response: $BODY"
echo ""

# Test 6: POST /fraud-score - Overflow values (should clamp)
echo "--- Test 6: Overflow values (clamp test) ---"
BODY=$(curl -s -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  -d '{
    "id": "tx-smoke-005",
    "transaction": {"amount": 50000, "installments": 48, "requested_at": "2026-06-01T23:59:59Z"},
    "customer": {"avg_amount": 100, "tx_count_24h": 100, "known_merchants": ["MERC-001"]},
    "merchant": {"id": "MERC-001", "mcc": "5411", "avg_amount": 99999},
    "terminal": {"is_online": false, "card_present": true, "km_from_home": 99999},
    "last_transaction": {"timestamp": "2026-05-01T00:00:00Z", "km_from_current": 99999}
  }')
assert_contains "Response has 'approved' field" '"approved"' "$BODY"
assert_contains "Response has 'fraud_score' field" '"fraud_score"' "$BODY"
echo "  Response: $BODY"
echo ""

# Test 7: Invalid JSON (should return fallback, not 500)
echo "--- Test 7: Invalid JSON (fallback) ---"
STATUS=$(curl -s -o /dev/null -w "%{http_code}" -X POST "$BASE_URL/fraud-score" \
  -H "Content-Type: application/json" \
  -d 'invalid json')
assert_status "Invalid JSON returns 200 (fallback)" "200" "$STATUS"
echo ""

# Summary
echo "=== Results ==="
echo "Passed: $PASS"
echo "Failed: $FAIL"
echo ""

if [ $FAIL -gt 0 ]; then
    echo "❌ SOME TESTS FAILED"
    exit 1
else
    echo "✅ ALL TESTS PASSED"
    exit 0
fi
