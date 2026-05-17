import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate, Trend } from 'k6/metrics';

const fraudRate = new Rate('fraud_detected');
const approvedRate = new Rate('approved');
const errorRate = new Rate('errors');
const fraudScoreTrend = new Trend('fraud_score');

export const options = {
  stages: [
    { duration: '10s', target: 10 },   // ramp up
    { duration: '30s', target: 50 },   // sustain
    { duration: '10s', target: 100 },  // peak
    { duration: '10s', target: 0 },    // ramp down
  ],
  thresholds: {
    http_req_duration: ['p(99)<2000'],  // p99 < 2s
    errors: ['rate<0.15'],              // error rate < 15%
  },
};

const BASE_URL = __ENV.BASE_URL || 'http://localhost:9999';

// Sample payloads for testing
const payloads = [
  // Legit transaction - low amount, known merchant, close to home
  {
    id: 'tx-load-001',
    transaction: { amount: 41.12, installments: 2, requested_at: '2026-03-11T18:45:53Z' },
    customer: { avg_amount: 82.24, tx_count_24h: 3, known_merchants: ['MERC-003', 'MERC-016'] },
    merchant: { id: 'MERC-016', mcc: '5411', avg_amount: 60.25 },
    terminal: { is_online: false, card_present: true, km_from_home: 29.23 },
    last_transaction: null,
  },
  // Fraud transaction - high amount, unknown merchant, far from home
  {
    id: 'tx-load-002',
    transaction: { amount: 9505.97, installments: 10, requested_at: '2026-03-14T05:15:12Z' },
    customer: { avg_amount: 81.28, tx_count_24h: 20, known_merchants: ['MERC-008', 'MERC-007', 'MERC-005'] },
    merchant: { id: 'MERC-068', mcc: '7802', avg_amount: 54.86 },
    terminal: { is_online: false, card_present: true, km_from_home: 952.27 },
    last_transaction: null,
  },
  // Medium risk - moderate amount, with last_transaction
  {
    id: 'tx-load-003',
    transaction: { amount: 384.88, installments: 3, requested_at: '2026-03-11T20:23:35Z' },
    customer: { avg_amount: 769.76, tx_count_24h: 3, known_merchants: ['MERC-009', 'MERC-001'] },
    merchant: { id: 'MERC-001', mcc: '5912', avg_amount: 298.95 },
    terminal: { is_online: false, card_present: true, km_from_home: 13.71 },
    last_transaction: { timestamp: '2026-03-11T14:58:35Z', km_from_current: 18.86 },
  },
  // Online transaction - unknown merchant
  {
    id: 'tx-load-004',
    transaction: { amount: 1500.00, installments: 6, requested_at: '2026-03-12T02:30:00Z' },
    customer: { avg_amount: 200.00, tx_count_24h: 8, known_merchants: ['MERC-001'] },
    merchant: { id: 'MERC-050', mcc: '7995', avg_amount: 500.00 },
    terminal: { is_online: true, card_present: false, km_from_home: 0 },
    last_transaction: { timestamp: '2026-03-12T01:00:00Z', km_from_current: 500.0 },
  },
  // Very low amount legit
  {
    id: 'tx-load-005',
    transaction: { amount: 15.50, installments: 1, requested_at: '2026-03-10T12:00:00Z' },
    customer: { avg_amount: 50.00, tx_count_24h: 1, known_merchants: ['MERC-010', 'MERC-011'] },
    merchant: { id: 'MERC-010', mcc: '5311', avg_amount: 30.00 },
    terminal: { is_online: false, card_present: true, km_from_home: 2.5 },
    last_transaction: { timestamp: '2026-03-09T18:00:00Z', km_from_current: 3.0 },
  },
];

export default function () {
  const payload = payloads[Math.floor(Math.random() * payloads.length)];

  const res = http.post(`${BASE_URL}/fraud-score`, JSON.stringify(payload), {
    headers: { 'Content-Type': 'application/json' },
    timeout: '2001ms',
  });

  const isError = res.status !== 200;
  errorRate.add(isError);

  if (!isError) {
    const body = JSON.parse(res.body);

    check(res, {
      'status is 200': (r) => r.status === 200,
      'has approved field': () => body.approved !== undefined,
      'has fraud_score field': () => body.fraud_score !== undefined,
      'fraud_score in range': () => body.fraud_score >= 0 && body.fraud_score <= 1,
    });

    fraudRate.add(!body.approved);
    approvedRate.add(body.approved);
    fraudScoreTrend.add(body.fraud_score);
  }
}

// Verify readiness before test
export function setup() {
  const res = http.get(`${BASE_URL}/ready`);
  if (res.status !== 200) {
    throw new Error(`API not ready: status ${res.status}`);
  }
  console.log('API is ready!');
}
