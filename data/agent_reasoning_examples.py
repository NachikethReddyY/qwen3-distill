#!/usr/bin/env python3
"""
25 high-quality agent and complex-reasoning training examples for model distillation.

Each dict has:
  - "prompt": a specific, complex, multi-step task requiring reasoning and planning
  - "answer": a detailed response containing a <thinking> step-by-step reasoning
              section, code or a structured approach, and an explanation of the
              reasoning process and tradeoffs.

Categories covered: system design, project planning, code refactoring strategies,
performance optimization, migration planning, troubleshooting complex systems,
and multi-component architecture.

To merge into the training pipeline, wrap each into the
{"messages": [system, user, assistant]} jsonl format used by the repo.
"""

SYSTEM_PROMPT = (
    "You are an expert staff engineer with deep knowledge of distributed systems, "
    "performance engineering, and software architecture. Think through problems step "
    "by step inside <thinking>...</thinking> tags, explaining your reasoning, the "
    "tradeoffs you weigh, and the alternatives you reject. Then give a final answer "
    "with concrete code or a structured plan."
)

examples = [
    {
        "prompt": "Design a URL shortener (like bit.ly) that must handle 100M new URLs per month and 10B redirects per month with sub-50ms p99 redirect latency. Walk through capacity estimation, the ID generation scheme, the storage layer, and the caching strategy. Justify each choice.",
        "answer": """<thinking>
Let me reason about scale first, then design backward from the constraints.

Capacity estimation:
- Writes: 100M/month ~= 40 writes/sec average, ~120/sec peak (3x). Trivial.
- Reads: 10B/month ~= 3,850 reads/sec average, ~12k/sec peak. This is the hot path.
- Read:write ratio is ~100:1, so the system is read-dominated -> cache aggressively.
- Storage: 100M URLs/month * 12 months * 5 years = 6B rows. At ~500 bytes/row = 3TB. Needs sharding or a horizontally scalable store.

ID generation -- the key design decision:
- Option A: hash(longURL) -> base62. Risk of collisions, need collision handling, but idempotent (same URL -> same short code, saves space).
- Option B: auto-increment counter -> base62. Predictable/enumerable (security/privacy leak), and a global counter is a bottleneck.
- Option C: distributed unique ID (e.g. a ticket/range server or Snowflake-like) -> base62. Scales, non-enumerable if we add randomness.
I'll choose a hybrid: a range-allocation service hands each app server a block of 10k IDs (e.g. via an atomic counter in a small coordination store). Servers convert IDs to base62 locally. This removes the per-write coordination bottleneck while keeping IDs short. 62^7 ~= 3.5 trillion, so 7 chars is plenty.

Storage:
- The access pattern is a simple key-value lookup by short code. A relational DB is overkill on the read path. Use a KV store (e.g. DynamoDB / Cassandra) keyed by short_code -> {long_url, created_at, owner}. Partition by short_code hash for even distribution.

Caching for the 50ms p99:
- A redirect is one KV GET. The 100:1 read skew + a power-law popularity distribution means a small cache absorbs most reads. Put a cache (Redis cluster) in front; cache short_code -> long_url. Hit ratio will be very high. On miss, read from KV and backfill. TTL + LRU.
- For p99, also add a CDN/edge layer that can do the redirect at the edge for the hottest links.

Putting numbers to the cache: even caching the top ~1% of codes likely covers >90% of traffic given typical link popularity.
</thinking>

## Architecture

```
Client -> CDN/Edge -> Load Balancer -> App Servers
                                          |  (cache miss)
                                          v
                                   Redis Cluster  ->  KV Store (DynamoDB/Cassandra)
                                          ^
                                  ID Range Service (counter)
```

### 1. Capacity
| Metric | Value |
|---|---|
| Avg writes | ~40/s (~120/s peak) |
| Avg reads | ~3.85k/s (~12k/s peak) |
| 5yr storage | ~3 TB |
| Read:write | ~100:1 (cache-friendly) |

### 2. ID generation (range allocation, base62)
```python
import string
ALPHABET = string.digits + string.ascii_letters  # 62 chars

def encode_base62(n: int) -> str:
    if n == 0:
        return ALPHABET[0]
    s = []
    while n:
        n, r = divmod(n, 62)
        s.append(ALPHABET[r])
    return "".join(reversed(s))

class IDAllocator:
    \"\"\"Each app server leases a contiguous block of IDs to avoid per-write coordination.\"\"\"
    BLOCK = 10_000
    def __init__(self, counter_store):
        self.store = counter_store
        self.next_id = 0
        self.end = 0
    def new_code(self) -> str:
        if self.next_id >= self.end:
            start = self.store.increment_by(self.BLOCK)  # atomic
            self.next_id, self.end = start, start + self.BLOCK
        code = encode_base62(self.next_id)
        self.next_id += 1
        return code
```

### 3. Redirect hot path (cache-aside)
```python
def redirect(short_code, cache, kv):
    url = cache.get(short_code)
    if url is None:                      # miss
        url = kv.get(short_code)         # single-key lookup
        if url is None:
            return 404
        cache.set(short_code, url, ttl=86400)
    return Response(status=301, headers={"Location": url})
```

## Reasoning & tradeoffs
- **Range allocation over a global counter**: removes the write-path bottleneck and DB round-trip per write at the cost of small ID gaps when a server restarts (acceptable, codes are cheap).
- **Hash-based IDs rejected**: collisions complicate the write path and gain little since storage is cheap.
- **301 vs 302**: 301 is cacheable by browsers/CDN (lowers our read load) but you lose click analytics; use 302 if analytics matter. State this tradeoff explicitly to the product owner.
- **The p99 budget is spent in the network + one cache GET**; the cache hit ratio is what makes 50ms achievable, so monitoring hit ratio is the key SLO leading indicator.""",
    },
    {
        "prompt": "We have a 400k-line Django monolith. The 'orders' domain is tangled with 'inventory' and 'billing' via direct ORM imports and shared models. Lay out a strategy to extract 'orders' into a separate service without a big-bang rewrite. Include the sequencing, how to handle the shared database, and how to de-risk each step.",
        "answer": """<thinking>
This is a strangler-fig migration. The dangerous parts are: (1) shared DB tables, (2) synchronous in-process calls becoming network calls, (3) distributed transactions that used to be ACID. I should never recommend a big-bang. The strategy must keep the system shippable at every step.

Sequencing principle: enforce the boundary *inside the monolith first* (logical decoupling), then physically split (separate deploy), then split the database last. Each phase is independently shippable and reversible.

Phase ordering reasoning:
1. Establish a module boundary: stop direct ORM cross-imports; route all cross-domain access through an explicit interface. This is the highest-leverage, lowest-risk step and can be done while everything is still one process.
2. Introduce an anti-corruption layer / facade so callers depend on an API, not the ORM.
3. Split the deployable: same DB, separate service. Now you have a network boundary but still one source of truth -> avoids the hardest problem (data split) while proving the API works.
4. Split the data: give orders its own DB. Cross-domain reads become API calls; cross-domain writes that were transactions become sagas/outbox events.

De-risking: dual-run / shadow traffic, feature flags, and the outbox pattern for eventual consistency. Measure before/after.

The DB split is the crux. Foreign keys from orders to inventory/billing must be broken. Replace FK joins with API calls or denormalized copies + events.
</thinking>

## Phased strangler-fig plan

### Phase 0 — Map & instrument
- Generate a dependency graph of cross-domain imports (`grep`/AST tooling). Catalog every place `orders` touches `inventory`/`billing` models and vice-versa.
- Add metrics/logging on those call sites so you can measure later.

### Phase 1 — Logical boundary (still one process, one DB)
Define an explicit interface module; ban direct cross-domain ORM imports via a lint rule.
```python
# orders/ports.py  -- the only sanctioned way other domains call orders
class OrdersService:
    def create_order(self, cmd: CreateOrder) -> OrderId: ...
    def get_order(self, order_id: OrderId) -> OrderDTO: ...

# enforce with an import-linter contract in CI:
# [importlinter:contract]
# type = forbidden
# source_modules = inventory, billing
# forbidden_modules = orders.models
```

### Phase 2 — Anti-corruption facade
Callers depend on DTOs, not ORM objects. Internally the facade still calls the ORM, so behavior is unchanged and shippable.

### Phase 3 — Split the deployable (shared DB)
Move `orders` to its own service. Replace in-process facade calls with HTTP/gRPC. **Keep the shared database** so you avoid data consistency problems while validating the network boundary. Use shadow traffic: send a copy of real requests to the new service and diff responses before cutover.

### Phase 4 — Split the database (the hard part)
```
Before: orders JOIN inventory (FK)        After: orders -> inventory API / cached read
Before: order+billing in one txn          After: outbox event -> billing consumes (saga)
```
Outbox pattern to avoid dual-write inconsistency:
```python
def place_order(cmd):
    with db.transaction():                     # single-DB atomic write
        order = Orders.create(cmd)
        Outbox.create(topic="order.placed",    # same txn = no lost events
                      payload=order.to_event())
    # async relay publishes Outbox rows to the broker, at-least-once
```
Compensating action (saga) if billing later fails: emit `order.cancelled`.

## De-risking checklist (every phase)
- **Reversible**: each phase ships behind a flag and can be rolled back.
- **Shadow/dual-run** before any cutover; diff outputs.
- **Strangler routing**: a façade/proxy decides old-path vs new-path per request.
- **SLO guardrails**: watch latency/error budgets; auto-rollback on regression.

## Tradeoffs
- Splitting the DB last is deliberate: it's the irreversible, consistency-breaking step, so you do it only after the API is proven. The cost is living with a shared DB (a coupling smell) temporarily.
- Moving to events trades strong consistency for availability + autonomy. Anything that truly needs ACID across domains (e.g. money) should stay in one service or use a carefully tested saga with idempotent compensations.""",
    },
    {
        "prompt": "A batch ETL job that processes 50M rows nightly has crept from 40 minutes to 6 hours over a year. Nothing about the data volume changed much (~10% growth). Give me a systematic methodology to find and fix the regression, and show the kind of code change you'd expect to make.",
        "answer": """<thinking>
A 9x slowdown from 10% data growth screams a complexity regression: something went from O(n) to O(n^2), or an N+1 query got introduced, or an index got dropped, or memory pressure caused swapping/GC thrashing. The 'crept over a year' detail suggests gradual accumulation OR a step-change that nobody noticed -- I should bisect by time.

Methodology: don't guess, measure. First, profile to localize. Second, characterize the curve -- is it superlinear? If I run on 10%, 50%, 100% of data and the time grows faster than linearly, it's an algorithmic problem. Third, bisect the git history / the runtime breakdown by stage.

Most common culprits in ETL:
1. N+1 queries (a lookup inside a per-row loop).
2. Lost index / changed query plan as table grew past a planner threshold.
3. Accidental O(n^2): `list.append` then `if x in list` membership checks; nested loops; repeated DataFrame concat.
4. Memory: dataset no longer fits, causing swap or chunk-by-chunk re-reads.

The classic fix shape: replace per-row DB lookups with a single bulk fetch into a dict (hash join in app code), turning O(n) queries into O(1).
</thinking>

## Systematic methodology

### Step 1 — Localize (where is the time going?)
Add per-stage timing before profiling line-by-line:
```python
import time, contextlib
@contextlib.contextmanager
def stage(name, stats):
    t = time.perf_counter()
    yield
    stats[name] = stats.get(name, 0) + time.perf_counter() - t
```
Then run a real profiler (`cProfile`, `py-spy dump` on the live process) to find the hot function.

### Step 2 — Characterize the curve
Run on 10%, 25%, 50%, 100% of rows. Plot time vs rows.
- Linear -> it's a constant-factor problem (slow I/O, no index) -> 10% growth shouldn't cause 9x, so look for a step change.
- **Superlinear -> algorithmic regression.** A 9x jump from 1.1x data is consistent with O(n^2) over the year's cumulative growth.

### Step 3 — Bisect
- `git log` the ETL module over the last year; `git bisect` against a fixed sample if reproducible.
- Check DB plan: `EXPLAIN ANALYZE` the queries the job runs; look for a seq scan where an index scan used to be (planner flips once a table crosses a row-count threshold).

### Step 4 — The likely fix (N+1 -> bulk hash join)
Before (O(n) round-trips, the regression):
```python
for row in source_rows:                       # 50M iterations
    cust = db.query(Customer).get(row.cust_id)  # one query PER row = 50M queries
    out.write(transform(row, cust))
```
After (one query, in-memory join):
```python
ids = {r.cust_id for r in source_rows}
customers = {c.id: c for c in db.query(Customer)
                                 .filter(Customer.id.in_(ids))}   # 1 bulk query
for row in source_rows:
    out.write(transform(row, customers[row.cust_id]))            # O(1) dict lookup
```

## Reasoning & tradeoffs
- **Measure before fixing**: a 9x regression has one dominant cause; guessing wastes nights. The curve-characterization in Step 2 is what distinguishes "buy a bigger box" (linear) from "fix the algorithm" (superlinear).
- **Bulk-fetch tradeoff**: loading all customers into memory uses RAM; if the lookup set is huge, batch it (chunks of 100k IDs) to bound memory. That's the second-order tradeoff to watch.
- **Index/plan fixes are cheap but fragile**: a forced index or updated statistics (`ANALYZE`) can restore the old plan, but the real fix is reducing query count.
- Add a **regression guard**: assert the job's rows/second stays above a threshold in CI-like canary runs so the next 2x creep is caught immediately.""",
    },
    {
        "prompt": "Design the architecture for a real-time collaborative document editor (like Google Docs) supporting 50 simultaneous editors per document. Address the concurrency model (OT vs CRDT), the server topology, presence/cursors, and offline editing. Justify the core data-model decision.",
        "answer": """<thinking>
The central decision is the concurrency model: Operational Transformation (OT) vs CRDT. Everything else hangs off it.

OT: operations are transformed against concurrent ops to converge. Requires a central server to order/transform (well-understood, what Google Docs actually uses), tricky transform functions, hard to get right for rich text.
CRDT: data structure mathematically guarantees convergence regardless of order, so it works peer-to-peer and offline-first without a central authority. Costs: metadata overhead (tombstones, per-character IDs), memory growth.

For 50 editors/doc with offline support, CRDT is the better fit: offline editing + merge-on-reconnect is native, and you don't need a perfectly-ordered central transform pipeline. The tradeoff is memory/GC of CRDT metadata, which modern sequence CRDTs (RGA / Yjs-style) handle well.

Topology: even with a CRDT, you want a server as a relay + persistence + auth, not pure p2p (NAT, fan-out, history). So: clients <-> WebSocket relay <-> shared CRDT doc, with periodic snapshot persistence.

Presence/cursors are ephemeral, not document state -> separate channel, no persistence, lossy is fine.

Offline: CRDT buffers local ops, syncs deltas on reconnect.
</thinking>

## Core decision: CRDT (sequence CRDT, e.g. RGA/Yjs-style)

```
Editor A ─┐
Editor B ─┼─ WebSocket ─> Relay/Sync Server ─> Doc CRDT (in mem) ─> Snapshot store
Editor C ─┘                     │                                   (S3 + op log)
                                └─> Presence channel (ephemeral, not persisted)
```

### Concurrency model
```javascript
// Each insert gets a globally-unique, ordered ID (siteId, clock).
// CRDT guarantees: applying the same set of ops in ANY order -> same doc.
function applyRemoteOp(op) {
  if (seen.has(op.id)) return;           // idempotent (at-least-once delivery ok)
  seen.add(op.id);
  doc.integrate(op);                     // commutative + associative + idempotent
  rerenderFrom(op.position);
}
// Local edit -> create op -> apply locally -> broadcast delta
function onLocalInsert(char, index) {
  const op = doc.localInsert(char, index);
  broadcast(op);                         // optimistic: UI updates immediately
}
```

### Presence & cursors (separate, ephemeral)
```javascript
// Lossy, high-frequency, never persisted. Throttle to ~20/s.
presenceChannel.publish({ userId, cursor: pos, selection, color });
// On disconnect, server emits a 'leave' so others clear the cursor.
```

### Offline editing
```javascript
// Offline: queue ops locally (IndexedDB). On reconnect, exchange state vectors
// and send only the delta the server is missing -> O(changes), not O(doc).
const missing = remoteStateVector.diff(localStateVector);
sync.send(doc.encodeDelta(missing));
```

### Persistence
- Append ops to a log for audit/recovery; periodically write a compacted CRDT snapshot so new joiners load a snapshot + tail instead of replaying all history.

## Reasoning & tradeoffs
- **CRDT over OT**: offline + multi-master merge is native; no central transform pipeline to get subtly wrong. Cost = per-character metadata and tombstones, mitigated by snapshot compaction and garbage collection of fully-acknowledged tombstones.
- **Relay server, not pure p2p**: solves NAT traversal, auth, fan-out to 50 peers, and durable history. We keep CRDT's offline benefit without p2p's operational pain.
- **Presence off the document path**: cursors change ~constantly; persisting or ordering them would swamp the doc sync. Treating them as lossy ephemeral state is the key scalability choice.
- **Optimistic local apply**: edits feel instant; convergence is guaranteed by the CRDT, so we never need to roll back local input (a big UX win over server-authoritative OT round-trips).""",
    },
    {
        "prompt": "Plan a zero-downtime migration of a 2TB PostgreSQL primary from on-prem to AWS RDS, where the application cannot tolerate more than 60 seconds of write downtime. Detail the cutover sequence, how you keep data in sync, the rollback plan, and the verification steps.",
        "answer": """<thinking>
Zero (well, <60s) write downtime for a 2TB DB across networks. A dump/restore would take hours of downtime -> rejected. The right approach is logical (or physical) replication: stand up the target, replicate continuously, then do a tiny cutover window where you stop writes, let replication catch up, flip the app, and resume.

Key constraints:
- 2TB over a network: initial sync is the long pole; it happens online while the source serves traffic.
- The 60s window is only for: quiesce writes -> drain replication lag to 0 -> repoint app -> resume.
- Rollback must be possible at every moment, including after cutover (so set up reverse replication or a fallback window).

Method choice: PostgreSQL logical replication (publication/subscription) works cross-version and lets the target be live/queryable, and AWS DMS is an alternative. Logical replication is cleaner if versions are compatible.

The riskiest things: replication lag at cutover, sequences not advancing on the new primary, and missing objects logical replication doesn't copy (sequences, large objects, DDL). I must explicitly handle sequences.

Verification: row counts + checksums on critical tables, and a read-only smoke test against the target before flipping writes.
</thinking>

## Cutover sequence

### Phase 1 — Prepare (online, days before)
1. Provision RDS target sized for headroom. Set `wal_level=logical` on source.
2. Create schema on target (pg_dump --schema-only). Disable target indexes/FKs you can rebuild after bulk load for speed.
3. Start initial sync via logical replication:
```sql
-- on source
CREATE PUBLICATION mig FOR ALL TABLES;
-- on target (RDS)
CREATE SUBSCRIPTION mig CONNECTION 'host=onprem ...' PUBLICATION mig;
```
4. Let the 2TB copy + ongoing CDC run for as long as needed. Monitor lag:
```sql
SELECT now() - pg_last_xact_replay_timestamp() AS replication_lag;
```

### Phase 2 — Pre-cutover verification (online)
- Reconcile row counts per table; checksum critical tables:
```sql
SELECT count(*), md5(string_agg(t::text, '' ORDER BY id)) FROM orders t;  -- run on both
```
- Run the app in **read-only** against the target as a smoke test.

### Phase 3 — Cutover (<60s window)
```
t+0s   Set app to maintenance / reject writes (read-only mode at the LB/app)
t+2s   Wait for replication_lag -> 0 (already near-zero from continuous CDC)
t+10s  Advance sequences on target:  SELECT setval(...) for each sequence  <-- critical
t+15s  Repoint app connection string / DNS to RDS endpoint
t+25s  Run write smoke test (insert+rollback) against RDS
t+35s  Re-enable writes
```
Most of the budget is buffer; steady-state lag should already be sub-second.

### Phase 4 — Rollback plan
- **Before flip**: trivial — do nothing, source is still primary.
- **After flip, if problems**: reverse-replicate RDS->on-prem was set up so the old primary stays warm; flip the connection string back. Keep this reverse path for ~24-48h, then decommission.

## Verification checklist
- [ ] Per-table row counts match
- [ ] Checksums match on money/critical tables
- [ ] **Sequences advanced** (the #1 forgotten step -> duplicate-key errors)
- [ ] FKs/indexes/extensions/large objects present on target
- [ ] App read smoke test (pre-flip) and write smoke test (post-flip) pass
- [ ] Monitoring/alerts repointed to RDS

## Reasoning & tradeoffs
- **Logical replication over dump/restore**: keeps downtime to the catch-up window (seconds) instead of hours, because the bulk copy happens online.
- **Sequences are the classic trap**: logical replication copies row data but not sequence positions; forgetting `setval` causes immediate PK collisions on the new primary. Explicitly scripted.
- **Reverse replication for rollback**: the real risk is *after* cutover; a warm old-primary turns a catastrophe into a 1-minute flip-back, at the cost of running both for a day.
- **DMS vs native**: DMS is more forgiving across major versions and can transform, but adds a moving part; native logical replication is simpler and lower-latency when versions are compatible.""",
    },
    {
        "prompt": "Our microservices platform has intermittent 5-second latency spikes on ~1% of requests, only in production, never reproducible in staging. Walk me through a rigorous debugging methodology to root-cause this, including what signals to gather and the hypotheses to rule in/out in order.",
        "answer": """<thinking>
Intermittent, ~1%, only in prod, exactly ~5 seconds: the '5 seconds' is a huge clue. Round-number latency almost always means a timeout-and-retry or a default timeout somewhere (connection pool, DNS, TCP retransmit ~3s, gRPC default deadline, a 5s client timeout). 1% + only-prod points at something prod-specific: scale, real traffic distribution, a noisy neighbor, GC, connection-pool exhaustion under load, or a single bad instance.

Methodology: this is a tail-latency problem, so averages lie -- I need percentiles and per-request tracing. The plan: (1) confirm it's tail not mean, (2) get distributed traces for the slow 1% to see WHICH span eats the 5s, (3) form ranked hypotheses from the span location, (4) correlate with infra signals.

Ranked hypotheses given the symptoms:
1. Connection-pool exhaustion: requests wait for a pooled connection; classic under prod load, invisible in staging. Manifests as time spent *before* the downstream call.
2. GC pause / stop-the-world: ~hundreds of ms usually, but full GC can hit seconds; correlates with heap.
3. Retry-after-timeout: a downstream call times out at ~5s default and the latency is exactly the timeout. The trace span shows the call taking ~5s.
4. DNS / service-discovery slow resolution.
5. A single unhealthy instance / noisy neighbor (cgroup throttling, CPU steal).

The discriminator is WHERE the 5s lives in the trace: in-queue (pool), in-GC (runtime metrics), in a downstream span (timeout), or pinned to one host (bad instance).
</thinking>

## Methodology

### Step 1 — Reframe as a tail problem
Stop looking at averages. Pull p50/p99/p99.9 per service and per endpoint. Confirm the 5s lives in the tail (p99.9) and find which endpoint(s) own it.

### Step 2 — Trace the slow 1% specifically
Enable/sample distributed tracing **biased toward slow requests** (tail-based sampling). For captured slow traces, find which span contains the ~5s:
```
Request total: 5,040 ms
├─ gateway              12 ms
├─ auth                  8 ms
├─ order-svc          5,005 ms   <-- the 5s lives here
│   ├─ db-pool-acquire 4,990 ms  <-- !!! waiting for a connection, not the query
│   └─ db-query           9 ms
└─ serialize             3 ms
```
The span breakdown *is* the diagnosis. The example above points squarely at pool exhaustion.

### Step 3 — Correlate with infra/runtime signals (gather these)
| Hypothesis | Signal to check | Confirming pattern |
|---|---|---|
| Pool exhaustion | pool active/idle/wait counts, wait time | spikes line up with the latency spikes |
| GC pause | GC pause duration, heap before/after | full-GC events at the same timestamps |
| Timeout+retry | downstream call duration, retry counter | spans pinned at exactly the timeout value |
| DNS | resolver latency | slow resolves on cache miss |
| Bad instance / noisy neighbor | per-host p99, CPU steal, cgroup throttle | spikes isolated to one host/AZ |

### Step 4 — Confirm by intervention
For the leading hypothesis (say, pool exhaustion): raise pool size or add a queue-wait metric+alert, deploy to one canary, and check the p99.9 on the canary vs the rest.

## Likely fix (pool exhaustion example)
```python
# Before: pool too small for prod concurrency; under load, requests block ~5s
#   for a free connection, which is also the client timeout -> the 5s spikes.
pool = create_pool(max_size=10, acquire_timeout=5)

# After: size the pool to peak concurrency, fail fast, and expose wait metrics.
pool = create_pool(max_size=peak_concurrent_queries * 1.2, acquire_timeout=1)
metrics.gauge("db.pool.wait_ms", pool.wait_time)   # so it's never invisible again
```

## Reasoning & tradeoffs
- **The round number is the tell**: ~5s almost never comes from "the work being slow"; it comes from a configured timeout/limit. I rank timeout/pool hypotheses first because of it.
- **Tail-based tracing is non-negotiable** here: at 1%, uniform sampling will miss the slow ones; you must bias sampling toward high latency.
- **Only-in-prod = load- or scale-dependent**: pool exhaustion, GC under real heap pressure, and noisy neighbors are all things staging can't reproduce, which is itself a clue that narrows the field.
- **Canary the fix**: don't trust a fix you can't reproduce; prove it on one instance against live traffic before fleet-wide rollout.""",
    },
    {
        "prompt": "Design a rate limiter for a public API that must enforce per-user quotas (e.g. 1000 req/min) across a fleet of 30 stateless API servers behind a load balancer. Compare the algorithm options, handle the distributed-counter problem, and address what happens when the central store is briefly unavailable.",
        "answer": """<thinking>
Two sub-problems: (1) which algorithm, (2) how to share state across 30 servers.

Algorithm options:
- Fixed window: simple counter per minute. Flaw: boundary burst -- a user can do 1000 at 11:59:59 and 1000 at 12:00:00 = 2000 in 2 seconds.
- Sliding window log: store every request timestamp. Accurate but memory-heavy (O(requests)).
- Sliding window counter: weighted blend of current+previous fixed window. Cheap and smooths the boundary burst. Good default.
- Token bucket: tokens refill at a rate, allows bursts up to bucket size, smooth average. Great when you want to permit bursts.
- Leaky bucket: smooths output to a constant rate.

For a public API quota with some burst tolerance, token bucket is the standard choice; sliding-window-counter if you want strict per-minute counts.

Distributed state: 30 stateless servers can't each keep local counters (a user hitting different servers would get 30x the limit). Need a shared store (Redis). The counter increment must be atomic -> Lua script or INCR with EXPIRE. To avoid a round-trip per request becoming the bottleneck, you can do approximate local buckets + periodic reconciliation, but start with central Redis for correctness.

Failure mode: if Redis is down, do you fail-open (allow all, risk abuse) or fail-closed (block all, outage)? For a public API, brief fail-open with a local fallback limiter is usually safer for availability, but it must be bounded.
</thinking>

## Algorithm comparison
| Algorithm | Burst handling | Memory | Notes |
|---|---|---|---|
| Fixed window | bad (2x boundary burst) | O(1) | simplest, avoid for strict limits |
| Sliding log | exact | O(n) | accurate but heavy |
| Sliding window counter | good | O(1) | great default for "per minute" |
| **Token bucket** | configurable burst | O(1) | chosen: smooth rate + allowed bursts |

## Distributed token bucket (atomic in Redis)
```lua
-- KEYS[1] = bucket key (per user)   ARGV: rate, capacity, now, requested
local data   = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens = tonumber(data[1]) or tonumber(ARGV[2])   -- start full
local ts     = tonumber(data[2]) or tonumber(ARGV[3])
local rate, cap, now, need = tonumber(ARGV[1]), tonumber(ARGV[2]), tonumber(ARGV[3]), tonumber(ARGV[4])

tokens = math.min(cap, tokens + (now - ts) * rate)       -- refill
local allowed = tokens >= need
if allowed then tokens = tokens - need end
redis.call('HMSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('EXPIRE', KEYS[1], math.ceil(cap / rate) + 1)
return allowed and 1 or 0
```
The Lua script runs atomically on Redis, so all 30 servers share one consistent bucket without race conditions.

## Store-unavailable behavior (graceful degradation)
```python
def allow(user):
    try:
        return redis_token_bucket(user)          # authoritative path
    except RedisUnavailable:
        # Fail-open but BOUNDED: each server enforces a local fallback at
        # limit/30 so the worst case across the fleet ~ the real limit.
        return local_fallback_bucket(user, rate=GLOBAL_RATE / N_SERVERS)
```

## Reasoning & tradeoffs
- **Token bucket** chosen because public APIs usually want to permit short bursts while bounding the sustained rate; capacity tunes burst tolerance independently from rate.
- **Atomicity is the crux**: read-modify-write across 30 servers without an atomic op = race that lets users exceed the limit. The Lua script collapses it into one atomic operation; pipelined INCR/EXPIRE has a subtle window, so Lua is preferred.
- **Fail-open vs fail-closed**: I fail *open but bounded*. Fail-closed turns a cache blip into a full API outage; pure fail-open invites abuse. Dividing the global limit across servers as a local fallback caps the blast radius while keeping the API up.
- **Hot-key risk**: a single very-high-volume user concentrates load on one Redis key/shard; mitigate with per-user local pre-aggregation (count locally, sync every N ms) accepting slight over-admission for big keys.""",
    },
    {
        "prompt": "I need to choose a database for a new IoT telemetry platform ingesting 500k sensor readings/second, where queries are mostly time-range aggregations over recent data and old data is rarely read. Reason through the candidate database types, the data modeling, and the retention/downsampling strategy. Recommend one and justify it.",
        "answer": """<thinking>
Workload signature: extreme write throughput (500k/s), append-only, time-ordered, queries are time-range aggregations (downsampled rollups), recent data hot / old data cold. This is the textbook time-series database (TSDB) workload.

Candidate types:
- Relational (Postgres): great for general queries but 500k writes/s needs heavy partitioning; possible with TimescaleDB (a Postgres extension purpose-built for this).
- Vanilla NoSQL KV (Cassandra/Scylla): excellent write throughput, time-bucketed partition keys work well; aggregations need app-side or Spark. Operationally heavier.
- Purpose-built TSDB (InfluxDB, TimescaleDB, ClickHouse): columnar/compressed, built-in downsampling, retention policies, time-range scans are native.
- ClickHouse: columnar OLAP, insane scan/aggregation speed, great compression; handles 500k/s with batching.

Decision drivers: 500k/s sustained needs columnar compression + batched ingest. Queries are aggregations -> columnar wins big (ClickHouse/Timescale). Old data cold -> tiered storage + downsampling + TTL.

Data modeling: wide vs narrow. Narrow (one row per (series, ts, value)) is flexible; columnar storage compresses repeated tags. Partition by time (e.g. daily) so retention = drop partition (O(1)) instead of DELETE (expensive).

Retention/downsampling: keep raw for N days, continuous-aggregate into 1m/1h rollups, expire raw via partition drop.
</thinking>

## Candidate comparison
| Option | 500k/s ingest | Time-range aggs | Ops cost | Verdict |
|---|---|---|---|---|
| Postgres (plain) | needs heavy partitioning | good | low | strains at this rate |
| Cassandra/Scylla | excellent | weak (app-side) | high | great writes, poor aggs |
| **ClickHouse** | excellent (batched) | excellent (columnar) | medium | strong fit |
| TimescaleDB | good | excellent + SQL | low | strong fit, SQL ergonomics |

## Recommendation: a columnar TSDB (ClickHouse, or TimescaleDB if SQL/Postgres ecosystem matters)

### Data model (narrow, time-partitioned, columnar)
```sql
CREATE TABLE readings (
    sensor_id  UInt32,
    metric     LowCardinality(String),
    ts         DateTime64(3),
    value      Float64
) ENGINE = MergeTree
PARTITION BY toYYYYMMDD(ts)          -- daily partitions -> O(1) retention drop
ORDER BY (sensor_id, metric, ts);    -- sorts for fast range scans + compression
```
- Ingest in **batches** (e.g. buffer 10–50k rows then bulk insert); per-row inserts will not sustain 500k/s.

### Downsampling (continuous/materialized aggregation)
```sql
CREATE MATERIALIZED VIEW readings_1m
ENGINE = AggregatingMergeTree PARTITION BY toYYYYMMDD(ts) ORDER BY (sensor_id, metric, ts)
AS SELECT sensor_id, metric, toStartOfMinute(ts) AS ts,
          avgState(value) AS avg_v, maxState(value) AS max_v
   FROM readings GROUP BY sensor_id, metric, ts;
```

### Retention (tiered)
- Raw: hot tier, 7–30 days, then **DROP PARTITION** (instant, no tombstones).
- 1m rollups: warm, 90 days. 1h rollups: cold/object storage, years.
- `TTL ts + INTERVAL 30 DAY` to automate, or move to S3-backed cold storage.

## Reasoning & tradeoffs
- **Columnar TSDB over KV**: queries are aggregations, and columnar compression + vectorized scans make time-range aggs orders of magnitude cheaper, while compression tames 500k/s of storage growth.
- **Batched ingest is mandatory**: no engine sustains 500k *individual* inserts/s; a write-ahead buffer trades a few seconds of durability-on-crash risk (mitigated by a Kafka ingestion buffer in front) for throughput.
- **Partition-by-time = cheap retention**: dropping a partition is O(1); `DELETE WHERE ts < x` would be catastrophic at this scale. This single modeling choice dominates retention cost.
- **TimescaleDB vs ClickHouse**: pick Timescale if you want full SQL/Postgres tooling and transactional metadata alongside; pick ClickHouse for raw aggregation/scan throughput at the cost of a less familiar operational model.
- **Kafka in front**: decouples ingest spikes from the DB and provides replay/durability — worth the extra component at this throughput.""",
    },
    {
        "prompt": "Refactor a 1200-line 'God class' OrderProcessor that handles validation, pricing, tax, inventory, payment, notification, and persistence. Give me a concrete decomposition strategy, the target design, the order in which to extract pieces safely, and how to keep the system green throughout.",
        "answer": """<thinking>
A God class mixes many responsibilities and is hard to test/change. Goal: separate concerns without breaking behavior. The safe path is characterization tests first, then incremental extraction, keeping tests green at every step (Working-Effectively-with-Legacy-Code style).

Decomposition: the responsibilities listed are basically a pipeline: validate -> price -> tax -> reserve inventory -> charge payment -> persist -> notify. That's a natural seam: extract each stage into its own collaborator behind an interface, and have OrderProcessor become a thin orchestrator (or a pipeline of steps).

Order of extraction matters: extract the *leaf* responsibilities with the fewest dependencies first (notification, tax calc are pure-ish), and the most entangled (persistence, payment side effects) later. Pure functions are safest to extract first.

Keeping it green: characterization tests pin current behavior; then for each extraction, move code into a new class, delegate, run tests. Use dependency injection so the orchestrator gets collaborators -> testable in isolation.

Watch for: hidden state shared across responsibilities (a mutable order object mutated everywhere), and ordering dependencies (payment must come after inventory reservation). The pipeline model makes ordering explicit.
</thinking>

## Target design: thin orchestrator + a pipeline of single-responsibility steps
```python
class OrderStep(Protocol):
    def apply(self, ctx: OrderContext) -> OrderContext: ...

class OrderProcessor:                       # was 1200 lines, now an orchestrator
    def __init__(self, steps: list[OrderStep]):
        self.steps = steps
    def process(self, order) -> Result:
        ctx = OrderContext(order)
        for step in self.steps:             # explicit, ordered, each replaceable
            ctx = step.apply(ctx)
        return ctx.result

# Wiring (DI):
processor = OrderProcessor([
    ValidationStep(), PricingStep(tax=TaxService()),
    InventoryStep(repo), PaymentStep(gateway),
    PersistenceStep(repo), NotificationStep(mailer),
])
```

## Safe extraction sequence
1. **Pin behavior**: write characterization tests that capture current outputs for representative orders (including edge cases). These are your safety net.
2. **Extract pure leaves first** (lowest risk): `PricingStep`, `TaxStep`, `ValidationStep` — these are mostly input->output with no side effects. Move code, delegate, run tests after each.
3. **Extract side-effecting steps** behind interfaces: `InventoryStep`, `PaymentStep`, `PersistenceStep`, `NotificationStep`. Inject their dependencies so you can fake them in tests.
4. **Introduce OrderContext** to carry state between steps explicitly, replacing scattered mutable fields.
5. **Invert ordering dependencies** into the pipeline order (inventory reserve before payment, payment before persist), making implicit sequencing explicit.
6. **Delete dead code** from the old class as each responsibility moves out; the class shrinks to the orchestrator.

```python
# Example of one safe extraction step (delegate, don't rewrite):
class OrderProcessor:
    def _calculate_price(self, order):
        return self._pricing.apply(order)   # was 150 inline lines; now delegates
```

## Reasoning & tradeoffs
- **Characterization tests before any change**: you cannot safely refactor untested legacy code; these tests let every subsequent step be verified green. This is the single most important discipline here.
- **Extract pure functions first**: lowest blast radius, builds confidence and test infrastructure before touching side-effecting code (payment/persistence) where mistakes are costly.
- **Pipeline + context object**: makes ordering and data flow explicit (the God class hid both), and each step becomes independently unit-testable with fakes.
- **Tradeoff — orchestration overhead**: a pipeline of steps adds indirection and small object allocation; for a hot loop that might matter, but for order processing the clarity/testability win dominates.
- **Don't over-abstract**: if two steps are always coupled and never reused, keep them together. Decompose along the seams that actually change independently, not every noun.""",
    },
    {
        "prompt": "Plan the rollout of a new recommendation model to production where it must not degrade revenue. Design the experimentation and deployment strategy: shadow mode, A/B testing, guardrail metrics, gradual ramp, and automated rollback. Include how you'd decide the model is safe to fully launch.",
        "answer": """<thinking>
This is ML deployment risk management. The model affects revenue, so we cannot just ship it. The standard safe progression: offline eval -> shadow mode (no user impact) -> small online A/B -> gradual ramp with guardrails -> full launch, with automated rollback at every online stage.

Each stage answers a different question:
- Offline: is the model better on historical data? (necessary, not sufficient — offline/online gap is real.)
- Shadow: does it run in prod without errors/latency issues, and do its live predictions look sane? (no user sees output.)
- A/B small %: does it actually move the target metric online, without hurting guardrails?
- Ramp: does the effect hold at scale, no rare-segment regressions?
- Full launch: decision based on stat-sig improvement in the primary metric AND no guardrail breach.

Guardrail metrics are key: not just revenue (primary) but latency, error rate, and counter-metrics (e.g. diversity, did we tank a segment, did engagement drop). Need a stopping rule and automated rollback.

Statistical rigor: pick primary metric + minimum detectable effect, run long enough for power, watch novelty effects, segment analysis.
</thinking>

## Staged rollout

### Stage 1 — Offline evaluation
- Backtest on held-out data; compare ranking metrics (NDCG, CTR proxy) and a revenue-proxy. Gate: must beat the incumbent offline. (Necessary but not sufficient — offline rarely predicts online perfectly.)

### Stage 2 — Shadow mode (0% user impact)
- Run the new model in parallel; log its predictions but **serve the old model**. Verify: latency p99 within budget, no errors, prediction distribution sane (no degenerate "recommend same item to everyone").
```python
def recommend(user):
    live = incumbent.predict(user)          # served to user
    shadow = candidate.predict(user)        # logged only
    log_compare(user, live, shadow)         # for offline/online gap analysis
    return live
```

### Stage 3 — A/B test (small %)
- Randomize users: 95% control / 5% treatment. Define **primary metric** (revenue/user) + **minimum detectable effect** + required sample size for power up front.
- Guardrails (auto-halt if breached): latency, error rate, and counter-metrics (engagement, catalog diversity, per-segment revenue).

### Stage 4 — Gradual ramp with automated rollback
```python
RAMP = [0.05, 0.10, 0.25, 0.50, 1.0]
for pct in RAMP:
    set_treatment_fraction(pct)
    metrics = observe(window="24h")
    if metrics.guardrail_breached() or metrics.revenue_delta < -GUARDRAIL:
        set_treatment_fraction(previous_pct)   # automated rollback
        alert("ramp halted at %s" % pct); break
    if not metrics.primary_significant(): wait_for_more_data()
```

### Stage 5 — Launch decision
Ship to 100% only when **all** hold:
- Primary metric improvement is statistically significant (and practically meaningful), after enough time to wash out novelty effects.
- No guardrail breached at any ramp step.
- No critical segment regressed (segment-level analysis, not just the aggregate).

## Reasoning & tradeoffs
- **Shadow before A/B**: catches operational failures (latency, crashes, degenerate outputs) with zero revenue risk, so the A/B only tests *quality*, not *plumbing*.
- **Guardrails + counter-metrics**: optimizing one metric can silently wreck another (a model can boost short-term clicks while tanking diversity/long-term retention). Counter-metrics catch the "winning the battle, losing the war" failure.
- **Gradual ramp over instant 100%**: rare-segment regressions and capacity effects only appear at scale; ramping bounds the blast radius and gives the auto-rollback room to act.
- **Statistical discipline**: predeclaring the metric, MDE, and duration prevents p-hacking and stopping early on noise. Novelty effects mean the first day's lift often overstates the true effect — wait it out.
- **Tradeoff — speed vs safety**: this whole process costs weeks. For low-risk surfaces you can compress stages; for revenue-critical surfaces the caution is worth it.""",
    },
    {
        "prompt": "Architect an event-driven order-fulfillment system spanning order, payment, inventory, and shipping services. The business requirement is that an order is never charged without inventory reserved, and never shipped without payment captured. Design the saga, handle failures/compensation, and ensure exactly-once-ish effects despite at-least-once delivery.",
        "answer": """<thinking>
Cross-service business invariant with no distributed ACID transaction available. This is a saga. Two flavors: orchestration (a central coordinator drives the steps) vs choreography (services react to each other's events). Given strict ordering invariants (reserve -> charge -> ship) and the need for clear compensation, orchestration is easier to reason about and audit.

The invariants:
- never charge without inventory reserved -> reserve must succeed BEFORE charge.
- never ship without payment captured -> capture must succeed BEFORE ship.
So the happy path is a strict sequence. Each step needs a compensating action if a later step fails:
- reserve inventory -> compensation: release reservation
- capture payment -> compensation: refund
- ship -> (terminal; hard to compensate, so it's last)

Delivery is at-least-once -> every handler must be idempotent (dedup by a key) so a redelivered "capture payment" doesn't double-charge. True exactly-once doesn't exist across systems; we get effectively-once via idempotency keys + the outbox pattern (atomic state+event write).

Failure handling: a saga step can fail transiently (retry) or terminally (compensate and abort). The orchestrator tracks saga state in a durable store.
</thinking>

## Saga: orchestrated, with compensation

```
OrderSaga (orchestrator, durable state machine):
  CREATED
   -> reserve_inventory  --fail--> ABORT (no comp needed yet)
   -> capture_payment    --fail--> compensate: release_inventory -> ABORTED
   -> create_shipment    --fail--> compensate: refund_payment,
                                                release_inventory -> ABORTED
   -> COMPLETED
```

```python
class OrderSaga:
    STEPS = [
        Step("reserve_inventory", compensate="release_inventory"),
        Step("capture_payment",   compensate="refund_payment"),
        Step("create_shipment",   compensate=None),   # terminal
    ]
    def advance(self, saga):
        for i, step in enumerate(self.STEPS):
            ok = self.execute(step, saga)              # idempotent + retried
            if not ok:
                self.compensate(self.STEPS[:i], saga)  # unwind in reverse
                saga.state = "ABORTED"; return
        saga.state = "COMPLETED"
```

## Idempotency under at-least-once delivery
```python
def capture_payment(cmd):
    # cmd.idempotency_key is stable per saga step (e.g. saga_id + "capture")
    if processed.contains(cmd.idempotency_key):
        return processed.result(cmd.idempotency_key)   # dedup redelivery
    result = gateway.charge(cmd.amount, key=cmd.idempotency_key)  # gateway also dedups
    processed.put(cmd.idempotency_key, result)
    return result
```

## Atomic state + event publish (outbox)
```python
def on_inventory_reserved(event):
    with db.transaction():                     # one DB, atomic
        saga = load(event.saga_id)
        saga.mark("inventory_reserved")
        outbox.add("capture_payment", saga.payment_cmd())  # same txn -> no lost/ghost events
```

## Reasoning & tradeoffs
- **Orchestration over choreography**: the invariant is a strict ordered sequence with explicit compensations; a central state machine makes the order, the current state, and the unwind path auditable. Choreography would scatter that logic across services and make "where is this order stuck?" hard to answer. The cost is a coordinator component and some coupling.
- **Compensation, not rollback**: there's no distributed transaction, so "undo" is a *business* action (refund, release) — and it must itself be idempotent and retried, because compensations can fail too.
- **Ordering enforces the invariants directly**: reserve-before-charge and capture-before-ship are encoded as step order, so the invariants hold by construction, not by hope.
- **Effectively-once, not exactly-once**: at-least-once + idempotency keys + outbox gives effectively-once side effects. The outbox prevents the dual-write problem (state saved but event lost, or event sent but state not saved).
- **Shipping is last and non-compensable**: irreversible steps go at the end of a saga so that everything reversible is already committed before you cross the point of no return.""",
    },
    {
        "prompt": "Our React single-page app has a 4.5MB JavaScript bundle and a 9-second time-to-interactive on mid-tier mobile. Give me a prioritized, measurement-driven optimization plan to get TTI under 3 seconds. Reason about what to attack first and quantify expected gains.",
        "answer": """<thinking>
Frontend perf. 4.5MB JS is the obvious villain — parsing/executing JS on mobile CPUs is the dominant cost, worse than download. TTI is gated by main-thread work. Prioritize by impact/effort.

Measure first: is it download-bound or CPU/parse-bound? On mid-tier mobile it's almost always CPU/parse-bound for a 4.5MB bundle. Need a bundle analysis to see what's in there (often: moment.js locales, lodash whole import, duplicate deps, a huge charting lib, unused polyfills).

Highest-leverage levers, roughly ordered:
1. Code splitting / lazy loading: don't ship the whole app for the first screen. Route-based + component-based splitting. Biggest TTI win because TTI only needs the initial route's code.
2. Tree-shaking / dependency diet: replace heavy libs (moment->date-fns/dayjs, lodash->lodash-es per-method, drop duplicate deps). Often 1-2MB.
3. Defer non-critical work: analytics, A/B, third-party scripts off the critical path.
4. Compression (brotli) + caching + CDN: cheap, download-side.
5. SSR/streaming or prerender for first paint, hydrate later.

Quantify: route-splitting can cut initial JS by 60-70% if the app has many routes. Dependency diet another chunk. The combination plausibly takes 4.5MB initial -> ~1MB initial, and parse time scales with bytes, so ~9s -> ~3s is realistic.
</thinking>

## Measurement-driven plan (highest impact first)

### Step 0 — Measure & analyze
- Run a bundle analyzer to see *what's* in the 4.5MB. Run Lighthouse on a throttled mid-tier device profile to confirm TTI is main-thread/parse-bound (it almost certainly is).

### Step 1 — Route-based code splitting (biggest win)
The first screen needs only its own code, not all routes.
```jsx
const Dashboard = lazy(() => import('./Dashboard'));   // separate chunk
const Reports   = lazy(() => import('./Reports'));
<Suspense fallback={<Skeleton/>}><Route path="/dashboard" element={<Dashboard/>}/></Suspense>
```
Expected: initial JS often drops 60–70% (e.g. 4.5MB -> ~1.5MB).

### Step 2 — Dependency diet (tree-shaking)
```js
// Before: pulls the whole library
import _ from 'lodash';            import moment from 'moment';
// After: per-method / lighter lib
import debounce from 'lodash-es/debounce';   import dayjs from 'dayjs';
```
- Deduplicate versions, drop unused polyfills, lazy-load the charting lib only on chart pages.
Expected: another ~0.5–1MB off initial.

### Step 3 — Defer non-critical work off the critical path
- Load analytics/A-B/3rd-party tags after `load`/idle; they steal main-thread time during TTI.

### Step 4 — Delivery (cheap wins)
- Brotli compression, long-lived cache headers with content hashing, HTTP/2, CDN. Reduces download time (secondary to parse on mobile but still helps).

### Step 5 — Consider SSR/streaming
- Server-render the first route for fast first paint; hydrate progressively. Bigger effort — only if Steps 1–4 don't get you under 3s.

## Expected trajectory
| After step | Initial JS | TTI (mid mobile) |
|---|---|---|
| baseline | 4.5 MB | ~9 s |
| + route splitting | ~1.5 MB | ~4.5 s |
| + dep diet | ~1.0 MB | ~3.3 s |
| + defer 3rd-party | ~1.0 MB | ~2.8 s |

## Reasoning & tradeoffs
- **Parse/execute, not download, dominates on mobile**: a phone CPU parses JS far slower than it downloads it, so *shipping fewer bytes of JS to the first screen* (code splitting) beats faster delivery. That's why splitting is first.
- **Measure to avoid wasted effort**: without the bundle analysis you might spend days micro-optimizing a 50KB component while a 1.5MB charting lib loads on every page.
- **Tradeoff — splitting adds requests/latency on navigation**: lazy chunks load on demand, so route transitions can show a spinner; mitigate with prefetching likely-next routes during idle.
- **SSR last**: it's high-effort and adds server cost/complexity; reach for it only if pure client-side budget can't hit the target. Quantifying expected gains first tells you whether you even need it.""",
    },
    {
        "prompt": "Design a multi-region active-active architecture for a social app's write-heavy 'post + feed' feature that must survive a full region outage with under 30 seconds of write unavailability. Address data replication, conflict resolution, routing, and the consistency model you're accepting. Justify the CAP tradeoff.",
        "answer": """<thinking>
Active-active multi-region + write-heavy + survive region loss in <30s. CAP: during a partition you must choose. They want availability under partition (survive outage, keep writing), so we accept AP -> eventual consistency, with conflict resolution. Strong global consistency (CP) would mean blocking writes during a partition -> violates the 30s requirement.

So the model: each region accepts writes locally (low latency, available), replicates asynchronously cross-region. Conflicts can happen (same user edits a post in two regions, or concurrent writes). For a social feed, conflicts are rare and tolerable -> last-write-wins with logical clocks, or CRDTs for counters (like counts).

Data:
- Posts are mostly append-only (immutable after create) -> conflicts rare; LWW on edits is fine.
- Feeds are derived; rebuild from posts -> eventually consistent is acceptable (seeing a post a few seconds late is fine socially).
- Counters (likes) -> use CRDT counters to avoid lost increments.

Routing: geo-DNS / anycast to nearest healthy region; health checks fail over in seconds. The user's writes go local; reads can be local (may be slightly stale).

Region failure: because writes are local + async replicated, losing a region doesn't block other regions. The <30s is just the DNS/health-check failover time for users in the dead region.

The honest tradeoff: a region can fail with un-replicated writes in flight -> small window of data loss. Mitigate with synchronous replication to at least one nearby region for durability-critical data, accepting a little latency.
</thinking>

## Architecture: AP, local-write + async cross-region replication

```
Users --geo-DNS/anycast--> nearest healthy region
  Region A (write-local) <==async replication==> Region B <==> Region C
  per-region: app + DB replica accepting writes; conflict resolution on merge
```

### Replication & conflict resolution
```python
# Posts: immutable -> no conflict. Edits: LWW via hybrid logical clock.
def merge_post(local, remote):
    return remote if remote.hlc > local.hlc else local   # last-write-wins

# Like counts: CRDT counter so concurrent increments in 2 regions both survive.
class GCounter:
    def __init__(self): self.per_region = defaultdict(int)
    def incr(self, region): self.per_region[region] += 1
    def value(self): return sum(self.per_region.values())   # merge = max per region
```

### Routing & failover
- Geo-DNS/anycast routes users to the nearest region; health checks detect a dead region and re-route within seconds (the <30s budget).
- Reads served locally (may be a few seconds stale); writes served locally then replicated.

### Surviving a region outage
- Other regions keep accepting writes (no global coordination on the write path), so the rest of the world has zero write downtime.
- Users in the dead region fail over via DNS in <30s.
- Durability gap mitigation: replicate durability-critical writes synchronously to one nearby region before ack, so an outage loses at most the in-flight async tail elsewhere.

## Reasoning & tradeoffs (the CAP call)
- **Chose AP (availability under partition) -> eventual consistency**, because the requirement *is* availability during a region loss. A CP design (e.g. global consensus/quorum across regions) would block writes during a partition, directly violating the <30s write-availability goal.
- **Why this is acceptable for social**: seeing a friend's post or a like-count a few seconds late has near-zero business cost; the social domain is tolerant of staleness. This is what makes AP the right call here, whereas a bank ledger would not be.
- **Conflict strategy fits the data shape**: posts are append-only (conflict-free), edits use LWW (rare, low-stakes), counters use CRDTs (so concurrent likes never get lost — LWW would drop increments). Matching the resolution to each data type is the key design move.
- **Honest cost — durability window**: pure async replication risks losing un-replicated writes if a region dies. Synchronous replication to one neighbor closes most of that gap at the price of a little write latency; that's the dial to tune per data criticality.
- **Operational complexity**: active-active is far harder to run and reason about than active-passive; only adopt it because the cross-region write-availability requirement demands it.""",
    },
    {
        "prompt": "We want to migrate ~300 services from a hand-rolled REST framework to gRPC over 6 months without freezing feature development. Produce a migration roadmap: sequencing, the dual-stack interop strategy, how to avoid a flag day, tooling/automation, and the metrics that tell you it's working.",
        "answer": """<thinking>
Large-scale migration with two hard constraints: no feature freeze, and 300 services is too many for a flag day. So this must be incremental, parallelizable across teams, and automated. The classic answer: dual-stack (serve both protocols during transition), migrate edges/leaves and high-value services first, and never require a synchronized cutover.

Sequencing logic: dependency order matters. If service A calls B, you can migrate B's *server* side to also speak gRPC while A still calls REST, then migrate A's client side. Leaf services (few callers) are easiest. Hot-path/high-value services give the most benefit early but are riskier — balance by piloting on a couple of medium-importance services to build the playbook.

Avoid flag day: each service runs both REST and gRPC simultaneously (dual-stack); callers migrate independently. Remove REST only when no caller uses it (measure traffic to prove zero usage before deletion).

No feature freeze: provide a shared library / codegen so teams add features in proto and get both stacks. Automation is essential at 300 services — a per-service migration toolkit, codegen from a single IDL, and a scorecard.

Metrics: % services dual-stack, % traffic on gRPC, latency/error deltas, and "REST endpoints with zero traffic" (ready to delete).
</thinking>

## Roadmap

### Phase 0 — Foundations (weeks 1–4)
- Define proto style guide + shared codegen so a service defines its API once and gets gRPC + a REST gateway. This is what lets teams keep shipping features without double work.
- Build a **migration toolkit**: scaffolding, codegen, a compatibility shim, and a per-service scorecard.
- Pilot on 2–3 medium-importance services to harden the playbook.

### Phase 1 — Dual-stack everything (the interop strategy)
Each service serves **both** protocols during transition:
```
        ┌──────────── Service X ────────────┐
REST ──>│ REST handler ─┐                    │
        │               ├─> shared business  │
gRPC ──>│ gRPC handler ─┘     logic          │
        └────────────────────────────────────┘
```
- Use a gRPC-gateway (gRPC <-> JSON/REST transcoding) so one implementation backs both, minimizing per-service work.

### Phase 2 — Migrate clients in dependency order
- Migrate **leaf services first** (fewest callers), then move up the call graph. A caller switches its client to gRPC independently — no coordinated cutover.
- Track per-edge: REST traffic vs gRPC traffic.

### Phase 3 — Decommission REST per service (proven safe)
- Delete a service's REST surface **only after telemetry shows zero REST traffic for N days**. This is how you avoid a flag day and avoid breaking an unknown caller.

## Automation / tooling
- Single IDL + codegen (no hand-written stubs at 300 services).
- A dashboard scorecard per service: dual-stack? % gRPC traffic? latency/error delta? REST-deletable?
- CI check: new endpoints must be defined in proto.

## Metrics that prove it's working
| Metric | Meaning |
|---|---|
| % services dual-stack | rollout coverage |
| % cross-service traffic on gRPC | adoption |
| p99 latency / error-rate delta (gRPC vs REST) | is it actually better/safe |
| # REST endpoints with 0 traffic | ready-to-delete = real progress |
| services fully on gRPC | completion |

## Reasoning & tradeoffs
- **Dual-stack is what kills the flag day**: every service speaking both protocols means callers migrate on their own schedule; there's never a moment where everything must switch at once. The cost is running two stacks temporarily (extra surface, slightly more memory).
- **Single IDL + codegen removes the feature-freeze conflict**: teams keep adding features in proto and get both protocols for free, so migration and feature work happen in the same edits instead of competing.
- **Dependency-ordered, leaf-first**: minimizes the number of callers you must coordinate per step; a server can advertise gRPC before any client uses it, decoupling the two sides of each edge.
- **Telemetry-gated deletion**: "zero traffic for N days" is the only safe signal that no forgotten caller still depends on REST — deleting on a guess is how you cause an incident.
- **Tradeoff — transition cost vs speed**: dual-stack doubles maintenance during the window; the 6-month bound forces prioritization (migrate high-traffic edges first for the biggest early benefit) rather than a uniform sweep.""",
    },
    {
        "prompt": "A production Kubernetes service is OOMKilled intermittently under load, but local load tests show stable memory. Memory grows slowly over hours then the pod is killed. Lay out a methodical investigation: what to measure, the leak-vs-limit-vs-fragmentation hypotheses, and how to confirm the root cause before changing code.",
        "answer": """<thinking>
Slow growth over hours -> killed = classic memory leak OR an unbounded cache/buffer OR a limit set too low for real traffic patterns. 'Local tests stable' means the leak is triggered by something only present in prod: real traffic diversity, long-lived connections, a specific request type, or simply longer uptime (a slow leak needs hours to manifest, local tests are short).

Hypotheses to separate:
1. True leak: heap grows unbounded -> eventually exceeds limit. Confirm via heap profiling over time (live set keeps growing).
2. Unbounded buffer/cache: an in-memory cache without eviction grows with unique keys seen (prod has many more unique keys than local). Looks like a leak but is "working as coded".
3. Limit too low / legit working set: traffic peaks need more memory than the limit allows; not a leak, just under-provisioned. Memory plateaus high but the limit is below it.
4. Fragmentation / allocator: RSS grows while live heap is flat (allocator doesn't return memory to OS). Common in some runtimes.
5. Off-heap / native: connection pools, mmap, native libs not visible in language heap profiler.

Discriminators: compare *live heap* (profiler) vs *RSS* (cgroup). If live heap grows -> leak/cache (1 or 2). If live heap flat but RSS grows -> fragmentation/native (4 or 5). If both plateau below kill but pod still dies -> wrong limit or a burst (3).

Method: measure before changing code. Capture heap profiles at intervals, diff them to find the growing allocation site.
</thinking>

## Methodical investigation

### Step 1 — Measure the right things over time
- Plot, per pod over hours: **RSS / cgroup memory** (what OOMKiller watches) vs **language live-heap** (from a profiler) vs request rate.
- Capture `kubectl describe pod` to confirm it's the container memory limit triggering the kill (OOMKilled reason) vs node pressure eviction.

### Step 2 — Discriminate with one key comparison
```
live heap rising  + RSS rising      -> leak or unbounded cache   (Step 3a)
live heap FLAT    + RSS rising      -> fragmentation / native    (Step 3b)
both plateau high, still killed     -> limit too low / burst     (Step 3c)
```

### Step 3a — Confirm a leak by diffing heap profiles
```bash
# capture heap snapshots an hour apart, then DIFF to find the growing site
# (pprof for Go, tracemalloc for Python, heap dump + MAT for JVM)
go tool pprof -base heap_t0.pb.gz heap_t1.pb.gz   # shows what grew between t0 and t1
```
The diff names the allocation site that keeps growing — that's the leak/cache. (e.g. a global dict/list appended per request, or a cache with no eviction.)

### Step 3b — Fragmentation/native
- If live heap is flat but RSS climbs: suspect allocator fragmentation (tune allocator, e.g. release-to-OS settings) or native allocations (connection pools, mmap). Profile native memory, not just the managed heap.

### Step 3c — Limit/burst
- If memory legitimately needs more than the limit at peak: raise the limit and/or fix the burst (e.g. unbounded concurrency loading big payloads into memory). Add a memory-based HPA or backpressure.

## Reasoning & tradeoffs
- **Measure before coding**: "OOMKilled" has at least four distinct causes that need *opposite* fixes (delete a cache vs raise a limit vs change allocator). Changing code on a guess often makes it worse.
- **The live-heap-vs-RSS comparison is the single most diagnostic signal**: it cleanly separates managed leaks from fragmentation/native growth, which language heap profilers alone can't tell apart.
- **Why local tests miss it**: slow leaks need hours and prod's key/traffic diversity; an unbounded cache only blows up when it sees prod's cardinality of unique keys. Reproduce by replaying prod-like traffic for hours, not a short synthetic burst.
- **Heap diff over single snapshot**: a snapshot shows what's *there*; the *diff* shows what's *growing*, which is what a leak actually is. That's the confirmation step before touching code.
- **Tradeoff in the fix**: an unbounded cache "leak" is fixed by adding eviction (bounded LRU) — but that trades memory safety for a lower hit ratio; size it from the heap-diff data, not arbitrarily.""",
    },
    {
        "prompt": "Design an idempotent, exactly-once-effect payment-processing pipeline that consumes a Kafka stream of payment requests. It must never double-charge even with consumer restarts, rebalances, and at-least-once delivery. Detail the idempotency mechanism, the commit ordering, and how you handle a crash between charging and committing the offset.",
        "answer": """<thinking>
The crux: Kafka gives at-least-once. A consumer can charge a card, then crash *before* committing the offset, so on restart it reprocesses the same message and charges again. We must make the *effect* exactly-once even though *delivery* is at-least-once. True exactly-once delivery isn't the goal; idempotent effects are.

Core technique: idempotency key. Each payment request carries (or derives) a stable unique key. Before charging, we atomically record "I am processing key K" in a durable store; the payment gateway is also called with that idempotency key so the gateway itself dedups. So even double-processing -> single charge.

Commit ordering matters. Options:
- Commit offset AFTER the side effect is durably recorded. If we crash after charging but before commit, on restart we see key K already done (in our store / gateway returns the prior result) -> skip the charge, then commit. Safe.
- The dangerous ordering is commit-before-effect (could lose a charge) — avoid for payments; prefer at-least-once + idempotency.

The atomic record + charge: ideally one transaction. But the charge is on an external system (gateway). Use the gateway's idempotency-key support so retries are deduped on their side; locally, write a "processed" record. Use an outbox/state table keyed by idempotency key with states: PENDING -> CHARGED -> COMMITTED.

Rebalances: another consumer may pick up the partition; same idempotency mechanism protects it.
</thinking>

## Pipeline design (at-least-once delivery, exactly-once effect)

```
Kafka topic --> consumer --> [idempotency store + gateway] --> commit offset
                                  (key = payment_id)
States per key: NONE -> PENDING -> CHARGED -> DONE
```

### Idempotency mechanism
```python
def handle(msg):
    key = msg.payment_id                         # stable, unique per request
    rec = store.get(key)
    if rec and rec.state == "DONE":
        return                                    # already charged -> skip (dedup)

    # Reserve intent durably BEFORE charging (so a crash leaves a trace)
    store.upsert(key, state="PENDING")

    # Gateway call is itself idempotent via the same key:
    # if we already charged and crashed, the gateway returns the SAME result,
    # not a second charge.
    result = gateway.charge(amount=msg.amount, idempotency_key=key)

    store.upsert(key, state="DONE", result=result)
```

### Commit ordering (the critical part)
```python
for msg in consumer:               # auto-commit OFF
    handle(msg)                    # side effect recorded durably first
    consumer.commit(msg.offset)    # commit ONLY after effect is durable
```

### Crash between charge and commit
```
charge succeeds -> store=DONE -> CRASH (offset not committed)
  restart: message redelivered
  handle(): store.get(key).state == "DONE" -> skip charge -> commit offset
  => single charge, despite redelivery
```

## Reasoning & tradeoffs
- **Idempotent effect, not exactly-once delivery**: chasing exactly-once *delivery* across Kafka + an external gateway is a fool's errand. Instead we accept at-least-once delivery and make the *charge* idempotent via a stable key, which is achievable and robust.
- **Commit after the effect, never before**: committing before the side effect risks *losing* a charge on crash (worse for some flows) — but for payments, at-least-once + idempotency is the safe combination: redelivery is harmless because the second attempt is deduped.
- **Gateway-side idempotency key is essential**: our local "DONE" record can't by itself prevent a double-charge if we crashed *after* the gateway charged but *before* writing DONE. Passing the idempotency key to the gateway means the gateway dedups that retry, closing the gap our local store can't.
- **PENDING state for forensics/recovery**: a key stuck in PENDING after a crash flags "we may have charged, verify with the gateway" — reconcile by querying the gateway with the key rather than blindly retrying.
- **Tradeoff — store write per message**: the durable idempotency record adds latency and a hot write path; mitigate with a fast KV store and TTL on old keys, accepting that the dedup window is bounded (fine, since redelivery happens quickly, not weeks later).""",
    },
    {
        "prompt": "Plan the technical due-diligence and integration approach for acquiring a startup whose product we must merge into our platform within 9 months. We get read access to their repos and infra. Lay out what to assess, the risks to surface, and an integration architecture decision framework (rebuild vs integrate vs strangle).",
        "answer": """<thinking>
This is part technical due diligence (assess what we're buying) and part integration planning (how to merge it). The deliverable is a framework, not code-heavy. Key risks in acquisitions: hidden tech debt, security/compliance landmines, key-person dependency, licensing, scalability ceilings, and data migration.

Due-diligence dimensions: code quality/test coverage, architecture, security & compliance, infra/cost, data model & migration difficulty, team/knowledge, licenses (open-source obligations), and operational maturity (monitoring, on-call, incidents).

Integration decision: rebuild (rewrite on our stack), integrate (adopt theirs as-is, connect via APIs), or strangle (run theirs, incrementally replace). Choose based on code quality, strategic value, divergence from our stack, and the 9-month constraint. A rebuild is risky under a deadline; integrate-then-strangle is usually safer.

Decision framework dimensions: code health, stack alignment, how core the product is, data coupling, regulatory constraints. Score each path.

Risks to surface to leadership early because they can kill the deal or blow the timeline: a GPL dependency in a proprietary product, no tests, a single engineer who knows everything, PII without compliance, a database that won't scale.
</thinking>

## Technical due diligence — what to assess
| Dimension | What to look for | Why it matters |
|---|---|---|
| Code quality | test coverage, churn hotspots, complexity | predicts integration cost & defect risk |
| Architecture | coupling, scalability ceiling, stack fit | rebuild vs integrate driver |
| Security/compliance | secrets in repo, PII handling, authz | deal-breakers; legal exposure |
| Licensing | GPL/AGPL in a proprietary product | can force rewrite or open-sourcing |
| Data | schema, volume, migration difficulty | often the long pole of integration |
| Infra & cost | cloud spend, IaC maturity, lock-in | run-rate + operability |
| Team/knowledge | bus factor, docs, on-call maturity | key-person risk post-acquisition |

Automate the first pass: dependency/license scan, secret scan, coverage report, cloud cost export, architecture diagram from the dependency graph.

## Integration decision framework
Score each candidate path across weighted criteria:
```
score(path) = Σ wᵢ · criterionᵢ
criteria: code_health, stack_alignment, strategic_coreness,
          data_coupling, regulatory_fit, time_to_value (9-mo fit)
```
| Path | Best when |
|---|---|
| **Integrate** (adopt as-is, connect via APIs) | code is healthy, stack-compatible, time-critical |
| **Strangle** (run theirs, replace incrementally) | strategically core but diverges from our stack |
| **Rebuild** (rewrite on our platform) | small/poor codebase, deep strategic value, scalability dead-end — but risky under 9 months |

## Recommended default: integrate, then strangle
- Ship fast by integrating via an anti-corruption layer (their API behind our facade), proving the product in our platform within the deadline.
- Then strangle: incrementally re-home the highest-value/most-divergent pieces onto our stack, deleting theirs as you go.

## Risks to surface early (to leadership)
- **Licensing landmine** (copyleft dep) — can force a rewrite; check first, it can change the price.
- **Bus factor** — retention packages for key engineers; document tribal knowledge during DD.
- **Compliance/PII gaps** — may block launch in regulated markets.
- **Data migration size** — frequently the real schedule risk.

## Reasoning & tradeoffs
- **Integrate-then-strangle over rebuild under a deadline**: a 9-month full rewrite of an unfamiliar product is the highest-risk option; integrating first delivers value and de-risks by *learning the system in production* before committing to replace it.
- **Surface deal-breakers in week 1**: licensing, PII/compliance, and bus factor can change the valuation or kill the deal — they're cheap to check and catastrophic to discover late.
- **Anti-corruption layer protects your platform**: it prevents the acquired system's assumptions from leaking into your codebase, so the eventual strangle/replacement is contained behind a stable interface.
- **Weighted scoring forces explicitness**: making the rebuild-vs-integrate call via stated criteria (and weights leadership agrees to) turns a religious argument into a transparent, defensible decision — and re-scoring later when you know more is cheap.
- **Tradeoff**: integrate-first carries temporary tech-debt and two stacks running at once; that's the deliberate price of hitting the deadline and reducing rewrite risk.""",
    },
    {
        "prompt": "Build the design for a feature-flag and experimentation system used by 200 engineers across many services, supporting gradual rollouts, targeting rules, and kill switches, with flag evaluation adding under 1ms of latency per request. Address evaluation location, config distribution, consistency, and the audit/safety model.",
        "answer": """<thinking>
Requirements: <1ms eval latency, many services, targeting + gradual rollout + kill switches, used org-wide. The <1ms constraint is the dominant design driver: you cannot do a network call to a flag service per evaluation (that's milliseconds+). So evaluation must be LOCAL — flags are evaluated in-process against a locally-cached ruleset. The flag service's job is to *distribute* config, not to evaluate per request.

Architecture: central control plane (UI, storage, audit) + config distribution (push/poll to SDKs) + local SDK that evaluates in-memory. This is how LaunchDarkly/Unleash work.

Distribution: SDK fetches the full flag ruleset on startup and subscribes to updates (streaming or short poll). Evaluation is a pure function over (flag rules, user context) -> in-memory, sub-microsecond.

Targeting + gradual rollout: rules = list of conditions; percentage rollout via hashing(user_id+flag) % 100 < pct -> deterministic bucketing so a user stays in the same bucket as you ramp.

Kill switch: a flag flip propagates via the streaming channel fast (seconds). Must be near-instant org-wide -> push, not slow poll.

Consistency: eventual — different services may see a flag change a second or two apart. For kill switches that's acceptable if propagation is fast. Strong consistency isn't needed and would cost latency.

Safety/audit: every change is audited (who/when/what), changes can be previewed, and there are guardrails (require approval for prod, automatic rollback hooks). Stale-config fallback: SDK uses last-known-good if the control plane is unreachable.
</thinking>

## Architecture
```
Control plane (UI + store + audit)  --push/stream-->  SDK in each service
                                                        |
                                          evaluate locally, in-memory  (<1ms, no network)
```

### Local evaluation (the <1ms requirement)
```python
def evaluate(flag, ctx) -> bool:
    if flag.killed: return flag.off_value          # kill switch short-circuit
    for rule in flag.targeting_rules:              # in-memory, no I/O
        if rule.matches(ctx):                      # e.g. country, plan, userId list
            return rule.value
    # gradual rollout: deterministic bucketing so users don't flap as % ramps
    bucket = stable_hash(ctx.user_id + flag.key) % 100
    return flag.on_value if bucket < flag.rollout_pct else flag.off_value
```
Evaluation never makes a network call — it reads a locally-cached ruleset, so it's sub-millisecond.

### Config distribution & consistency
- SDK loads the full ruleset on startup, then subscribes to a **streaming update channel**; changes propagate in seconds. Eventual consistency across services is acceptable.
- **Stale-config fallback**: if the control plane is unreachable, the SDK keeps serving last-known-good (and bootstraps from a local file), so a control-plane outage never takes down request handling.

### Kill switch
- A single boolean checked first in evaluation, propagated over the push channel for near-instant org-wide effect.

### Audit & safety model
- Every change is recorded (who, when, before/after) and reversible.
- Prod changes require approval; risky flags support scheduled auto-rollback.
- Changes are previewable ("who would this affect?") before applying.

## Reasoning & tradeoffs
- **Local in-process evaluation is forced by the latency budget**: a per-request call to a flag service is milliseconds; only evaluating against a locally cached ruleset meets <1ms. The control plane therefore *distributes config*, it doesn't *evaluate*.
- **Push/streaming over polling for kill switches**: a kill switch must take effect in seconds org-wide; slow polling would leave a dangerous window where a broken feature stays on.
- **Eventual consistency is acceptable and cheaper**: forcing all services to flip a flag at the exact same instant would require coordination that adds latency and fragility; a few seconds of skew is fine for rollouts and even for kill switches given fast push.
- **Deterministic bucketing** (hash of user+flag) keeps a user in a stable cohort as you ramp 5%->50%, so users don't flicker in and out of the feature — essential for clean experiments and good UX.
- **Stale-config fallback = availability**: the flag system must never be a single point of failure for request handling; serving last-known-good on control-plane outage is the safety net that makes org-wide adoption acceptable.
- **Tradeoff**: SDKs in every language are real maintenance cost, and local caches mean a brief inconsistency window — accepted in exchange for the latency and resilience wins.""",
    },
    {
        "prompt": "A nightly Spark job joining two large datasets (5TB and 200GB) runs for 4 hours and frequently fails with executor OOM and severe data skew on one join key. Diagnose the skew, and give a concrete optimization plan with the techniques in priority order and the expected effect of each.",
        "answer": """<thinking>
Spark join, OOM + skew on one key. Classic. A join shuffles by key; if one key has a huge share of rows (a 'hot key', e.g. null or a default/anonymous user id), all its rows land on one executor -> that task OOMs and straggles, dragging the whole job to 4 hours.

Diagnosis: look at the Spark UI — one task processes vastly more records/shuffle bytes than others (the skew signature). Find the hot key by grouping/counting the join key. Often it's NULL or a sentinel.

Optimization techniques, priority order:
1. Handle the skew directly:
   - Filter/special-case nulls if they shouldn't join.
   - Salting: append a random suffix to the hot key on both sides to spread it across partitions, then aggregate. Standard skew fix.
   - Spark AQE skew join (adaptive) auto-splits skewed partitions — enable it first, it's free.
2. Broadcast join: the 200GB side is too big to broadcast normally, but if a *filtered* version fits, broadcast it to avoid the shuffle entirely. If neither side is small, can't broadcast.
3. Reduce data before join: project only needed columns, push filters down, pre-aggregate. Less shuffle = less OOM.
4. Tune partitions/memory: more shuffle partitions so each is smaller; raise executor memory; spill config. Band-aid, not a fix for skew.

Priority: enable AQE (cheap), then salt the hot key (real fix), then reduce columns/filter, then broadcast if a side fits, then memory tuning last.

Expected: skew handling turns the 4-hour straggler-bound job into something balanced; could drop to <1 hour.
</thinking>

## Diagnosis: confirm the skew
```python
# One task in the Spark UI processes 100x the records of others = skew signature.
# Find the hot key:
df.groupBy("join_key").count().orderBy(desc("count")).show()
# Frequently the hot key is NULL or a sentinel/default id.
```

## Optimization plan (priority order, with expected effect)

### 1. Enable Adaptive Query Execution skew handling (free, first)
```python
spark.conf.set("spark.sql.adaptive.enabled", "true")
spark.conf.set("spark.sql.adaptive.skewJoin.enabled", "true")
```
Effect: Spark auto-splits oversized skewed partitions. Often resolves moderate skew with zero code change.

### 2. Handle the hot key directly (the real fix)
```python
# 2a. If nulls shouldn't join, separate them out:
matched = df.filter(col("join_key").isNotNull()).join(dim, "join_key")

# 2b. Salting: spread the hot key across N partitions on both sides
N = 50
big = big.withColumn("salt", (rand()*N).cast("int"))
small = small.withColumn("salt", explode(array([lit(i) for i in range(N)])))
joined = big.join(small, ["join_key", "salt"])   # hot key now split N ways
```
Effect: the single overloaded task becomes N balanced tasks -> kills the OOM and the straggler.

### 3. Shrink data before the shuffle
```python
big = big.select("join_key", "needed_col")          # project early
big = big.filter(col("date") >= cutoff)             # push filters down
```
Effect: less shuffle volume -> less memory pressure, faster everything.

### 4. Broadcast if a side fits after filtering
```python
joined = big.join(broadcast(small_filtered), "join_key")  # no shuffle at all
```
Effect: eliminates the shuffle entirely *if* the small side fits in executor memory.

### 5. Memory/partition tuning (last, band-aid)
- Raise `spark.sql.shuffle.partitions` so partitions are smaller; bump executor memory. Helps the OOM but doesn't fix skew's root cause.

## Reasoning & tradeoffs
- **Diagnose before tuning**: throwing memory at a skewed job is the common mistake — one task is overloaded, so the cluster sits idle while one executor dies. Bigger executors just delay the OOM. The fix is *balancing the work*, not enlarging the box.
- **AQE first because it's free**: zero code change and often enough; only escalate to salting if it isn't.
- **Salting is the definitive skew fix** but adds complexity and a post-join aggregation; you pay code complexity to convert one giant task into many balanced ones. Use it for the hot key, not blanket.
- **Reduce-before-shuffle is high-leverage**: shuffle is the expensive, OOM-prone phase; projecting columns and pushing filters down means you never move bytes you'll throw away.
- **Broadcast only when a side truly fits**: it removes the shuffle entirely (huge win) but OOMs if the "small" side is too big — hence it comes after filtering shrinks it, and after the skew is already handled.""",
    },
    {
        "prompt": "Design an observability strategy for a 150-service distributed system where on-call engineers currently can't tell, during an incident, which service is the root cause. Cover metrics, logging, tracing, the alerting philosophy, and how the three pillars connect to cut mean-time-to-resolution. Justify what you'd build first.",
        "answer": """<thinking>
The pain is RCA during incidents across 150 services — a symptom shows up at the edge but the cause is buried N hops deep. The three pillars (metrics, logs, traces) only help if they're *connected*; disconnected, an on-call hops between dashboards and grep, which is exactly the slow path described.

The key insight: distributed tracing is what answers "which service is the root cause?" because a trace shows the full request path and where latency/errors originate. So if I must build one thing first for *this specific pain*, it's tracing with trace-id propagation. But tracing alone doesn't tell you *when* something's wrong — metrics/alerts do. And to debug a specific failing request you need logs correlated to the trace.

So the connective tissue is a shared correlation id (trace id) flowing through traces, logs, and exemplar metrics. That's what collapses three tools into one investigation.

Alerting philosophy: alert on symptoms (SLOs/user-facing), not causes. Cause-based alerting on 150 services = alert storms where every downstream service pages during one upstream failure. Symptom-based + a clear severity model reduces noise; the trace then finds the cause.

Build order: 1) trace-id propagation everywhere (the enabler), 2) symptom-based SLO alerts, 3) correlate logs+traces, 4) RED/USE dashboards + service dependency map. Tracing first because it directly attacks the stated RCA problem.
</thinking>

## Strategy: connect the three pillars via a shared trace id

```
request --> [trace id generated, propagated through all 150 hops]
   metrics (RED) ── exemplars ──► traces (full path, finds root cause)
                                      │
                                  logs tagged with trace_id (debug the exact request)
```

### Tracing (build first — it answers "which service?")
```python
# Propagate a trace context on every inbound/outbound call.
def handle(request):
    ctx = extract_trace_context(request.headers)      # or start a new root
    with tracer.start_span("svc.handle", ctx) as span:
        result = downstream.call(inject_trace_context(headers, span))
        return result
# A single trace shows: edge -> A -> B -> C(5s, error) -> the root cause is C.
```

### Metrics — RED per service, alert on symptoms
- RED: Rate, Errors, Duration per service/endpoint. Attach **exemplars** (trace ids of slow/failed requests) so a spiking latency chart links straight to an offending trace.

### Logging — structured, correlated
- Every log line carries `trace_id`. During an incident you pivot from a bad trace to *its* logs instantly, instead of grepping 150 services.

### Alerting philosophy — symptoms, not causes
```
ALERT on: user-facing SLO burn (error rate, p99 latency at the edge)   <-- few, actionable
NOT on:   every internal service's CPU/each dependency failing         <-- alert storm
```
A symptom alert fires once; the trace then localizes which of the 150 services is the cause — no 50-page storm.

## Build order & justification
1. **Trace-id propagation across all services** — directly fixes the stated RCA problem; nothing else localizes a cause across 150 hops.
2. **Symptom/SLO-based alerting** — stops the alert storms so on-call trusts pages.
3. **Log<->trace correlation** — turns "which request?" into "here's exactly what it did."
4. **RED/USE dashboards + auto-generated service dependency map** — situational awareness.

## Reasoning & tradeoffs
- **Tracing first, against the instinct to start with metrics**: the specific pain is *root-cause localization across many hops*, and only traces show the request path. Metrics tell you *that* something's wrong; traces tell you *where* — and "where" is what's broken today.
- **Correlation id is the multiplier**: the pillars are weak in isolation and strong when joined by a shared trace id; that connection is what turns three tools and tab-hopping into a single investigation, which is the actual MTTR win.
- **Symptom-based alerting kills storms**: cause-based alerts on 150 services mean one upstream failure pages dozens of teams — desensitizing on-call. Alert on user-facing SLOs and let the trace find the cause; this is the difference between an actionable page and noise.
- **Tradeoffs**: tracing has overhead and storage cost — mitigate with tail-based sampling (keep the interesting/slow/error traces). Full propagation requires touching every service (instrumentation libraries/auto-instrumentation reduce the toil). Storage cost of high-cardinality logs/traces is real, so sample and retain by value.""",
    },
    {
        "prompt": "We need to deduplicate ~2 billion customer records (name, email, address, phone) that come from 12 source systems with inconsistent formatting and typos, at scale. Design the entity-resolution pipeline: blocking, matching, the scoring model, conflict resolution, and how you avoid the O(n^2) comparison explosion. Justify the accuracy-vs-cost tradeoffs.",
        "answer": """<thinking>
Entity resolution at 2B records. Naive pairwise comparison is O(n^2) = 4×10^18 comparisons — utterly infeasible. The whole game is (1) avoid comparing everything to everything via *blocking*, (2) score candidate pairs with a similarity model, (3) resolve into clusters, (4) merge with conflict resolution.

Blocking: group records into blocks that likely contain matches, only compare within blocks. Keys: e.g. soundex(last_name)+zip prefix, or first 3 chars of email, etc. This reduces comparisons by orders of magnitude. Risk: a too-tight block misses true matches (recall loss); too-loose block = big blocks = back toward O(n^2). Use multiple blocking keys (a record can be in several blocks) to recover recall, then dedup candidate pairs.

Matching/scoring: for candidate pairs, compute field-level similarities (Jaro-Winkler for names/typos, edit distance, phonetic, email normalization, address standardization). Combine via weighted score or a trained classifier (logistic/GBM) on labeled pairs -> probability of match. Threshold into match / non-match / human-review band.

Clustering/transitive closure: matches form a graph; connected components = entities. But transitivity can over-merge (A~B, B~C, but A≠C) — use clustering with constraints / correlation clustering, and a review band.

Conflict resolution / golden record: when merging, pick surviving values by rules (most recent, most complete, most trusted source).

Tradeoffs: blocking trades recall for tractability; threshold trades precision vs recall; human review for the uncertain band trades cost for accuracy.
</thinking>

## Entity-resolution pipeline

```
2B records
  └─ 1. Standardize/normalize (emails lowercased, addresses parsed, phones E.164)
  └─ 2. Blocking (multiple keys) -> candidate pairs   <-- kills O(n^2)
  └─ 3. Pairwise similarity scoring -> match probability
  └─ 4. Clustering / transitive closure -> entities
  └─ 5. Golden-record merge (conflict resolution)
```

### 1–2. Blocking (the O(n^2) killer)
```python
# Only compare records that share a blocking key. Use SEVERAL keys for recall.
def blocking_keys(r):
    return {
        ("name_zip", soundex(r.last_name) + r.zip[:3]),     # phonetic + geo
        ("email3",  r.email_norm[:3]),
        ("phone",   r.phone_e164[-7:]),
    }
# Records sharing any key form candidate pairs; dedup pairs across keys.
```
Effect: 2B² comparisons -> roughly (records × avg block size), many orders of magnitude fewer.

### 3. Pairwise scoring
```python
def score(a, b):
    return model.predict_proba([
        jaro_winkler(a.name, b.name),        # robust to typos/transpositions
        a.email_norm == b.email_norm,
        address_sim(a.addr, b.addr),         # after standardization
        phone_match(a.phone, b.phone),
    ])
# thresholds: >0.9 auto-match | <0.4 non-match | between -> human review band
```

### 4. Clustering with transitivity guard
- Build a graph of matched pairs; take connected components as entities, but apply correlation/constrained clustering so weak transitive chains don't over-merge distinct people.

### 5. Golden record (conflict resolution)
```python
# Survivorship rules per field
golden.email   = most_recent(non_null_emails)
golden.address = most_complete(addresses)
golden.name    = highest_trust_source(names)
```

## Reasoning & tradeoffs
- **Blocking is mandatory, not optional**: at 2B records, O(n²) is physically impossible; blocking is the only thing that makes ER tractable. Everything downstream operates on the tiny candidate set blocking produces.
- **Multi-key blocking trades a little cost for recall**: a single block key will miss matches that disagree on that key (a typo'd last name hides from a name-based block). Several keys recover those at the cost of more candidate pairs — the central recall/cost dial.
- **Threshold bands trade precision vs recall vs human cost**: auto-match high-confidence, auto-reject low, and route the uncertain middle to human review. Widening the review band buys accuracy with labor; narrowing it saves money but risks bad merges. Tune to the business cost of a wrong merge (merging two real customers is often worse than missing a dup).
- **Transitivity is a trap**: naive connected components over-merge (A~B~C chains link unrelated people); constrained clustering guards against creating a "super-entity" that silently corrupts data.
- **Standardization before matching is cheap and high-impact**: normalizing emails/phones/addresses turns many fuzzy comparisons into exact ones, raising both precision and speed before any expensive similarity math runs.""",
    },
    {
        "prompt": "Plan the technical strategy to bring a B2B SaaS product from 'mostly works' to a state where it can pass SOC 2 Type II and serve enterprise customers, in 4 months, without halting the product roadmap. Cover the gap assessment, the controls to implement, sequencing, and how to embed compliance into engineering rather than bolting it on.",
        "answer": """<thinking>
SOC 2 Type II is about operating controls over a *period* (typically the audit observes controls working over months), not a point-in-time snapshot. That changes the strategy: I need controls *in place and generating evidence* early, because the audit period must have continuous evidence. So sequencing front-loads control implementation; the clock on the observation window starts once controls are live.

Trust Service Criteria: Security (required), plus Availability, Confidentiality, Processing Integrity, Privacy as scoped. Most start with Security.

Gap assessment first: where do we stand vs the controls? Access control, change management, encryption, logging/monitoring, incident response, vendor management, backups/DR, onboarding/offboarding.

Key constraint: 4 months + don't halt roadmap. So embed controls into existing engineering workflow (compliance-as-code) rather than as parallel bureaucracy. E.g. change management = enforce via PR reviews + CI gates you already have; access control = SSO + least privilege via IaC; evidence = automated collection, not screenshots.

The "embed not bolt-on" theme: controls that engineers experience as friction will rot. Make the compliant path the easy path (mandatory PR review, automated provisioning, audit logs by default).

Sequencing: gap assessment (weeks 1-2) -> implement controls + start evidence collection (weeks 2-8) -> operate/observe + remediate (weeks 8-16) -> readiness assessment -> Type II observation. Note 4 months is tight for a full Type II period; realistically you get controls live and begin the observation window, possibly a short Type II period or Type I first then Type II.
</thinking>

## Strategy

### Weeks 1–2 — Gap assessment
Map current state vs SOC 2 Security criteria. Likely gaps and the control that closes each:
| Area | Control |
|---|---|
| Access | SSO + MFA, least-privilege RBAC, quarterly access reviews |
| Change mgmt | mandatory PR review + CI gates + linked tickets |
| Encryption | TLS in transit, encryption at rest, key management |
| Monitoring | centralized audit logs, alerting, log retention |
| Incident response | documented IR plan, on-call, postmortems |
| Vendors | vendor risk register, DPAs |
| Resilience | backups + tested restore, DR runbook |
| People | onboarding/offboarding checklists, security training |

### Weeks 2–8 — Implement controls AS CODE (start the evidence clock)
```yaml
# Change management enforced in CI (evidence = the PR/CI record, auto-collected)
required_checks: [code_review_approved, tests_passed, security_scan]
# = your existing workflow, now an auditable control. No new bureaucracy.
```
```hcl
# Access provisioned via IaC = least privilege + an audit trail by construction
resource "iam_role" "engineer" { policy = least_privilege_policy }
```
Crucially, controls go live now so they accumulate the continuous evidence Type II requires.

### Weeks 8–16 — Operate, observe, remediate
- Controls run continuously; automated evidence collection (logs, access reviews, change records) instead of manual screenshots.
- Run a readiness/gap-remediation pass; fix exceptions before the auditor sees them.

### Reality check on the 4-month timeline
- Type II audits an *operating period*. In 4 months you realistically: get all controls live, begin the observation window, and either pursue a **Type I** (point-in-time) now to unblock sales while the Type II observation period accrues, then convert to Type II. Be explicit with leadership about this.

## Embedding compliance into engineering (not bolting on)
- **Compliant path = easy path**: PR review and CI gates you already use *become* the change-management control; provisioning via SSO/IaC *is* the access control. Engineers feel no new process.
- **Evidence is automated**: audit logs, access reviews, and change records are collected by tooling, so quarter-end isn't a screenshot scramble.

## Reasoning & tradeoffs
- **Front-load controls because Type II observes a period**: the binding constraint isn't building controls, it's *operating them long enough to evidence*; every week of delay in turning controls on is a week the audit window can't count. Hence implementation precedes everything.
- **Compliance-as-code over manual process**: bolted-on controls (manual approvals, screenshot evidence) rot under roadmap pressure and fail the "operating effectively" test. Encoding controls into existing CI/IaC makes them continuous and self-evidencing — and doesn't halt the roadmap, satisfying both constraints.
- **Type I-then-Type II is the honest timeline tradeoff**: promising a full Type II in 4 months oversells; a Type I unblocks enterprise sales immediately while the Type II window matures. Surfacing this protects credibility.
- **Scope to Security first**: adding Availability/Confidentiality/etc. multiplies controls; start with the required criterion and expand once the program operates, rather than boiling the ocean under a deadline.""",
    },
    {
        "prompt": "Design a job-scheduling system that runs ~10 million heterogeneous jobs/day (seconds to hours long) across a cluster, with priorities, retries, dependencies (DAGs), and fairness across tenants, while preventing any one tenant from starving others. Detail the scheduler architecture, the queueing/fairness model, and how you handle a worker dying mid-job.",
        "answer": """<thinking>
10M jobs/day ~= 115 jobs/sec average, bursty. Heterogeneous durations (seconds to hours) is a key complication: long jobs can hold resources and starve short ones, and naive FIFO is terrible. Need priorities, retries, DAG dependencies, and multi-tenant fairness.

Architecture: separate the *control plane* (scheduler deciding what runs where) from the *data plane* (workers executing). A durable queue/store holds job state. The scheduler is the brain; it must be HA (leader election) and its state durable so a scheduler crash doesn't lose jobs.

Queueing & fairness: a single global priority queue lets a high-volume tenant dominate. Need fair scheduling — e.g. weighted fair queuing / DRF (dominant resource fairness) across tenants, or per-tenant queues with a fair dispatcher. To prevent starvation of long-waiting low-priority jobs, add aging (priority increases with wait time).

Heterogeneous durations: separate pools or admission control so a flood of hour-long jobs doesn't block seconds-long ones; reserve capacity for short jobs, or use multiple queues by expected duration.

DAGs: a job with dependencies is only schedulable when parents complete. Maintain a dependency graph; on completion, enqueue now-ready children. Detect cycles at submit.

Worker death: jobs are leased with a heartbeat/lease timeout. If a worker stops heartbeating, the lease expires and the job is rescheduled. Jobs must be idempotent or checkpointed so re-execution is safe (at-least-once execution).

Retries: bounded retries with backoff; distinguish transient vs permanent failures; dead-letter after max retries.
</thinking>

## Architecture (control plane / data plane split)
```
Submit -> [Durable job store + DAG] -> Scheduler (HA, leader-elected)
                                           | dispatch (lease)
                                           v
                                     Worker pool (heartbeat)
```

### Fairness & queueing (prevent tenant starvation)
```python
# Per-tenant queues + weighted-fair dispatch (DRF-style), with aging.
def pick_next():
    candidates = [q.peek() for q in tenant_queues if q.nonempty()]
    # fair share: pick the tenant furthest BELOW its fair resource share
    tenant = min(candidates, key=lambda j: usage[j.tenant] / weight[j.tenant])
    job = tenant_queues[tenant].pop()
    job.effective_priority += aging(job.wait_time)   # long-waiters rise -> no starvation
    return job
```
- **Heterogeneous durations**: reserve a pool (or a queue) for short jobs so a flood of hour-long jobs can't block them; admission control caps concurrent long jobs per tenant.

### DAG dependencies
```python
def on_complete(job):
    for child in dag.children(job):
        child.pending_deps -= 1
        if child.pending_deps == 0:
            enqueue(child)            # becomes schedulable only when all parents done
# cycles rejected at submit time.
```

### Worker death (lease + heartbeat)
```python
# Dispatch grants a lease; worker heartbeats. No heartbeat -> lease expires -> reschedule.
def reaper():
    for job in running:
        if now - job.last_heartbeat > LEASE_TIMEOUT:
            job.attempts += 1
            requeue(job) if job.attempts < MAX else dead_letter(job)
```
- Execution is **at-least-once**, so jobs must be idempotent or checkpoint progress to be safely re-run.

### Retries
- Bounded retries with exponential backoff; transient vs permanent failure classification; dead-letter queue after max attempts for inspection.

## Reasoning & tradeoffs
- **Fair scheduling over global priority FIFO**: a single priority queue lets the highest-volume (or highest-priority) tenant monopolize the cluster. Per-tenant fair share (weighted/DRF) bounds each tenant's footprint — the core anti-starvation mechanism — while weights still let you favor premium tenants.
- **Aging prevents the *other* starvation**: pure fairness/priority can leave low-priority jobs waiting forever; raising effective priority with wait time guarantees eventual execution. The tradeoff is occasionally running a low-value job ahead of a fresh high-value one.
- **Duration-aware pooling**: mixing seconds-long and hours-long jobs in one queue causes head-of-line blocking; isolating short jobs (reserved capacity) keeps latency-sensitive work responsive at the cost of some utilization.
- **Lease/heartbeat over RPC liveness**: a worker can hang or partition without "crashing"; lease expiry is the robust way to reclaim its job. The price is at-least-once execution, which pushes an idempotency/checkpointing requirement onto jobs — a deliberate, well-understood tradeoff.
- **Durable scheduler state + leader election**: the scheduler is the brain; losing its state loses jobs. HA via leader election plus a durable store keeps a control-plane crash from becoming data loss, at the cost of more operational complexity.""",
    },
    {
        "prompt": "Our search feature returns results in 800ms p95 and users complain it feels slow. We use Elasticsearch over 50M documents. Walk through a layered diagnosis and optimization plan covering query structure, index design, caching, and the relevance-vs-latency tradeoff. Show what a problematic query looks like and how you'd fix it.",
        "answer": """<thinking>
Search latency over 50M docs. 800ms p95 is slow for ES which should do most queries in tens of ms. Need layered diagnosis: is it the query (expensive clauses), the index (mapping/sharding), the cluster (resources, GC), or downstream (rendering/over-fetching)?

Common ES latency killers:
1. Expensive query constructs: wildcard/leading-wildcard, regex, script queries, deep pagination (from+size deep), large aggregations, fielddata on text fields, sorting on un-optimized fields.
2. Over-fetching: returning huge _source, requesting more hits than rendered, no source filtering.
3. Index design: wrong number of shards (too many small shards = overhead; too few = hot shards), missing keyword sub-fields, analyzing fields that should be keyword, not using doc_values.
4. No caching: same filters re-run; ES filter cache helps if you use filter context (not query context) for non-scoring constraints.
5. Relevance scoring cost: complex function_score, many should clauses.

Diagnosis: use the Profile API to see where time goes per query component. Slow log to find offending queries.

The classic fix: move non-scoring constraints into filter context (cacheable, no scoring), avoid leading wildcards (use ngram/edge-ngram index instead), replace deep from/size pagination with search_after, and source-filter to return only needed fields.

Relevance vs latency: heavy scoring (function_score, script_score) improves relevance but costs latency; can precompute signals at index time or use a cheaper first pass + rerank top-k.
</thinking>

## Layered diagnosis & fixes

### Step 0 — Profile to localize
Use the ES **Profile API** + slow log to see which query component eats the time (query vs aggregation vs fetch).

### Layer 1 — Query structure (usual culprit)
Problematic query:
```json
{ "query": { "bool": { "must": [
    { "wildcard": { "title": "*phone*" }},          // leading wildcard = scans everything
    { "match": { "body": "case" }}
]}},
"from": 10000, "size": 10 }                           // deep pagination = very expensive
```
Fixed:
```json
{ "query": { "bool": {
    "must":   [{ "match": { "body": "case" }}],       // scoring clause
    "filter": [{ "term": { "category": "phones" }}]    // FILTER context = cacheable, no scoring
}},
"search_after": [last_sort_value], "size": 10,        // search_after, not deep from/size
"_source": ["title","price"] }                         // return only needed fields
```
- Replace leading-wildcard contains-search with an **edge/n-gram analyzer at index time**, so prefix/substring matching is a cheap term lookup instead of a full scan.

### Layer 2 — Index design
- Right-size shards (avoid hundreds of tiny shards or one giant hot shard for 50M docs).
- Use `keyword` for exact-match/sort/aggregate fields; rely on `doc_values` (default) for sorting/aggs; don't enable `fielddata` on text.

### Layer 3 — Caching
- Put non-scoring constraints in **filter context** so ES caches the filter bitset; repeated filtered searches then skip re-evaluation.

### Layer 4 — Relevance vs latency
- If `function_score`/`script_score` is the cost: precompute ranking signals (popularity, recency) **at index time**, do a cheap first-pass retrieval, then rerank only the top-K. Big latency win for a small relevance cost.

## Reasoning & tradeoffs
- **Profile before optimizing**: ES latency can live in query, fetch, or aggregation phases that need different fixes; the Profile API tells you which, avoiding wasted effort.
- **Filter vs query context is the highest-leverage cheap fix**: non-scoring constraints in filter context are both *cacheable* and skip scoring — often a large win with a one-line change. Leaving them in query context pays scoring cost on every constraint for nothing.
- **Leading wildcards / deep pagination are latency cliffs**: a `*term*` wildcard or `from: 10000` forces near-full scans; moving the cost to index time (n-grams) or to cursor pagination (search_after) removes the cliff. The tradeoff: n-grams enlarge the index (storage for speed).
- **Relevance-vs-latency is a genuine dial**: richer scoring improves results but costs ms per query. Two-phase retrieve-then-rerank and index-time signal precomputation buy most of the relevance at a fraction of the latency — accept slightly less flexible scoring for big speedups.
- **Return less**: source filtering and not over-fetching hits cut serialization and network time, which the user feels directly even when ES "query time" looks fine.""",
    },
    {
        "prompt": "We must process and store user-uploaded video (up to 4GB files), transcode to multiple resolutions, generate thumbnails, run content moderation, and serve globally — at a scale of 100k uploads/day. Design the end-to-end pipeline, the storage and CDN strategy, how to make the long-running transcode reliable and observable, and where you'd spend money vs save it.",
        "answer": """<thinking>
Media pipeline. 100k uploads/day ~= 1.2 uploads/sec avg, bursty; files up to 4GB. Key challenges: large uploads (can't go through the app server; use direct-to-storage multipart upload with presigned URLs), long-running transcode (minutes; must be async, resumable, observable), fan-out of work (transcode N resolutions + thumbnails + moderation in parallel), and global serving (CDN).

Architecture stages:
1. Upload: client -> presigned multipart upload directly to object storage (S3). App never proxies 4GB. Resumable.
2. Trigger: object-created event -> queue -> orchestrator kicks off a processing workflow (DAG: transcode renditions, thumbnails, moderation can run in parallel).
3. Transcode: async workers (autoscaled, possibly spot/GPU). Long jobs -> need idempotency, retries, progress tracking. Use a workflow engine (Step Functions/Temporal) for durable orchestration + visibility.
4. Moderation: run an ML/3rd-party check; gate publishing on it.
5. Storage: originals in cheap cold storage after processing; renditions in standard; serve via CDN with origin = object storage.
6. Serve: CDN globally, signed URLs, adaptive bitrate (HLS/DASH) so clients pick resolution.

Reliability/observability for transcode: durable workflow with per-step retries, idempotent steps, progress events, dead-letter for poison files, and metrics (queue depth, transcode duration, failure rate).

Cost: spend on CDN (user-facing, dominates egress cost but essential) and reliable transcode; save by using spot instances for transcode (interruptible, retry-safe), tiering originals to cold storage, and only generating renditions actually requested (lazy/on-demand transcode for rare resolutions).
</thinking>

## End-to-end pipeline
```
Client --presigned multipart upload--> Object storage (originals)
   └─ object-created event -> Queue -> Workflow orchestrator (durable)
                                          ├─ transcode 1080p/720p/480p (parallel)
                                          ├─ generate thumbnails
                                          └─ content moderation (gate publish)
   renditions -> Object storage -> CDN (global) -> users (HLS/DASH adaptive)
```

### Upload (handle 4GB without proxying)
- Client uploads **directly to object storage via presigned multipart URLs** (resumable, parallel parts). The app server only issues URLs and receives the completion event — it never streams 4GB.

### Processing (long-running, reliable, observable)
```python
# Durable workflow (Temporal/Step Functions): each step retried, idempotent, tracked.
@workflow
def process_video(object_key):
    moderation = start(moderate, object_key)            # parallel
    renditions = [start(transcode, object_key, r) for r in ["1080p","720p","480p"]]
    thumbs     = start(thumbnails, object_key)
    wait_all(renditions + [thumbs])
    if wait(moderation).blocked:
        quarantine(object_key); return
    publish(object_key)        # only after moderation passes + renditions ready
```
- Workers **autoscale on queue depth**; idempotent steps + retries handle transient failures; **poison files dead-letter** instead of looping forever.
- Observability: queue depth, per-step duration, failure rate, and per-video status (the workflow engine gives end-to-end visibility for support).

### Storage & serving
- Originals -> tier to **cold storage** after processing (rarely re-read).
- Renditions -> standard storage, fronted by **CDN**; serve adaptive bitrate (HLS/DASH) with **signed URLs**.

## Where to spend vs save
| Spend | Save |
|---|---|
| CDN (global egress) — it's the UX | Transcode on **spot/preemptible** (retry-safe) |
| Durable orchestration + observability | Tier originals to **cold storage** |
| Moderation (compliance/legal) | **Lazy-transcode** rare resolutions on demand |

## Reasoning & tradeoffs
- **Direct-to-storage upload is non-negotiable at 4GB**: proxying multi-GB files through app servers wastes bandwidth, ties up request threads, and isn't resumable. Presigned multipart pushes the bytes straight to storage and survives flaky mobile networks.
- **Durable workflow over ad-hoc queue chaining**: transcode is minutes-long with multiple parallel steps and a moderation gate; a workflow engine gives retries, idempotency, and *per-video visibility* (crucial for support and for not silently dropping a user's upload). The cost is another platform component.
- **Spot for transcode is the big cost lever**: transcode is interruptible and already retry-safe, so it's the perfect spot workload — large savings with no UX risk. CDN, by contrast, is where you *should* spend, because egress and global latency directly shape perceived quality.
- **Lazy-transcode rare renditions**: pre-generating every resolution for every video wastes compute/storage on resolutions few watch; generate common ones eagerly and rare ones on first request — trading a one-time latency for ongoing savings.
- **Moderation gates publish**: making publishing depend on moderation prevents harmful content going live, accepting a small processing delay for legal/brand safety.""",
    },
]


if __name__ == "__main__":
    print(f"Loaded {len(examples)} agent/complex-reasoning examples.")
    # Sanity checks: structure + required content in every example.
    for i, ex in enumerate(examples):
        assert set(ex) == {"prompt", "answer"}, f"example {i} wrong keys"
        assert ex["prompt"].strip(), f"example {i} empty prompt"
        a = ex["answer"]
        assert "<thinking>" in a and "</thinking>" in a, f"example {i} missing thinking"
        assert "```" in a, f"example {i} missing code/structured block"
        assert "tradeoff" in a.lower() or "Reasoning" in a, f"example {i} missing reasoning"
    print("All examples passed structural validation.")
