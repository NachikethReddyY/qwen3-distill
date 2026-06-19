#!/usr/bin/env python3
"""
High-quality tools & integration training examples for model distillation.

Each example pairs a specific, realistic prompt about tools, APIs, CLIs,
configuration, and integration with a detailed answer that includes a
<thinking> section, complete practical examples, and notes on best
practices and common pitfalls.

To merge into the training set, write each dict as a chat-format JSONL row:

    import json
    from pathlib import Path
    from tools_integration_examples import examples, SYSTEM_PROMPT

    out = Path("data/raw/tools_examples.jsonl")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as f:
        for ex in examples:
            f.write(json.dumps({"messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["answer"]},
            ]}) + "\n")
"""

SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of React, JavaScript, Python, animations, and data visualization.
Think through problems step by step inside <thinking>...</thinking> tags, explaining your reasoning and approach.
Then give your final answer with clean, well-structured code and detailed explanations.
Be thorough, correct, and explain complex concepts clearly."""

examples = [
    # 1. Docker multi-stage build
    {
        "prompt": "Write a production-ready multi-stage Dockerfile for a Node.js (TypeScript) service that compiles in a build stage and runs as a non-root user in a minimal final image. Explain how to keep the image small and secure.",
        "answer": """<thinking>
A good production Node image needs: (1) a build stage with full toolchain to install dev deps and compile TS, (2) a runtime stage that only carries production deps and compiled JS, (3) a non-root user, (4) deterministic installs with `npm ci`, and (5) proper layer caching by copying package files before source. I'll also add a healthcheck and use `node` directly (not `npm start`) so signals propagate for graceful shutdown.
</thinking>

```dockerfile
# ---- Build stage ----
FROM node:20-bookworm-slim AS build
WORKDIR /app

# Copy manifests first for better layer caching
COPY package.json package-lock.json ./
RUN npm ci

COPY tsconfig.json ./
COPY src ./src
RUN npm run build          # emits to ./dist

# Prune to production deps only
RUN npm ci --omit=dev

# ---- Runtime stage ----
FROM node:20-bookworm-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app

# Run as the built-in non-root `node` user
COPY --from=build --chown=node:node /app/node_modules ./node_modules
COPY --from=build --chown=node:node /app/dist ./dist
COPY --chown=node:node package.json ./

USER node
EXPOSE 3000

HEALTHCHECK --interval=30s --timeout=3s --retries=3 \\
  CMD node -e "require('http').get('http://127.0.0.1:3000/health',r=>process.exit(r.statusCode===200?0:1)).on('error',()=>process.exit(1))"

# exec form -> PID 1 is node, so SIGTERM reaches the app
CMD ["node", "dist/index.js"]
```

**Setup and Explanation**

- **Layer caching:** copy `package*.json` and run `npm ci` *before* copying source. Source changes won't invalidate the dependency layer.
- **`npm ci` vs `npm install`:** `ci` is deterministic, respects the lockfile exactly, and fails if `package.json` and the lock are out of sync.
- **Small image:** the runtime stage never sees dev dependencies, the TS compiler, or the `src/` tree. Use `*-slim` (or distroless) base images.
- **Non-root:** the `node` image ships a `node` user; running as it limits blast radius if the process is compromised.
- **Graceful shutdown:** exec-form `CMD` makes `node` PID 1 so orchestrators' SIGTERM reaches your app for clean connection draining.

**Common pitfalls**
- Using `CMD npm start` (shell form) means npm is PID 1 and swallows SIGTERM.
- Forgetting a `.dockerignore` with `node_modules`, `.git`, `dist` bloats the build context.
- Baking secrets via `ENV` — they persist in image layers; use build secrets or runtime env injection instead.""",
    },

    # 2. docker-compose local stack
    {
        "prompt": "Create a docker-compose.yml for local development with a Postgres database, a Redis cache, and a web app, including healthchecks, dependency ordering, and persistent volumes.",
        "answer": """<thinking>
For a reliable local stack I need: named volumes so DB data survives `down`, healthchecks on Postgres and Redis, and `depends_on` with `condition: service_healthy` so the app waits for ready dependencies rather than just started containers. Env vars should come from a `.env` file. I'll expose only what's useful locally.
</thinking>

```yaml
services:
  db:
    image: postgres:16
    environment:
      POSTGRES_USER: ${POSTGRES_USER:-app}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-secret}
      POSTGRES_DB: ${POSTGRES_DB:-app}
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER:-app}"]
      interval: 5s
      timeout: 3s
      retries: 5

  cache:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes"]
    ports:
      - "6379:6379"
    volumes:
      - redisdata:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 3s
      retries: 5

  web:
    build: .
    environment:
      DATABASE_URL: postgres://${POSTGRES_USER:-app}:${POSTGRES_PASSWORD:-secret}@db:5432/${POSTGRES_DB:-app}
      REDIS_URL: redis://cache:6379
    ports:
      - "3000:3000"
    depends_on:
      db:
        condition: service_healthy
      cache:
        condition: service_healthy

volumes:
  pgdata:
  redisdata:
```

**Setup and Explanation**

- Put defaults in `.env`; Compose auto-loads it. `${VAR:-default}` provides fallbacks.
- Inside the network, services reach each other by name (`db`, `cache`) — never `localhost`.
- `condition: service_healthy` waits for the healthcheck to pass, avoiding "connection refused" races at startup.
- Named volumes (`pgdata`, `redisdata`) persist across `docker compose down`; use `down -v` to wipe.

**Common pitfalls**
- `depends_on` *without* a condition only waits for the container to *start*, not to be *ready*.
- Mapping host ports for every service causes collisions; only publish what you actually hit from the host.
- Hardcoding `localhost` in `DATABASE_URL` — from `web`'s perspective the DB host is `db`.""",
    },

    # 3. REST API design with pagination
    {
        "prompt": "Design a paginated REST endpoint `GET /api/orders` using cursor-based pagination. Show the request/response contract and a Node/Express implementation. Why cursor over offset?",
        "answer": """<thinking>
Offset pagination (`LIMIT/OFFSET`) is simple but degrades on large tables and yields duplicates/skips when rows are inserted between page loads. Cursor pagination encodes the position of the last seen row (e.g. created_at + id) so the next query is a keyset lookup — stable and O(log n) with an index. I'll return an opaque base64 cursor and a `nextCursor`.
</thinking>

```js
// GET /api/orders?limit=20&cursor=<opaque>
import express from "express";
const router = express.Router();

const encode = (o) => Buffer.from(JSON.stringify(o)).toString("base64url");
const decode = (c) => JSON.parse(Buffer.from(c, "base64url").toString());

router.get("/api/orders", async (req, res) => {
  const limit = Math.min(parseInt(req.query.limit ?? "20", 10), 100);
  const cursor = req.query.cursor ? decode(req.query.cursor) : null;

  // Keyset predicate: (created_at, id) strictly after the cursor
  const rows = await db.query(
    `SELECT id, created_at, total, status
       FROM orders
      WHERE ($1::timestamptz IS NULL)
         OR (created_at, id) < ($1, $2)
      ORDER BY created_at DESC, id DESC
      LIMIT $3`,
    [cursor?.created_at ?? null, cursor?.id ?? null, limit + 1]
  );

  const hasMore = rows.length > limit;
  const page = hasMore ? rows.slice(0, limit) : rows;
  const last = page[page.length - 1];

  res.json({
    data: page,
    pageInfo: {
      hasMore,
      nextCursor: hasMore ? encode({ created_at: last.created_at, id: last.id }) : null,
    },
  });
});
```

**Response contract**
```json
{ "data": [ { "id": "...", "total": 4200, "status": "paid" } ],
  "pageInfo": { "hasMore": true, "nextCursor": "eyJ..." } }
```

**Setup and Explanation**
- Fetch `limit + 1` rows to detect `hasMore` without a separate `COUNT`.
- The composite key `(created_at, id)` breaks ties deterministically; index it: `CREATE INDEX ON orders (created_at DESC, id DESC)`.
- The cursor is **opaque** — clients must treat it as a token, never parse it.

**Why cursor over offset**
- Stable under concurrent inserts (no duplicate/skipped rows).
- Constant performance: `OFFSET 1000000` scans and discards a million rows; keyset jumps via the index.

**Pitfalls:** clamp `limit` to a max; reject malformed cursors with `400`; cursors are tied to the sort order — changing `ORDER BY` invalidates them.""",
    },

    # 4. GraphQL schema + resolver, N+1
    {
        "prompt": "I have a GraphQL API where querying authors with their posts causes an N+1 query problem. Show the schema, the naive resolver, and how to fix it with DataLoader.",
        "answer": """<thinking>
The N+1 happens because the `posts` field resolver runs once per author, each firing its own DB query. DataLoader batches the keys collected within a single tick and issues one `WHERE author_id IN (...)` query, then maps results back per key. I'll create a per-request loader to avoid cross-request cache leakage.
</thinking>

```graphql
type Author { id: ID!  name: String!  posts: [Post!]! }
type Post   { id: ID!  title: String!  authorId: ID! }
type Query  { authors: [Author!]! }
```

```js
import DataLoader from "dataloader";

// Batch fn: receives [authorId, ...], returns posts[] aligned to that order
function makePostLoader(db) {
  return new DataLoader(async (authorIds) => {
    const rows = await db.query(
      `SELECT * FROM posts WHERE author_id = ANY($1)`,
      [authorIds]
    );
    const byAuthor = new Map(authorIds.map((id) => [id, []]));
    for (const r of rows) byAuthor.get(r.author_id)?.push(r);
    return authorIds.map((id) => byAuthor.get(id)); // order MUST match input
  });
}

// One loader per request — created in context
const context = ({ req }) => ({ db, loaders: { posts: makePostLoader(db) } });

const resolvers = {
  Query: { authors: (_p, _a, { db }) => db.query("SELECT * FROM authors") },
  Author: {
    posts: (author, _a, { loaders }) => loaders.posts.load(author.id),
  },
};
```

**Setup and Explanation**
- Without DataLoader, 1 query for authors + N queries for posts = N+1. With it, you get exactly 2 queries.
- The batch function **must** return an array the same length and order as the input keys; missing keys map to `[]` (or `null`), not omitted.
- Create loaders **per request** in `context` so DataLoader's memoization cache doesn't leak data between users.

**Common pitfalls**
- Returning rows in DB order instead of key order silently mismatches authors and posts.
- Reusing a single global loader caches stale data across requests/users.
- Forgetting `ANY($1)` / `IN` and falling back to per-id queries inside the batch fn defeats the purpose.""",
    },

    # 5. Build a CLI with argparse/subcommands
    {
        "prompt": "Build a Python CLI tool called `deployctl` with subcommands `deploy` and `rollback`, proper argument parsing, exit codes, and a `--dry-run` flag. Make it scriptable and testable.",
        "answer": """<thinking>
A well-behaved CLI needs: subcommands via argparse subparsers, a `func` callback per subcommand, meaningful exit codes (0 ok, non-zero on failure), `--dry-run` to preview, and a `main(argv)` that returns an int so it's unit-testable without spawning a process. I'll keep side effects out of arg parsing.
</thinking>

```python
import argparse, sys

def cmd_deploy(args) -> int:
    if args.dry_run:
        print(f"[dry-run] would deploy {args.service} to {args.env}")
        return 0
    print(f"Deploying {args.service} -> {args.env}")
    # ... real work; return 1 on failure
    return 0

def cmd_rollback(args) -> int:
    print(f"Rolling back {args.service} in {args.env} to {args.to}")
    return 0

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="deployctl")
    p.add_argument("--env", default="staging", choices=["staging", "prod"])
    p.add_argument("--dry-run", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    d = sub.add_parser("deploy", help="Deploy a service")
    d.add_argument("service")
    d.set_defaults(func=cmd_deploy)

    r = sub.add_parser("rollback", help="Roll back a service")
    r.add_argument("service")
    r.add_argument("--to", required=True, help="target revision")
    r.set_defaults(func=cmd_rollback)
    return p

def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
```

**Usage**
```bash
deployctl --env prod deploy api
deployctl deploy api --dry-run
deployctl rollback api --to v1.4.2
```

**Setup and Explanation**
- `set_defaults(func=...)` dispatches to the right handler — no big `if/elif` chain.
- `main(argv=None)` lets tests call `main(["deploy", "api", "--dry-run"])` and assert the return code.
- `sys.exit(main())` propagates the int as the process exit code; CI and shell `&&` chains rely on it.

**Pitfalls:** `subparsers(required=True)` is needed (older Python doesn't require a subcommand by default); don't `print` errors to stdout — use stderr and reserve stdout for machine-parseable output; expose `--dry-run` consistently.""",
    },

    # 6. PostgreSQL transaction + indexing
    {
        "prompt": "Show how to safely transfer money between two accounts in PostgreSQL avoiding race conditions, and explain locking, isolation levels, and the right index.",
        "answer": """<thinking>
A transfer is the classic concurrency hazard: two transactions reading then writing the same balance can lose updates. I'll wrap it in a transaction and use `SELECT ... FOR UPDATE` to take row locks, locking rows in a consistent order (by id) to prevent deadlocks. I'll discuss isolation levels and the supporting index.
</thinking>

```sql
BEGIN;

-- Lock both rows; ORDER BY id ensures consistent lock ordering (deadlock-safe)
SELECT id, balance FROM accounts
WHERE id IN (101, 202)
ORDER BY id
FOR UPDATE;

-- Guard against overdraft inside the locked section
UPDATE accounts SET balance = balance - 5000
WHERE id = 101 AND balance >= 5000;

-- Verify exactly one row updated; if 0, ROLLBACK in app code
UPDATE accounts SET balance = balance + 5000 WHERE id = 202;

INSERT INTO ledger(from_id, to_id, amount) VALUES (101, 202, 5000);

COMMIT;
```

**Setup and Explanation**
- `FOR UPDATE` takes a row lock so a concurrent transfer touching the same accounts blocks until this commits — no lost updates.
- Locking in a **deterministic order** (`ORDER BY id`) prevents the A-locks-101-waits-202 / B-locks-202-waits-101 deadlock.
- The `balance >= 5000` predicate makes the debit conditional; check the affected row count in the app and roll back if zero (insufficient funds).
- `accounts.id` is the primary key, so it's already indexed for the lookup. The `ledger` table benefits from indexes on `from_id`/`to_id` for reporting.

**Isolation levels**
- Default `READ COMMITTED` + `FOR UPDATE` is sufficient here.
- `REPEATABLE READ`/`SERIALIZABLE` add snapshot guarantees but can throw serialization failures (`40001`) — wrap callers in a retry loop.

**Pitfalls:** never read balance into the app, compute, then write back unconditionally (lost update). Avoid long transactions holding locks. Always log to a `ledger` for auditability.""",
    },

    # 7. GitHub Actions CI/CD
    {
        "prompt": "Write a GitHub Actions workflow that runs lint + tests on every PR, caches dependencies, runs jobs in a matrix across Node versions, and deploys to production only on pushes to main.",
        "answer": """<thinking>
I want two logical concerns: a `test` job that runs on PRs and pushes across a Node matrix with dependency caching, and a `deploy` job gated to `main` pushes that `needs` the test job and uses a protected environment. `actions/setup-node` has built-in npm caching. I'll scope deploy with an `if` on ref and use environments for approval/secrets.
</thinking>

```yaml
name: ci
on:
  pull_request:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        node: [18, 20, 22]
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with:
          node-version: ${{ matrix.node }}
          cache: npm                 # caches ~/.npm keyed on package-lock.json
      - run: npm ci
      - run: npm run lint
      - run: npm test -- --ci

  deploy:
    needs: test                       # only after all matrix tests pass
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: production           # enables required reviewers + scoped secrets
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-node@v4
        with: { node-version: 20, cache: npm }
      - run: npm ci --omit=dev
      - run: npm run deploy
        env:
          DEPLOY_TOKEN: ${{ secrets.DEPLOY_TOKEN }}
```

**Setup and Explanation**
- `cache: npm` in `setup-node` is simpler than a manual `actions/cache` step and keys on the lockfile hash automatically.
- The matrix runs lint+test on 18/20/22 in parallel; a failure in any cancels the deploy via `needs: test`.
- `if: github.ref == 'refs/heads/main' && github.event_name == 'push'` ensures deploy never runs on PRs (where it would also lack prod secrets).
- `environment: production` lets you require manual approval and scope `DEPLOY_TOKEN` to prod only.

**Pitfalls:** without `needs`, deploy could run before tests; secrets are not available to workflows triggered by PRs from forks (by design); use `concurrency` groups to cancel superseded runs and avoid overlapping deploys.""",
    },

    # 8. Prometheus metrics instrumentation
    {
        "prompt": "Instrument an Express service with Prometheus metrics: request count, latency histogram, and in-flight gauge, exposed at /metrics. Then write a PromQL query for p95 latency.",
        "answer": """<thinking>
I need the `prom-client` library: a default registry with default process metrics, a Counter for total requests labeled by method/route/status, a Histogram for latency with sensible buckets, and a Gauge for in-flight requests. Middleware records these. Critically, labels must use the *route pattern* (`/users/:id`), not the raw URL, to avoid cardinality explosion. p95 over a histogram uses `histogram_quantile`.
</thinking>

```js
import express from "express";
import client from "prom-client";

const register = new client.Registry();
client.collectDefaultMetrics({ register });

const httpRequests = new client.Counter({
  name: "http_requests_total",
  help: "Total HTTP requests",
  labelNames: ["method", "route", "status"],
  registers: [register],
});
const httpDuration = new client.Histogram({
  name: "http_request_duration_seconds",
  help: "Request latency",
  labelNames: ["method", "route", "status"],
  buckets: [0.01, 0.05, 0.1, 0.3, 0.5, 1, 2, 5],
  registers: [register],
});
const inFlight = new client.Gauge({
  name: "http_requests_in_flight",
  help: "In-flight requests",
  registers: [register],
});

const app = express();
app.use((req, res, next) => {
  inFlight.inc();
  const end = httpDuration.startTimer();
  res.on("finish", () => {
    const route = req.route?.path ?? req.path; // pattern, not raw URL
    const labels = { method: req.method, route, status: res.statusCode };
    httpRequests.inc(labels);
    end(labels);
    inFlight.dec();
  });
  next();
});

app.get("/metrics", async (_req, res) => {
  res.set("Content-Type", register.contentType);
  res.end(await register.metrics());
});
```

**PromQL: p95 latency over 5m, per route**
```promql
histogram_quantile(
  0.95,
  sum(rate(http_request_duration_seconds_bucket[5m])) by (le, route)
)
```

**Setup and Explanation**
- `collectDefaultMetrics` adds CPU, memory, event-loop lag — free and valuable.
- Histograms expose `_bucket`, `_sum`, `_count`; `histogram_quantile` interpolates the quantile from buckets. Always keep `le` in the `by()` clause.
- `rate(...[5m])` turns the monotonically increasing counters into per-second rates before quantiling.

**Pitfalls:** **label cardinality** is the #1 trap — never label with user IDs, raw URLs, or unbounded values; use route patterns. Choose buckets that bracket your real latencies, or quantiles will be inaccurate.""",
    },

    # 9. Structured logging
    {
        "prompt": "Set up structured JSON logging in a Node service with correlation IDs that flow through async requests, using pino. Show how to propagate a request ID and what to log (and not log).",
        "answer": """<thinking>
Structured logs are JSON so they're machine-parseable in ELK/Loki. I'll use pino for speed, an HTTP middleware to attach a request-scoped child logger carrying a correlation/request ID, and AsyncLocalStorage so any code in the call chain can grab the current logger/ID without threading it through every function. I'll note redaction of secrets.
</thinking>

```js
import pino from "pino";
import { AsyncLocalStorage } from "node:async_hooks";
import { randomUUID } from "node:crypto";

const als = new AsyncLocalStorage();
export const logger = pino({
  level: process.env.LOG_LEVEL ?? "info",
  redact: ["req.headers.authorization", "password", "*.token"],
});

export const log = () => als.getStore()?.logger ?? logger;

// Express middleware: bind a child logger with a request id for this async context
export function requestLogger(req, res, next) {
  const reqId = req.headers["x-request-id"] ?? randomUUID();
  res.setHeader("x-request-id", reqId);
  const child = logger.child({ reqId, method: req.method, path: req.path });
  als.run({ logger: child }, () => {
    const start = Date.now();
    res.on("finish", () =>
      child.info({ status: res.statusCode, ms: Date.now() - start }, "request")
    );
    next();
  });
}
```

```js
// Anywhere deeper in the stack — no plumbing needed:
async function chargeCard(amount) {
  log().info({ amount }, "charging card");   // automatically carries reqId
}
```

**Setup and Explanation**
- `AsyncLocalStorage` keeps the per-request logger available across `await` boundaries, so the `reqId` appears on every log line for that request without passing it manually.
- Honor an inbound `x-request-id` (set by upstream/gateway) for end-to-end tracing; otherwise generate one and echo it back in the response header.
- `redact` strips secrets before they hit disk.

**What to log:** request lifecycle, status, duration, error stacks, business events. **What NOT to log:** passwords, tokens, full card numbers, PII. Log levels: `info` for lifecycle, `warn` for recoverable, `error` for failures with `err` field.

**Pitfalls:** plain `console.log` strings aren't queryable; logging full request bodies leaks PII; high-volume `debug` in prod is expensive — make level env-driven.""",
    },

    # 10. OAuth2 / API auth
    {
        "prompt": "Explain and implement the OAuth2 client credentials flow to call a third-party API from a backend service in Python, including token caching and refresh-on-expiry.",
        "answer": """<thinking>
Client credentials is the machine-to-machine flow: the service exchanges its client_id/secret for an access token, no user involved. The key engineering concern is caching the token and refreshing it shortly before expiry rather than on every call. I'll build a small token manager with an expiry buffer and thread-safety, plus a requests session that auto-attaches the bearer.
</thinking>

```python
import time, threading, requests

class TokenManager:
    def __init__(self, token_url, client_id, client_secret, scope=None, skew=30):
        self.token_url, self.client_id, self.client_secret = token_url, client_id, client_secret
        self.scope, self.skew = scope, skew
        self._token, self._exp, self._lock = None, 0.0, threading.Lock()

    def _fetch(self):
        data = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        if self.scope:
            data["scope"] = self.scope
        r = requests.post(self.token_url, data=data, timeout=10)
        r.raise_for_status()
        body = r.json()
        self._token = body["access_token"]
        self._exp = time.time() + body.get("expires_in", 3600)

    def token(self):
        with self._lock:
            # Refresh if missing or within `skew` seconds of expiry
            if not self._token or time.time() >= self._exp - self.skew:
                self._fetch()
            return self._token

tm = TokenManager(
    "https://auth.example.com/oauth/token",
    client_id="svc-123", client_secret="...", scope="orders:read",
)

def call_api(path):
    r = requests.get(
        f"https://api.example.com{path}",
        headers={"Authorization": f"Bearer {tm.token()}"},
        timeout=10,
    )
    if r.status_code == 401:        # token revoked early -> force refresh once
        tm._token = None
        r = requests.get(
            f"https://api.example.com{path}",
            headers={"Authorization": f"Bearer {tm.token()}"}, timeout=10)
    r.raise_for_status()
    return r.json()
```

**Setup and Explanation**
- The `skew` buffer (30s) refreshes *before* expiry so requests don't fail at the boundary.
- The lock prevents a thundering herd of concurrent refreshes.
- Handle a `401` by invalidating and retrying once — covers early revocation or clock drift.

**Pitfalls:** never log the token or secret; store `client_secret` in a secrets manager, not source; set timeouts on every HTTP call; client credentials has **no refresh token** — you just request a new access token. Respect `expires_in` rather than hardcoding a TTL.""",
    },

    # 11. Nginx reverse proxy config
    {
        "prompt": "Write an Nginx config that reverse-proxies to an upstream app on port 3000, terminates TLS, redirects HTTP to HTTPS, sets the right forwarding headers, and supports WebSockets.",
        "answer": """<thinking>
A correct reverse proxy needs: a port-80 server that 301s to HTTPS, a 443 server with the cert/key, an upstream block, `proxy_pass`, and the forwarding headers (`Host`, `X-Forwarded-For`, `X-Forwarded-Proto`) so the app knows the original client and scheme. WebSocket upgrade requires `Upgrade`/`Connection` headers and HTTP/1.1. I'll add sane timeouts.
</thinking>

```nginx
upstream app { server 127.0.0.1:3000; keepalive 32; }

# HTTP -> HTTPS redirect
server {
    listen 80;
    server_name example.com;
    return 301 https://$host$request_uri;
}

server {
    listen 443 ssl http2;
    server_name example.com;

    ssl_certificate     /etc/letsencrypt/live/example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/example.com/privkey.pem;
    ssl_protocols       TLSv1.2 TLSv1.3;

    location / {
        proxy_pass http://app;
        proxy_http_version 1.1;

        # Preserve original request info for the app
        proxy_set_header Host              $host;
        proxy_set_header X-Real-IP         $remote_addr;
        proxy_set_header X-Forwarded-For   $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support
        proxy_set_header Upgrade    $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_connect_timeout 5s;
        proxy_read_timeout    60s;
    }
}
```

**Setup and Explanation**
- `X-Forwarded-Proto $scheme` lets the app know the original request was HTTPS even though Nginx talks plain HTTP to it — essential for generating correct redirect URLs and secure-cookie logic.
- `proxy_add_x_forwarded_for` appends the client IP to any existing chain.
- WebSockets need `proxy_http_version 1.1` plus the `Upgrade`/`Connection` headers; without them the handshake fails with HTTP/1.0 buffering.
- `keepalive` in the upstream reuses connections, reducing latency.

**Pitfalls:** trust `X-Forwarded-*` only from your proxy (an internet-facing app must set `trust proxy`/equivalent carefully or attackers spoof IPs). Hardcoding `Connection: upgrade` for non-WS routes is fine here, but a `map` of `$http_upgrade` is cleaner for mixed traffic. Reload, don't restart: `nginx -t && nginx -s reload`.""",
    },

    # 12. Kubernetes deployment manifest
    {
        "prompt": "Write a Kubernetes Deployment + Service manifest for a web app with resource limits, liveness/readiness probes, rolling-update strategy, and config from a ConfigMap and Secret. Explain probes.",
        "answer": """<thinking>
I need a Deployment with replicas, a rolling update strategy with surge/unavailable controls, resource requests/limits (requests for scheduling, limits to cap usage), liveness vs readiness probes (distinct purposes), and env from ConfigMap/Secret. A ClusterIP Service fronts the pods. I'll explain why liveness != readiness.
</thinking>

```yaml
apiVersion: apps/v1
kind: Deployment
metadata: { name: web }
spec:
  replicas: 3
  strategy:
    type: RollingUpdate
    rollingUpdate: { maxSurge: 1, maxUnavailable: 0 }   # zero-downtime
  selector: { matchLabels: { app: web } }
  template:
    metadata: { labels: { app: web } }
    spec:
      containers:
        - name: web
          image: registry.example.com/web:1.4.2
          ports: [{ containerPort: 3000 }]
          resources:
            requests: { cpu: "100m", memory: "128Mi" }
            limits:   { cpu: "500m", memory: "256Mi" }
          envFrom:
            - configMapRef: { name: web-config }
            - secretRef:    { name: web-secrets }
          readinessProbe:
            httpGet: { path: /ready, port: 3000 }
            initialDelaySeconds: 5
            periodSeconds: 5
          livenessProbe:
            httpGet: { path: /healthz, port: 3000 }
            initialDelaySeconds: 15
            periodSeconds: 10
---
apiVersion: v1
kind: Service
metadata: { name: web }
spec:
  selector: { app: web }
  ports: [{ port: 80, targetPort: 3000 }]
```

**Setup and Explanation**
- **Readiness** gates traffic: a failing readiness probe removes the pod from the Service endpoints but doesn't restart it. Use it for "warming up" or "dependency down."
- **Liveness** gates restarts: a failing liveness probe kills and restarts the container. Use it for deadlocks/unrecoverable states.
- `maxUnavailable: 0, maxSurge: 1` adds a new pod before removing an old one — zero downtime.
- `requests` drive scheduling; `limits` cap usage (CPU is throttled, memory over-limit is OOM-killed).

**Pitfalls:** pointing liveness at a deep dependency check causes restart storms when a DB blips — keep liveness shallow, put dependency checks in readiness. Memory `limits` too low → OOMKills; missing `requests` → poor bin-packing. Secrets in a `Secret` are base64, not encrypted at rest unless you enable encryption providers.""",
    },

    # 13. Webhook receiver with signature verification
    {
        "prompt": "Implement a secure webhook receiver (e.g. for Stripe-style events) in Node that verifies the HMAC signature, guards against replay attacks, and is idempotent. Explain each safeguard.",
        "answer": """<thinking>
Webhook security has three pillars: (1) verify the HMAC signature over the *raw* body using a constant-time compare, (2) reject stale timestamps to stop replay, (3) dedupe on event id for idempotency since providers retry. I must capture the raw body before JSON parsing, because re-serializing changes bytes and breaks the signature.
</thinking>

```js
import express from "express";
import crypto from "node:crypto";

const SECRET = process.env.WEBHOOK_SECRET;
const seen = new Set(); // use Redis with TTL in production

const app = express();
// Capture the RAW body for this route — required for signature checks
app.post("/webhooks", express.raw({ type: "application/json" }), (req, res) => {
  const sig = req.header("X-Signature");        // "t=1700000000,v1=abc..."
  const parts = Object.fromEntries(sig.split(",").map((p) => p.split("=")));
  const signedPayload = `${parts.t}.${req.body.toString("utf8")}`;

  const expected = crypto.createHmac("sha256", SECRET)
    .update(signedPayload).digest("hex");

  // Constant-time compare prevents timing attacks
  const ok = crypto.timingSafeEqual(Buffer.from(expected), Buffer.from(parts.v1));
  if (!ok) return res.status(400).send("bad signature");

  // Replay protection: reject events older than 5 minutes
  if (Math.abs(Date.now() / 1000 - Number(parts.t)) > 300)
    return res.status(400).send("stale");

  const event = JSON.parse(req.body.toString("utf8"));
  if (seen.has(event.id)) return res.status(200).send("dup"); // idempotent
  seen.add(event.id);

  // ... process event ...
  res.status(200).send("ok");   // 2xx fast; do heavy work async
});
```

**Setup and Explanation**
- **Raw body:** `express.raw` keeps exact bytes. If you use `express.json()` first, the parsed-then-restringified payload won't match the signature.
- **`timingSafeEqual`:** comparing with `===` leaks timing info that can be used to forge signatures; constant-time compare closes that.
- **Timestamp window:** rejecting old timestamps prevents replaying a captured valid request later.
- **Idempotency:** providers retry on non-2xx or timeouts; deduping on `event.id` (persisted in Redis/DB with TTL) avoids double-processing.

**Pitfalls:** the in-memory `Set` resets on restart — use Redis/DB. Return `2xx` quickly and offload slow work to a queue, or the provider times out and retries. Never trust the body before verifying the signature.""",
    },

    # 14. Terraform AWS resource
    {
        "prompt": "Write Terraform to provision an S3 bucket with versioning, encryption, and public access blocked, plus an IAM policy granting a role read/write. Explain state and idempotency.",
        "answer": """<thinking>
I'll declare the S3 bucket plus the separate resources Terraform now requires (versioning, SSE, public access block are their own resources in the AWS provider v4+). Then an IAM policy scoped to that bucket's ARN and a role attachment. I'll explain how Terraform state enables idempotent plan/apply and why remote state + locking matters.
</thinking>

```hcl
resource "aws_s3_bucket" "data" {
  bucket = "acme-app-data-prod"
}

resource "aws_s3_bucket_versioning" "data" {
  bucket = aws_s3_bucket.data.id
  versioning_configuration { status = "Enabled" }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "data" {
  bucket = aws_s3_bucket.data.id
  rule {
    apply_server_side_encryption_by_default { sse_algorithm = "aws:kms" }
  }
}

resource "aws_s3_bucket_public_access_block" "data" {
  bucket                  = aws_s3_bucket.data.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

data "aws_iam_policy_document" "rw" {
  statement {
    actions   = ["s3:GetObject", "s3:PutObject"]
    resources = ["${aws_s3_bucket.data.arn}/*"]
  }
  statement {
    actions   = ["s3:ListBucket"]
    resources = [aws_s3_bucket.data.arn]
  }
}

resource "aws_iam_role_policy" "app_rw" {
  name   = "app-s3-rw"
  role   = aws_iam_role.app.id
  policy = data.aws_iam_policy_document.rw.json
}
```

**Remote state + locking**
```hcl
terraform {
  backend "s3" {
    bucket         = "acme-tfstate"
    key            = "prod/app.tfstate"
    region         = "us-east-1"
    dynamodb_table = "tf-locks"   # state locking
    encrypt        = true
  }
}
```

**Setup and Explanation**
- **State:** Terraform records what it created in state. `plan` diffs desired config vs state vs real infrastructure, so re-running `apply` with no changes does nothing — that's idempotency.
- **Remote state + DynamoDB lock** prevents two engineers from corrupting state with concurrent applies.
- Note `s3:GetObject` needs the `/*` object ARN, while `s3:ListBucket` needs the bucket ARN — a common scoping mistake.

**Pitfalls:** never commit `terraform.tfstate` (it holds secrets) — use a remote backend. Don't edit infra by hand (drift); reconcile with `plan`. Order matters: the versioning/SSE/public-access resources reference the bucket, so Terraform builds the dependency graph automatically — don't hardcode names.""",
    },

    # 15. Redis caching with TTL & invalidation
    {
        "prompt": "Implement a cache-aside pattern in Python with Redis: read-through with TTL, stampede protection, and explicit invalidation on writes. Explain trade-offs.",
        "answer": """<thinking>
Cache-aside (lazy loading): on read, check cache; on miss, load from DB, set with TTL. The hazards are (a) thundering herd / stampede when a hot key expires and many requests hit the DB at once, and (b) stale data after writes. I'll add a short lock to serialize regeneration and invalidate the key on write. I'll discuss TTL vs invalidation trade-offs.
</thinking>

```python
import json, time, redis

r = redis.Redis(decode_responses=True)
TTL = 300

def get_user(uid, db):
    key = f"user:{uid}"
    cached = r.get(key)
    if cached is not None:
        return json.loads(cached)

    # Stampede protection: only one caller regenerates; others briefly wait
    lock_key = f"lock:{key}"
    if r.set(lock_key, "1", nx=True, ex=10):
        try:
            user = db.fetch_user(uid)        # source of truth
            r.set(key, json.dumps(user), ex=TTL)
            return user
        finally:
            r.delete(lock_key)
    else:
        time.sleep(0.05)                      # let the lock holder fill cache
        cached = r.get(key)
        return json.loads(cached) if cached else db.fetch_user(uid)

def update_user(uid, fields, db):
    db.update_user(uid, fields)
    r.delete(f"user:{uid}")                   # invalidate; next read repopulates
```

**Setup and Explanation**
- **Cache-aside** keeps the DB as source of truth; the cache is a best-effort accelerator.
- **TTL** bounds staleness even if an invalidation is missed — defense in depth.
- **Stampede protection** via an `SET NX EX` lock means only one request rebuilds a hot key; the rest wait briefly instead of dogpiling the DB.
- **Invalidate on write** (`DELETE`, not update-in-place) avoids serializing fresh values into cache that might race with concurrent writes.

**Trade-offs / pitfalls**
- Delete-on-write is simpler and safer than write-through but causes a cache miss right after a write.
- Caching `None`/misses (negative caching) prevents repeated DB hits for non-existent keys — but pick a short TTL.
- Add jitter to TTLs so many keys don't expire simultaneously.
- The lock's `ex=10` must exceed your DB fetch time, or two callers regenerate.""",
    },

    # 16. pytest fixtures + mocking
    {
        "prompt": "Write pytest tests for a function that calls an external HTTP API. Show fixtures, mocking the HTTP call, parametrized cases, and testing error paths without hitting the network.",
        "answer": """<thinking>
The goal is fast, deterministic tests that never touch the network. I'll mock at the boundary (the `requests` call or use `responses`/`monkeypatch`), use a fixture for the client, parametrize success/failure inputs, and assert both happy path and error handling. I'll show `responses` since it intercepts at the HTTP layer cleanly.
</thinking>

```python
# weather.py
import requests

def get_temp(city):
    resp = requests.get(f"https://api.example.com/weather/{city}", timeout=5)
    resp.raise_for_status()
    return resp.json()["temp_c"]
```

```python
# test_weather.py
import pytest, responses, requests
from weather import get_temp

@pytest.fixture
def base_url():
    return "https://api.example.com/weather"

@responses.activate
@pytest.mark.parametrize("city,payload,expected", [
    ("london", {"temp_c": 12}, 12),
    ("cairo",  {"temp_c": 35}, 35),
])
def test_get_temp_success(base_url, city, payload, expected):
    responses.get(f"{base_url}/{city}", json=payload, status=200)
    assert get_temp(city) == expected

@responses.activate
def test_get_temp_http_error(base_url):
    responses.get(f"{base_url}/nowhere", json={"error": "not found"}, status=404)
    with pytest.raises(requests.HTTPError):
        get_temp("nowhere")

@responses.activate
def test_get_temp_malformed(base_url):
    responses.get(f"{base_url}/london", json={}, status=200)  # missing temp_c
    with pytest.raises(KeyError):
        get_temp("london")
```

**Setup and Explanation**
- `@responses.activate` intercepts outgoing `requests` calls; no real network, fully deterministic.
- `@pytest.mark.parametrize` runs the same test body over multiple inputs — one test, many cases, clear failure reporting per case.
- Error paths (`404` → `HTTPError`, missing field → `KeyError`) are tested explicitly; untested error branches are where bugs hide.
- Fixtures (`base_url`) centralize shared setup; scope them (`function`/`module`/`session`) to balance isolation vs speed.

**Pitfalls:** mocking too deep (patching internal implementation instead of the HTTP boundary) makes tests brittle — mock at the edge. Forgetting `@responses.activate` lets a test hit the real API and flake. Always set timeouts in the code under test so a hung mock/real call can't stall the suite.""",
    },

    # 17. Makefile for build orchestration
    {
        "prompt": "Write a Makefile that orchestrates a typical project: install, lint, test, build, and docker-build targets, with a help target and proper handling of phony targets and variables.",
        "answer": """<thinking>
A good Makefile uses `.PHONY` for non-file targets, variables for tweakable values, a self-documenting `help` target that parses `##` comments, and dependency chaining between targets. I'll guard against the silent-failure of recipe lines and make `help` the default.
</thinking>

```makefile
.DEFAULT_GOAL := help
IMAGE := registry.example.com/app
TAG   := $(shell git rev-parse --short HEAD)

.PHONY: help install lint test build docker-build clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
	  | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies
	npm ci

lint: install ## Run linters
	npm run lint

test: install ## Run tests
	npm test

build: lint test ## Compile the app (after lint + test pass)
	npm run build

docker-build: build ## Build and tag the Docker image
	docker build -t $(IMAGE):$(TAG) -t $(IMAGE):latest .

clean: ## Remove build artifacts
	rm -rf dist node_modules
```

**Setup and Explanation**
- **`.PHONY`** tells Make these targets don't produce a file of that name, so they always run (otherwise a file/dir called `test` would shadow the target).
- **Dependency chaining:** `build: lint test` runs lint and test first; `make build` is a one-liner gate.
- **Self-documenting help:** the `awk`/`grep` one-liner prints every target with a `## comment`, so `make` (or `make help`) lists the API.
- **Variables:** `TAG` derives from git SHA for traceable images; override on the CLI: `make docker-build TAG=v1.0`.

**Pitfalls:** Make requires **tab** indentation, not spaces (a classic "missing separator" error). Each recipe line runs in its own shell — `cd foo` doesn't persist to the next line; chain with `&&` or `.ONESHELL`. Without `.PHONY`, targets sharing a name with a file silently no-op.""",
    },

    # 18. gRPC service definition + client
    {
        "prompt": "Define a gRPC service in protobuf for a user lookup with both unary and server-streaming RPCs, then show a Python server and client. Explain why protobuf/gRPC over REST here.",
        "answer": """<thinking>
I'll define a `.proto` with a unary `GetUser` and a server-streaming `ListUsers`. Then generate stubs and implement a server (servicer) and a client. I'll explain the contract-first nature, HTTP/2 multiplexing, and binary efficiency vs REST, plus the codegen workflow.
</thinking>

```protobuf
// user.proto
syntax = "proto3";
package users;

message UserId { string id = 1; }
message User   { string id = 1; string name = 2; string email = 3; }
message ListReq { int32 limit = 1; }

service UserService {
  rpc GetUser   (UserId)  returns (User);            // unary
  rpc ListUsers (ListReq) returns (stream User);     // server-streaming
}
```

```bash
python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. user.proto
```

```python
# server.py
from concurrent import futures
import grpc, user_pb2, user_pb2_grpc

class UserService(user_pb2_grpc.UserServiceServicer):
    def GetUser(self, request, context):
        if request.id != "1":
            context.abort(grpc.StatusCode.NOT_FOUND, "no such user")
        return user_pb2.User(id="1", name="Ada", email="ada@x.io")

    def ListUsers(self, request, context):
        for i in range(request.limit):           # stream rows back
            yield user_pb2.User(id=str(i), name=f"u{i}", email=f"u{i}@x.io")

server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
user_pb2_grpc.add_UserServiceServicer_to_server(UserService(), server)
server.add_insecure_port("[::]:50051")
server.start(); server.wait_for_termination()
```

```python
# client.py
import grpc, user_pb2, user_pb2_grpc
with grpc.insecure_channel("localhost:50051") as ch:
    stub = user_pb2_grpc.UserServiceStub(ch)
    print(stub.GetUser(user_pb2.UserId(id="1")))
    for u in stub.ListUsers(user_pb2.ListReq(limit=3)):  # iterate the stream
        print(u.name)
```

**Setup and Explanation**
- **Contract-first:** the `.proto` is the single source of truth; codegen produces typed stubs for every language, eliminating hand-written serialization drift.
- **Server streaming:** `yield`ing messages sends them incrementally over one HTTP/2 stream — great for large result sets or live feeds.
- **Why gRPC over REST:** binary protobuf is smaller/faster than JSON, HTTP/2 multiplexes many calls over one connection, and the schema gives strong typing and backward-compat rules (field numbers).

**Pitfalls:** never reuse or renumber existing field numbers — it breaks wire compat; use `reserved` for removed fields. Use TLS (`secure_channel`) in production, not `insecure`. Set deadlines on client calls so a slow server doesn't hang the caller.""",
    },

    # 19. Environment configuration / 12-factor
    {
        "prompt": "How should I manage configuration across dev/staging/prod following 12-factor principles? Show a typed config loader in Python that reads from environment with validation and sensible defaults.",
        "answer": """<thinking>
12-factor says config lives in the environment, not in code, and there's strict separation between code and config so the same artifact runs in every environment. I'll build a typed loader (pydantic-settings) that reads env vars, validates types/required fields, fails fast on misconfig, and supports `.env` for local dev only. I'll stress not committing secrets.
</thinking>

```python
# config.py
from pydantic import Field, PostgresDsn, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_")

    environment: str = Field(default="development")
    debug: bool = Field(default=False)
    database_url: PostgresDsn                      # required; validated as a DSN
    redis_url: str = "redis://localhost:6379"
    max_workers: int = Field(default=4, ge=1, le=64)
    secret_key: str                                # required, no default

    @field_validator("environment")
    @classmethod
    def known_env(cls, v):
        if v not in {"development", "staging", "production"}:
            raise ValueError(f"unknown environment: {v}")
        return v

settings = Settings()   # raises at import time if required vars missing/invalid
```

```bash
# .env  (local dev only — gitignored)
APP_ENVIRONMENT=development
APP_DATABASE_URL=postgres://app:secret@localhost:5432/app
APP_SECRET_KEY=dev-only-not-real
```

**Setup and Explanation**
- **Config in the environment:** dev/staging/prod differ only by env vars; the same Docker image ships everywhere (build once, promote).
- **Fail fast:** instantiating `Settings()` at startup throws immediately on a missing `SECRET_KEY` or malformed `DATABASE_URL` — better than a `KeyError` deep in a request hours later.
- **Typed + validated:** `int`, `bool`, `PostgresDsn`, and bounds (`ge/le`) catch "MAX_WORKERS=banana" at boot.
- **`.env` for local only:** real environments inject vars via the orchestrator/secrets manager; `.env` is gitignored.

**Pitfalls:** never commit secrets or bake them into images (they persist in layers). Don't read `os.environ` ad hoc throughout the code — centralize in one typed object. Prefix env vars (`APP_`) to avoid collisions. Keep `debug=False` and proper logging in prod; gate debug features on `environment`.""",
    },

    # 20. Database migrations
    {
        "prompt": "Explain a safe zero-downtime database migration to add a NOT NULL column with a default to a large Postgres table, using Alembic. Why is the naive approach dangerous?",
        "answer": """<thinking>
The danger: on older Postgres, adding a NOT NULL column with a default rewrote the whole table under an `ACCESS EXCLUSIVE` lock — blocking all reads/writes for the duration on a big table. Even on modern PG (11+), a *volatile* default or a NOT NULL backfill can lock or bloat. The safe, expand/contract pattern: add nullable column, backfill in batches, add the constraint with `NOT VALID` then `VALIDATE`. I'll show Alembic ops.
</thinking>

```python
# Migration A: add column as NULLABLE (fast, metadata-only)
def upgrade():
    op.add_column("orders", sa.Column("region", sa.Text(), nullable=True))
```

```python
# Migration B: backfill in batches (run outside a single huge transaction)
def upgrade():
    conn = op.get_bind()
    while True:
        res = conn.execute(sa.text(
            "WITH cte AS ("
            "  SELECT id FROM orders WHERE region IS NULL "
            "  LIMIT 5000 FOR UPDATE SKIP LOCKED"
            ") "
            "UPDATE orders o SET region = 'us' "
            "FROM cte WHERE o.id = cte.id"))
        if res.rowcount == 0:
            break
```

```python
# Migration C: enforce NOT NULL without a long lock
def upgrade():
    op.execute("ALTER TABLE orders ADD CONSTRAINT orders_region_nn CHECK (region IS NOT NULL) NOT VALID")
    op.execute("ALTER TABLE orders VALIDATE CONSTRAINT orders_region_nn")  # scans, but only SHARE UPDATE lock
    op.alter_column("orders", "region", nullable=False)  # now instant; constraint already proves it
    op.execute("ALTER TABLE orders ALTER COLUMN region SET DEFAULT 'us'")
```

**Setup and Explanation**
- **Expand/contract:** add the column nullable first (metadata-only, instant), backfill incrementally, then tighten the constraint — application keeps running throughout.
- **Batched backfill** with `SKIP LOCKED` avoids a single giant `UPDATE` that locks every row and bloats WAL.
- **`NOT VALID` then `VALIDATE`** lets you add the check without the full-table `ACCESS EXCLUSIVE` lock; validation takes only a `SHARE UPDATE EXCLUSIVE` lock.

**Why naive is dangerous:** `ALTER TABLE orders ADD COLUMN region TEXT NOT NULL DEFAULT 'us'` in one shot can take an exclusive lock and/or rewrite the table on a large dataset — a multi-minute outage.

**Pitfalls:** deploy code that tolerates the column being null *before* enforcing NOT NULL. Run backfill batches outside one transaction. Keep migrations forward-only and reversible where feasible; always test on a prod-sized copy.""",
    },

    # 21. Rate limiting middleware
    {
        "prompt": "Implement a distributed rate limiter (token bucket) for an API using Redis so it works across multiple app instances. Show the Lua script for atomicity and the middleware.",
        "answer": """<thinking>
Per-instance in-memory limiting fails when you scale horizontally — each instance has its own counter. Redis centralizes state. Token bucket allows bursts up to capacity while enforcing a refill rate. The check-and-decrement must be atomic, so I'll do it in a single Lua script (Redis runs scripts atomically) to avoid race conditions between GET and SET. Middleware returns 429 with retry headers.
</thinking>

```lua
-- token_bucket.lua : KEYS[1]=bucket  ARGV: capacity, refill_rate, now, requested
local cap   = tonumber(ARGV[1])
local rate  = tonumber(ARGV[2])     -- tokens per second
local now   = tonumber(ARGV[3])
local need  = tonumber(ARGV[4])

local data   = redis.call("HMGET", KEYS[1], "tokens", "ts")
local tokens = tonumber(data[1]) or cap
local ts     = tonumber(data[2]) or now

tokens = math.min(cap, tokens + (now - ts) * rate)   -- refill since last call
local allowed = tokens >= need
if allowed then tokens = tokens - need end

redis.call("HMSET", KEYS[1], "tokens", tokens, "ts", now)
redis.call("EXPIRE", KEYS[1], math.ceil(cap / rate) * 2)
return { allowed and 1 or 0, tokens }
```

```js
import { createClient } from "redis";
const redis = createClient(); await redis.connect();
const sha = await redis.scriptLoad(fs.readFileSync("token_bucket.lua", "utf8"));

const CAP = 20, RATE = 5;  // burst 20, sustained 5 req/s
export async function rateLimit(req, res, next) {
  const key = `rl:${req.ip}`;
  const [allowed, remaining] = await redis.evalSha(sha, {
    keys: [key], arguments: [String(CAP), String(RATE), String(Date.now()/1000), "1"],
  });
  res.set("X-RateLimit-Remaining", String(Math.floor(remaining)));
  if (!allowed) {
    res.set("Retry-After", String(Math.ceil(1 / RATE)));
    return res.status(429).json({ error: "rate limited" });
  }
  next();
}
```

**Setup and Explanation**
- **Why Redis:** all app instances share one bucket per client, so the limit holds globally, not per-process.
- **Why Lua:** the refill-check-decrement sequence must be atomic. Doing it with separate `GET`/`SET` commands races under load and lets clients exceed the limit. Redis executes a script atomically.
- **Token bucket vs fixed window:** allows short bursts (up to `CAP`) while bounding the sustained rate (`RATE`), and avoids the boundary spikes fixed windows suffer.
- `EXPIRE` reclaims memory for idle keys.

**Pitfalls:** key on a stable identity (API key/user) not just IP behind a proxy (use `X-Forwarded-For` carefully). Return `429` + `Retry-After` so clients back off. Beware clock skew across instances — pass server-side time or use Redis `TIME`.""",
    },

    # 22. Message queue consumer (RabbitMQ/Kafka)
    {
        "prompt": "Write a reliable RabbitMQ consumer in Python that processes messages with manual acks, prefetch limiting, a dead-letter queue for poison messages, and graceful shutdown. Explain at-least-once semantics.",
        "answer": """<thinking>
Reliability essentials: manual acknowledgements (ack only after success), prefetch (QoS) so one consumer doesn't hoard the queue, a dead-letter exchange so messages that repeatedly fail don't block the queue forever, and signal handling for graceful shutdown so in-flight messages finish. This gives at-least-once delivery — I must make handlers idempotent.
</thinking>

```python
import pika, json, signal, sys

conn = pika.BlockingConnection(pika.ConnectionParameters("localhost"))
ch = conn.channel()

# Main queue routes failures to a dead-letter exchange after rejection
ch.exchange_declare("dlx", "direct", durable=True)
ch.queue_declare("tasks.dlq", durable=True)
ch.queue_bind("tasks.dlq", "dlx", routing_key="tasks")
ch.queue_declare("tasks", durable=True, arguments={
    "x-dead-letter-exchange": "dlx",
    "x-dead-letter-routing-key": "tasks",
})

ch.basic_qos(prefetch_count=10)      # at most 10 unacked in flight per consumer

def handle(ch, method, props, body):
    try:
        task = json.loads(body)
        process(task)                # idempotent business logic
        ch.basic_ack(method.delivery_tag)          # success -> remove from queue
    except PoisonError:
        ch.basic_nack(method.delivery_tag, requeue=False)  # -> dead-letter queue
    except Exception:
        ch.basic_nack(method.delivery_tag, requeue=True)   # transient -> retry

ch.basic_consume("tasks", handle)

def shutdown(*_):
    ch.stop_consuming()              # finish in-flight, then exit cleanly
    conn.close(); sys.exit(0)
signal.signal(signal.SIGTERM, shutdown)
signal.signal(signal.SIGINT, shutdown)

ch.start_consuming()
```

**Setup and Explanation**
- **Manual ack:** the broker keeps the message until you `basic_ack`. If the consumer crashes mid-process, the message is redelivered — nothing is silently lost.
- **Prefetch (`basic_qos`)** caps unacked messages so work spreads across consumers instead of one greedily buffering the queue.
- **Dead-letter queue:** `requeue=False` routes poison messages (malformed/permanently failing) to `tasks.dlq` for inspection instead of looping forever.
- **Graceful shutdown:** `stop_consuming()` lets in-flight handlers finish on SIGTERM (e.g. during a deploy).

**At-least-once semantics:** crashes between processing and ack cause redelivery, so a message may be handled more than once. **Make handlers idempotent** (dedupe on a message id, use upserts). Exactly-once is generally impractical; idempotency is the pragmatic answer.

**Pitfalls:** acking before processing risks loss (becomes at-most-once). No DLQ means a single poison message blocks/retries forever. Use durable queues + persistent messages to survive broker restarts.""",
    },

    # 23. curl / httpie API debugging
    {
        "prompt": "Show how to debug a failing REST API call from the command line with curl: sending JSON with auth, inspecting headers and timing, following redirects, and saving output. Include httpie equivalents.",
        "answer": """<thinking>
This is about practical CLI HTTP debugging. I'll cover: a JSON POST with a bearer token, `-i`/`-v` to see headers, `-w` for timing breakdown, `-L` for redirects, `-o` to save, and `-f` to fail on HTTP errors for scripts. Then the httpie equivalents which are terser. I'll note gotchas like quoting and `--data` not setting content-type.
</thinking>

```bash
# POST JSON with auth, show response headers (-i) and fail on >=400 (-f)
curl -fi -X POST https://api.example.com/orders \\
  -H "Authorization: Bearer $TOKEN" \\
  -H "Content-Type: application/json" \\
  -d '{"item":"sku-1","qty":2}'

# Full verbose trace incl. TLS + request headers
curl -v https://api.example.com/health

# Timing breakdown (DNS, connect, TLS, TTFB, total)
curl -s -o /dev/null -w \\
  'dns=%{time_namelookup} connect=%{time_connect} tls=%{time_appconnect} ttfb=%{time_starttransfer} total=%{time_total}\\n' \\
  https://api.example.com/health

# Follow redirects (-L) and save body to a file (-o)
curl -L -o resp.json https://api.example.com/export

# Send a file as the JSON body
curl -X POST https://api.example.com/bulk -H "Content-Type: application/json" --data @payload.json
```

```bash
# HTTPie equivalents (auto-sets JSON content-type, pretty-prints, colorizes)
http POST api.example.com/orders item=sku-1 qty:=2 "Authorization: Bearer $TOKEN"
http --print=Hh GET api.example.com/health        # H=req headers, h=resp headers
http --follow GET api.example.com/export > resp.json
```

**Setup and Explanation**
- `-i` shows response headers with the body; `-v` shows the full request/response incl. TLS handshake — use `-v` when a request mysteriously fails.
- `-w` with `%{time_*}` variables pinpoints where latency lives (DNS vs connect vs server time).
- `-f` makes curl exit non-zero on HTTP `>=400`, essential in scripts and CI so failures don't pass silently.
- HTTPie infers JSON: `qty:=2` sends a number (`:=` = raw JSON), `qty=2` would be a string.

**Pitfalls:** `-d` defaults to `application/x-www-form-urlencoded`; you must set `Content-Type: application/json` explicitly (httpie does it for you). Single-quote JSON to avoid shell expansion of `$`/`"`. Don't put secrets in shell history — use an env var (`$TOKEN`) or `--netrc`. `-k` disables TLS verification — only for local debugging, never in production scripts.""",
    },

    # 24. Feature flags / config integration
    {
        "prompt": "Implement a simple feature-flag system in a Node app that supports boolean flags, percentage rollouts, and per-user targeting, with the ability to change flags without redeploying. Explain the integration pattern.",
        "answer": """<thinking>
Feature flags decouple deploy from release. I'll model flags with: a boolean kill-switch, a percentage rollout (deterministic by hashing user id so a user stays in/out consistently), and explicit allow-lists. Config comes from an external store (here a JSON loaded/polled; in prod a service like LaunchDarkly/Unleash or Redis) so flips don't need a redeploy. The evaluation must be deterministic and cheap.
</thinking>

```js
import crypto from "node:crypto";

// Loaded from a remote store and refreshed periodically (no redeploy to flip)
let flags = {};
async function refresh(store) { flags = await store.fetch(); }
setInterval(() => refresh(store), 30_000);

// Deterministic 0-99 bucket for (flag, user): same user -> same bucket
function bucket(flagKey, userId) {
  const h = crypto.createHash("md5").update(`${flagKey}:${userId}`).digest();
  return h.readUInt32BE(0) % 100;
}

export function isEnabled(flagKey, user = {}) {
  const f = flags[flagKey];
  if (!f || f.enabled === false) return false;        // global kill-switch
  if (f.allowUsers?.includes(user.id)) return true;    // explicit targeting
  if (f.blockUsers?.includes(user.id)) return false;
  if (typeof f.rolloutPct === "number")                // percentage rollout
    return bucket(flagKey, user.id ?? "anon") < f.rolloutPct;
  return Boolean(f.on);
}
```

```json
{
  "new-checkout": { "enabled": true, "rolloutPct": 25, "allowUsers": ["u-42"] },
  "dark-mode":    { "enabled": true, "on": true }
}
```

```js
// Usage at the call site
if (isEnabled("new-checkout", { id: req.user.id })) {
  renderNewCheckout();
} else {
  renderLegacyCheckout();
}
```

**Setup and Explanation**
- **Deploy ≠ release:** ship code dark, then flip the flag in the store. Roll back instantly by toggling `enabled:false` — no redeploy.
- **Deterministic bucketing:** hashing `flag:user` means a given user is stably in or out of a 25% rollout (not re-rolled each request), so the experience is consistent and you can ramp 5% → 25% → 100%.
- **Targeting:** allow/block lists let you enable for internal users first.
- **External store + polling** is what makes flips instant; cache locally so evaluation stays in-memory and fast.

**Pitfalls:** clean up stale flags or they rot into permanent dead `if` branches. Don't hash on a missing user id (everyone lands in the same bucket) — fall back gracefully. Failure mode should default to *off* (or the safe legacy path) if the flag store is unreachable. Keep evaluation side-effect-free and synchronous at the call site.""",
    },

    # 25. OpenTelemetry distributed tracing
    {
        "prompt": "Set up distributed tracing with OpenTelemetry in a Python service: auto-instrument incoming requests and outgoing HTTP/DB calls, propagate context across service boundaries, and export to an OTLP collector. Explain spans, traces, and context propagation.",
        "answer": """<thinking>
OpenTelemetry models a request as a trace (a tree of spans). I'll configure a TracerProvider with an OTLP exporter, use auto-instrumentation for the web framework, requests, and the DB driver so spans are created and context is propagated via W3C `traceparent` headers automatically. I'll add a manual span for custom business logic and explain how context flows across services.
</thinking>

```python
# tracing.py
from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

def setup_tracing(service_name="orders"):
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(BatchSpanProcessor(
        OTLPSpanExporter(endpoint="http://otel-collector:4317", insecure=True)
    ))
    trace.set_tracer_provider(provider)
```

```python
# app.py
from tracing import setup_tracing
setup_tracing()

from opentelemetry.instrumentation.flask import FlaskInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor
from opentelemetry.instrumentation.psycopg2 import Psycopg2Instrumentor
from opentelemetry import trace
import flask, requests

app = flask.Flask(__name__)
FlaskInstrumentor().instrument_app(app)   # span per incoming request
RequestsInstrumentor().instrument()       # span + traceparent on outgoing calls
Psycopg2Instrumentor().instrument()       # span per DB query

tracer = trace.get_tracer(__name__)

@app.route("/checkout")
def checkout():
    with tracer.start_as_current_span("validate_cart") as span:   # custom span
        span.set_attribute("cart.items", 3)
        # this outgoing call automatically carries the trace context downstream:
        requests.get("http://inventory:8000/reserve", timeout=5)
    return {"ok": True}
```

**Setup and Explanation**
- **Trace** = the whole request journey across services; **span** = one timed operation (an HTTP handler, a DB query, a function). Spans nest into a tree under one trace id.
- **Auto-instrumentation** wraps Flask, `requests`, and psycopg2 so spans are created and timed for you, and the `requests` instrumentor injects the W3C `traceparent` header on outgoing calls.
- **Context propagation:** the downstream `inventory` service (also OTel-instrumented) reads `traceparent`, continues the *same* trace, and its spans appear under the same trace in the collector — that's how you see end-to-end latency across services.
- **BatchSpanProcessor** buffers and exports asynchronously so tracing doesn't add latency to the request path.

**Pitfalls:** without context propagation each service produces disconnected traces — always propagate (auto-instrumentation handles HTTP; for queues/Kafka you must inject/extract manually). Sample in high-traffic systems (head/tail sampling) to control cost. Don't put high-cardinality or sensitive data in span attributes. Ensure the collector endpoint and protocol (gRPC `:4317` vs HTTP `:4318`) match the exporter.""",
    },
]


def to_jsonl_rows():
    """Yield chat-format dicts ready to write to a JSONL training file."""
    for ex in examples:
        yield {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }


if __name__ == "__main__":
    print(f"Loaded {len(examples)} tools & integration training examples.")
    for i, ex in enumerate(examples, 1):
        assert ex["prompt"] and ex["answer"], f"example {i} incomplete"
        assert "<thinking>" in ex["answer"], f"example {i} missing <thinking>"
        assert "```" in ex["answer"], f"example {i} missing code block"
    print("All examples validated (prompt, <thinking>, code block present).")
