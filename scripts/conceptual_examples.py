#!/usr/bin/env python3
"""
Conceptual / reasoning training examples for model distillation.

Each example pairs a realistic software-architecture question with a detailed,
reasoning-first answer. Answers open with a <thinking> section that shows the
chain of reasoning, followed by a structured explanation: comparisons, when to
use, tradeoffs, and real-world applications. Explanation-focused (no code).

Topics covered: design patterns, architecture decisions, authentication /
authorization, caching strategies, database choices, API design, scalability,
security, and performance optimization.
"""

examples = [
    # 1 -------------------------------------------------------------------
    {
        "prompt": "When should I use the Strategy pattern instead of a simple chain of if/else or switch statements? What are the real tradeoffs?",
        "answer": """<thinking>
The question is really about *when the indirection of Strategy pays for itself*. A switch is cheaper to read until the branching logic starts to (a) repeat in many places, (b) change for reasons unrelated to the calling code, or (c) need to be extended by code that can't touch the original file. I should frame this around the Open/Closed Principle and the cost of the extra abstraction, not just recite the GoF definition. I'll compare concretely and give a decision rule.
</thinking>

**What Strategy actually is**

Strategy encapsulates a family of interchangeable algorithms behind a common interface, letting the caller select behavior at runtime by swapping the object rather than branching on a flag.

**Strategy vs. switch/if-else**

| Concern | switch / if-else | Strategy |
|---|---|---|
| Adding a new variant | Edit the existing function (risk of regressions) | Add a new class; nothing else changes |
| Testability | Test all branches through one entry point | Each strategy tested in isolation |
| Runtime swapping | Awkward | First-class (inject/replace the object) |
| Discoverability | All logic in one place (easy to read) | Logic scattered across classes |
| Overhead | None | Extra types, indirection, wiring |

**When to use Strategy**
- The branching represents genuinely independent algorithms (e.g. pricing rules, compression codecs, payment providers) that evolve on their own schedules.
- You need to choose behavior at runtime or via configuration/DI.
- Third parties or plugins must add variants without modifying your code (Open/Closed).

**When a switch is better**
- 2-3 stable branches that rarely change.
- The branches are short and share most of their logic.
- Readability of "everything in one place" matters more than extensibility.

**Tradeoffs**
The hidden cost of Strategy is *fragmentation*: a reader must hop across files to understand the whole behavior, and you still need somewhere (a factory or map) that picks the strategy — so you haven't eliminated branching, you've relocated it to the edge of the system. The win is that the relocated branching changes far less often than the algorithms themselves.

**Real-world**: payment gateways (Stripe/PayPal/Adyen), AWS S3 storage-class selection, sort comparators, retry/backoff policies. Frameworks like Spring lean on it heavily via dependency injection.

**Decision rule**: start with a switch. Refactor to Strategy the moment a third caller needs the same branching, or an external module needs to add a branch.""",
    },
    # 2 -------------------------------------------------------------------
    {
        "prompt": "Explain the difference between authentication and authorization, and why conflating them causes security bugs.",
        "answer": """<thinking>
People mix these up constantly, and the mix-up is a real source of vulnerabilities (e.g. IDOR / broken access control, which sits at the top of the OWASP list). I should define each crisply, show where each lives in a request lifecycle, and then explain the specific bug classes that arise when one is treated as a proxy for the other.
</thinking>

**Definitions**
- **Authentication (AuthN)** — *Who are you?* Verifying identity. Outputs a trusted principal (user id, claims).
- **Authorization (AuthZ)** — *What are you allowed to do?* Deciding whether that principal may perform a specific action on a specific resource.

**Order and placement**
AuthN happens first and once (login, token validation). AuthZ happens on *every* protected operation, because permissions are contextual: a user authenticated globally may still be forbidden from this particular record.

| | Authentication | Authorization |
|---|---|---|
| Question | Who? | What may they do? |
| Frequency | Once per session/request | Every protected action |
| Failure mode | 401 Unauthorized | 403 Forbidden |
| Mechanisms | Passwords, OAuth, MFA, certs | RBAC, ABAC, ACLs, policy engines |

**Why conflation creates bugs**
The classic mistake is assuming "logged in" implies "allowed." This produces **Insecure Direct Object Reference (IDOR)**: an authenticated user requests `/invoices/123` and the server returns it without checking that invoice 123 belongs to *them*. AuthN succeeded; AuthZ was skipped. Another variant is **privilege escalation**, where the UI hides an admin button but the endpoint never re-checks the role server-side — client-side hiding is not authorization.

**Real-world guidance**
- Enforce AuthZ at the data/service layer, not just the route or the UI.
- Default-deny: no rule means no access.
- Make ownership checks explicit (`resource.owner_id == principal.id`) rather than implicit.
- Treat tokens as identity, never as permission — re-derive permissions from a source of truth.""",
    },
    # 3 -------------------------------------------------------------------
    {
        "prompt": "Compare cache-aside, write-through, and write-back caching strategies. How do I choose between them?",
        "answer": """<thinking>
These three differ along two axes: who populates the cache, and how writes propagate to the backing store. The choice is driven by the read/write ratio, tolerance for staleness, and tolerance for data loss. I'll lay out each one's flow, then map them to consistency and durability properties, then give selection criteria.
</thinking>

**The three patterns**

- **Cache-aside (lazy loading)** — Application checks cache; on miss it reads the DB, then populates the cache. Writes go to the DB and *invalidate* (or update) the cache entry.
- **Write-through** — Writes go to the cache, which synchronously writes to the DB before acknowledging. Reads are always cache hits for written data.
- **Write-back (write-behind)** — Writes go to the cache and are acknowledged immediately; the cache flushes to the DB asynchronously (batched).

**Comparison**

| Property | Cache-aside | Write-through | Write-back |
|---|---|---|---|
| Cache populated by | Read misses | Writes | Writes |
| Write latency | DB latency | DB + cache (highest) | Cache only (lowest) |
| Staleness risk | Yes (until invalidated) | Low | Low for cache reads |
| Data-loss risk | None | None | Yes (unflushed on crash) |
| Cold-cache penalty | High (every miss hits DB) | High | High |
| Complexity | Low | Medium | High (durability handling) |

**When to use each**
- **Cache-aside**: read-heavy workloads where occasional staleness is fine and you want the cache decoupled from the DB. The default for most web apps (Redis + a SQL DB).
- **Write-through**: read-after-write consistency matters and write latency is acceptable. Good when reads vastly outnumber writes and you can't tolerate serving stale data right after a write.
- **Write-back**: write-heavy, latency-sensitive workloads that can tolerate some data-loss risk (metrics, counters, buffering). Often paired with replication/AOF to bound loss.

**Tradeoffs to weigh**
Cache-aside has the famous *invalidation race* (a read repopulates a stale value between DB update and cache delete) — mitigate with short TTLs or versioned keys. Write-back trades durability for speed; never use it for data you can't reconstruct. Write-through wastes cache space if you write data that's never read.

**Real-world**: CDNs are essentially cache-aside; database buffer pools are write-back; many CQRS read models behave like write-through projections.""",
    },
    # 4 -------------------------------------------------------------------
    {
        "prompt": "SQL vs NoSQL: how should I actually decide which to use for a new service, beyond the usual 'it depends'?",
        "answer": """<thinking>
The honest answer is that "SQL vs NoSQL" is a false binary — the real questions are about your access patterns, consistency needs, and how your data is shaped. I should replace the vague dichotomy with concrete decision drivers and name the sub-families of NoSQL, because "NoSQL" lumps together very different tools.
</thinking>

**Reframe the question**
"NoSQL" isn't one thing. It includes document stores (MongoDB), wide-column (Cassandra), key-value (DynamoDB, Redis), and graph (Neo4j) databases — each optimized for different access patterns. The useful comparison is relational vs. a *specific* NoSQL family.

**Decision drivers**

| Driver | Favors relational (SQL) | Favors NoSQL |
|---|---|---|
| Data shape | Highly relational, many-to-many | Hierarchical/denormalized, self-contained docs |
| Query flexibility | Ad-hoc joins, aggregations | Known, fixed access patterns |
| Consistency | Strong transactions (ACID) needed | Eventual consistency acceptable |
| Scale axis | Vertical, moderate horizontal | Massive horizontal write scale |
| Schema | Stable, enforced | Rapidly evolving / heterogeneous |

**Key principles**
- **Model around access patterns** for NoSQL: you design the schema to serve specific queries, because joins are expensive or unavailable. With SQL you normalize first and query flexibly later.
- **ACID vs BASE**: relational DBs give you atomic multi-row transactions and strong consistency; many NoSQL stores trade that for availability and partition tolerance (CAP).
- **Don't pick NoSQL for scale you don't have.** A well-indexed Postgres handles millions of rows comfortably and now offers JSONB for document-style flexibility — often the pragmatic middle ground.

**When NoSQL clearly wins**
- Single-digit-millisecond key lookups at huge scale (DynamoDB for shopping carts).
- Time-series / write-heavy ingest with predictable queries (Cassandra).
- Naturally graph-shaped data with deep traversals (Neo4j for fraud/recommendations).

**When SQL clearly wins**
- Financial/transactional integrity, reporting, and evolving ad-hoc queries.
- Strong relational integrity with foreign keys and constraints.

**Real-world**: most systems are *polyglot* — Postgres as the system of record, Redis for caching/sessions, Elasticsearch for search, and maybe DynamoDB for one hot, well-understood access pattern. Start relational; introduce a specialized store only when a measured access pattern demands it.""",
    },
    # 5 -------------------------------------------------------------------
    {
        "prompt": "Explain REST vs GraphQL vs gRPC for API design. What are the decision criteria for each?",
        "answer": """<thinking>
These three solve overlapping problems but optimize for different consumers and constraints: REST for resource-oriented HTTP simplicity, GraphQL for flexible client-driven queries, gRPC for high-performance service-to-service calls. I'll compare on transport, payload, evolution, and tooling, then map to scenarios.
</thinking>

**The three paradigms**
- **REST** — Resource-oriented over HTTP/JSON; server defines endpoints, client composes URLs. Stateless, cacheable, ubiquitous.
- **GraphQL** — A single endpoint with a typed schema; the *client* specifies exactly which fields it wants in one query.
- **gRPC** — Contract-first RPC over HTTP/2 using Protocol Buffers (binary); strongly typed, streaming-capable.

**Comparison**

| Dimension | REST | GraphQL | gRPC |
|---|---|---|---|
| Transport / format | HTTP / JSON (text) | HTTP / JSON | HTTP/2 / Protobuf (binary) |
| Data fetching | Fixed per endpoint (over/under-fetch) | Client-shaped, exact fields | Fixed per RPC method |
| Caching | Native (HTTP caching) | Hard (POST, custom layer) | Manual |
| Browser support | Native | Native | Needs proxy (grpc-web) |
| Streaming | Limited (SSE/polling) | Subscriptions | First-class bidi streaming |
| Best consumer | Public/3rd-party APIs | Rich UIs, mobile, aggregation | Internal microservices |

**Decision criteria**
- **Choose REST** when you want a simple, cacheable, broadly compatible public API, or your resources map cleanly to CRUD. The lowest cognitive and operational overhead.
- **Choose GraphQL** when diverse clients (web/mobile) need different data shapes, you're aggregating multiple backends (BFF pattern), or over/under-fetching is hurting mobile performance. You pay with query-complexity/N+1 and caching challenges.
- **Choose gRPC** for low-latency, high-throughput internal service-to-service communication where both ends are yours and you want strict contracts and streaming. Avoid for public/browser-facing APIs.

**Tradeoffs**
GraphQL moves complexity to the server (query planning, depth limiting, the N+1 problem solved via dataloaders). gRPC's binary efficiency costs you human-readability and browser-friendliness. REST's simplicity costs you precise data fetching and forces versioning discipline.

**Real-world**: GitHub offers both REST and GraphQL; Netflix and Google use gRPC internally; most companies expose REST publicly while using gRPC between services and GraphQL as a BFF for their apps.""",
    },
    # 6 -------------------------------------------------------------------
    {
        "prompt": "What is the difference between horizontal and vertical scaling, and why do most large systems eventually need horizontal scaling?",
        "answer": """<thinking>
Vertical = bigger machine; horizontal = more machines. The deeper point is that vertical scaling hits hard physical and economic ceilings and offers no fault tolerance, while horizontal scaling introduces distributed-systems complexity in exchange for near-unbounded capacity and resilience. I should cover the ceilings, the new problems horizontal introduces, and what makes a system horizontally scalable (statelessness).
</thinking>

**Definitions**
- **Vertical scaling (scale up)** — Add resources (CPU, RAM, faster disk) to a single node.
- **Horizontal scaling (scale out)** — Add more nodes and distribute load across them.

**Comparison**

| Property | Vertical | Horizontal |
|---|---|---|
| Ceiling | Hardware limits (finite) | Practically unbounded |
| Fault tolerance | None (single point of failure) | Built-in (lose a node, survive) |
| Complexity | Low (no code changes) | High (distribution, coordination) |
| Cost curve | Super-linear at the high end | Roughly linear with commodity HW |
| Downtime to scale | Often requires restart | Add nodes live |

**Why large systems go horizontal**
1. **Physical ceiling** — There's a biggest machine you can buy; beyond it you must add machines.
2. **Cost** — Top-end hardware is exponentially more expensive per unit of capacity than commodity boxes.
3. **Availability** — A single big server is a single point of failure. Horizontal gives redundancy; this is usually the real driver, not raw capacity.

**What it requires**
Horizontal scaling demands **stateless services** (no in-memory session affinity), so any node can serve any request behind a load balancer. State is pushed to shared stores (databases, Redis, object storage). The database often becomes the bottleneck, addressed with read replicas, sharding, or caching.

**New problems it introduces**
Distributed systems bring consistency challenges (CAP), network partitions, the need for idempotency, distributed tracing, and coordination overhead. This is why you scale up *first* (cheap, simple) and scale out only when you hit ceilings or need HA.

**Real-world**: a startup runs on one beefy DB + a couple of app servers (mostly vertical). At scale, app servers go fully horizontal (trivially, since stateless), and the DB layer is the hard part — solved with replicas, sharding (Vitess), or distributed databases (Spanner, CockroachDB).""",
    },
    # 7 -------------------------------------------------------------------
    {
        "prompt": "Explain the CAP theorem and the common misconceptions about it.",
        "answer": """<thinking>
CAP is widely cited and widely misunderstood. The biggest misconception is "pick 2 of 3" as if you freely choose among C, A, P at all times. Reality: partitions (P) are not optional in a distributed system, so the real choice is C vs A *during a partition*. I should state it precisely, debunk the myths, and connect to PACELC which captures the no-partition tradeoff CAP ignores.
</thinking>

**The theorem**
In a distributed data store, during a **network partition** you can guarantee at most one of:
- **Consistency (C)** — every read sees the latest write (linearizability).
- **Availability (A)** — every request gets a non-error response.

Partition tolerance (P) means the system keeps operating despite dropped/delayed messages between nodes.

**The key correction**
You do **not** get to "pick 2 of 3." In any real distributed system, network partitions *will* happen — P is a given, not a choice. So CAP reduces to a single decision: **when a partition occurs, do you sacrifice consistency or availability?**
- **CP systems** reject/block requests to avoid serving stale data (e.g. HBase, ZooKeeper, etcd).
- **AP systems** stay available but may return stale data (e.g. Cassandra, DynamoDB in eventually-consistent mode).

**Common misconceptions**
1. *"Pick any 2."* — No; P is mandatory, so it's C vs A under partition.
2. *"It's a permanent property of the database."* — Many systems are tunable per-operation (DynamoDB, Cassandra quorums).
3. *"CA databases exist."* — Only on a single node / no network; not meaningful for distributed systems.
4. *"Consistency = ACID's C."* — No; CAP's C is linearizability, a different concept.

**PACELC — the fuller picture**
CAP only describes behavior *during* a partition. PACELC adds: **E**lse (no partition), you still trade **L**atency vs **C**onsistency. A system might be PA/EL (Cassandra: available under partition, low-latency otherwise) or PC/EC (strongly consistent always). This better captures everyday tradeoffs, since partitions are rare but latency-vs-consistency is constant.

**Real-world guidance**: choose CP for systems of record where correctness is non-negotiable (inventory, balances, leader election). Choose AP for high-availability, write-heavy, geo-distributed data where eventual consistency is acceptable (shopping carts, social feeds, telemetry).""",
    },
    # 8 -------------------------------------------------------------------
    {
        "prompt": "What's the difference between session-based authentication and JWT-based (token) authentication? When is each appropriate?",
        "answer": """<thinking>
The core tradeoff is *stateful server-side sessions vs. stateless self-contained tokens*. Sessions are easy to revoke but require shared state; JWTs scale statelessly but are hard to revoke before expiry. I should cover the mechanics, the revocation problem (the crux), and where each fits — and dispel the myth that JWTs are strictly more modern/better.
</thinking>

**Session-based**
Server creates a session record (in memory/Redis/DB) and hands the client an opaque session ID (usually a cookie). Every request, the server looks up the session to resolve identity and state.

**JWT-based**
Server issues a signed token containing the claims (user id, roles, expiry). The client sends it on each request; the server *verifies the signature* and trusts the contents without a lookup — no server-side session store needed.

**Comparison**

| Property | Sessions | JWT |
|---|---|---|
| Server state | Stateful (lookup each request) | Stateless (verify signature) |
| Revocation | Easy (delete the record) | Hard (valid until expiry) |
| Horizontal scaling | Needs shared session store | Trivial (no shared state) |
| Payload size | Tiny (just an ID) | Larger (claims travel each request) |
| Cross-domain / mobile | Cookie friction | Clean (Authorization header) |

**The crux: revocation**
A session can be killed instantly server-side. A JWT, once issued, is valid until it expires — you can't easily "log someone out" or revoke a compromised token without adding back state (a denylist or token-version check), which partly defeats the stateless benefit. This is why robust JWT systems use **short-lived access tokens + long-lived refresh tokens**: the access token expires in minutes, and revocation is enforced at refresh time.

**When to use each**
- **Sessions**: traditional server-rendered web apps, monoliths, when instant revocation and simplicity matter. A single Redis-backed session store scales fine for most apps.
- **JWT**: stateless microservices, APIs consumed by mobile/SPAs, service-to-service auth, and federated/OAuth scenarios where you don't want a central session lookup.

**Tradeoff summary**: JWTs aren't universally better — they trade easy revocation and small payloads for statelessness. Choose based on whether your pain is *scaling shared state* (favor JWT) or *needing instant control over sessions* (favor sessions). Many systems combine both: opaque session for the web frontend, JWTs between services.""",
    },
    # 9 -------------------------------------------------------------------
    {
        "prompt": "Explain the difference between monolithic and microservices architectures, and warn me about premature microservices.",
        "answer": """<thinking>
This is as much an organizational decision as a technical one (Conway's Law). The common failure is adopting microservices for prestige before you have the operational maturity or the domain understanding to draw boundaries — distributed monoliths are the result. I'll compare, then be direct about when NOT to use microservices.
</thinking>

**Definitions**
- **Monolith** — One deployable unit containing all functionality; modules call each other in-process.
- **Microservices** — Many small, independently deployable services, each owning a bounded context, communicating over the network.

**Comparison**

| Dimension | Monolith | Microservices |
|---|---|---|
| Deployment | One unit | Independent per service |
| Scaling | Whole app together | Per-service, targeted |
| Tech diversity | Usually one stack | Per-service freedom |
| Operational overhead | Low | High (orchestration, observability) |
| Failure isolation | Weak (one bug can crash all) | Strong (bulkheads) |
| Data | Shared DB, easy transactions | DB-per-service, distributed txns |
| Team autonomy | Coupled releases | Independent teams |

**Why microservices appeal**
Independent deployment, fault isolation, targeted scaling, and team autonomy — they let large orgs ship in parallel without coordinating one giant release.

**The premature-microservices trap**
Microservices solve *organizational and scale* problems and introduce *engineering* ones: network latency and failures, distributed transactions (sagas), eventual consistency, service discovery, distributed tracing, and far harder local development. If you adopt them before you understand your domain boundaries, you get a **distributed monolith** — services so chatty and coupled that you have all the distributed-systems pain with none of the independence.

**Guidance**
- **Start with a (well-modularized) monolith.** It's faster to build, easier to refactor, and lets boundaries emerge from real usage. Keep modules clean so extraction is cheap later.
- **Extract a service when you have a concrete reason**: a part needs independent scaling, a separate team needs autonomy, or a component has a genuinely different lifecycle.
- **Prerequisites for microservices**: solid CI/CD, observability (logs/metrics/traces), and infra automation. Without them, the operational tax is brutal.

**Real-world**: Amazon and Netflix run thousands of services — but they got there by *evolving* monoliths, with the platform tooling to support it. Most companies are best served by a "modular monolith" until clear scaling or team-boundary pressure justifies splitting.""",
    },
    # 10 ------------------------------------------------------------------
    {
        "prompt": "What is idempotency in API design, why does it matter, and how do I implement idempotent operations?",
        "answer": """<thinking>
Idempotency is the property that doing an operation multiple times has the same effect as doing it once. It matters because networks are unreliable: clients retry, and without idempotency a retry can double-charge a card or create duplicate orders. I'll cover which HTTP methods are idempotent by spec, the at-least-once delivery reality, and the idempotency-key pattern for making non-idempotent operations safe.
</thinking>

**Definition**
An operation is **idempotent** if applying it multiple times produces the same result as applying it once. The *response* may differ, but the *state* converges.

**Why it matters**
Distributed systems guarantee at best *at-least-once* delivery. Timeouts and retries mean the same request can arrive twice. Without idempotency, a retried "charge $50" becomes "charge $100." Idempotency makes retries *safe*, which is the foundation of reliable distributed communication.

**HTTP methods by spec**

| Method | Idempotent? | Why |
|---|---|---|
| GET | Yes | Read-only |
| PUT | Yes | Full replace → same end state |
| DELETE | Yes | Deleting twice = still deleted |
| POST | **No** | Creates a new resource each call |
| PATCH | Depends | Idempotent only if the patch is absolute, not relative |

**The problem child: POST**
"Create order" / "charge payment" are inherently non-idempotent. The standard fix is the **idempotency key**:
1. Client generates a unique key (UUID) per logical operation and sends it in a header.
2. Server records the key with the result on first processing.
3. On a retry with the same key, the server returns the stored result *without re-executing* the side effect.
4. Keys expire after a window (e.g. 24h).

This requires atomic "check-and-store" (a unique constraint or a transaction) to avoid races where two concurrent retries both think they're first.

**Other techniques**
- Make operations naturally idempotent: "set status to shipped" instead of "increment counter."
- Use conditional requests (ETags / `If-Match`) for optimistic concurrency.
- Dedupe at the consumer for message queues using a processed-message log.

**Real-world**: Stripe's API requires an `Idempotency-Key` header on charges precisely so a network blip never double-charges a customer. Payment systems, order placement, and any "exactly-once-feeling" workflow depend on this pattern.""",
    },
    # 11 ------------------------------------------------------------------
    {
        "prompt": "Explain the Repository pattern and how it differs from a Data Access Object (DAO). Is the abstraction worth it?",
        "answer": """<thinking>
These overlap and people use the terms interchangeably, but there's a meaningful distinction in intent: a DAO abstracts a *data source / table*, while a Repository abstracts a *collection of domain objects* and lives in the domain layer. The "is it worth it" question is the contested part — there's a real critique that Repository over an ORM is redundant. I'll be balanced.
</thinking>

**Repository pattern**
Mediates between the domain and data-mapping layers, presenting a *collection-like* interface for domain objects: `add(order)`, `findById(id)`, `remove(order)`. The domain code talks to an interface and is ignorant of persistence details.

**DAO**
A lower-level abstraction over a specific data source — typically one DAO per table — exposing CRUD operations close to the storage shape. Returns data structures/records rather than rich domain aggregates.

**Key differences**

| Aspect | DAO | Repository |
|---|---|---|
| Granularity | Per table / data source | Per aggregate / domain concept |
| Mental model | "Database operations" | "In-memory collection of objects" |
| Layer | Persistence layer | Domain layer |
| Returns | Records / DTOs | Domain entities / aggregates |
| Origin | Java/persistence tradition | Domain-Driven Design |

In practice a Repository may *use* DAOs internally; DAO is table-centric, Repository is domain-centric.

**Is the abstraction worth it?**
**Benefits**: testability (mock the repository, no DB in unit tests), a single place for query logic, swappable persistence, and a domain layer free of SQL/ORM leakage.

**The critique**: ORMs like Hibernate or Entity Framework already implement the Unit of Work and a repository-ish abstraction. Wrapping them in a hand-rolled Repository can be a leaky, redundant layer that mostly forwards calls — and worse, it tempts you to add a generic `getAll`/`query` that re-exposes the ORM and defeats the encapsulation.

**Guidance**
- **Worth it** when you have a rich domain model (DDD), need to keep the domain pure and testable, or genuinely anticipate swapping persistence.
- **Skip it (or keep it thin)** for simple CRUD apps where the ORM's own abstractions suffice — a Repository there is ceremony.
- If you do use it, keep the interface *intention-revealing* (`findActiveSubscribers()`), not a generic query passthrough.

**Real-world**: Spring Data and EF Core provide repository abstractions out of the box; DDD-heavy systems define explicit repositories per aggregate root to enforce consistency boundaries.""",
    },
    # 12 ------------------------------------------------------------------
    {
        "prompt": "What are the main differences between OAuth 2.0 and OpenID Connect, and what problem does each actually solve?",
        "answer": """<thinking>
The single most common confusion: OAuth is for *authorization* (delegated access), not *authentication*, yet people use it to log users in. OIDC is the layer that adds authentication on top of OAuth correctly. I should explain what each provides, why using bare OAuth for login is a known anti-pattern, and the token types involved.
</thinking>

**OAuth 2.0 — delegated authorization**
OAuth lets a user grant a *third-party app* limited access to their resources on another service *without sharing credentials*. The output is an **access token** that says "this app may call these APIs on this user's behalf." It answers *what an app can do*, not *who the user is*.

**OpenID Connect (OIDC) — authentication layer on top**
OIDC is a thin identity layer built on OAuth 2.0. It adds an **ID token** (a JWT with verified identity claims: subject, email, name) and a standard `/userinfo` endpoint. It answers *who the user is* in a standardized, verifiable way.

**Tokens**

| Token | Spec | Purpose | Audience |
|---|---|---|---|
| Access token | OAuth | Authorize API calls | The resource server |
| ID token | OIDC | Prove identity | The client app |
| Refresh token | OAuth | Get new access tokens | The auth server |

**The "OAuth for login" anti-pattern**
Plain OAuth gives you an access token, which proves an app *can access a resource* — not *who the user is*. Apps historically inferred identity by calling a provider-specific "me" API with the access token, but this is non-standard and has subtle security pitfalls (e.g. token-substitution / confused-deputy issues). OIDC standardizes identity with a *signed, audience-bound* ID token, closing those gaps. **Rule: use OIDC for authentication, OAuth for authorization.**

**Flows**
The recommended flow today is **Authorization Code with PKCE** for both web and mobile/SPA, which protects against authorization-code interception. The Implicit flow is deprecated; client-credentials is for machine-to-machine.

**Real-world**: "Sign in with Google/Apple/Microsoft" is OIDC. "Allow this app to access your Google Calendar" is OAuth authorization. Identity providers (Auth0, Okta, Keycloak) implement both — you typically want OIDC for SSO and OAuth scopes for API access.""",
    },
    # 13 ------------------------------------------------------------------
    {
        "prompt": "Explain database indexing: how do B-tree indexes work, when do they help, and when can they hurt?",
        "answer": """<thinking>
The mental model is the index of a book vs. reading every page. B-trees keep data sorted and balanced so lookups are logarithmic. But indexes aren't free: they cost write performance and storage, and the optimizer can ignore them if they're not selective. I should cover how B-trees enable range + equality + ordering, the write tradeoff, composite-index column order, and cases where indexes hurt.
</thinking>

**How a B-tree index works**
A B-tree is a balanced, sorted tree. Each node holds keys and pointers; leaf nodes point to (or contain) rows. Because it stays balanced, lookups, insertions, and range scans are **O(log n)** instead of the O(n) of a full table scan. The sorted structure is why one B-tree serves equality (`=`), range (`<`, `BETWEEN`), prefix (`LIKE 'abc%'`), and `ORDER BY` efficiently.

**When indexes help**
- High-selectivity filters on large tables (return a small fraction of rows).
- Join columns and foreign keys.
- Columns used for sorting and range queries.
- *Covering indexes* — when the index includes all columns a query needs, the DB never touches the table (index-only scan).

**Composite index column order matters**
A `(a, b, c)` index supports queries filtering on `a`, `a+b`, or `a+b+c` (a left-prefix), but **not** `b` alone. Order columns by: equality filters first, then range, then sort columns. Getting this wrong means the index goes unused.

**When indexes hurt**
- **Writes slow down**: every INSERT/UPDATE/DELETE must also update every affected index. Over-indexing a write-heavy table is a real cost.
- **Storage**: indexes can rival the table in size.
- **Low selectivity**: an index on a boolean or a column where most rows share a value is often worse than a scan — the optimizer may correctly ignore it.
- **Maintenance**: indexes fragment and need occasional rebuilds; stale statistics lead the planner astray.

**Tradeoff summary**
Indexing trades write throughput and storage for read speed. The art is indexing exactly the columns your real queries filter/sort on — no more. Use `EXPLAIN`/`EXPLAIN ANALYZE` to confirm the planner actually uses an index and isn't doing a sequential scan.

**Real-world**: a slow query is most often a missing index on a WHERE/JOIN column; a slow write-heavy ingest path is often *too many* indexes. Specialized cases use other structures — hash indexes for pure equality, GIN/GiST for full-text and JSON, BRIN for naturally-ordered huge tables.""",
    },
    # 14 ------------------------------------------------------------------
    {
        "prompt": "Compare optimistic vs pessimistic locking for concurrency control. How do I choose?",
        "answer": """<thinking>
The decision hinges on contention: pessimistic locking assumes conflicts are likely and prevents them by locking up front (cost: blocking, deadlocks, reduced throughput); optimistic assumes conflicts are rare and detects them at commit (cost: retries on conflict). I'll cover mechanics, the version/ETag approach, and the contention-based decision rule.
</thinking>

**Pessimistic locking**
Assume conflicts are likely, so *prevent* them: acquire a lock on the data before reading/modifying it (`SELECT ... FOR UPDATE`). Other transactions block until you release. Correctness is guaranteed by exclusion.

**Optimistic locking**
Assume conflicts are rare, so *detect* them: read freely, and at write time verify nothing changed since you read — typically via a `version` column or timestamp. If the version changed, the write fails and the caller retries.

**Comparison**

| Property | Pessimistic | Optimistic |
|---|---|---|
| Assumption | Conflicts likely | Conflicts rare |
| Mechanism | Lock before access | Version check at commit |
| Cost when no conflict | Lock overhead, reduced concurrency | None |
| Cost when conflict | None (you waited) | Retry / merge work |
| Risks | Deadlocks, blocking, timeouts | Wasted work, retry storms |
| Throughput | Lower under load | Higher under low contention |

**How to choose — it's about contention**
- **High contention** (many writers hitting the same rows, e.g. inventory for a hot item, seat booking): pessimistic locking avoids endless retry loops. Optimistic here can livelock as everyone keeps colliding.
- **Low contention** (writes rarely overlap, e.g. a user editing their own profile): optimistic locking gives better throughput and avoids holding locks across user think-time.
- **Long-lived / web transactions**: prefer optimistic — you must *never* hold a DB lock while waiting on a human (it kills concurrency and risks timeouts). The version-number/ETag pattern is the standard for "edit form" workflows.

**Tradeoffs**
Pessimistic risks deadlocks and serializes access, hurting scalability. Optimistic shifts the cost to conflict handling — you need a retry/merge strategy, and under high contention it degrades badly.

**Real-world**: most web apps use optimistic concurrency (Hibernate's `@Version`, HTTP ETags with `If-Match`) because requests span user interaction. Pessimistic locking is reserved for short, hot, server-side critical sections like decrementing limited stock.""",
    },
    # 15 ------------------------------------------------------------------
    {
        "prompt": "What is eventual consistency, and how do I design systems and UX around it without confusing users?",
        "answer": """<thinking>
Eventual consistency means replicas converge given no new writes, but reads in the meantime may be stale. The engineering challenge is real, but the *UX* challenge is underappreciated — users get confused when they write data and don't immediately see it. I'll cover the guarantee, the consistency-model spectrum, and concrete patterns (read-your-writes, optimistic UI) to hide staleness.
</thinking>

**Definition**
**Eventual consistency** guarantees that if no new updates are made, all replicas will *eventually* converge to the same value. It does **not** guarantee that a read right after a write sees that write. It's the price of high availability and low latency in distributed/replicated systems (AP in CAP terms).

**The consistency spectrum**

| Model | Guarantee |
|---|---|
| Strong / linearizable | Reads always see the latest write |
| Read-your-writes | A client sees its *own* writes immediately |
| Monotonic reads | A client never sees data go "backwards" |
| Eventual | Converges sometime; no ordering guarantees |

Often you don't need full strong consistency — a weaker *session guarantee* like read-your-writes is enough for good UX.

**Designing the system**
- Use **quorum reads/writes** (R + W > N) when you need stronger guarantees on specific operations.
- Resolve conflicts deliberately: last-write-wins (simple, lossy), vector clocks, or CRDTs (mergeable, no lost updates — great for collaborative editing).
- Make operations **idempotent and commutative** so reordering doesn't corrupt state.

**Designing the UX — the underrated part**
- **Optimistic UI**: update the screen immediately as if the write succeeded, then reconcile when the server confirms (and roll back on failure). This hides replication lag entirely for the actor.
- **Read-your-writes routing**: pin a user to the primary/freshest replica for a short window after they write, so *they* always see their own change even if other users lag.
- **Honest affordances**: "Saving…", "Pending", or relative timestamps signal in-flight state instead of pretending everything is instantaneous.

**Real-world**: DNS is eventually consistent (propagation delay). Amazon's cart, social media like counts, and S3 historically used eventual consistency. Collaborative tools (Google Docs, Figma) use CRDTs/OT to make concurrent edits converge cleanly. The trick is matching the consistency model to each operation — strong for money, eventual for likes.""",
    },
    # 16 ------------------------------------------------------------------
    {
        "prompt": "Explain the Circuit Breaker pattern and how it differs from simple retries. Why are naive retries dangerous?",
        "answer": """<thinking>
Retries and circuit breakers are complementary resilience patterns, but naive retries actively make outages worse (retry storms, thundering herd amplifying load on an already-struggling service). The circuit breaker stops hammering a failing dependency. I'll explain the three states, contrast with retries, and cover exponential backoff + jitter as the safe retry strategy.
</thinking>

**Why naive retries are dangerous**
When a downstream service is overloaded, immediate retries *add* load to a system that's already failing — a **retry storm**. Synchronized retries cause a **thundering herd** that can keep a recovering service down. Retries also amplify load multiplicatively across a call chain (3 services each retrying 3x = 27x load). Retrying without limits or delay turns a brief blip into a cascading outage.

**Safe retries**
- **Exponential backoff**: wait 1s, 2s, 4s… between attempts to give the dependency room to recover.
- **Jitter**: randomize the delay so clients don't all retry in lockstep (decorrelated jitter).
- **Bounded attempts** and only retry *idempotent*, *transient* failures (timeouts, 503) — never retry a 400.

**Circuit Breaker**
A breaker wraps a call to a dependency and tracks failures. It has three states:
- **Closed** — calls flow normally; failures are counted.
- **Open** — once failures cross a threshold, the breaker trips and *fails fast* for a cooldown period, not even attempting the call. This sheds load from the struggling dependency and gives callers instant errors instead of piling up on timeouts.
- **Half-open** — after cooldown, it lets a trickle of test requests through; success closes the circuit, failure re-opens it.

**Retries vs Circuit Breaker**

| | Retries | Circuit Breaker |
|---|---|---|
| Goal | Recover from *transient* failures | Protect against *sustained* failures |
| Behavior | Try again | Stop trying, fail fast |
| Risk if naive | Amplifies load | (Designed to reduce load) |
| Scope | Per request | Per dependency, shared state |

They're **complementary**: retry transient blips with backoff; trip the breaker when failures are sustained so you stop retrying into a wall.

**Companion patterns**: bulkheads (isolate resource pools so one failing dependency can't exhaust all threads), timeouts (never wait forever), and fallbacks (serve cached/default data when the breaker is open).

**Real-world**: Netflix's Hystrix popularized this; Resilience4j, Polly, and service meshes (Istio/Envoy) provide breakers + backoff today. Essential for any service with network dependencies.""",
    },
    # 17 ------------------------------------------------------------------
    {
        "prompt": "What's the difference between message queues and publish/subscribe (pub/sub)? When do I use each?",
        "answer": """<thinking>
The core distinction is delivery semantics: a queue delivers each message to exactly one consumer (work distribution), while pub/sub broadcasts each message to all subscribers (event fan-out). People blur these because brokers like Kafka and RabbitMQ can do both. I'll separate the patterns, then map to use cases, and touch on ordering/competing consumers.
</thinking>

**Message Queue (point-to-point)**
A producer puts messages on a queue; **one** consumer (from a pool of competing consumers) processes each message. The goal is **work distribution and load leveling** — spread tasks across workers, and buffer bursts so a slow consumer doesn't drop work.

**Publish/Subscribe (fan-out)**
A publisher emits an event to a topic; **every** subscriber gets its own copy. The goal is **decoupled broadcast** — one event triggers many independent reactions, and the publisher doesn't know or care who's listening.

**Comparison**

| Property | Queue | Pub/Sub |
|---|---|---|
| Delivery | One consumer per message | All subscribers get a copy |
| Coupling | Producer ↔ worker pool | Fully decoupled (N subscribers) |
| Primary use | Task/work distribution | Event broadcast, fan-out |
| Scaling consumers | Add workers → more throughput | Add subscribers → more reactions |
| Mental model | To-do list | Newsletter |

**When to use a queue**
- Background jobs (send email, resize image, process payment).
- Smoothing traffic spikes (buffer requests for downstream).
- Ensuring each task is done once by one worker (competing consumers).

**When to use pub/sub**
- One event must trigger many independent services (e.g. "OrderPlaced" → inventory, billing, notifications, analytics each react).
- Event-driven architecture / decoupling producers from an unknown set of consumers.
- Audit logs, cache invalidation broadcasts, real-time notifications.

**Nuances**
- **Ordering**: queues often preserve FIFO per queue; pub/sub ordering is per-partition/topic and harder globally. Kafka guarantees order *within a partition* only.
- **Consumer groups** (Kafka) blur the line: a topic delivers each message once *per group* (pub/sub across groups, queue-like within a group).
- **Delivery guarantees**: at-least-once is the norm → consumers must be idempotent.

**Real-world**: RabbitMQ/SQS shine at queue-style work distribution; Kafka/Google Pub/Sub/SNS excel at high-throughput event streaming and fan-out. Many systems use both — a queue to dispatch work, pub/sub to broadcast domain events.""",
    },
    # 18 ------------------------------------------------------------------
    {
        "prompt": "Explain the difference between symmetric and asymmetric encryption, and why TLS uses both.",
        "answer": """<thinking>
The key insight is a tradeoff: symmetric is fast but has a key-distribution problem; asymmetric solves key distribution but is slow. TLS combines them — asymmetric to securely establish a shared symmetric key, then symmetric for the bulk data. I should also separate confidentiality from authenticity/signatures, and mention forward secrecy.
</thinking>

**Symmetric encryption**
One shared secret key both encrypts and decrypts (AES, ChaCha20). **Fast** and efficient for large data. The problem: both parties must already share the secret key — and securely distributing that key over an insecure network is the hard part.

**Asymmetric encryption**
A mathematically linked **key pair**: a public key (shareable freely) and a private key (kept secret). Data encrypted with the public key can only be decrypted with the private key — and vice versa for signatures (RSA, ECC). This **solves key distribution**: anyone can encrypt to you using your public key. But it's computationally **expensive** and impractical for bulk data.

**Comparison**

| Property | Symmetric | Asymmetric |
|---|---|---|
| Keys | One shared secret | Public + private pair |
| Speed | Fast | Slow (orders of magnitude) |
| Key distribution | Hard (the core problem) | Solved (publish the public key) |
| Use | Bulk data encryption | Key exchange, signatures |

**Why TLS uses both (hybrid)**
TLS gets the best of each:
1. **Handshake (asymmetric)**: the client and server use asymmetric crypto / Diffie-Hellman to *authenticate the server* (via its certificate) and *securely agree on a shared symmetric session key* — without ever transmitting that key in the clear.
2. **Bulk transfer (symmetric)**: all subsequent application data is encrypted with the fast symmetric session key.

So asymmetric solves the chicken-and-egg key-distribution problem *once* per connection; symmetric handles the high-volume traffic efficiently.

**Related concepts**
- **Digital signatures** use asymmetric crypto in reverse (sign with private, verify with public) for *authenticity and integrity* — this is how certificates prove a server's identity.
- **Forward secrecy** (ephemeral Diffie-Hellman) ensures each session uses a fresh key, so compromising the server's long-term private key later can't decrypt past recorded sessions.

**Real-world**: every HTTPS connection, SSH, signed software packages, and end-to-end messaging (Signal) all rely on this hybrid model — asymmetric to bootstrap trust and keys, symmetric for throughput.""",
    },
    # 19 ------------------------------------------------------------------
    {
        "prompt": "What is the N+1 query problem, how do I detect it, and what are the ways to fix it?",
        "answer": """<thinking>
N+1 is the canonical ORM performance bug: one query to fetch a list, then one additional query per item to fetch a relation. It's invisible in code but devastating at scale. I'll explain the mechanism, detection, and the fixes (eager loading/joins, batching/dataloader, projections), plus the tradeoff that over-eager fetching causes the opposite problem.
</thinking>

**The problem**
You fetch a list of N parent records with 1 query, then access a related field on each, triggering 1 query *per parent* — N+1 total. Example: load 100 blog posts (1 query), then access `post.author` in a loop (100 more queries) = 101 queries where 2 would do. The killer is that the code looks innocent — lazy-loaded relations fire queries silently as you iterate.

**Why it's so damaging**
Each query has fixed round-trip overhead. 101 small queries can be 50-100x slower than 1-2 well-formed ones, and it scales *with your data* — fine in dev with 10 rows, catastrophic in prod with 10,000.

**How to detect it**
- Enable SQL logging and watch for repeated, near-identical queries differing only by an id.
- Use query-count assertions in tests or tools (Django Debug Toolbar, Bullet, Hibernate statistics, n+1 detectors in APM like Datadog/New Relic).
- A request issuing hundreds of queries is the tell.

**Fixes**

| Technique | How it works | When |
|---|---|---|
| Eager loading / JOIN | Fetch parents + relations in one query (`JOIN FETCH`, `includes`, `selectinload`) | Known relations needed up front |
| Batching (DataLoader) | Collect ids, fetch all relations in one `WHERE id IN (...)` | GraphQL, dynamic access patterns |
| Projection / select_related | Pull only needed columns, avoid extra round-trips | Read-heavy list views |
| Caching | Serve hot relations from cache | Repeated reads of the same data |

**The opposite trap**
Over-eager loading causes the **Cartesian/over-fetch problem**: JOINing many to-many relations multiplies rows and pulls megabytes you don't need. The fix isn't "always eager" — it's "fetch exactly what this view needs." DataLoader-style batching (one query per relation type, not per row) is often the sweet spot.

**Real-world**: this is the single most common ORM performance bug across Django, Rails, Hibernate, and especially GraphQL resolvers (where Facebook's DataLoader was invented to solve it). The discipline is to make data access explicit and measure query counts.""",
    },
    # 20 ------------------------------------------------------------------
    {
        "prompt": "Explain Role-Based Access Control (RBAC) vs Attribute-Based Access Control (ABAC). When does RBAC stop scaling?",
        "answer": """<thinking>
RBAC groups permissions into roles; it's simple and auditable but suffers from "role explosion" when access depends on context (resource owner, time, location, department). ABAC evaluates policies over attributes and is far more expressive but harder to reason about and audit. I'll compare, explain where RBAC breaks down, and note hybrid approaches (ReBAC/Zanzibar).
</thinking>

**RBAC**
Permissions are bundled into **roles**; users are assigned roles; access = the union of their roles' permissions. Simple, intuitive, easy to audit ("who has the admin role?").

**ABAC**
Access decisions are computed by **policies evaluated over attributes** of the subject, resource, action, and environment — e.g. "allow if user.department == resource.department AND time is business hours AND user.clearance >= resource.classification." Highly dynamic and fine-grained.

**Comparison**

| Property | RBAC | ABAC |
|---|---|---|
| Basis | Static roles | Dynamic attributes/policies |
| Granularity | Coarse | Very fine, context-aware |
| Auditability | Easy ("list role members") | Hard (must evaluate policies) |
| Flexibility | Limited | Very high |
| Complexity | Low | High (policy engine needed) |
| Performance | Cheap lookups | Policy evaluation per request |

**When RBAC stops scaling — role explosion**
RBAC encodes context by *creating more roles*. The moment access depends on *which* resource, you can't have a global role — you'd need "editor-of-project-A," "editor-of-project-B," etc., and the number of roles explodes combinatorially with tenants, regions, departments, and resource instances. You also can't express conditions like "only during business hours" or "only for records you own" without contorting the model. This is the signal that you've outgrown pure RBAC.

**The middle ground**
- **RBAC + resource scoping**: roles *within* a context ("admin of *this* org") — covers most multi-tenant SaaS without full ABAC.
- **ABAC** when decisions genuinely depend on many runtime attributes (healthcare, finance, government clearance).
- **ReBAC (relationship-based)**: model access as a graph of relationships ("user is owner of doc, which is in folder shared with team"). Google's Zanzibar / OpenFGA / SpiceDB popularized this for fine-grained, scalable authorization.

**Guidance**: start with RBAC for its simplicity and auditability. Add resource-scoped roles as you go multi-tenant. Reach for ABAC/ReBAC and a dedicated policy engine (OPA, Cedar, OpenFGA) only when context-dependent rules make roles unmanageable.""",
    },
    # 21 ------------------------------------------------------------------
    {
        "prompt": "Compare server-side rendering (SSR), client-side rendering (CSR), and static site generation (SSG). How do I choose for a given page?",
        "answer": """<thinking>
This is really about *where and when* HTML is produced, which drives time-to-content, SEO, server cost, and interactivity. The modern answer is per-page/hybrid (and now streaming/RSC), not a whole-app choice. I'll compare on the axes that matter and give a per-page decision rule.
</thinking>

**The three approaches**
- **CSR** — Server sends a near-empty HTML shell + JS bundle; the browser fetches data and renders. Rendering happens *in the client, at request time*.
- **SSR** — Server renders full HTML *per request*, then the client hydrates it for interactivity. Rendering happens *on the server, at request time*.
- **SSG** — HTML is rendered *at build time* and served as static files (often via CDN). Rendering happens *ahead of time*.

**Comparison**

| Property | CSR | SSR | SSG |
|---|---|---|---|
| Render time | Client, on request | Server, on request | Build time |
| Time to first content | Slow (wait for JS) | Fast | Fastest (CDN) |
| SEO | Weak (needs JS) | Strong | Strong |
| Server load | Low | High (render per request) | None (static) |
| Data freshness | Real-time | Real-time | Stale until rebuild |
| Best for | App-like, auth'd dashboards | Dynamic + SEO content | Mostly-static content |

**Per-page decision rule**
- **SSG** for content that's the same for everyone and changes infrequently: marketing pages, blogs, docs. Cheapest, fastest, most cacheable.
- **SSR** for pages that are dynamic/personalized *and* need SEO or fast first paint: e-commerce product pages, news, search results.
- **CSR** for highly interactive, authenticated app shells where SEO doesn't matter and data is user-specific: dashboards, internal tools.

**Modern hybrids**
- **ISR (Incremental Static Regeneration)**: SSG that revalidates in the background — static speed with bounded freshness.
- **Streaming SSR + React Server Components**: send HTML in chunks and ship less JS, blurring SSR/CSR and reducing hydration cost.

**Tradeoffs**: CSR shifts work and latency to the client (bad on weak devices, bad for SEO) but is cheap to host. SSR gives great UX/SEO but costs server CPU per request and adds hydration complexity. SSG is unbeatable for speed/cost but can't show per-request data without a hybrid.

**Real-world**: Next.js/Nuxt let you pick per route — SSG the blog, SSR the product page, CSR the logged-in dashboard. The right answer is rarely one mode for the whole app.""",
    },
    # 22 ------------------------------------------------------------------
    {
        "prompt": "Explain database sharding: strategies, the problems it introduces, and when I should avoid it.",
        "answer": """<thinking>
Sharding is horizontal partitioning of data across nodes — the heavyweight scaling tool of last resort. It works, but it sacrifices cross-shard joins/transactions and makes operations far harder. I'll cover strategies (range, hash, directory/geo), the hard problems (rebalancing, hot shards, cross-shard queries), and emphasize exhausting cheaper options first.
</thinking>

**What sharding is**
Splitting a dataset *horizontally* across multiple database nodes (shards), each holding a subset of rows. Unlike read replicas (which copy all data), sharding distributes the data itself, scaling *writes* and storage beyond one machine.

**Strategies**

| Strategy | How | Strength | Weakness |
|---|---|---|---|
| Range-based | Partition by key ranges (A-M, N-Z) | Efficient range queries | Hot spots if data skews |
| Hash-based | Shard = hash(key) % N | Even distribution | Range queries scatter; resharding is painful |
| Directory/lookup | A service maps key → shard | Flexible, rebalanceable | Lookup is a bottleneck/SPOF |
| Geo / entity | By region or tenant | Locality, isolation | Uneven sizes |

**Choosing a shard key** is the most consequential decision: it must spread load evenly *and* keep related data co-located so common queries hit one shard. A bad key gives you **hot shards** (one node overloaded) or queries that fan out to every shard.

**Problems it introduces**
- **Cross-shard joins/transactions**: data on different shards can't be joined or transactionally updated easily — you need application-level joins or distributed-transaction patterns (sagas, 2PC), both costly.
- **Rebalancing**: adding a shard often requires moving data; hash-mod resharding is especially painful (consistent hashing helps).
- **Hot shards**: skew concentrates load; the celebrity/viral row problem.
- **Operational complexity**: backups, migrations, monitoring, and unique-id generation all get harder across N nodes.

**When to avoid it (almost always, at first)**
Sharding is a *last resort*. Before sharding, exhaust:
1. **Vertical scaling** — bigger box.
2. **Read replicas** — offload reads.
3. **Caching** — Redis in front of hot reads.
4. **Indexing / query tuning** — fix the real bottleneck.
5. **Archiving / partitioning within one DB** — table partitioning without distribution.

Only when *write* throughput or dataset size genuinely exceeds a single node should you shard — and pick the shard key based on real access patterns.

**Real-world**: large platforms shard (Instagram by user id, Slack/Notion by workspace). Tooling like Vitess (MySQL) automates much of it, and distributed SQL databases (CockroachDB, Spanner, YugabyteDB) shard transparently so you get horizontal scale without hand-rolling it — increasingly the preferred path over manual sharding.""",
    },
    # 23 ------------------------------------------------------------------
    {
        "prompt": "What is the Saga pattern for distributed transactions, and how does it compare to two-phase commit (2PC)?",
        "answer": """<thinking>
Across microservices with a DB per service, you can't use a single ACID transaction. 2PC offers atomicity but is blocking and fragile at scale; sagas trade atomicity for availability by breaking a transaction into local steps with compensating actions. I'll explain both saga styles (orchestration vs choreography), the compensation model, and the consistency tradeoff (no isolation → must handle intermediate states).
</thinking>

**The problem**
In microservices with a database per service, a business operation spanning services (place order → reserve inventory → charge payment) can't be a single ACID transaction. You need *distributed* coordination.

**Two-Phase Commit (2PC)**
A coordinator asks all participants to *prepare* (phase 1); if all vote yes, it tells them to *commit* (phase 2), else abort. It gives **atomicity** across services.
- **Problem**: it's **blocking** — participants hold locks while awaiting the coordinator's decision. If the coordinator dies mid-protocol, participants are stuck. This kills availability and scalability, so 2PC is rarely used in modern microservices.

**Saga pattern**
Break the distributed transaction into a sequence of **local transactions**, each in one service. If a step fails, run **compensating transactions** to semantically undo the prior steps (e.g. "refund payment," "release inventory"). There's no global lock — each local step commits independently.

Two coordination styles:
- **Orchestration**: a central orchestrator tells each service what to do and triggers compensations on failure. Easier to reason about and monitor; the orchestrator is a (logical) single point of control.
- **Choreography**: services emit events and react to each other's events, no central coordinator. More decoupled but harder to trace and reason about as the flow grows.

**Comparison**

| Property | 2PC | Saga |
|---|---|---|
| Atomicity | Yes (all-or-nothing) | No — *eventual* consistency via compensation |
| Locking | Holds locks (blocking) | No distributed locks |
| Availability | Poor (coordinator SPOF) | High |
| Failure recovery | Stuck if coordinator dies | Compensations roll back |
| Complexity | Protocol overhead | Compensation logic, intermediate states |

**The key tradeoff**
Sagas give up **isolation**: because steps commit independently, other transactions can observe *intermediate* states (an order that's created but not yet paid). You must design for this — use states like "PENDING," make compensations idempotent, and accept eventual consistency. Compensation isn't a true rollback; it's a *semantic* undo (a refund, not erasing history).

**Real-world**: e-commerce checkout, travel booking (flight + hotel + car), and most cross-service workflows use sagas, often via orchestrators like Temporal, AWS Step Functions, or Camunda. 2PC survives mainly in tightly-coupled, lower-scale systems (some XA/JTA enterprise setups) where its blocking nature is tolerable.""",
    },
    # 24 ------------------------------------------------------------------
    {
        "prompt": "How should I think about API versioning strategies (URL, header, media-type), and how do I avoid breaking clients?",
        "answer": """<thinking>
Versioning is fundamentally about managing change without breaking existing consumers. The strategies differ in visibility and HTTP-purity, but the deeper skill is distinguishing breaking from non-breaking changes and minimizing the need to version at all. I'll compare strategies, define what counts as breaking, and cover deprecation discipline.
</thinking>

**Why version at all**
Once external clients depend on your API, you can't change its contract freely — a breaking change orphans existing integrations. Versioning lets you evolve while honoring the old contract for a transition period.

**Strategies**

| Strategy | Example | Pros | Cons |
|---|---|---|---|
| URL path | `/v1/users` | Obvious, cache-friendly, easy to route | Pollutes URIs; "v2" duplicates routes |
| Query param | `/users?version=1` | Simple, optional | Easy to forget; caching quirks |
| Custom header | `X-API-Version: 1` | Clean URLs | Hidden, harder to test in a browser |
| Media-type (content negotiation) | `Accept: application/vnd.api.v1+json` | RESTful, per-resource | Complex, poor tooling/discoverability |

**Pragmatic take**: URL versioning is the most common and operationally simplest (clear, cacheable, debuggable in a browser). Header/media-type versioning is "more RESTful" but harder for consumers to discover and test. Choose based on your audience's sophistication; for most public APIs, `/v1/` wins.

**The real skill: minimize breaking changes**
Define what's breaking vs. additive:
- **Non-breaking (no version bump needed)**: adding a new endpoint, adding an *optional* field, adding an enum value clients ignore. Good clients tolerate unknown fields.
- **Breaking (needs a version)**: removing/renaming a field, changing a type, making an optional field required, changing semantics or error codes.

Design for evolvability so you rarely *need* a new version: be liberal in what you accept and conservative in what you emit, never repurpose fields, and make clients ignore unknown fields.

**Deprecation discipline**
When you must version: run old and new in parallel, announce a timeline, signal deprecation in responses (`Deprecation` / `Sunset` headers), monitor old-version usage, and only retire once traffic drains. Never break a live version without notice.

**Real-world**: Stripe versions by date and pins each account to the version it integrated against, transparently transforming responses — so old clients never break. GitHub, Twilio, and others use explicit `/vN` paths with long deprecation windows. The best version is the one you didn't need because you designed the API to grow additively.""",
    },
    # 25 ------------------------------------------------------------------
    {
        "prompt": "Explain the tradeoffs of denormalization in databases. When is duplicating data the right call?",
        "answer": """<thinking>
Normalization optimizes for write integrity and storage; denormalization deliberately duplicates data to optimize reads at the cost of update complexity and consistency risk. The decision is driven by read/write ratio and query patterns. I'll frame normalization's purpose, denormalization's payoff, the consistency burden it creates, and concrete cases — plus how it relates to NoSQL and CQRS.
</thinking>

**The baseline: normalization**
Normalization structures data to eliminate redundancy — each fact lives in exactly one place, related via foreign keys. This guarantees **write integrity** (update a value once, everywhere is correct) and minimizes storage. The cost: reads often require **joins**, which get expensive at scale.

**Denormalization**
Deliberately introduce redundancy — duplicate data or precompute aggregates — so reads avoid expensive joins and serve from a single fetch. You trade write/storage cost for **read speed and simplicity**.

**The core tradeoff**

| | Normalized | Denormalized |
|---|---|---|
| Reads | Joins (slower) | Single fetch (faster) |
| Writes | Update once | Update every copy |
| Storage | Minimal | Redundant |
| Consistency | Guaranteed by structure | Your responsibility |
| Best for | Write-heavy, integrity-critical | Read-heavy, query-shaped |

**The hidden cost: consistency burden**
When you duplicate a value (say a `username` cached on every comment), *every* update must touch *every* copy — and if you miss one, the data is now inconsistent. You take on responsibility the database used to enforce for you. This is manageable but must be deliberate (triggers, application logic, or async propagation), and it's where denormalization bugs hide.

**When duplicating data is the right call**
- **Read-heavy workloads** where the same join runs constantly and dominates latency.
- **Precomputed aggregates** (comment counts, totals) too expensive to compute on every read.
- **Reporting / analytics** read models where data is mostly append-only or refreshed in batches.
- **NoSQL by design**: document stores *expect* denormalization — you embed related data to serve an access pattern in one read, because joins are costly or absent.
- **CQRS read models**: maintain a denormalized projection optimized for queries, kept in sync from the normalized write model via events.

**Guidance**: normalize first — it's the safe default and keeps writes correct. Denormalize *surgically* when profiling shows a specific join or aggregate is the bottleneck, and only where the read benefit clearly outweighs the added write/consistency cost. Treat each denormalization as a maintained cache with an explicit update strategy.

**Real-world**: social feeds precompute timelines (fan-out on write), e-commerce stores denormalize product data onto order records (so an order reflects the price *at purchase time* — here duplication is also *correctness*), and analytics warehouses (star schemas) denormalize heavily for query speed.""",
    },
]

if __name__ == "__main__":
    print(f"Total examples: {len(examples)}")
    for i, ex in enumerate(examples, 1):
        assert "prompt" in ex and "answer" in ex, f"Example {i} missing keys"
        assert "<thinking>" in ex["answer"], f"Example {i} missing <thinking>"
        assert len(ex["answer"]) > 500, f"Example {i} answer too short"
    print("All examples validated.")
