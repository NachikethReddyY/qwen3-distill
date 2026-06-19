#!/usr/bin/env python3
"""
High-quality Python training examples for model distillation.

Each example is a dict with:
  - "prompt": a specific, realistic Python engineering question
  - "answer": a detailed response containing a <thinking> reasoning section,
              production-grade code, and an explanation of best practices.

Topics covered: algorithms, data structures, decorators, generators, async,
pandas, numpy, visualization, design patterns, and optimization. Each example
is designed to teach a distinct, transferable concept.
"""

examples = [
    # 1 -- Algorithms: binary search on answer space
    {
        "prompt": "Implement a function that finds the minimum eating speed (bananas per hour) Koko needs to finish all piles within h hours. Given piles: list[int] and h: int, return the minimum integer speed. Explain why binary search applies here.",
        "answer": """<thinking>
This is the classic "binary search on the answer" pattern. The answer (eating speed k)
is monotonic: if speed k lets Koko finish in time, any speed > k also works. That
monotonicity is exactly what binary search needs.

- Lower bound for k is 1 (can't eat 0/hour).
- Upper bound is max(piles): eating faster than the biggest pile never helps because
  each pile takes at least one hour regardless.
- For a candidate k, hours needed = sum(ceil(pile / k)). Use integer math
  (pile + k - 1) // k to avoid float rounding errors.
- Search the smallest k where hours_needed(k) <= h.
</thinking>

```python
from math import ceil


def min_eating_speed(piles: list[int], h: int) -> int:
    \"\"\"Smallest integer speed so all piles are eaten within h hours.\"\"\"
    if h < len(piles):
        raise ValueError("h must be at least the number of piles")

    def hours_needed(speed: int) -> int:
        # Integer ceil avoids floating-point edge cases.
        return sum((pile + speed - 1) // speed for pile in piles)

    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if hours_needed(mid) <= h:
            hi = mid          # mid works; try slower
        else:
            lo = mid + 1      # too slow; speed up
    return lo
```

**Explanation**

The key insight is recognizing a *monotonic predicate* over the search space:
`hours_needed(k) <= h` is False for small k and flips to True once k is large enough,
never flipping back. Binary search finds that boundary in O(N log M) where N = len(piles)
and M = max(piles), versus O(N * M) for a linear scan. Best practices shown:
integer ceil instead of `math.ceil(pile / k)` (avoids float precision bugs at scale),
the half-open invariant `lo < hi` with `hi = mid` / `lo = mid + 1` (terminates cleanly
and converges on the boundary), and input validation up front.""",
    },

    # 2 -- Data structures: LRU cache from scratch
    {
        "prompt": "Implement an LRU cache class with O(1) get and put, without using functools.lru_cache or OrderedDict. Support a capacity limit.",
        "answer": """<thinking>
O(1) get and put requires two structures working together:
- A hash map for O(1) lookup by key.
- A doubly linked list to track recency order, so we can move a node to the
  "most recently used" end and evict from the "least recently used" end in O(1).

A singly linked list can't unlink a node in O(1) (no prev pointer), so it must be
doubly linked. I'll use sentinel head/tail nodes to eliminate null-checks at the
boundaries -- a common technique that removes a whole class of edge-case bugs.
</thinking>

```python
class _Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    \"\"\"Least-recently-used cache with O(1) get and put.\"\"\"

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._map: dict = {}
        # Sentinels: head.next is MRU, tail.prev is LRU.
        self._head = _Node()
        self._tail = _Node()
        self._head.next = self._tail
        self._tail.prev = self._head

    def _remove(self, node: _Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _add_front(self, node: _Node) -> None:
        node.prev = self._head
        node.next = self._head.next
        self._head.next.prev = node
        self._head.next = node

    def get(self, key):
        node = self._map.get(key)
        if node is None:
            return None
        self._remove(node)
        self._add_front(node)
        return node.value

    def put(self, key, value) -> None:
        if key in self._map:
            node = self._map[key]
            node.value = value
            self._remove(node)
            self._add_front(node)
            return
        if len(self._map) >= self.capacity:
            lru = self._tail.prev
            self._remove(lru)
            del self._map[lru.key]
        node = _Node(key, value)
        self._map[key] = node
        self._add_front(node)
```

**Explanation**

The hash map gives O(1) key lookup; the doubly linked list gives O(1) reordering
and eviction. Sentinel head/tail nodes mean every real node always has non-null
`prev` and `next`, so `_remove` and `_add_front` never special-case the ends.
`__slots__` on the node trims per-node memory (no `__dict__`), which matters when
the cache holds many entries. This is essentially how `OrderedDict` and
`functools.lru_cache` work internally.""",
    },

    # 3 -- Decorators: retry with exponential backoff
    {
        "prompt": "Write a decorator that retries a function on specified exceptions with exponential backoff and jitter. It should be configurable (max attempts, base delay, exceptions) and preserve the wrapped function's metadata.",
        "answer": """<thinking>
A configurable decorator needs an extra layer: a decorator factory that takes
parameters and returns the actual decorator. I'll use functools.wraps to preserve
__name__, __doc__, etc. Backoff = base * 2**attempt, plus random jitter to avoid
the "thundering herd" problem where many clients retry in lockstep. I should re-raise
the last exception after attempts are exhausted, and only catch the exception types
the caller opts into -- catching bare Exception hides bugs.
</thinking>

```python
import functools
import random
import time
from typing import Callable, Type


def retry(
    max_attempts: int = 3,
    base_delay: float = 0.5,
    exceptions: tuple[Type[Exception], ...] = (Exception,),
    max_delay: float = 30.0,
):
    \"\"\"Retry on the given exceptions with exponential backoff + jitter.\"\"\"
    if max_attempts < 1:
        raise ValueError("max_attempts must be >= 1")

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            last_exc = None
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except exceptions as exc:
                    last_exc = exc
                    if attempt == max_attempts - 1:
                        break
                    backoff = min(base_delay * (2 ** attempt), max_delay)
                    delay = random.uniform(0, backoff)  # full jitter: [0, backoff]
                    time.sleep(delay)
            raise last_exc

        return wrapper

    return decorator


@retry(max_attempts=4, base_delay=0.2, exceptions=(ConnectionError, TimeoutError))
def fetch(url: str) -> str:
    ...
```

**Explanation**

Three-layer structure (`retry` -> `decorator` -> `wrapper`) is the standard pattern
for parameterized decorators. `functools.wraps` is essential -- without it the wrapped
function loses its name and docstring, breaking introspection, logging, and Sphinx docs.
Full jitter (`random.uniform(0, backoff)`, the AWS-recommended form) spreads retries out
so failed services aren't hit by synchronized retry storms. We only catch caller-specified exception types so genuine
bugs (e.g. `TypeError`) surface immediately instead of being silently retried, and we
re-raise the final exception so callers still see real failures.""",
    },

    # 4 -- Generators: streaming large file processing
    {
        "prompt": "Show how to process a multi-gigabyte CSV log file to compute the average response time per status code, without loading the whole file into memory. Use generators.",
        "answer": """<thinking>
The constraint is memory: the file is larger than RAM, so I must stream it line by
line and aggregate incrementally. Generators are ideal -- they yield one record at a
time and hold constant memory. I'll build a small pipeline: read lines lazily, parse
each into a record, then fold into running sums. I only need running totals (sum and
count) per status code to compute averages, so memory stays O(number of distinct
status codes), not O(rows).
</thinking>

```python
import csv
from collections import defaultdict
from typing import Iterator


def read_rows(path: str) -> Iterator[dict]:
    \"\"\"Lazily yield rows from a CSV; holds one row in memory at a time.\"\"\"
    with open(path, newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            yield row


def average_response_time_by_status(path: str) -> dict[str, float]:
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)

    for row in read_rows(path):
        try:
            status = row["status"]
            rt = float(row["response_ms"])
        except (KeyError, ValueError):
            continue  # skip malformed lines, keep streaming
        sums[status] += rt
        counts[status] += 1

    return {status: sums[status] / counts[status] for status in sums}
```

**Explanation**

`read_rows` is a generator: the `with` block stays open across yields, and each row
is produced on demand, so peak memory is one row regardless of file size. The
aggregation keeps only running `sums` and `counts` keyed by status code, which is the
key to constant memory -- we never materialize the full dataset. `csv.DictReader`
handles quoting and embedded commas correctly (a hand-written `line.split(",")` would
break on quoted fields). Malformed rows are skipped rather than crashing the whole
job, which is important for real log files that always contain some garbage lines.""",
    },

    # 5 -- Async: bounded concurrent HTTP fetching
    {
        "prompt": "Using asyncio and aiohttp, fetch a list of URLs concurrently but limit concurrency to N simultaneous requests. Return results in input order, with failures captured rather than crashing the whole batch.",
        "answer": """<thinking>
Unbounded concurrency (gather over thousands of URLs) exhausts sockets and gets you
rate-limited. The standard fix is an asyncio.Semaphore as a concurrency gate. I want
results in input order, so asyncio.gather is ideal -- it preserves the order of the
awaitables passed in. To avoid one failure cancelling everything, I'll catch exceptions
inside each task and return a result object, rather than letting gather propagate.
</thinking>

```python
import asyncio
from dataclasses import dataclass
import aiohttp


@dataclass
class FetchResult:
    url: str
    status: int | None
    body: str | None
    error: str | None = None


async def _fetch_one(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    url: str,
) -> FetchResult:
    async with sem:  # acquire a concurrency slot
        try:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
                body = await resp.text()
                return FetchResult(url, resp.status, body)
        except Exception as exc:
            return FetchResult(url, None, None, error=str(exc))


async def fetch_all(urls: list[str], concurrency: int = 10) -> list[FetchResult]:
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [_fetch_one(session, sem, url) for url in urls]
        return await asyncio.gather(*tasks)  # preserves input order
```

**Explanation**

The `Semaphore(concurrency)` bounds how many `session.get` calls run at once: tasks
beyond the limit await the `async with sem` line until a slot frees up. A single shared
`ClientSession` reuses the underlying connection pool -- creating a session per request
is a common and costly mistake. Each task catches its own exceptions and returns a
`FetchResult`, so one bad URL doesn't abort the batch; this is cleaner than
`gather(..., return_exceptions=True)` because the result type is uniform and typed.
`gather` preserves input order, so callers can zip results back to their inputs.""",
    },

    # 6 -- Pandas: idiomatic groupby transform without loops
    {
        "prompt": "Given a pandas DataFrame of sales with columns ['store', 'date', 'revenue'], compute each row's revenue as a percentage of that store's total revenue, and a 7-day rolling average of revenue per store. Avoid Python loops.",
        "answer": """<thinking>
Two vectorized operations per store group:
1. Share of store total: groupby('store')['revenue'].transform('sum') gives a Series
   aligned to the original index, so I can divide element-wise. transform (not agg)
   is the key -- it broadcasts the group result back to every row.
2. Rolling 7-day average per store: sort by date, group by store, then apply a rolling
   window. I must sort first so the window is chronological, and group so windows don't
   bleed across stores.
</thinking>

```python
import pandas as pd


def enrich_sales(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["store", "date"]).reset_index(drop=True)

    # Share of each store's total revenue (transform broadcasts back to rows).
    store_total = df.groupby("store")["revenue"].transform("sum")
    df["pct_of_store"] = df["revenue"] / store_total * 100

    # 7-day rolling average within each store, chronological.
    df["revenue_7d_avg"] = (
        df.groupby("store")["revenue"]
        .rolling(window=7, min_periods=1)
        .mean()
        .reset_index(level=0, drop=True)  # drop the group key from the index
    )
    return df
```

**Explanation**

`transform('sum')` is the idiomatic way to compute a group aggregate and align it back
to the original rows in one vectorized pass -- far faster than merging a separate
groupby result. Sorting by `['store', 'date']` before the rolling window guarantees
windows are chronological and never span two stores. The `reset_index(level=0, drop=True)`
after `groupby().rolling()` is the easy-to-miss step: rolling on a groupby produces a
MultiIndex (group key + original index), and dropping the group level realigns the
result with `df` so the assignment lands on the right rows. `min_periods=1` yields a
value even before 7 days have accumulated.""",
    },

    # 7 -- NumPy: vectorization vs loops
    {
        "prompt": "Compute pairwise Euclidean distances between two sets of points (M points and N points, each D-dimensional) using NumPy broadcasting, with no Python loops. Explain the broadcasting and a numerical-stability concern.",
        "answer": """<thinking>
Naive double loop is O(M*N) Python-level iterations -- slow. Two vectorized approaches:
1. Broadcasting: reshape A to (M, 1, D) and B to (1, N, D); subtract to get (M, N, D),
   square, sum over D. Clear but allocates an (M, N, D) temporary.
2. The expansion ||a-b||^2 = ||a||^2 + ||b||^2 - 2 a.b avoids the big temporary
   (only an (M, N) matrix), but floating-point subtraction can produce tiny negatives
   before the sqrt, so I must clip at 0.
I'll show the memory-efficient expansion since it scales better, and note the clip.
</thinking>

```python
import numpy as np


def pairwise_distances(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    \"\"\"Euclidean distances between rows of a (M,D) and b (N,D) -> (M,N).\"\"\"
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    a_sq = np.sum(a ** 2, axis=1)[:, None]   # (M, 1)
    b_sq = np.sum(b ** 2, axis=1)[None, :]   # (1, N)
    cross = a @ b.T                           # (M, N)

    sq_dist = a_sq + b_sq - 2.0 * cross
    np.maximum(sq_dist, 0.0, out=sq_dist)     # clip tiny negatives from rounding
    return np.sqrt(sq_dist)
```

**Explanation**

The identity `||a - b||^2 = ||a||^2 + ||b||^2 - 2 a.b` turns the distance computation
into a single matrix multiply (`a @ b.T`, which BLAS executes in highly optimized C)
plus broadcasted addition. Broadcasting aligns `a_sq` of shape (M, 1) with `b_sq` of
shape (1, N) to produce an (M, N) grid without explicit loops. The numerical subtlety:
floating-point rounding can make `sq_dist` slightly negative for near-identical points,
and `sqrt` of a negative yields NaN -- so `np.maximum(sq_dist, 0)` clips them. This is
roughly what scikit-learn's `euclidean_distances` does internally and is much faster and
lower-memory than the (M, N, D) broadcasting subtraction for large inputs.""",
    },

    # 8 -- Visualization: publication-quality matplotlib
    {
        "prompt": "Write matplotlib code to plot two time series (model prediction vs ground truth) with a shaded confidence band around the prediction, proper labels, a legend, and a clean style suitable for a paper. Use the object-oriented API.",
        "answer": """<thinking>
For reproducible, publication-quality figures the object-oriented API (fig, ax) is
preferred over the stateful pyplot interface -- it's explicit about which axes you draw
on and composes into subplots. I'll use fill_between for the confidence band, set
labels/title/legend, and tidy the spines. I'll keep styling explicit rather than relying
on global rcParams so the function is self-contained.
</thinking>

```python
import numpy as np
import matplotlib.pyplot as plt


def plot_forecast(t, truth, pred, lower, upper, title="Forecast"):
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)

    ax.fill_between(t, lower, upper, color="#1f77b4", alpha=0.2,
                    label="95% interval")
    ax.plot(t, pred, color="#1f77b4", lw=2, label="Prediction")
    ax.plot(t, truth, color="#333333", lw=1.5, ls="--", label="Ground truth")

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(frameon=False, loc="upper left")

    # Clean up chart junk.
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, ax


if __name__ == "__main__":
    t = np.linspace(0, 10, 200)
    truth = np.sin(t)
    pred = np.sin(t) + np.random.normal(0, 0.1, t.size)
    band = 0.2
    fig, _ = plot_forecast(t, truth, pred, pred - band, pred + band)
    fig.savefig("forecast.png", bbox_inches="tight")
```

**Explanation**

Using the object-oriented API (`fig, ax = plt.subplots()`) instead of `plt.plot()` makes
the code explicit and composable -- the same function works inside a grid of subplots.
`fill_between` with low alpha draws the uncertainty band behind the lines, so the
prediction stays readable. Returning `(fig, ax)` rather than calling `plt.show()` lets
callers save, embed, or further customize the figure. Removing the top/right spines and
using a light horizontal grid reduces "chart junk" (Tufte's term) so the data dominates.
`bbox_inches="tight"` on save trims surrounding whitespace -- important for figures
dropped into LaTeX.""",
    },

    # 9 -- Design pattern: Strategy
    {
        "prompt": "Refactor a function with a big if/elif chain selecting a discount calculation into the Strategy pattern using a registry, so new discount types can be added without modifying existing code.",
        "answer": """<thinking>
The big if/elif violates the Open/Closed Principle: every new discount edits the same
function. The Strategy pattern extracts each branch into its own callable and selects
by key. A registry dict (optionally populated by a decorator) lets new strategies
register themselves without touching the dispatcher -- that's the OCP win. Using a
Protocol documents the strategy interface without forcing inheritance.
</thinking>

```python
from typing import Callable, Protocol


class DiscountStrategy(Protocol):
    def __call__(self, price: float) -> float: ...


_REGISTRY: dict[str, DiscountStrategy] = {}


def register(name: str) -> Callable[[DiscountStrategy], DiscountStrategy]:
    def deco(fn: DiscountStrategy) -> DiscountStrategy:
        _REGISTRY[name] = fn
        return fn
    return deco


@register("none")
def _no_discount(price: float) -> float:
    return price


@register("percent10")
def _ten_percent(price: float) -> float:
    return price * 0.90


@register("bulk")
def _bulk(price: float) -> float:
    return price * 0.80 if price > 1000 else price


def apply_discount(price: float, strategy: str) -> float:
    try:
        return _REGISTRY[strategy](price)
    except KeyError:
        raise ValueError(f"unknown strategy: {strategy!r}") from None
```

**Explanation**

Each discount rule is now an isolated, independently testable function, and the
`@register` decorator adds it to the lookup table at import time. To add a new discount
you write a new function with a decorator -- you never touch `apply_discount` or any
existing rule, satisfying the Open/Closed Principle and eliminating merge conflicts in a
growing if/elif. The `Protocol` defines the strategy's call signature for type checkers
without requiring a base class (structural typing). The `from None` suppresses the
chained `KeyError`, giving callers a clean, domain-specific error message.""",
    },

    # 10 -- Optimization: memoization for recursive DP
    {
        "prompt": "Solve the classic 'minimum coin change' problem (fewest coins to make an amount) with a clean dynamic-programming solution. Show both the recursive memoized version and the iterative bottom-up version, and discuss the trade-off.",
        "answer": """<thinking>
Coin change is a DP problem with optimal substructure: min_coins(amount) =
1 + min over coins c of min_coins(amount - c). Two equivalent formulations:
- Top-down: recursion + memoization (lru_cache), natural to write, only computes
  reachable subproblems, but risks deep recursion / stack limits for large amounts.
- Bottom-up: a 1-D table filled iteratively, no recursion, predictable memory, usually
  a bit faster due to no call overhead.
I'll return float('inf') sentinel for unreachable amounts and convert to -1 at the end.
</thinking>

```python
from functools import lru_cache


def min_coins_topdown(coins: tuple[int, ...], amount: int) -> int:
    @lru_cache(maxsize=None)
    def solve(rem: int) -> float:
        if rem == 0:
            return 0
        if rem < 0:
            return float("inf")
        return min((1 + solve(rem - c) for c in coins), default=float("inf"))

    result = solve(amount)
    return -1 if result == float("inf") else int(result)


def min_coins_bottomup(coins: list[int], amount: int) -> int:
    dp = [0] + [float("inf")] * amount
    for sub in range(1, amount + 1):
        for c in coins:
            if c <= sub and dp[sub - c] + 1 < dp[sub]:
                dp[sub] = dp[sub - c] + 1
    return -1 if dp[amount] == float("inf") else dp[amount]
```

**Explanation**

Both run in O(amount * len(coins)) time and O(amount) space. The top-down version is
the most direct translation of the recurrence and only evaluates subproblems that are
actually reachable, but for very large `amount` it can hit Python's recursion limit and
carries call overhead. The bottom-up version trades that for an explicit loop: no
recursion, cache-friendly sequential array access, and easy to extend (e.g. to also
reconstruct *which* coins). Note `coins` must be hashable (a tuple) for `lru_cache` to
key on it. Rule of thumb: prototype with memoized recursion, ship the bottom-up table
when the input range is large or the recurrence depth is a concern.""",
    },

    # 11 -- Algorithms: graph BFS shortest path
    {
        "prompt": "Implement breadth-first search to find the shortest path (as a list of nodes) between two nodes in an unweighted graph represented as an adjacency dict. Return None if unreachable.",
        "answer": """<thinking>
BFS explores level by level, so the first time we reach the target we've found a
shortest path in an unweighted graph. To reconstruct the path I'll track each node's
predecessor in a dict, then walk back from target to source. I must mark nodes visited
when they're enqueued (not when dequeued) to avoid enqueuing the same node many times,
which would blow up the queue.
</thinking>

```python
from collections import deque


def shortest_path(graph: dict, start, goal) -> list | None:
    if start == goal:
        return [start]
    visited = {start}
    parent = {start: None}
    queue = deque([start])

    while queue:
        node = queue.popleft()
        for nbr in graph.get(node, ()):
            if nbr in visited:
                continue
            visited.add(nbr)
            parent[nbr] = node
            if nbr == goal:
                return _reconstruct(parent, goal)
            queue.append(nbr)
    return None


def _reconstruct(parent: dict, goal) -> list:
    path = []
    node = goal
    while node is not None:
        path.append(node)
        node = parent[node]
    return path[::-1]
```

**Explanation**

A `deque` gives O(1) `popleft`; a plain list's `pop(0)` is O(n) and would make BFS
quadratic. Marking a node visited at *enqueue* time (not dequeue) is the critical detail
-- it guarantees each node enters the queue at most once, keeping the queue bounded by
the number of nodes. The `parent` dict records the breadth-first tree, and `_reconstruct`
walks predecessors from goal back to start, reversing to get start->goal order. Because
BFS processes nodes in non-decreasing distance order, the first time `goal` is reached is
guaranteed to be along a shortest path (for unweighted graphs -- weighted graphs need
Dijkstra).""",
    },

    # 12 -- Context managers: resource handling and __exit__
    {
        "prompt": "Create a context manager (both the class-based and the contextlib.contextmanager generator-based form) that times a code block and guarantees the timer is recorded even if the block raises. Explain when to use each form.",
        "answer": """<thinking>
A context manager guarantees cleanup via __exit__/the finally clause even on exception.
I'll show both forms. For the class form, __exit__ receives exception info; returning
False (the default) re-raises any exception, which is what we want -- the timer should
record but not swallow errors. For the generator form, contextlib.contextmanager wraps
a single yield in try/finally so the timing after the yield always runs.
</thinking>

```python
import time
from contextlib import contextmanager


class Timer:
    \"\"\"Class-based timer context manager.\"\"\"

    def __init__(self, label: str = "block"):
        self.label = label
        self.elapsed = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        self.elapsed = time.perf_counter() - self._start
        print(f"{self.label}: {self.elapsed:.4f}s")
        return False  # propagate any exception


@contextmanager
def timer(label: str = "block"):
    \"\"\"Generator-based equivalent.\"\"\"
    start = time.perf_counter()
    try:
        yield
    finally:
        print(f"{label}: {time.perf_counter() - start:.4f}s")
```

**Explanation**

`time.perf_counter()` is the right clock for measuring durations -- it's monotonic and
high-resolution, unlike `time.time()` which can jump backward on NTP adjustments. In the
class form, `__exit__` runs even when the block raises; returning `False` means any
exception still propagates (returning `True` would silently swallow it -- almost always a
bug). The generator form puts the timing in a `finally`, giving the same guarantee with
less boilerplate. Use the **class form** when you need to store state on the manager
(here, `self.elapsed` is readable after the block) or implement reentrancy; use the
**generator form** for simple setup/teardown where you don't need an object to inspect
afterward.""",
    },

    # 13 -- Dataclasses and immutability
    {
        "prompt": "Model a 2D vector with a frozen dataclass supporting addition, scalar multiplication, and magnitude. Explain why immutability and __slots__ help here, and the caveat about mutable default fields.",
        "answer": """<thinking>
A vector is a value object: two vectors with the same components should be equal and
hashable, and operations should return new instances rather than mutate. A frozen
dataclass gives __eq__, __hash__, and immutability for free. Adding slots=True removes
per-instance __dict__ for memory and speed -- meaningful when you create millions of
vectors. I'll implement __add__ and __mul__ returning new Vec2 instances. I'll also note
the classic mutable-default trap even though this class doesn't need it.
</thinking>

```python
from dataclasses import dataclass
from math import hypot


@dataclass(frozen=True, slots=True)
class Vec2:
    x: float
    y: float

    def __add__(self, other: "Vec2") -> "Vec2":
        return Vec2(self.x + other.x, self.y + other.y)

    def __mul__(self, scalar: float) -> "Vec2":
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__  # allow 3 * v as well as v * 3

    @property
    def magnitude(self) -> float:
        return hypot(self.x, self.y)
```

**Explanation**

`frozen=True` makes instances immutable, which auto-generates `__hash__` so `Vec2` can
be a dict key or set member, and prevents accidental mutation -- exactly the semantics you
want for a mathematical value type where `a + b` should never alter `a`. `slots=True`
drops the per-instance `__dict__`, cutting memory and speeding attribute access (valuable
in tight numeric loops). `hypot(x, y)` computes the magnitude without intermediate
overflow that `sqrt(x*x + y*y)` can suffer. Defining `__rmul__` lets scalars appear on
either side. The general dataclass caveat to remember: never use a mutable default like
`field=[]` directly -- all instances would share one list; use
`field: list = field(default_factory=list)` instead. (This frozen value type sidesteps
that entirely.)""",
    },

    # 14 -- Itertools: efficient combinatorics
    {
        "prompt": "Using itertools, write a function that groups consecutive equal elements in an iterable into (value, count) pairs (like run-length encoding), and a second function that flattens a list of lists lazily. Explain why itertools is preferable to manual loops here.",
        "answer": """<thinking>
Run-length encoding maps directly onto itertools.groupby, which groups *consecutive*
equal elements -- exactly the semantics needed (unlike SQL GROUP BY, it doesn't sort).
The count is len() of the group, but groupby's group is a one-shot iterator, so I sum a
generator of 1s to avoid materializing it. For flattening, itertools.chain.from_iterable
lazily concatenates sublists without building an intermediate list.
</thinking>

```python
from itertools import chain, groupby
from typing import Iterable, Iterator


def run_length_encode(seq: Iterable) -> list[tuple]:
    \"\"\"[a,a,b,a] -> [(a,2),(b,1),(a,1)].\"\"\"
    return [(value, sum(1 for _ in group)) for value, group in groupby(seq)]


def flatten(nested: Iterable[Iterable]) -> Iterator:
    \"\"\"Lazily flatten one level of nesting.\"\"\"
    return chain.from_iterable(nested)
```

**Explanation**

`itertools.groupby` is purpose-built for collapsing *consecutive* runs -- note it groups
only adjacent equal elements, so `[a, b, a]` yields three groups, which is the correct
run-length semantics. The group it yields is a sub-iterator that shares the underlying
stream, so I consume it immediately with `sum(1 for _ in group)`; trying to reuse it
later would give empty results -- a classic groupby gotcha. `chain.from_iterable` flattens
lazily in C, producing elements on demand with no intermediate list, so it works on
infinite or streaming inputs and uses constant extra memory. Reaching for itertools over
hand-rolled index loops gives you tested, C-speed building blocks and code that reads as
intent rather than mechanics.""",
    },

    # 15 -- Async: producer/consumer with asyncio.Queue
    {
        "prompt": "Build a producer/consumer pipeline with asyncio where one producer puts work items on a queue and multiple workers consume them concurrently. Ensure clean shutdown when all items are processed.",
        "answer": """<thinking>
asyncio.Queue coordinates producer and consumers with backpressure (maxsize bounds it).
The tricky part is clean shutdown: workers loop forever on queue.get(), so I need a way
to tell them to stop. The idiomatic approach is queue.join() (waits until every put has a
matching task_done) plus cancelling the worker tasks afterward. Each worker must call
task_done() after processing so join() can unblock.
</thinking>

```python
import asyncio


async def producer(queue: asyncio.Queue, items):
    for item in items:
        await queue.put(item)


async def worker(name: str, queue: asyncio.Queue, results: list):
    while True:
        item = await queue.get()
        try:
            await asyncio.sleep(0.01)      # simulate work
            results.append((name, item * 2))
        finally:
            queue.task_done()              # signal completion even on error


async def run_pipeline(items, num_workers: int = 4) -> list:
    queue: asyncio.Queue = asyncio.Queue(maxsize=100)
    results: list = []

    workers = [
        asyncio.create_task(worker(f"w{i}", queue, results))
        for i in range(num_workers)
    ]
    await producer(queue, items)
    await queue.join()                     # wait until every item is processed

    for w in workers:                      # shut down idle workers
        w.cancel()
    await asyncio.gather(*workers, return_exceptions=True)
    return results
```

**Explanation**

`asyncio.Queue` decouples the producer from the consumers and, with `maxsize`, provides
backpressure -- the producer blocks on `put` if workers fall behind, preventing unbounded
memory growth. The shutdown pattern is the important part: every worker calls
`queue.task_done()` in a `finally` (so even a failed item is counted), and `queue.join()`
blocks until completed tasks equal enqueued items. Only then do we `cancel()` the workers,
which are otherwise parked forever on `queue.get()`. Gathering the cancelled tasks with
`return_exceptions=True` absorbs the resulting `CancelledError` cleanly. This avoids both
the deadlock of forgetting `task_done()` and the leak of orphaned worker tasks.""",
    },

    # 16 -- Pandas: efficient categorical and memory reduction
    {
        "prompt": "I have a pandas DataFrame with millions of rows and string columns with low cardinality (e.g. 'country', 'status'). Show how to drastically reduce memory usage and explain the techniques and their trade-offs.",
        "answer": """<thinking>
The big memory wins on a wide DataFrame:
1. Convert low-cardinality string (object) columns to 'category' dtype -- stores codes +
   a small dictionary instead of one Python string object per row.
2. Downcast numeric columns: int64 -> the smallest int that fits, float64 -> float32 when
   precision allows.
I'll write a function that does both, measuring before/after. Trade-offs: category is a
loss if cardinality is high (overhead of the index), and float32 loses precision.
</thinking>

```python
import pandas as pd


def optimize_memory(df: pd.DataFrame, cat_threshold: float = 0.5) -> pd.DataFrame:
    df = df.copy()
    for col in df.columns:
        kind = df[col].dtype.kind
        if kind == "O":  # object/string
            n_unique = df[col].nunique(dropna=False)
            if n_unique / max(len(df), 1) < cat_threshold:
                df[col] = df[col].astype("category")
        elif kind == "i":
            df[col] = pd.to_numeric(df[col], downcast="integer")
        elif kind == "f":
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def report(df: pd.DataFrame) -> str:
    mb = df.memory_usage(deep=True).sum() / 1024 ** 2
    return f"{mb:.1f} MB"
```

**Explanation**

The dominant cost in a string column is that pandas stores a pointer to a separate Python
`str` object per row. Converting to `category` replaces those with small integer codes
plus a single dictionary of distinct values, so a column with 3 distinct values across a
million rows shrinks dramatically. The `cat_threshold` guard avoids the anti-pattern of
categorizing high-cardinality columns (e.g. unique IDs), where the category index adds
overhead instead of saving. Numeric downcasting (`to_numeric(downcast=...)`) picks the
smallest dtype that holds the data. The trade-off to flag: `float32` halves memory but
loses precision, so don't downcast columns feeding sensitive financial or scientific math.
Always use `memory_usage(deep=True)` -- the default `deep=False` undercounts object
columns because it ignores the referenced string objects.""",
    },

    # 17 -- Metaclasses / __init_subclass__: plugin registration
    {
        "prompt": "Create a base class that automatically registers every subclass into a registry by a 'name' attribute, so plugins can be discovered by name. Prefer __init_subclass__ over a metaclass and explain why.",
        "answer": """<thinking>
Auto-registering subclasses is a textbook metaclass use case, but since Python 3.6
__init_subclass__ does the same thing with far less complexity and composes with other
metaclasses. It's called automatically when a subclass is defined. I'll register by a
required `name` attribute, validating it exists and isn't duplicated. This gives a
plugin-discovery system without any metaclass.
</thinking>

```python
class Plugin:
    registry: dict[str, type["Plugin"]] = {}

    name: str  # subclasses must set this

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        name = getattr(cls, "name", None)
        if not name:
            raise TypeError(f"{cls.__name__} must define a non-empty 'name'")
        if name in Plugin.registry:
            raise ValueError(f"duplicate plugin name: {name!r}")
        Plugin.registry[name] = cls

    @classmethod
    def create(cls, name: str, *args, **kwargs) -> "Plugin":
        try:
            return cls.registry[name](*args, **kwargs)
        except KeyError:
            raise ValueError(f"no plugin named {name!r}") from None


class CsvExporter(Plugin):
    name = "csv"


class JsonExporter(Plugin):
    name = "json"
```

**Explanation**

`__init_subclass__` runs on the *parent* whenever a subclass is created, so each plugin
self-registers at definition/import time -- no manual list to maintain. It's preferred
over a metaclass because metaclasses are heavier, harder to reason about, and conflict
when a class needs to combine multiple metaclasses; `__init_subclass__` is a plain hook
that composes cleanly. Validating the `name` and rejecting duplicates at class-creation
time turns silent misconfiguration into an immediate, loud error. The `create` factory
then instantiates plugins by string name -- the foundation of a discoverable plugin
architecture (the same idea powers many serializer and backend registries). Always call
`super().__init_subclass__(**kwargs)` to stay cooperative in multi-level hierarchies.""",
    },

    # 18 -- Algorithms: two-pointer / sliding window
    {
        "prompt": "Find the length of the longest substring without repeating characters using the sliding window technique. Explain the amortized complexity.",
        "answer": """<thinking>
Sliding window with two pointers. Keep a window [left, right] containing only unique
characters. As right advances, if the new char is already in the window, move left past
its previous occurrence. Track the last index of each char in a dict so left can jump
directly instead of stepping one at a time. Each character is added once and the window
only moves forward, giving O(n).
</thinking>

```python
def longest_unique_substring(s: str) -> int:
    last_seen: dict[str, int] = {}
    left = 0
    best = 0
    for right, ch in enumerate(s):
        # If ch is in the current window, shrink from the left past it.
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        best = max(best, right - left + 1)
    return best
```

**Explanation**

The window `[left, right]` always holds distinct characters. The dict `last_seen` records
each character's most recent index, letting `left` *jump* directly past a repeat rather
than sliding one position at a time -- the key optimization. The guard
`last_seen[ch] >= left` matters: a character seen earlier but already outside the current
window (index < left) must not pull `left` backward. Although there's a nested notion of
two pointers, each of the n characters causes `right` to advance once and `left` to
advance at most n times total, so the algorithm is O(n) time and O(min(n, alphabet))
space -- a clean improvement over the O(n^2) brute-force of checking every substring.""",
    },

    # 19 -- Optimization: profiling and the right data structure
    {
        "prompt": "I have a function that checks membership of items against a growing collection in a loop and it's slow. Show how to profile it and explain the fix, including a benchmark comparing list vs set membership.",
        "answer": """<thinking>
The classic performance bug: `x in some_list` is O(n), so doing it in a loop is O(n*m).
Switching the collection to a set makes membership O(1) average, turning the loop into
O(m). I'll show how to measure with timeit and cProfile, then the structural fix. The
teaching point is: profile to find the hot line, then pick the data structure whose
complexity matches the access pattern.
</thinking>

```python
import cProfile
import pstats
from timeit import timeit


def count_known_list(items, known: list) -> int:
    return sum(1 for x in items if x in known)      # O(len(items) * len(known))


def count_known_set(items, known_set: set) -> int:
    return sum(1 for x in items if x in known_set)   # O(len(items))


def benchmark() -> None:
    known = list(range(10_000))
    known_set = set(known)
    items = list(range(5_000))

    t_list = timeit(lambda: count_known_list(items, known), number=20)
    t_set = timeit(lambda: count_known_set(items, known_set), number=20)
    print(f"list membership: {t_list:.3f}s")
    print(f"set membership:  {t_set:.3f}s")


def profile() -> None:
    known = list(range(10_000))
    items = list(range(5_000))
    pr = cProfile.Profile()
    pr.enable()
    count_known_list(items, known)
    pr.disable()
    pstats.Stats(pr).sort_stats("cumulative").print_stats(5)
```

**Explanation**

`timeit` gives a stable wall-clock comparison (run multiple iterations to amortize noise),
and `cProfile` + `pstats` show *where* time goes by cumulative cost -- always profile
before optimizing rather than guessing. The fix is structural, not micro: `in` on a list
scans linearly (O(n)), while `in` on a set or dict is hash-based O(1) average. Done inside
a loop over m items, that's the difference between O(m*n) and O(m) -- often orders of
magnitude on real data. Build the set once outside the loop; rebuilding it each iteration
would erase the gain. The broader lesson: most Python speedups come from matching the data
structure to the access pattern, not from rewriting in C.""",
    },

    # 20 -- Design pattern: Observer / pub-sub
    {
        "prompt": "Implement a lightweight event emitter (publish/subscribe) supporting subscribe, unsubscribe, and emit, where a failing subscriber doesn't break the others. Make it usable as a decorator for handlers.",
        "answer": """<thinking>
Pub/sub decouples emitters from listeners. I need: a mapping event-name -> list of
callbacks; subscribe/unsubscribe; emit that calls each handler. Robustness requirement:
one handler raising must not stop the rest, so I'll catch and collect exceptions per
handler. I'll also support a decorator form `@bus.on("event")` for ergonomic registration.
Returning an unsubscribe function is a nice touch for cleanup.
</thinking>

```python
from collections import defaultdict
from typing import Callable


class EventBus:
    def __init__(self):
        self._subs: dict[str, list[Callable]] = defaultdict(list)

    def subscribe(self, event: str, handler: Callable) -> Callable[[], None]:
        self._subs[event].append(handler)
        return lambda: self.unsubscribe(event, handler)

    def unsubscribe(self, event: str, handler: Callable) -> None:
        if handler in self._subs.get(event, []):
            self._subs[event].remove(handler)

    def on(self, event: str) -> Callable[[Callable], Callable]:
        def deco(handler: Callable) -> Callable:
            self.subscribe(event, handler)
            return handler
        return deco

    def emit(self, event: str, *args, **kwargs) -> list[Exception]:
        errors: list[Exception] = []
        for handler in list(self._subs.get(event, [])):  # copy: handlers may unsubscribe
            try:
                handler(*args, **kwargs)
            except Exception as exc:
                errors.append(exc)
        return errors


bus = EventBus()


@bus.on("user.created")
def send_welcome(user):
    print(f"welcome {user}")
```

**Explanation**

The Observer pattern lets producers fire events without knowing who listens -- new
subscribers attach without touching the emitter. The `emit` loop wraps each handler in
try/except so a single misbehaving subscriber can't abort delivery to the others; the
collected exceptions are returned for the caller to log or inspect rather than silently
discarded. Iterating over a `list(...)` copy of the handlers is a subtle but important
guard: a handler that unsubscribes itself (or others) during `emit` would otherwise mutate
the list mid-iteration. `subscribe` returns an unsubscribe closure for easy teardown, and
the `on` decorator gives the clean registration syntax familiar from JS event emitters and
Flask routes.""",
    },

    # 21 -- NumPy: structured masking and where
    {
        "prompt": "Given a NumPy array of sensor readings with some NaNs and some out-of-range outliers, write a vectorized cleaning function that replaces NaNs with the column mean and clips outliers to a given percentile range. No Python loops.",
        "answer": """<thinking>
Fully vectorized cleaning:
1. Compute per-column means ignoring NaN with np.nanmean.
2. Replace NaNs: build a boolean mask with np.isnan and use np.where to substitute the
   column mean (broadcast across rows).
3. Clip outliers: compute lower/upper percentiles per column with np.nanpercentile, then
   np.clip each column to that range.
All operations are array-level, so no loops. I must be careful that nanmean over an
all-NaN column warns/returns NaN -- I'll note that edge case.
</thinking>

```python
import numpy as np


def clean_readings(arr: np.ndarray, lo_pct=1.0, hi_pct=99.0) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float64).copy()

    # 1. Fill NaNs with column means (broadcast).
    col_means = np.nanmean(arr, axis=0)
    nan_mask = np.isnan(arr)
    arr = np.where(nan_mask, col_means, arr)

    # 2. Clip outliers to per-column percentile range.
    lower = np.percentile(arr, lo_pct, axis=0)
    upper = np.percentile(arr, hi_pct, axis=0)
    arr = np.clip(arr, lower, upper)
    return arr
```

**Explanation**

Every step works on the whole array at once. `np.nanmean(arr, axis=0)` computes a mean
per column while ignoring NaNs; `np.where(nan_mask, col_means, arr)` then substitutes
those means wherever a value is missing, with `col_means` (shape `(ncols,)`) broadcasting
across all rows. After imputation there are no NaNs, so plain `np.percentile` gives the
per-column clipping bounds, and `np.clip(arr, lower, upper)` caps extreme values -- both
`lower` and `upper` broadcast column-wise. This is far faster than iterating rows and
keeps the logic declarative. One edge case to handle in production: a column that is
entirely NaN makes `nanmean` return NaN (and emit a warning), so for fully-missing columns
you'd fall back to a global default before clipping.""",
    },

    # 22 -- Generators: building a lazy pipeline with composition
    {
        "prompt": "Compose a data-processing pipeline as a chain of generator functions (read -> filter -> transform -> batch) so the whole thing streams lazily. Show how to yield fixed-size batches from an arbitrary iterable.",
        "answer": """<thinking>
Generator pipelines let each stage pull from the previous one lazily, so memory stays
flat and you can process infinite streams. The interesting piece is batching: islice from
itertools grabs up to n items at a time from an iterator; when it returns an empty list,
the source is exhausted and we stop. Each stage is just a generator function that takes an
iterable and yields transformed items, making them composable.
</thinking>

```python
from itertools import islice
from typing import Callable, Iterable, Iterator


def filter_stage(source: Iterable, pred: Callable) -> Iterator:
    for item in source:
        if pred(item):
            yield item


def map_stage(source: Iterable, fn: Callable) -> Iterator:
    for item in source:
        yield fn(item)


def batched(source: Iterable, size: int) -> Iterator[list]:
    it = iter(source)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            return
        yield chunk


def build_pipeline(raw: Iterable) -> Iterator[list]:
    kept = filter_stage(raw, lambda x: x % 2 == 0)
    doubled = map_stage(kept, lambda x: x * 2)
    return batched(doubled, size=3)
```

**Explanation**

Each stage is a generator that consumes the previous stage's output one item at a time,
so nothing is materialized until the final consumer iterates -- the pipeline streams end
to end and works on inputs larger than memory or even infinite. `batched` is the reusable
primitive: `islice(it, size)` pulls the next `size` items from the *same* iterator each
call (note `it = iter(source)` is created once, outside the loop, so successive slices
advance rather than restart), and an empty chunk signals exhaustion. This composition
style is the foundation of streaming ETL; Python 3.12 even adds `itertools.batched` as a
built-in equivalent of the `batched` helper here.""",
    },

    # 23 -- Type system: generics and Protocols for reusable code
    {
        "prompt": "Write a generic, type-safe function that returns the item with the maximum value according to a key function, that works for any comparable result. Use typing generics and a Protocol for the comparable constraint. Explain the typing decisions.",
        "answer": """<thinking>
I want max_by(items, key) returning the element whose key is largest. To be type-safe and
reusable, the element type T is generic, and the key returns something orderable. I'll
define a SupportsLessThan Protocol to constrain "orderable", and a TypeVar bound to it for
the key's return type. This lets a type checker verify both the element type flows through
and the key result is actually comparable.
</thinking>

```python
from typing import Callable, Iterable, Protocol, TypeVar, runtime_checkable


@runtime_checkable
class SupportsLessThan(Protocol):
    def __lt__(self, other: object) -> bool: ...


T = TypeVar("T")
C = TypeVar("C", bound=SupportsLessThan)


def max_by(items: Iterable[T], key: Callable[[T], C]) -> T:
    iterator = iter(items)
    try:
        best = next(iterator)
    except StopIteration:
        raise ValueError("max_by() arg is an empty iterable") from None
    best_key = key(best)
    for item in iterator:
        k = key(item)
        if best_key < k:
            best = item
            best_key = k
    return best
```

**Explanation**

Two type variables carry the safety: `T` is the element type, so the return type matches
the input element type exactly (call it on `list[User]` and the checker knows you get a
`User` back). `C` is bound to the `SupportsLessThan` Protocol, which encodes the only
capability we actually need -- that key results can be `<`-compared -- via structural
typing, so any class implementing `__lt__` qualifies without inheriting anything. Caching
`best_key` avoids recomputing the (possibly expensive) key per comparison. Raising
`ValueError` on an empty iterable mirrors built-in `max`'s behavior, and `from None`
suppresses the confusing chained `StopIteration`. This is essentially how `max(..., key=)`
is typed in the standard library's stubs.""",
    },

    # 24 -- Concurrency: CPU-bound work with multiprocessing
    {
        "prompt": "Explain when to use multiprocessing instead of threading in Python, then implement a parallel map over a CPU-bound function using ProcessPoolExecutor, including a chunked variant for many small tasks.",
        "answer": """<thinking>
The GIL means CPU-bound Python code doesn't speed up with threads -- only one thread
executes bytecode at a time. multiprocessing sidesteps the GIL by using separate
processes, each with its own interpreter. The cost is process startup and pickling
arguments/results across the boundary, so for many tiny tasks you want chunksize to
amortize that overhead. I'll show ProcessPoolExecutor.map with chunksize and note the
__main__ guard requirement.
</thinking>

```python
from concurrent.futures import ProcessPoolExecutor
from math import isqrt


def is_prime(n: int) -> bool:
    if n < 2:
        return False
    for d in range(2, isqrt(n) + 1):
        if n % d == 0:
            return False
    return True


def parallel_filter_primes(numbers: list[int], workers: int | None = None) -> list[int]:
    with ProcessPoolExecutor(max_workers=workers) as pool:
        # chunksize batches tasks to amortize per-task IPC/pickle overhead.
        flags = pool.map(is_prime, numbers, chunksize=1000)
        return [n for n, prime in zip(numbers, flags) if prime]


if __name__ == "__main__":  # required: child processes re-import this module
    primes = parallel_filter_primes(list(range(2, 200_000)))
    print(len(primes))
```

**Explanation**

Use **threading** for I/O-bound work (network, disk) where threads wait and release the
GIL; use **multiprocessing** for CPU-bound work like this primality test, because separate
processes each get their own GIL and run truly in parallel across cores. The catch is that
arguments and results are pickled and shipped between processes, so a flood of tiny tasks
spends more time on IPC than computing -- `chunksize` batches many inputs per dispatch to
amortize that, often the single biggest tuning knob. The `if __name__ == "__main__"` guard
is mandatory on the spawn start method (default on macOS and Windows): child processes
re-import the module, and without the guard they'd recursively spawn more pools. For
pure-numeric work, also consider vectorizing with NumPy first -- it's often faster than
multiprocessing and far simpler.""",
    },

    # 25 -- Pandas + visualization: end-to-end analysis
    {
        "prompt": "Given a DataFrame of daily temperatures with columns ['date', 'city', 'temp_c'], resample to monthly averages per city and plot each city's monthly trend on a single chart. Handle missing days correctly.",
        "answer": """<thinking>
End-to-end: parse dates, set a DatetimeIndex (required for resample), then group by city
and resample monthly with mean -- which naturally ignores missing days rather than
treating them as zero. Pivot to wide form (months x cities) for a clean multi-line plot.
I'll use the OO matplotlib API and plot each city as a column. Resample('MS') gives
month-start labels.
</thinking>

```python
import pandas as pd
import matplotlib.pyplot as plt


def monthly_trends(df: pd.DataFrame):
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Monthly mean per city; mean() skips NaN/missing days automatically.
    monthly = (
        df.set_index("date")
        .groupby("city")["temp_c"]
        .resample("MS")
        .mean()
        .reset_index()
    )

    # Wide form: index=month, one column per city.
    wide = monthly.pivot(index="date", columns="city", values="temp_c")

    fig, ax = plt.subplots(figsize=(10, 5), dpi=120)
    for city in wide.columns:
        ax.plot(wide.index, wide[city], marker="o", label=city)

    ax.set_xlabel("Month")
    ax.set_ylabel("Avg temperature (°C)")
    ax.set_title("Monthly temperature trend by city")
    ax.legend(frameon=False)
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()        # rotate date ticks
    fig.tight_layout()
    return wide, fig
```

**Explanation**

`resample` requires a `DatetimeIndex`, so `set_index("date")` comes first; combining it
with `groupby("city")` resamples each city independently. Using `.mean()` is what makes
missing-day handling correct -- the average is over the days that exist, so absent days
don't drag the value toward zero the way a `sum()` or a fill-with-0 would. `"MS"` labels
each bucket by month start for clean axis ticks. Pivoting to wide form turns the long,
tidy result into a months-by-cities matrix that's trivial to plot as one line per column.
The object-oriented matplotlib API plus `fig.autofmt_xdate()` (which rotates and aligns
date labels) yields a readable multi-series chart. Keep data wrangling in pandas and only
hand the final shaped frame to matplotlib -- that separation keeps each stage simple.""",
    },
]


if __name__ == "__main__":
    # Sanity checks on the dataset shape and quality.
    assert len(examples) == 25, f"expected 25 examples, got {len(examples)}"
    for i, ex in enumerate(examples):
        assert ex["prompt"].strip(), f"example {i} has empty prompt"
        ans = ex["answer"]
        assert "<thinking>" in ans and "</thinking>" in ans, f"example {i} missing thinking"
        assert "```python" in ans, f"example {i} missing python code block"
        assert "**Explanation**" in ans, f"example {i} missing explanation"
    print(f"OK: {len(examples)} validated Python distillation examples.")
