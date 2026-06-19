#!/usr/bin/env python3
"""
25 high-quality JavaScript training examples for model distillation.

Each example is a dict with:
  - "prompt": a specific, realistic JavaScript question
  - "answer": a detailed response containing a <thinking> reasoning section,
              production-grade code in a ```javascript block, and an
              explanation of best practices.

Topics covered: algorithms, async/await, promises, closures, prototypes,
DOM manipulation, events, performance optimization, data structures, and
design patterns. Each example teaches something unique.

These integrate with the existing pipeline (generate_synthetic_data.py /
add_examples.py): wrap each {prompt, answer} as a chat message triple using
SYSTEM_PROMPT, or import `examples` directly.
"""

SYSTEM_PROMPT = (
    "You are an expert software engineer with deep knowledge of React, "
    "JavaScript, Python, animations, and data visualization.\n"
    "Think through problems step by step inside <thinking>...</thinking> tags, "
    "explaining your reasoning and approach.\n"
    "Then give your final answer with clean, well-structured code and detailed "
    "explanations.\nBe thorough, correct, and explain complex concepts clearly."
)

examples = [
    # 1 — Algorithms: binary search with correctness edge cases
    {
        "prompt": "Implement an iterative binary search in JavaScript that returns the index of a target in a sorted array, or -1 if absent. Avoid the classic integer-overflow midpoint bug and handle empty arrays.",
        "answer": """<thinking>
Binary search halves the search space each iteration, giving O(log n) time.
Key correctness concerns:
1. Use a half-open or closed interval consistently. I'll use closed [lo, hi].
2. The classic `(lo + hi) / 2` can overflow in fixed-width languages. JS numbers
   are 64-bit doubles so true overflow is unlikely for array indices, but
   `lo + ((hi - lo) >> 1)` is the canonical overflow-safe form and signals intent.
3. Loop condition must be `lo <= hi` so a single-element range is still checked.
4. Empty array: hi starts at -1, loop never runs, returns -1. Correct by construction.
</thinking>

```javascript
/**
 * Iterative binary search over a sorted (ascending) array.
 * @param {number[]} arr - sorted array
 * @param {number} target - value to locate
 * @returns {number} index of target, or -1 if not found
 */
function binarySearch(arr, target) {
  let lo = 0;
  let hi = arr.length - 1;

  while (lo <= hi) {
    // Overflow-safe midpoint: never computes lo + hi directly.
    const mid = lo + ((hi - lo) >> 1);
    const value = arr[mid];

    if (value === target) return mid;
    if (value < target) lo = mid + 1;
    else hi = mid - 1;
  }

  return -1;
}
```

**Explanation**
- **Closed interval `[lo, hi]`** with `lo <= hi` ensures every candidate, including the final single element, is examined. An off-by-one here is the #1 source of binary-search bugs.
- **`lo + ((hi - lo) >> 1)`** computes the midpoint without ever forming the sum `lo + hi`, the overflow-safe idiom. `>> 1` is an integer divide-by-two.
- **Empty array** is handled implicitly: `hi = -1` makes the loop body never execute.
- **Complexity:** O(log n) time, O(1) space. Requires the array to be pre-sorted; if it is not, sort first (O(n log n)) or use a linear scan.""",
    },

    # 2 — Async/await: concurrent fetch with error isolation
    {
        "prompt": "I have an array of URLs. Write an async function that fetches all of them concurrently, but where one failed request does not abort the others. Return an array of { url, status, data } or { url, error } results.",
        "answer": """<thinking>
The requirement is "fail-soft" concurrency: run all requests in parallel, but
isolate failures so one rejection doesn't sink the whole batch.
- `Promise.all` rejects as soon as any promise rejects — wrong for this.
- `Promise.allSettled` waits for every promise and reports each outcome — exactly
  what's needed. It never rejects.
- I'll wrap each fetch in an async helper that normalizes both success and failure
  into a plain result object, then map over the settled results.
- Use AbortController-free simple version; add response.ok check since fetch only
  rejects on network errors, not HTTP 4xx/5xx.
</thinking>

```javascript
/**
 * Fetch many URLs concurrently with per-request error isolation.
 * @param {string[]} urls
 * @returns {Promise<Array<{url:string,status?:number,data?:any,error?:string}>>}
 */
async function fetchAll(urls) {
  const tasks = urls.map(async (url) => {
    try {
      const res = await fetch(url);
      if (!res.ok) {
        // fetch resolves on 4xx/5xx — treat them as errors explicitly.
        throw new Error(`HTTP ${res.status}`);
      }
      const data = await res.json();
      return { url, status: res.status, data };
    } catch (err) {
      return { url, error: err.message };
    }
  });

  // allSettled never rejects; every task already resolves to a result object.
  const settled = await Promise.allSettled(tasks);
  return settled.map((s) => s.value);
}
```

**Explanation**
- **`Promise.allSettled`** is the right primitive for fail-soft batches; `Promise.all` would reject on the first failure and discard successful results.
- I **catch inside each task** and return a normalized object, so the outer `.map(s => s.value)` is always safe — none of the tasks reject.
- **`res.ok` check:** `fetch` only rejects on network-level failures (DNS, offline). A 404 or 500 still resolves, so HTTP errors must be detected manually.
- **Concurrency:** all fetches start before any `await` resolves them, so they run in parallel rather than sequentially.
- For very large URL lists, add a concurrency limit (a pool) to avoid exhausting sockets — unbounded parallelism can overwhelm the server or the browser's connection pool.""",
    },

    # 3 — Promises: implement a concurrency-limited pool
    {
        "prompt": "Write an async function `mapLimit(items, limit, asyncFn)` that runs `asyncFn` over items with at most `limit` running concurrently, preserving output order.",
        "answer": """<thinking>
This is a worker-pool / semaphore pattern. Naive `Promise.all(items.map(fn))`
runs everything at once — bad for rate-limited APIs or memory.
Design:
1. Maintain an index cursor shared by N workers.
2. Spawn `limit` workers; each loops, grabs the next index, runs asyncFn, stores
   the result at that index (preserving order), repeats until items are exhausted.
3. Wait for all workers with Promise.all.
Order is preserved because each result is written to `results[i]`, not pushed.
Edge cases: limit larger than items.length (cap it), empty input.
</thinking>

```javascript
/**
 * Map an async function over items with bounded concurrency, order-preserving.
 * @param {T[]} items
 * @param {number} limit - max in-flight calls
 * @param {(item:T, index:number) => Promise<R>} asyncFn
 * @returns {Promise<R[]>}
 */
async function mapLimit(items, limit, asyncFn) {
  const results = new Array(items.length);
  let cursor = 0;

  async function worker() {
    while (cursor < items.length) {
      const index = cursor++;           // claim an index atomically (single-threaded JS)
      results[index] = await asyncFn(items[index], index);
    }
  }

  const poolSize = Math.min(limit, items.length);
  const workers = Array.from({ length: poolSize }, () => worker());
  await Promise.all(workers);

  return results;
}
```

**Explanation**
- **`cursor++` is atomic in JS** because execution is single-threaded; no two workers ever claim the same index, even though they run "concurrently" via the event loop.
- **Order preservation:** results are written to `results[index]`, so the output array matches input order regardless of which task finishes first.
- **Bounded concurrency:** exactly `poolSize` workers run; each pulls new work only after its previous call settles, keeping in-flight count at or below `limit`.
- **`Math.min(limit, items.length)`** avoids spawning idle workers when there are fewer items than the limit.
- This pattern is essential for rate-limited APIs, database connection pools, and avoiding memory blowups from unbounded parallelism.""",
    },

    # 4 — Closures: private counter / module pattern
    {
        "prompt": "Explain closures by building a `createCounter` factory with truly private state, supporting increment, decrement, and reset to the initial value. Show why the state can't be tampered with from outside.",
        "answer": """<thinking>
A closure is a function bundled with references to its surrounding lexical scope.
The teaching goal: demonstrate genuine encapsulation — `count` should be
unreachable except through the returned methods.
Design:
- `createCounter(initial)` holds `count` in its local scope.
- Return an object of methods that close over `count`.
- `reset` needs the original `initial`, which is also captured by the closure.
Because `count` is a local variable (not a property on the returned object), there
is no reference path to it from outside — only the methods can touch it.
</thinking>

```javascript
function createCounter(initial = 0) {
  let count = initial; // private: lives only in this scope

  return {
    increment() { return ++count; },
    decrement() { return --count; },
    reset()     { count = initial; return count; },
    value()     { return count; },
  };
}

// Usage
const counter = createCounter(10);
counter.increment(); // 11
counter.increment(); // 12
counter.decrement(); // 11
counter.value();     // 11
counter.reset();     // 10

// There is no `counter.count` to read or overwrite:
console.log(counter.count); // undefined — state is truly private
```

**Explanation**
- **Lexical capture:** every returned method "closes over" the same `count` binding. They share one variable, not copies — incrementing then reading reflects the latest value.
- **True privacy:** `count` is a local variable, not a property of the returned object, so outside code has no reference to it. `counter.count` is `undefined`; you cannot corrupt the invariant.
- **`reset` captures `initial`** from the enclosing scope, so it restores the original seed without storing it on the public object.
- This is the classic **module pattern** — the foundation of encapsulation in pre-class JavaScript and still useful when you want privacy without `#private` fields or want multiple independent instances cheaply.""",
    },

    # 5 — Prototypes: prototypal inheritance done correctly
    {
        "prompt": "Without using the `class` keyword, set up prototypal inheritance: an `Animal` constructor with a `speak` method, and a `Dog` that inherits from it and overrides `speak`. Explain the prototype chain and the common mistakes.",
        "answer": """<thinking>
Goal: show inheritance with raw constructor functions + prototypes, the model
`class` syntax sugars over. Two classic mistakes to avoid:
1. `Dog.prototype = new Animal()` — runs the parent constructor too early and
   shares parent instance state. Use `Object.create(Animal.prototype)` instead.
2. Forgetting to reset `Dog.prototype.constructor`, which then wrongly points at Animal.
Also must call the parent constructor with `Animal.call(this, ...)` to init fields.
</thinking>

```javascript
function Animal(name) {
  this.name = name;
}
Animal.prototype.speak = function () {
  return `${this.name} makes a sound`;
};

function Dog(name, breed) {
  Animal.call(this, name); // run parent constructor on the new instance
  this.breed = breed;
}

// Link prototypes WITHOUT invoking the Animal constructor:
Dog.prototype = Object.create(Animal.prototype);
// Restore the constructor reference (Object.create overwrote it):
Dog.prototype.constructor = Dog;

// Override speak on Dog's prototype:
Dog.prototype.speak = function () {
  return `${this.name} barks`;
};

const d = new Dog('Rex', 'Labrador');
d.speak();                      // "Rex barks"
d instanceof Dog;               // true
d instanceof Animal;            // true
Object.getPrototypeOf(d) === Dog.prototype; // true
```

**Explanation**
- **The chain:** `d` → `Dog.prototype` → `Animal.prototype` → `Object.prototype` → `null`. Property lookups walk this chain until found, which is how `instanceof Animal` is also true.
- **`Object.create(Animal.prototype)`** builds a fresh object whose prototype is `Animal.prototype` *without* calling `Animal`. The old anti-pattern `Dog.prototype = new Animal()` runs the parent constructor prematurely and can leak shared mutable state onto the prototype.
- **`Animal.call(this, name)`** runs the parent's field-initialization logic on the new `Dog` instance — the analog of `super(name)`.
- **Resetting `.constructor`** keeps `d.constructor === Dog`, which some libraries and reflection code rely on.
- `class Dog extends Animal` compiles down to exactly this machinery; understanding it demystifies `super`, method resolution, and `instanceof`.""",
    },

    # 6 — DOM manipulation: efficient batched rendering
    {
        "prompt": "I need to append 10,000 list items to the DOM. Doing it in a loop with appendChild is janky. Show the performant approach and explain why it's faster.",
        "answer": """<thinking>
The performance killer is layout thrashing / repeated reflow: each `appendChild`
to a live, rendered element can trigger style recalculation and reflow.
Best practices:
1. Build off-DOM using a DocumentFragment, then insert once — a single reflow.
2. Avoid reading layout properties (offsetHeight, getBoundingClientRect) inside the
   loop, which would force synchronous reflows interleaved with writes.
3. Set textContent (not innerHTML in a loop) to avoid re-parsing and XSS.
I'll show the fragment approach and contrast it with the naive version.
</thinking>

```javascript
/**
 * Append many items to a container with a single reflow.
 * @param {HTMLElement} container
 * @param {string[]} labels
 */
function renderList(container, labels) {
  const fragment = document.createDocumentFragment();

  for (const label of labels) {
    const li = document.createElement('li');
    li.textContent = label;          // textContent: safe + fast, no HTML parsing
    fragment.appendChild(li);        // appends to off-DOM fragment, no reflow
  }

  container.appendChild(fragment);   // ONE insertion → one reflow/repaint
}

// Usage
const data = Array.from({ length: 10000 }, (_, i) => `Item ${i + 1}`);
renderList(document.getElementById('list'), data);
```

**Explanation**
- **DocumentFragment is off-DOM:** mutating it costs nothing in layout because it isn't part of the rendered tree. The single `container.appendChild(fragment)` moves all children in at once, triggering **one** reflow instead of 10,000.
- **`textContent` over `innerHTML`:** in a loop, `innerHTML +=` re-parses the entire string each iteration (quadratic) and risks XSS. `textContent` is escaped and fast.
- **Avoid layout reads in the loop:** touching `offsetWidth`, `getBoundingClientRect`, etc. forces a synchronous reflow to return a fresh value, defeating batching. Keep reads and writes separated.
- **For truly huge lists**, even this isn't enough — use **virtualization** (render only the visible window, e.g. via IntersectionObserver or a library) so the DOM never holds 10,000 nodes at once.""",
    },

    # 7 — Events: delegation
    {
        "prompt": "I'm attaching a click handler to every button in a dynamically-growing table and it's getting unwieldy and leaky. Show me event delegation and explain its advantages.",
        "answer": """<thinking>
The problem: one listener per element doesn't scale, must be re-bound when new
rows are added, and risks leaks if elements are removed without removing listeners.
Solution: event delegation — attach a single listener to a stable ancestor and
use event bubbling. Inside the handler, use `event.target.closest(selector)` to
find the relevant element, and ignore clicks that don't match.
Benefits: O(1) listeners, works for elements added later, simpler teardown.
</thinking>

```javascript
/**
 * One delegated listener handles clicks for all current AND future buttons.
 * @param {HTMLElement} table - stable container present at bind time
 * @param {(rowId:string) => void} onDelete
 */
function attachRowActions(table, onDelete) {
  function handler(event) {
    // Find the nearest delete button from the actual click target:
    const btn = event.target.closest('button.delete');
    if (!btn || !table.contains(btn)) return; // click wasn't on a delete button

    const row = btn.closest('tr');
    onDelete(row.dataset.id);
  }

  table.addEventListener('click', handler);

  // Return a cleanup function to remove the single listener.
  return () => table.removeEventListener('click', handler);
}

// Usage
const cleanup = attachRowActions(document.querySelector('#users'), (id) => {
  console.log('delete user', id);
});
```

**Explanation**
- **One listener, many targets:** the click bubbles from the button up to the table, where the single handler intercepts it. Buttons added *after* binding are handled automatically — no re-binding.
- **`event.target.closest('button.delete')`** walks up from the exact clicked node (which might be an icon inside the button) to find the matching element, then we early-return if it isn't one. This is the delegation core.
- **`table.contains(btn)`** guards against `closest` matching something outside the container in nested layouts.
- **Memory & teardown:** instead of removing N listeners, you remove one. Returning a `cleanup` closure makes lifecycle management trivial (e.g., in a component unmount).
- **When not to delegate:** non-bubbling events (`focus`, `blur` — use `focusin`/`focusout` instead) or when you need per-element listener options like `{ passive: true }` on each.""",
    },

    # 8 — Optimization: throttle vs debounce, with leading/trailing
    {
        "prompt": "Build a `throttle(fn, wait)` that guarantees `fn` runs at most once per `wait` ms, fires on the leading edge, and also fires once on the trailing edge if calls happened during the cooldown. Explain how it differs from debounce.",
        "answer": """<thinking>
Throttle caps invocation rate; debounce waits for quiet. The tricky bit is
combining leading + trailing:
- Leading: invoke immediately on the first call.
- During the cooldown window, remember the latest args.
- Trailing: when the window ends, if there were suppressed calls, invoke once more
  with the most recent args, and restart the window.
State needed: last-invoke timestamp, a pending timer, and stored args/context.
</thinking>

```javascript
/**
 * Throttle with leading + trailing invocation.
 * @param {Function} fn
 * @param {number} wait - minimum ms between invocations
 */
function throttle(fn, wait) {
  let lastCall = 0;       // timestamp of last actual invocation
  let timer = null;       // pending trailing-edge timer
  let savedArgs = null;
  let savedThis = null;

  function invoke(now) {
    lastCall = now;
    fn.apply(savedThis, savedArgs);
    savedArgs = savedThis = null;
  }

  return function throttled(...args) {
    const now = Date.now();
    const remaining = wait - (now - lastCall);
    savedArgs = args;
    savedThis = this;

    if (remaining <= 0) {
      // Leading edge: enough time has passed, fire now.
      if (timer) { clearTimeout(timer); timer = null; }
      invoke(now);
    } else if (!timer) {
      // Schedule the trailing edge with the latest args.
      timer = setTimeout(() => {
        timer = null;
        invoke(Date.now());
      }, remaining);
    }
  };
}
```

**Explanation**
- **Leading edge:** the first call (when `remaining <= 0`) invokes immediately, giving responsive UI feedback.
- **Trailing edge:** calls arriving during the cooldown overwrite `savedArgs` and arm a single timer that fires once when the window expires — so a burst of N calls yields invocations at the start and the end, never more than once per `wait`.
- **Throttle vs debounce:** *throttle* guarantees a steady max rate (ideal for scroll, resize, mousemove, drag); *debounce* waits until activity stops before firing once (ideal for search-as-you-type, validating on input pause). Throttle fires *during* a burst; debounce fires *after* it.
- **`savedThis`/`savedArgs`** preserve the call context and most-recent arguments for the deferred invocation.
- For production, consider `requestAnimationFrame`-based throttling for visual updates so you align with the browser's paint cadence.""",
    },

    # 9 — Data structures: LRU cache
    {
        "prompt": "Implement an LRU (Least Recently Used) cache with O(1) get and put. It should evict the least-recently-used entry when capacity is exceeded.",
        "answer": """<thinking>
LRU needs O(1) for both get and put. The elegant trick: JavaScript's built-in
`Map` preserves insertion order AND lets us delete + re-insert to mark recency.
- get(key): if present, delete and re-set it so it becomes the newest entry; return value.
- put(key, value): if present, delete first (so re-insert updates order); set it.
  If size exceeds capacity, evict the oldest = `map.keys().next().value` (first key).
This avoids hand-rolling a doubly linked list while keeping O(1) amortized ops.
</thinking>

```javascript
class LRUCache {
  /** @param {number} capacity - max entries before eviction */
  constructor(capacity) {
    if (capacity <= 0) throw new RangeError('capacity must be positive');
    this.capacity = capacity;
    this.map = new Map(); // insertion order == recency order (oldest first)
  }

  /** @returns {*} value or undefined */
  get(key) {
    if (!this.map.has(key)) return undefined;
    const value = this.map.get(key);
    this.map.delete(key);     // remove...
    this.map.set(key, value); // ...and re-insert as most-recently-used
    return value;
  }

  put(key, value) {
    if (this.map.has(key)) this.map.delete(key); // refresh ordering
    this.map.set(key, value);

    if (this.map.size > this.capacity) {
      const oldestKey = this.map.keys().next().value; // first = least recent
      this.map.delete(oldestKey);
    }
  }
}

// Usage
const cache = new LRUCache(2);
cache.put('a', 1);
cache.put('b', 2);
cache.get('a');      // 1  -> 'a' is now most recent
cache.put('c', 3);   // evicts 'b' (least recently used)
cache.get('b');      // undefined
```

**Explanation**
- **`Map` insertion order is the recency order.** The oldest entry is whatever was inserted first, retrievable in O(1) via `map.keys().next().value`.
- **delete-then-set on access** moves an entry to the end (newest), which is how we record "recently used" without a linked list.
- **All operations are O(1):** `Map` get/has/delete/set are constant time, and we only ever touch one key per call.
- **Eviction** triggers only after a put pushes size past capacity, removing exactly the front (least recently used) entry.
- The textbook alternative is a hash map + doubly linked list; the `Map` approach gets the same complexity with far less code, which is why it's preferred in interviews and production JS.""",
    },

    # 10 — Design patterns: Observer / pub-sub EventEmitter
    {
        "prompt": "Implement a small typed-ish EventEmitter (pub/sub) with on, once, off, and emit. Make sure unsubscribing during emit is safe and `once` listeners are removed reliably.",
        "answer": """<thinking>
Observer pattern. Pitfalls to handle:
1. Mutating the listener array while iterating during emit — iterate over a copy.
2. `once` must remove itself after firing exactly one time, even if it throws.
3. `off` must remove the correct wrapper for `once` listeners (store original fn
   reference on the wrapper so off(name, original) works).
4. `on` returns an unsubscribe function for ergonomic cleanup.
</thinking>

```javascript
class EventEmitter {
  constructor() {
    this.listeners = new Map(); // event -> Set of handler fns
  }

  on(event, handler) {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event).add(handler);
    return () => this.off(event, handler); // unsubscribe closure
  }

  once(event, handler) {
    const wrapper = (...args) => {
      this.off(event, wrapper);
      handler(...args);
    };
    wrapper._original = handler; // allow off(event, handler) to find the wrapper
    return this.on(event, wrapper);
  }

  off(event, handler) {
    const set = this.listeners.get(event);
    if (!set) return;
    // Remove direct match or a once-wrapper whose original is `handler`.
    for (const fn of set) {
      if (fn === handler || fn._original === handler) set.delete(fn);
    }
    if (set.size === 0) this.listeners.delete(event);
  }

  emit(event, ...args) {
    const set = this.listeners.get(event);
    if (!set) return false;
    // Iterate a snapshot so handlers can safely on/off during emit.
    for (const fn of [...set]) fn(...args);
    return true;
  }
}
```

**Explanation**
- **Snapshot iteration (`[...set]`)** in `emit` prevents bugs when a handler subscribes or unsubscribes during dispatch — the loop runs over a frozen copy, matching Node's EventEmitter semantics.
- **`once`** wraps the handler so it unsubscribes *before* invoking the user fn; removing first guarantees it fires exactly once even if the handler throws.
- **`wrapper._original`** lets `off(event, handler)` cancel a `once` subscription using the original function reference, not the internal wrapper.
- **`on` returns an unsubscribe closure**, the modern ergonomic pattern (mirrors `addEventListener` cleanup and React effects).
- A `Set` avoids duplicate registrations and gives O(1) add/delete. This decoupled pub/sub is the backbone of event-driven architectures, state stores, and DOM-free domain events.""",
    },

    # 11 — Algorithms: deep clone with cycles
    {
        "prompt": "Write a deep clone function that handles nested objects/arrays, Date, Map, Set, and circular references. Explain why JSON.parse(JSON.stringify(x)) is insufficient.",
        "answer": """<thinking>
JSON round-trip loses Dates (become strings), Maps/Sets (become {}), functions,
undefined, and throws on cycles. A robust clone needs:
1. A WeakMap to track already-cloned objects -> handles cycles and shared refs.
2. Type dispatch for Date, Map, Set, Array, plain object.
3. Recursion that consults the WeakMap before cloning.
structuredClone is the modern built-in, but I'll implement it to show the mechanics
and because it doesn't clone functions/prototypes either. Mention structuredClone.
</thinking>

```javascript
function deepClone(value, seen = new WeakMap()) {
  // Primitives and functions are returned as-is.
  if (value === null || typeof value !== 'object') return value;

  // Cycle / shared-reference guard.
  if (seen.has(value)) return seen.get(value);

  if (value instanceof Date) return new Date(value.getTime());

  if (value instanceof Map) {
    const result = new Map();
    seen.set(value, result);
    for (const [k, v] of value) result.set(deepClone(k, seen), deepClone(v, seen));
    return result;
  }

  if (value instanceof Set) {
    const result = new Set();
    seen.set(value, result);
    for (const v of value) result.add(deepClone(v, seen));
    return result;
  }

  if (Array.isArray(value)) {
    const result = [];
    seen.set(value, result);          // register BEFORE recursing (cycles)
    for (let i = 0; i < value.length; i++) result[i] = deepClone(value[i], seen);
    return result;
  }

  // Plain object: preserve prototype.
  const result = Object.create(Object.getPrototypeOf(value));
  seen.set(value, result);
  for (const key of Reflect.ownKeys(value)) {
    result[key] = deepClone(value[key], seen);
  }
  return result;
}
```

**Explanation**
- **Why not JSON:** `JSON.parse(JSON.stringify(x))` silently corrupts `Date` (→ ISO string), `Map`/`Set` (→ `{}`), drops `undefined` and functions, mangles `NaN`/`Infinity` (→ `null`), and **throws on circular references**.
- **`WeakMap` `seen`** maps each source object to its clone. Registering the clone *before* recursing into its children means a cycle (`a.self = a`) resolves to the already-created clone instead of recursing forever. It also preserves shared references (two properties pointing at the same object stay shared).
- **Type dispatch** reconstructs `Date`, `Map`, and `Set` properly instead of producing empty objects.
- **`Object.create(getPrototypeOf)` + `Reflect.ownKeys`** preserves the prototype and copies symbol keys too, which a naive `for...in` would miss.
- **Modern alternative:** the built-in `structuredClone(value)` handles all of the above and more — prefer it when available. This hand-rolled version is worth understanding for the algorithm and for environments/edge cases where you need custom type handling.""",
    },

    # 12 — Async: cancellable fetch with AbortController + timeout
    {
        "prompt": "Wrap fetch so a request automatically aborts after a timeout, and the caller can also cancel manually. Distinguish a timeout from a user-initiated cancel in the error.",
        "answer": """<thinking>
AbortController is the idiom for cancellation. To support BOTH a timeout and a
manual cancel, I can:
- Create a timeout controller internally.
- Accept an external signal and link it, OR return an abort function.
Cleanest: create an internal controller, abort it on timeout with a custom reason,
and expose a `.cancel()` so the caller can abort with a different reason. Then I can
inspect `signal.reason` (or the controller) to tell timeout from manual cancel.
Must clear the timeout in a finally to avoid leaks.
</thinking>

```javascript
/**
 * fetch with an automatic timeout and manual cancellation.
 * @returns {{ promise: Promise<Response>, cancel: () => void }}
 */
function fetchWithTimeout(url, { timeout = 5000, ...options } = {}) {
  const controller = new AbortController();
  const cancel = () => controller.abort(new DOMException('Cancelled', 'AbortError'));

  const timer = setTimeout(
    () => controller.abort(new DOMException('Timeout', 'TimeoutError')),
    timeout
  );

  const promise = fetch(url, { ...options, signal: controller.signal })
    .catch((err) => {
      // Distinguish the abort cause by the reason we set.
      if (controller.signal.aborted) {
        const reason = controller.signal.reason;
        throw reason instanceof Error ? reason : new Error(String(reason));
      }
      throw err; // genuine network error
    })
    .finally(() => clearTimeout(timer)); // always clean up the timer

  return { promise, cancel };
}

// Usage
const { promise, cancel } = fetchWithTimeout('/api/data', { timeout: 3000 });
promise
  .then((res) => res.json())
  .catch((err) => {
    if (err.name === 'TimeoutError') console.warn('Request timed out');
    else if (err.name === 'AbortError') console.warn('User cancelled');
    else console.error('Network error', err);
  });

// cancel(); // call to abort manually
```

**Explanation**
- **`AbortController` + `signal`** is the standard cancellation mechanism; passing `signal` to `fetch` makes it reject with an abort error when `abort()` is called.
- **`abort(reason)`** accepts a custom reason. By aborting with a `TimeoutError` on the timer and an `AbortError` on manual cancel, the caller can branch on `err.name` to tell the two apart.
- **`finally(clearTimeout)`** prevents a dangling timer (and a stray late abort) once the request settles — a common leak if forgotten.
- **Returning `{ promise, cancel }`** gives the caller imperative control without exposing the internal controller.
- The same `signal` can be forwarded to other abortable APIs (e.g., `addEventListener(..., { signal })`) to cancel a whole group of operations at once.""",
    },

    # 13 — Closures gotcha: var in loops
    {
        "prompt": "This loop logs `3, 3, 3` instead of `0, 1, 2`. Explain exactly why with the var version, and show three correct fixes. `for (var i = 0; i < 3; i++) setTimeout(() => console.log(i), 0)`",
        "answer": """<thinking>
Classic closure-over-loop-variable bug. Root cause: `var` is function-scoped, so
all three callbacks close over the SAME `i` binding. By the time the timeouts fire
(after the synchronous loop completes), `i` is 3.
Three fixes:
1. `let` — block-scoped, fresh binding per iteration (the modern fix).
2. IIFE — capture the current value in a new function scope.
3. Pass i as an argument / bind it — `setTimeout(fn, 0, i)`.
Explain the per-iteration binding semantics of `let` in for-loops specifically.
</thinking>

```javascript
// THE BUG: one shared `i`, captured by reference, equals 3 when callbacks run.
for (var i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0); // logs 3, 3, 3
}

// FIX 1 — `let`: block-scoped, the for-loop creates a NEW binding each iteration.
for (let i = 0; i < 3; i++) {
  setTimeout(() => console.log(i), 0); // logs 0, 1, 2
}

// FIX 2 — IIFE: capture the value in a fresh function scope.
for (var i = 0; i < 3; i++) {
  ((j) => setTimeout(() => console.log(j), 0))(i); // 0, 1, 2
}

// FIX 3 — pass the value as a setTimeout argument (bound at call time).
for (var i = 0; i < 3; i++) {
  setTimeout((j) => console.log(j), 0, i); // 0, 1, 2
}
```

**Explanation**
- **Why `var` fails:** `var i` is hoisted to one function-scoped binding. All three arrow functions close over that single variable *by reference*, not by value. The synchronous loop runs to completion (`i === 3`) before any timeout callback executes, so all log `3`.
- **Fix 1 (`let`)** is special: the spec says a `for (let ...)` loop creates a **fresh binding per iteration** and copies the value forward, so each closure captures its own `i`. This is the idiomatic modern solution.
- **Fix 2 (IIFE)** predates `let`: invoking a function immediately creates a new scope whose parameter `j` holds the current value, which the inner closure captures.
- **Fix 3** uses `setTimeout`'s extra-args feature: the value of `i` is evaluated and passed *now*, so the callback receives the correct per-iteration value.
- Takeaway: prefer `let`/`const` over `var` precisely to avoid this whole class of capture bugs.""",
    },

    # 14 — Algorithms: memoization decorator
    {
        "prompt": "Write a generic `memoize` higher-order function that caches results of a pure function by its arguments. Handle multiple arguments correctly and let me supply a custom key resolver.",
        "answer": """<thinking>
Memoization trades memory for speed on pure functions. Design points:
1. Multi-arg keying — JSON.stringify of args is a decent default but breaks on
   functions/circular/order-insensitive cases. Allow a custom resolver.
2. Use a Map keyed by the resolved key.
3. Preserve `this` so it works on methods.
4. Cache the result even if it's undefined (distinguish "absent" from "cached undefined")
   by using map.has, not truthiness.
Mention pitfalls: only memoize pure functions; unbounded cache growth.
</thinking>

```javascript
/**
 * Memoize a pure function.
 * @param {Function} fn
 * @param {(args:any[]) => string} [resolver] - custom cache-key builder
 */
function memoize(fn, resolver = (args) => JSON.stringify(args)) {
  const cache = new Map();

  function memoized(...args) {
    const key = resolver(args);
    if (cache.has(key)) return cache.get(key); // .has handles cached `undefined`
    const result = fn.apply(this, args);       // preserve `this` for methods
    cache.set(key, result);
    return result;
  }

  memoized.clear = () => cache.clear(); // allow manual cache invalidation
  return memoized;
}

// Usage: expensive pure computation
const slowFib = (n) => (n < 2 ? n : slowFib(n - 1) + slowFib(n - 2));
const fib = memoize(slowFib);
fib(30); // computed once
fib(30); // returned from cache

// Custom resolver: order-insensitive args
const add = memoize((a, b) => a + b, (args) => [...args].sort().join(','));
```

**Explanation**
- **`cache.has(key)` not `cache.get(key)`:** checking presence (not truthiness) means a legitimately cached `undefined`, `0`, `false`, or `null` is still served from cache rather than recomputed.
- **Custom `resolver`:** the default `JSON.stringify(args)` works for primitive/serializable args but fails on functions, circular refs, and treats argument order as significant. A custom resolver handles object identity, order-insensitivity, or selecting a single significant argument.
- **`fn.apply(this, args)`** preserves the receiver so the wrapper can decorate object methods, not just free functions.
- **Caveats:** only memoize **pure** functions (same input → same output, no side effects). The cache grows unbounded — for hot paths with many distinct inputs, pair this with an LRU eviction policy. Use `WeakMap` keying when the cache key is a single object argument so entries can be garbage-collected.""",
    },

    # 15 — Async: serial vs parallel, and the await-in-loop trap
    {
        "prompt": "My code `for (const id of ids) { results.push(await fetchUser(id)) }` is slow. Explain the await-in-loop trap, and show when to keep it serial versus parallelize it.",
        "answer": """<thinking>
The trap: awaiting inside a for-loop serializes requests — each waits for the prior
to finish, so total time ~= sum of latencies. Often you want them concurrent.
Teach:
- Parallel with Promise.all(ids.map(fetchUser)) when requests are independent.
- Keep serial when each step depends on the previous, or to respect rate limits /
  ordering / backpressure.
- Show a bounded-concurrency middle ground reference.
Also note ordering: Promise.all preserves input order regardless of completion order.
</thinking>

```javascript
// SLOW: serial — each await blocks the next request. Total ~ sum of latencies.
async function serial(ids) {
  const results = [];
  for (const id of ids) {
    results.push(await fetchUser(id)); // waits for THIS before starting next
  }
  return results;
}

// FAST: parallel — fire all requests, then await together.
async function parallel(ids) {
  // map creates all promises immediately (concurrent), Promise.all preserves order.
  return Promise.all(ids.map((id) => fetchUser(id)));
}

// WHEN ORDER/DEPENDENCY MATTERS: keep it serial deliberately.
async function dependentChain(ids) {
  let token = null;
  const out = [];
  for (const id of ids) {
    const user = await fetchUser(id, token); // each call needs the previous result
    token = user.nextToken;
    out.push(user);
  }
  return out;
}
```

**Explanation**
- **The await-in-loop trap:** `await` pauses the loop until the promise settles, so independent requests run back-to-back. With 10 requests at 200 ms each, serial takes ~2 s; parallel takes ~200 ms.
- **Parallelize with `Promise.all(map(...))`** when calls are independent. The `.map` starts every promise *before* any `await`, so they overlap. `Promise.all` returns results in **input order**, not completion order — so indexing stays correct.
- **Keep it serial on purpose** when: each step needs the previous step's output (pagination tokens, dependent writes), you must preserve side-effect ordering, or you're respecting a strict rate limit / avoiding overwhelming a backend.
- **Middle ground:** for large `ids`, unbounded `Promise.all` can exhaust connections — use a concurrency-limited pool (see `mapLimit`) to run, say, 5 at a time.
- Rule of thumb: reach for `Promise.all` by default for independent async work; use a sequential loop only when there's a real dependency or backpressure reason.""",
    },

    # 16 — Data structures: min-heap / priority queue
    {
        "prompt": "Implement a binary min-heap (priority queue) with push and pop in O(log n), supporting a custom comparator. Explain the sift-up/sift-down logic.",
        "answer": """<thinking>
A binary heap stored in an array: for node at index i, children are 2i+1 and 2i+2,
parent is (i-1)>>1. Min-heap invariant: parent <= children.
- push: append at end, sift-up (bubble the new element toward root while smaller
  than parent).
- pop: swap root with last, remove last, sift-down (sink the new root while larger
  than its smallest child).
Accept a comparator so it works for arbitrary priorities. Default min-heap.
</thinking>

```javascript
class MinHeap {
  /** @param {(a:any, b:any) => number} compare - <0 means a has higher priority */
  constructor(compare = (a, b) => a - b) {
    this.heap = [];
    this.compare = compare;
  }

  get size() { return this.heap.length; }
  peek() { return this.heap[0]; }

  push(value) {
    this.heap.push(value);
    this._siftUp(this.heap.length - 1);
  }

  pop() {
    const top = this.heap[0];
    const last = this.heap.pop();
    if (this.heap.length > 0) {
      this.heap[0] = last;      // move last element to root
      this._siftDown(0);        // restore heap property
    }
    return top;
  }

  _siftUp(i) {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (this.compare(this.heap[i], this.heap[parent]) >= 0) break;
      [this.heap[i], this.heap[parent]] = [this.heap[parent], this.heap[i]];
      i = parent;
    }
  }

  _siftDown(i) {
    const n = this.heap.length;
    while (true) {
      let smallest = i;
      const left = 2 * i + 1, right = 2 * i + 2;
      if (left < n && this.compare(this.heap[left], this.heap[smallest]) < 0) smallest = left;
      if (right < n && this.compare(this.heap[right], this.heap[smallest]) < 0) smallest = right;
      if (smallest === i) break;
      [this.heap[i], this.heap[smallest]] = [this.heap[smallest], this.heap[i]];
      i = smallest;
    }
  }
}

// Usage: task queue by priority number (lower = higher priority)
const pq = new MinHeap((a, b) => a.priority - b.priority);
pq.push({ priority: 5, job: 'email' });
pq.push({ priority: 1, job: 'page-oncall' });
pq.pop(); // { priority: 1, job: 'page-oncall' }
```

**Explanation**
- **Array-encoded tree:** index math (`parent = (i-1)>>1`, `children = 2i+1, 2i+2`) avoids node objects and gives cache-friendly contiguous storage.
- **`push` → sift-up:** append at the end (an O(1) array push), then bubble it upward, swapping with its parent while it has higher priority. At most `log n` swaps.
- **`pop` → sift-down:** the root is the answer; replace it with the last element and sink that element down, repeatedly swapping with its higher-priority child until the invariant holds. Again `O(log n)`.
- **Custom comparator** generalizes it to max-heaps (negate the comparator) or complex priorities, as in the task-queue example.
- Priority queues power Dijkstra's algorithm, A* search, event simulators, scheduling, and "top-k" streaming problems. JS lacks a built-in heap, so this is a frequently needed building block.""",
    },

    # 17 — Patterns: function currying & partial application
    {
        "prompt": "Implement a generic `curry` function so that `curry(fn)(a)(b)(c)`, `curry(fn)(a, b)(c)`, and `curry(fn)(a, b, c)` all work for a 3-arg function. Explain currying vs partial application.",
        "answer": """<thinking>
Currying transforms f(a,b,c) into a chain that can be supplied in any grouping
until enough args accumulate. Implementation:
- Track collected args; if we have >= fn.length, invoke. Otherwise return a function
  that collects more.
- Use fn.length (arity) to know when we're "full".
- Preserve `this`.
Distinguish: currying = always unary-step transformation (conceptually); partial
application = fixing some args producing a function of the rest. My flexible version
blends both, which is what people usually want in JS.
</thinking>

```javascript
/**
 * Curry a fixed-arity function. Args can be supplied one at a time or in groups.
 * @param {Function} fn - must have a meaningful .length (no rest params)
 */
function curry(fn) {
  return function curried(...args) {
    if (args.length >= fn.length) {
      return fn.apply(this, args);           // enough args -> invoke
    }
    // Not enough yet: return a collector that appends future args.
    return function (...next) {
      return curried.apply(this, args.concat(next));
    };
  };
}

// Usage
const sum3 = (a, b, c) => a + b + c;
const csum = curry(sum3);

csum(1)(2)(3);   // 6  — one at a time
csum(1, 2)(3);   // 6  — grouped
csum(1)(2, 3);   // 6
csum(1, 2, 3);   // 6  — all at once

// Partial application falls out naturally:
const add10 = csum(10);     // waiting for two more args
add10(5, 5);                // 20
```

**Explanation**
- **Arity check via `fn.length`:** once the accumulated args reach the function's declared parameter count, we invoke; otherwise we return a collector that concatenates the next batch and recurses through `curried`. This makes every supply pattern work.
- **Currying vs partial application:** *currying* converts an n-ary function into a sequence of applications (classically one argument each); *partial application* pre-binds some arguments and returns a function awaiting the rest. This flexible `curry` supports both — `csum(10)` is effectively partial application.
- **`this` preservation** via `apply` lets it decorate methods.
- **Practical value:** currying enables point-free composition and reusable specialized functions (`const half = divide(_, 2)`-style helpers), heavily used in functional libraries like Ramda and lodash/fp.
- **Caveat:** it relies on `fn.length`, which excludes rest/default params; for variadic functions, pass an explicit arity instead.""",
    },

    # 18 — DOM/Observer: IntersectionObserver lazy loading
    {
        "prompt": "Implement lazy-loading of images using IntersectionObserver so images only load when near the viewport. Include cleanup and a fallback for unsupported browsers.",
        "answer": """<thinking>
Lazy loading via IntersectionObserver beats scroll-event listeners (no jank, the
browser computes intersections off the main thread).
Plan:
1. Select images with data-src placeholders.
2. Observer with a rootMargin so images preload slightly before entering view.
3. On intersect: set src from data-src, then unobserve that image (one-shot).
4. Fallback: if IntersectionObserver is missing, load everything eagerly.
5. Return a disconnect() for cleanup.
Also mention native loading="lazy" as the simplest option.
</thinking>

```javascript
/**
 * Lazy-load <img data-src="..."> elements within `root`.
 * @returns {() => void} disconnect function for cleanup
 */
function lazyLoadImages(root = document) {
  const images = root.querySelectorAll('img[data-src]');

  // Fallback: no IntersectionObserver -> just load everything.
  if (!('IntersectionObserver' in window)) {
    images.forEach((img) => { img.src = img.dataset.src; });
    return () => {};
  }

  const observer = new IntersectionObserver(
    (entries, obs) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const img = entry.target;
        img.src = img.dataset.src;
        delete img.dataset.src;
        obs.unobserve(img); // one-shot: stop watching once loaded
      }
    },
    {
      root: null,            // viewport
      rootMargin: '200px',   // start loading 200px before it enters view
      threshold: 0.01,
    }
  );

  images.forEach((img) => observer.observe(img));
  return () => observer.disconnect(); // cleanup all at once
}

// Usage
const stop = lazyLoadImages();
// stop(); // e.g. on route change / component unmount
```

**Explanation**
- **IntersectionObserver vs scroll listeners:** the browser computes visibility asynchronously and efficiently, avoiding the layout-thrash and jank of `scroll`-event polling with `getBoundingClientRect`.
- **`rootMargin: '200px'`** inflates the detection box so images begin loading *before* they scroll into view, hiding network latency from the user.
- **One-shot `unobserve`:** once an image's real `src` is set, we stop observing it to avoid redundant work; `disconnect()` tears down everything at once for cleanup.
- **Progressive enhancement:** the feature-detect fallback loads all images eagerly where the API is unavailable, so nothing breaks.
- **Even simpler when applicable:** native `<img loading="lazy">` requires no JS and is the right default; reach for IntersectionObserver when you need custom thresholds, preloading distance, or to lazy-load non-image content (analytics, infinite scroll, animations).""",
    },

    # 19 — Algorithms: group/partition with reduce, immutably
    {
        "prompt": "Given an array of transaction objects, group them by category and compute the total amount per category, without mutating the inputs. Show an idiomatic reduce-based solution and note the new Object.groupBy.",
        "answer": """<thinking>
Aggregation with reduce. Requirements: group by a key, sum a numeric field, keep
it immutable (don't mutate input objects).
Approach with reduce into an accumulator object/Map. Use a Map for arbitrary keys
and clean iteration. Build new objects rather than mutating.
Also mention the modern Object.groupBy / Map.groupBy if available.
Edge: missing/zero amounts, ensure numeric coercion is intentional.
</thinking>

```javascript
/**
 * Sum `amount` per `category`. Pure: inputs are never mutated.
 * @param {{category:string, amount:number}[]} transactions
 * @returns {Record<string, { count:number, total:number }>}
 */
function totalsByCategory(transactions) {
  return transactions.reduce((acc, tx) => {
    const current = acc[tx.category] ?? { count: 0, total: 0 };
    // Build a NEW entry object instead of mutating `current`/`tx`.
    acc[tx.category] = {
      count: current.count + 1,
      total: current.total + Number(tx.amount || 0),
    };
    return acc;
  }, {});
}

// Usage
const txns = [
  { category: 'food', amount: 12.5 },
  { category: 'travel', amount: 40 },
  { category: 'food', amount: 7.5 },
];
totalsByCategory(txns);
// { food: { count: 2, total: 20 }, travel: { count: 1, total: 40 } }

// Modern built-in for the grouping step (where supported):
const grouped = Object.groupBy(txns, (tx) => tx.category);
// { food: [ {...}, {...} ], travel: [ {...} ] }
```

**Explanation**
- **`reduce` as an aggregator:** the accumulator carries running per-category stats. Using `acc[key] ?? default` initializes lazily on first sight of a category.
- **Immutability:** we never write back into `tx` or reuse the previous entry object — each update creates a fresh `{ count, total }`. The input array and its objects are untouched, which keeps the function pure and safe to call repeatedly or in concurrent React renders.
- **`Number(tx.amount || 0)`** makes coercion explicit and guards against `undefined`/`null` amounts producing `NaN`.
- **`Object.groupBy(items, keyFn)`** (and `Map.groupBy` for non-string keys) is the modern, declarative way to do the *grouping* step; combine it with a sum if you only needed buckets. Prefer it when targeting recent runtimes for readability.
- For very large datasets or non-string keys, use a `Map` accumulator to avoid prototype-key collisions (e.g., a category literally named `"constructor"`).""",
    },

    # 20 — `this` binding & arrow functions
    {
        "prompt": "Explain how `this` is determined in JavaScript across regular functions, methods, arrow functions, and event handlers. Give a concrete bug where an arrow function or .bind fixes a lost `this`.",
        "answer": """<thinking>
`this` rules, in priority order:
1. `new` -> the new instance.
2. explicit bind/call/apply -> the given object.
3. method call obj.fn() -> obj.
4. plain call fn() -> undefined (strict) / global (sloppy).
5. arrow functions -> NO own `this`; they capture lexically from the enclosing scope.
Common bug: passing a method as a callback (e.g., to setTimeout or addEventListener)
loses the receiver; the call becomes a plain call. Fix with arrow (lexical) or .bind.
I'll show a class with the bug and two fixes.
</thinking>

```javascript
class Timer {
  constructor() {
    this.seconds = 0;
  }

  // BUG: passing `this.tick` strips its receiver -> `this` is undefined inside.
  startBroken() {
    setInterval(this.tick, 1000); // `tick` called as a plain function
  }
  tick() {
    this.seconds++; // TypeError: Cannot read properties of undefined
  }

  // FIX 1 — arrow wrapper captures `this` lexically from startFixed's scope:
  startFixed() {
    setInterval(() => this.tick(), 1000);
  }

  // FIX 2 — bind the receiver permanently (e.g., in the constructor):
  startBound() {
    setInterval(this.tick.bind(this), 1000);
  }
}

// Demonstrating the rules:
const obj = {
  name: 'obj',
  regular() { return this.name; },      // method call -> `this` is obj
  arrow: () => this,                    // arrow -> lexical `this` (module/undefined)
};
obj.regular();                           // 'obj'
const detached = obj.regular;
// detached();                           // `this` is undefined (strict) -> throws
```

**Explanation**
- **Binding rules (highest priority first):** `new` > explicit `.call/.apply/.bind` > method invocation (`obj.fn()` → `obj`) > plain call (`fn()` → `undefined` in strict mode). **Arrow functions ignore all of this** — they have no own `this` and capture it lexically from where they're *defined*.
- **The lost-`this` bug:** `setInterval(this.tick, 1000)` passes the function value alone. When the timer later invokes it, there's no `obj.` in front, so it's a plain call and `this` is `undefined`.
- **Fix 1 (arrow wrapper):** `() => this.tick()` is defined inside a method, so its lexical `this` is the instance; it then calls `tick` as a method, restoring the receiver.
- **Fix 2 (`.bind(this)`):** permanently fixes the receiver, returning a new function whose `this` can't be re-bound. Useful for passing the same bound reference to both `addEventListener` and `removeEventListener`.
- **Rule of thumb:** use arrow functions for callbacks that should inherit the surrounding `this` (class field handlers, React effects); use regular methods when you want dynamic `this` (the object the method is called on).""",
    },

    # 21 — Generators & lazy iteration
    {
        "prompt": "Use a generator to produce an infinite sequence lazily, and write a `take(n, iterable)` helper. Show generating the first 5 Fibonacci numbers without precomputing or storing the whole sequence.",
        "answer": """<thinking>
Generators yield values on demand, enabling lazy/infinite sequences with O(1) memory.
- `fibonacci()` is an infinite generator that yields successive Fibonacci numbers.
- `take(n, iterable)` pulls n values then stops — works on ANY iterable/iterator,
  including infinite ones, because it never asks for more than n.
Key teaching point: laziness — nothing is computed until `next()` is called, so an
infinite generator is safe as long as consumers bound it.
</thinking>

```javascript
/** Infinite lazy sequence of Fibonacci numbers. */
function* fibonacci() {
  let [a, b] = [0, 1];
  while (true) {
    yield a;
    [a, b] = [b, a + b]; // advance only when next() is pulled
  }
}

/**
 * Take the first n values from any iterable (safe on infinite ones).
 * @param {number} n
 * @param {Iterable<T>} iterable
 * @returns {T[]}
 */
function take(n, iterable) {
  const out = [];
  const iterator = iterable[Symbol.iterator]();
  while (out.length < n) {
    const { value, done } = iterator.next();
    if (done) break;        // finite iterable shorter than n
    out.push(value);
  }
  return out;
}

// Usage
take(5, fibonacci()); // [0, 1, 1, 2, 3]  — only 5 values ever computed
```

**Explanation**
- **Lazy evaluation:** the generator's body suspends at each `yield` and resumes only when the consumer calls `next()`. So `fibonacci()` can loop forever yet costs O(1) memory and computes exactly as many values as are pulled.
- **`take` bounds the infinite source:** it requests at most `n` values, so pairing a bounded consumer with an unbounded producer is perfectly safe — the `while (true)` never runs away.
- **Works on any iterable:** `take` uses the `Symbol.iterator` protocol, so it also handles arrays, Sets, Maps, and custom iterables; it stops early via the `done` flag for finite sources shorter than `n`.
- **Why generators:** they express streaming pipelines (parsing, paginated APIs, event streams) without materializing entire collections, and they make state machines and cooperative iteration readable.
- Generators also compose: you can build `map`/`filter` generator helpers that stay lazy end-to-end, only doing work as the final consumer pulls.""",
    },

    # 22 — Patterns: immutable state update (Redux-style reducer)
    {
        "prompt": "Show how to update deeply nested state immutably in a reducer (no mutation), e.g. toggling a todo's completed flag inside a list inside an object. Explain why immutability matters and the spread pitfalls.",
        "answer": """<thinking>
Immutable updates underpin Redux/React change detection (reference equality). The
challenge is nested updates: you must create new objects/arrays along the entire
path from the root to the changed leaf, sharing untouched branches.
Pitfall: shallow spread copies only one level; nested objects remain shared, so a
naive `{...state, todos: state.todos}` still aliases the inner todos.
Show a reducer that spreads at each level and maps the array to replace one item.
Mention structuredClone is wrong here (breaks sharing/perf) and immer as a tool.
</thinking>

```javascript
const initialState = {
  user: { name: 'Ana' },
  todos: [
    { id: 1, text: 'Write code', completed: false },
    { id: 2, text: 'Tests', completed: false },
  ],
};

function reducer(state, action) {
  switch (action.type) {
    case 'TOGGLE_TODO':
      return {
        ...state,                          // new root object
        todos: state.todos.map((todo) =>   // new array
          todo.id === action.id
            ? { ...todo, completed: !todo.completed } // new object for the ONE changed item
            : todo                          // untouched items keep their reference
        ),
      };

    case 'RENAME_USER':
      return {
        ...state,
        user: { ...state.user, name: action.name }, // new user object, rest shared
      };

    default:
      return state; // SAME reference -> signals "no change"
  }
}
```

**Explanation**
- **Why immutability:** React/Redux detect changes by **reference equality** (`prevState !== nextState`). If you mutate in place, the reference is unchanged, so memoized components and `useMemo`/`React.memo` won't re-render. Returning a new object only along the changed path makes change detection cheap and correct.
- **Structural sharing:** only objects on the path root → changed leaf are recreated; sibling branches keep their old references. In `TOGGLE_TODO`, unchanged todos are returned as-is, so `===` comparisons on them stay true (good for list memoization).
- **Shallow-spread pitfall:** `{ ...state }` copies one level only. Nested objects/arrays are still shared, so you must spread *at every level* you intend to change — that's why `user` and the matched `todo` each get their own spread.
- **`default: return state`** returns the identical reference, the canonical "nothing changed" signal that prevents needless re-renders.
- **Tools:** for deeply nested updates, **Immer** (`produce(state, draft => { draft.todos[0].completed = true })`) lets you write mutative-looking code while producing a correctly structure-shared immutable result. Avoid `structuredClone` for this — it deep-copies everything, destroying structural sharing and hurting performance.""",
    },

    # 23 — Algorithms: sliding window (longest substring without repeats)
    {
        "prompt": "Find the length of the longest substring without repeating characters. Give an optimal O(n) sliding-window solution and explain the window invariant.",
        "answer": """<thinking>
Brute force checks all substrings: O(n^2) or worse. Sliding window achieves O(n):
- Maintain a window [left, right] that always contains unique characters.
- Use a Map from char -> last index seen.
- As right advances, if the char was seen AND its last index is >= left, jump left
  to (lastIndex + 1) to drop the duplicate.
- Track max window size.
Invariant: the substring s[left..right] never contains a repeat.
Edge cases: empty string -> 0; all same chars -> 1.
</thinking>

```javascript
/**
 * Length of the longest substring with all-unique characters. O(n) time, O(k) space.
 * @param {string} s
 * @returns {number}
 */
function lengthOfLongestSubstring(s) {
  const lastSeen = new Map(); // char -> most recent index
  let left = 0;
  let best = 0;

  for (let right = 0; right < s.length; right++) {
    const ch = s[right];

    // If we've seen ch inside the current window, shrink from the left.
    if (lastSeen.has(ch) && lastSeen.get(ch) >= left) {
      left = lastSeen.get(ch) + 1;
    }

    lastSeen.set(ch, right);
    best = Math.max(best, right - left + 1); // current window size
  }

  return best;
}

// Examples
lengthOfLongestSubstring('abcabcbb'); // 3  ("abc")
lengthOfLongestSubstring('bbbbb');    // 1  ("b")
lengthOfLongestSubstring('pwwkew');   // 3  ("wke")
lengthOfLongestSubstring('');         // 0
```

**Explanation**
- **Window invariant:** at every step, `s[left..right]` contains no repeated character. The whole algorithm is about maintaining this as `right` advances.
- **Why move `left` to `lastSeen.get(ch) + 1`:** when the new character duplicates one already inside the window, the only way to restore uniqueness is to drop everything up to and including the previous occurrence. We never move `left` backward, which is what guarantees linear time.
- **The `>= left` guard** is crucial: a duplicate that last appeared *before* `left` is already outside the window and must be ignored, otherwise `left` could jump backward and break the O(n) bound.
- **Complexity:** each index is visited by `right` once and `left` only moves forward, so O(n) time; the Map holds at most the alphabet size, O(min(n, charset)) space.
- Sliding window is the go-to pattern for "longest/shortest contiguous subarray/substring satisfying a constraint" problems — recognizing it turns many O(n²) problems into O(n).""",
    },

    # 24 — Async: retry with exponential backoff and jitter
    {
        "prompt": "Implement an async `retry` helper with exponential backoff and jitter, a max-attempts cap, and a predicate to decide which errors are retryable. Explain why jitter matters.",
        "answer": """<thinking>
Robust retry needs:
1. Max attempts to avoid infinite loops.
2. Exponential backoff: delay = base * 2^attempt, capped at a max.
3. Jitter: randomize the delay to avoid thundering-herd / synchronized retries.
4. A `shouldRetry(error)` predicate so non-retryable errors (e.g., 400/401) fail fast.
5. Respect an AbortSignal optionally; at least a sleep helper.
I'll implement full jitter: delay = random(0, cappedExp). Re-throw the last error
after exhausting attempts.
</thinking>

```javascript
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Retry an async operation with exponential backoff + full jitter.
 * @param {() => Promise<T>} operation
 * @param {object} [opts]
 * @param {number} [opts.maxAttempts=5]
 * @param {number} [opts.baseDelay=200] - ms
 * @param {number} [opts.maxDelay=10000] - ms cap
 * @param {(err:any) => boolean} [opts.shouldRetry] - retry predicate
 * @returns {Promise<T>}
 */
async function retry(operation, opts = {}) {
  const {
    maxAttempts = 5,
    baseDelay = 200,
    maxDelay = 10000,
    shouldRetry = () => true,
  } = opts;

  let lastError;
  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (err) {
      lastError = err;
      // Don't retry non-retryable errors or after the final attempt.
      if (!shouldRetry(err) || attempt === maxAttempts - 1) break;

      const expDelay = Math.min(maxDelay, baseDelay * 2 ** attempt);
      const jittered = Math.random() * expDelay; // full jitter: [0, expDelay)
      await sleep(jittered);
    }
  }
  throw lastError;
}

// Usage
await retry(() => fetch('/api/flaky').then((r) => {
  if (!r.ok) throw Object.assign(new Error('HTTP'), { status: r.status });
  return r.json();
}), {
  maxAttempts: 4,
  shouldRetry: (err) => err.status >= 500 || err.status === 429, // retry server/rate-limit only
});
```

**Explanation**
- **Exponential backoff** (`baseDelay * 2^attempt`, capped at `maxDelay`) gives a struggling service progressively more breathing room instead of hammering it at a fixed rate.
- **Jitter matters because** without it, many clients that failed at the same instant retry in lockstep, re-creating the load spike that caused the failure (the "thundering herd"). Randomizing each delay spreads retries out, smoothing load. This uses **full jitter** (`random(0, expDelay)`), which AWS's research found minimizes contention.
- **`shouldRetry` predicate** fails fast on non-retryable errors — retrying a `400 Bad Request` or `401 Unauthorized` just wastes time and may lock accounts. Typically retry only `5xx`, `429`, and network errors.
- **`maxAttempts` cap + re-throwing `lastError`** guarantees termination and surfaces the real failure to the caller.
- **Production extras:** thread an `AbortSignal` to cancel mid-wait, add a circuit breaker for sustained failures, and make timeouts idempotent so retried writes don't double-apply.""",
    },

    # 25 — Patterns: Proxy for validation / reactive object
    {
        "prompt": "Use a Proxy to create a validated object: setting a property runs a per-key validator and throws on invalid values; reading a missing property throws instead of returning undefined. Explain Proxy traps and when to use them.",
        "answer": """<thinking>
Proxy lets us intercept fundamental operations via traps. Goal:
- `set` trap: look up a validator for the key; if it fails, throw; else assign.
- `get` trap: if the key doesn't exist, throw a clear error (strict access) instead
  of silently returning undefined — catches typos early.
- `deleteProperty` optionally guard required keys.
Reflect is used to perform the default behavior cleanly.
Note: Proxies have overhead; use for boundaries (config, public API), not hot paths.
</thinking>

```javascript
/**
 * Create an object that validates writes and forbids reading unknown keys.
 * @param {object} target - initial object
 * @param {Record<string, (value:any) => boolean>} validators - per-key validators
 */
function createValidatedObject(target, validators) {
  return new Proxy(target, {
    set(obj, key, value, receiver) {
      const validate = validators[key];
      if (validate && !validate(value)) {
        throw new TypeError(`Invalid value for "${String(key)}": ${value}`);
      }
      return Reflect.set(obj, key, value, receiver); // perform default assignment
    },

    get(obj, key, receiver) {
      if (typeof key === 'string' && !(key in obj) && key !== 'then') {
        // `then` excluded so the proxy isn't mistaken for a thenable/Promise.
        throw new ReferenceError(`Unknown property "${key}"`);
      }
      return Reflect.get(obj, key, receiver);
    },
  });
}

// Usage
const user = createValidatedObject(
  { name: 'Ana', age: 30 },
  {
    name: (v) => typeof v === 'string' && v.length > 0,
    age: (v) => Number.isInteger(v) && v >= 0 && v < 150,
  }
);

user.age = 31;        // ok
// user.age = -5;     // TypeError: Invalid value for "age": -5
// console.log(user.email); // ReferenceError: Unknown property "email"
```

**Explanation**
- **Proxy traps** intercept fundamental object operations: `set` for assignment, `get` for reads, plus `has`, `deleteProperty`, `ownKeys`, `apply`, `construct`, and more. Each trap can validate, transform, log, or veto the operation.
- **`set` validation** centralizes invariants at the data boundary — no value violating the schema can ever be stored, regardless of which code path writes it.
- **Strict `get`** turns typo'd property reads (`user.emial`) into immediate `ReferenceError`s instead of silent `undefined`, surfacing bugs early. (We special-case `then` so the object isn't accidentally treated as a Promise by `await`.)
- **`Reflect.*`** performs the default operation correctly, including respecting the `receiver` for inherited setters/getters — cleaner and safer than `obj[key] = value` directly.
- **When to use Proxies:** validation layers, reactive state (Vue 3's reactivity, MobX, valtio are Proxy-based), API mocking, access logging, and negative/lazy data structures. **When not to:** hot inner loops — every trapped operation adds indirection overhead, so keep Proxies at architectural boundaries rather than in performance-critical paths.""",
    },
]


# ---------------------------------------------------------------------------
# Convenience: convert to the chat-message training format used by this repo.
# ---------------------------------------------------------------------------
def to_chat_entries(items=examples, system_prompt=SYSTEM_PROMPT):
    """Wrap each {prompt, answer} as a system/user/assistant message triple."""
    return [
        {
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": ex["prompt"]},
                {"role": "assistant", "content": ex["answer"]},
            ]
        }
        for ex in items
    ]


if __name__ == "__main__":
    import json
    from pathlib import Path

    assert len(examples) == 25, f"expected 25 examples, got {len(examples)}"

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "javascript_examples.jsonl"

    with open(out_file, "w") as f:
        for entry in to_chat_entries():
            f.write(json.dumps(entry) + "\n")

    print(f"Wrote {len(examples)} JavaScript examples -> {out_file}")
