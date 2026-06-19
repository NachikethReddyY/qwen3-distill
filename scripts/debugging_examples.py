#!/usr/bin/env python3
"""
25 high-quality debugging training examples for model distillation.

Each example is a dict with:
  - "prompt": a specific, realistic debugging question
  - "answer": a <thinking> section followed by the four-part structure
             **Problem** -> **Root Cause** -> **Solution with Code** -> **Prevention**

Coverage (25 total):
  React (hooks/renders/state) ... 6
  JavaScript (async/this/closures) 5
  Python (imports/exceptions) .... 4
  Performance .................... 3
  Memory leaks ................... 2
  Race conditions ................ 2
  Type errors .................... 3

To integrate into the training pipeline, append each dict to
data/raw/debugging_examples.jsonl wrapped in the {"messages": [...]} chat
format (system/user/assistant), then run merge_data.py to rebuild splits.
"""

examples = [
    # ====================== REACT (6) ======================
    {
        "prompt": "My React component re-renders on every keystroke and the whole page feels sluggish when typing in a search box, even though only one input changed. How do I debug and fix this?",
        "answer": """<thinking>
A search input that re-renders the whole tree on each keystroke usually means state is lifted too high or an expensive child re-renders without memoization. I should: confirm where the state lives, see what re-renders with React DevTools "Highlight updates", then either co-locate state or memoize children/computations.
</thinking>

**Problem**
Typing in a search box re-renders the entire page. With React DevTools "Highlight updates when components render" enabled, large unrelated subtrees flash on every keystroke, and CPU spikes.

**Root Cause**
The `searchTerm` state lives in a top-level component that also renders an expensive list. Every keystroke calls `setSearchTerm`, which re-renders the parent and *all* its children, because plain child components re-render whenever their parent does. The filtering work and the list rows are recomputed on each keystroke.

**Solution with Code**
```jsx
// ❌ BEFORE: state at the top forces everything below to re-render
function Page({ items }) {
  const [searchTerm, setSearchTerm] = useState('');
  const filtered = items.filter(i => i.name.includes(searchTerm)); // re-runs every keystroke

  return (
    <>
      <input value={searchTerm} onChange={e => setSearchTerm(e.target.value)} />
      <ExpensiveSidebar />        {/* unrelated, but re-renders */}
      <List rows={filtered} />    {/* re-renders + re-filters */}
    </>
  );
}

// ✅ AFTER: co-locate input state, memoize the heavy computation and children
function Page({ items }) {
  const [searchTerm, setSearchTerm] = useState('');

  const filtered = useMemo(
    () => items.filter(i => i.name.includes(searchTerm)),
    [items, searchTerm]
  );

  return (
    <>
      <SearchInput value={searchTerm} onChange={setSearchTerm} />
      <ExpensiveSidebar />      {/* now skips re-render via React.memo */}
      <List rows={filtered} />
    </>
  );
}

const ExpensiveSidebar = React.memo(function ExpensiveSidebar() { /* ... */ });
const List = React.memo(function List({ rows }) {
  return rows.map(r => <Row key={r.id} {...r} />);
});
```

**Prevention**
- Keep state as low in the tree as possible ("co-location"); only lift it when truly shared.
- Wrap pure, expensive children in `React.memo` and stabilize their props with `useMemo`/`useCallback`.
- Use the React DevTools Profiler and "Highlight updates" early to *see* what re-renders rather than guessing.
- For very large lists, virtualize with `react-window`/`react-virtualized` so only visible rows render."""
    },
    {
        "prompt": "I have a setInterval inside useEffect that increments a counter, but the counter is always stuck at 1. Why does the interval keep seeing the old value?",
        "answer": """<thinking>
Classic stale-closure-in-useEffect bug. The effect runs once with `[]`, so the callback closes over `count` from the first render (0). It computes 0 + 1 = 1 forever. Fixes: functional updater, or a ref, or correct deps. The functional updater is cleanest because it doesn't recreate the interval.
</thinking>

**Problem**
A counter driven by `setInterval` inside `useEffect([])` increments to 1 and then never moves, even though the interval keeps firing.

**Root Cause**
The effect runs only once (empty dependency array), so its callback captures `count` by closure at its value on the first render: `0`. Every tick computes `setCount(0 + 1)` -> always `1`. The interval is never re-created with an updated `count`, so it permanently references a *stale* value.

**Solution with Code**
```jsx
// ❌ BEFORE: stale closure over count
useEffect(() => {
  const id = setInterval(() => {
    setCount(count + 1); // count is always 0 here
  }, 1000);
  return () => clearInterval(id);
}, []); // count not in deps, so the closure never refreshes

// ✅ FIX (preferred): functional updater reads the latest state
useEffect(() => {
  const id = setInterval(() => {
    setCount(prev => prev + 1); // no dependency on count
  }, 1000);
  return () => clearInterval(id);
}, []);

// ✅ ALTERNATIVE: if you must read other live values, use a ref
const countRef = useRef(count);
countRef.current = count;
useEffect(() => {
  const id = setInterval(() => setCount(countRef.current + 1), 1000);
  return () => clearInterval(id);
}, []);
```

**Prevention**
- When the next state depends on the previous, always use the functional updater `setX(prev => ...)`.
- Treat the dependency array as a contract: anything from render scope that you *read* belongs in it.
- Enable `eslint-plugin-react-hooks` (`exhaustive-deps`); it flags exactly this missing dependency."""
    },
    {
        "prompt": "My React app's memory grows over time and I see 'Can't perform a React state update on an unmounted component' warnings. I think a component isn't cleaning up. How do I find and fix it?",
        "answer": """<thinking>
The warning plus growing memory points at subscriptions/timers/listeners created in an effect without a cleanup return, or a fetch that resolves after unmount and calls setState. Need to return a cleanup function and/or guard async setState. Note: in React 18 the unmounted-setState warning was removed, but the leak (uncancelled work) is still real.
</thinking>

**Problem**
Navigating between routes leaks memory and logs "Can't perform a React state update on an unmounted component." Detached DOM nodes and listeners accumulate in the heap snapshot.

**Root Cause**
An effect subscribes to something (event listener, interval, or in-flight fetch) but does not unsubscribe. When the component unmounts, the subscription/timer survives, keeps the component closure alive, and its callback later calls `setState` on a dead component. Repeated mounts pile up uncollected closures.

**Solution with Code**
```jsx
// ❌ BEFORE: no cleanup; listener + fetch outlive the component
useEffect(() => {
  window.addEventListener('resize', onResize);
  fetch(`/api/user/${id}`).then(r => r.json()).then(setUser);
}, [id]);

// ✅ AFTER: remove the listener and ignore late async results
useEffect(() => {
  const onResize = () => setWidth(window.innerWidth);
  window.addEventListener('resize', onResize);

  const controller = new AbortController();
  fetch(`/api/user/${id}`, { signal: controller.signal })
    .then(r => r.json())
    .then(setUser)
    .catch(err => { if (err.name !== 'AbortError') throw err; });

  return () => {
    window.removeEventListener('resize', onResize); // unsubscribe
    controller.abort();                             // cancel in-flight fetch
  };
}, [id]);
```

**Prevention**
- Every effect that *subscribes* (listeners, timers, sockets, observers) must *return* a cleanup that unsubscribes.
- Use `AbortController` for fetch, or an `ignore` flag, so resolved promises don't touch unmounted state.
- Verify with Chrome DevTools Memory tab: mount/unmount the route repeatedly, take heap snapshots, and confirm detached nodes don't grow."""
    },
    {
        "prompt": "I added an onClick that reads state inside a useEffect-registered native event listener, but it always logs the initial state value, not the current one. What's going on?",
        "answer": """<thinking>
This is a stale closure in an event handler registered once. The listener was attached in an effect with empty deps, so it closes over state from the first render. Options: re-register on change (deps), or read from a ref that's kept current. A ref avoids re-binding listeners constantly.
</thinking>

**Problem**
A `document.addEventListener('keydown', handler)` registered inside `useEffect([], )` always sees the state value from the first render, no matter how the UI has updated since.

**Root Cause**
Because the effect ran once, `handler` is a closure created on the initial render and bound to that render's `state`. State updates create *new* renders with *new* closures, but the document still holds the original handler. So the handler keeps reading the stale value.

**Solution with Code**
```jsx
// ❌ BEFORE: handler frozen with first-render state
useEffect(() => {
  const handler = () => console.log(value); // always initial value
  document.addEventListener('keydown', handler);
  return () => document.removeEventListener('keydown', handler);
}, []);

// ✅ FIX A: keep current value in a ref, read it at call time
const valueRef = useRef(value);
valueRef.current = value;
useEffect(() => {
  const handler = () => console.log(valueRef.current); // always current
  document.addEventListener('keydown', handler);
  return () => document.removeEventListener('keydown', handler);
}, []);

// ✅ FIX B: re-register when value changes (fine for cheap handlers)
useEffect(() => {
  const handler = () => console.log(value);
  document.addEventListener('keydown', handler);
  return () => document.removeEventListener('keydown', handler);
}, [value]);
```

**Prevention**
- Remember that handlers attached imperatively capture the render they were created in.
- Use a ref ("latest value" pattern) for values that change often but where you don't want to re-bind listeners.
- Let `exhaustive-deps` warn you; if you intentionally use `[]`, document *why* the closure is safe."""
    },
    {
        "prompt": "I call setCount(count + 1) three times in a row in one click handler expecting +3, but the count only goes up by 1. Why doesn't React apply all three updates?",
        "answer": """<thinking>
React batches state updates within an event handler, and each setCount(count + 1) uses the same stale count from the current render. So all three compute the same value. Functional updater fixes it because each call gets the latest queued state.
</thinking>

**Problem**
```jsx
const handleClick = () => {
  setCount(count + 1);
  setCount(count + 1);
  setCount(count + 1);
};
```
Clicking once increments by 1, not 3.

**Root Cause**
`count` is a constant captured from the current render. All three calls read the *same* `count` (say 0) and queue `setCount(1)` three times. React batches updates inside the handler and the last write wins at the same value, so the result is 1. State is not mutated synchronously between the calls.

**Solution with Code**
```jsx
// ❌ BEFORE: all three read the same stale `count`
setCount(count + 1);
setCount(count + 1);
setCount(count + 1); // ends at count + 1

// ✅ AFTER: functional updater receives the latest queued value
setCount(prev => prev + 1);
setCount(prev => prev + 1);
setCount(prev => prev + 1); // ends at count + 3
```
The functional form passes each update the result of the previous one in the queue, so increments compose correctly.

**Prevention**
- Use `setX(prev => ...)` whenever the new value derives from the old one.
- Understand that setState is asynchronous and batched; the variable in scope won't change mid-handler.
- For complex multi-field transitions, prefer `useReducer`, which makes the "compute next from previous" model explicit."""
    },
    {
        "prompt": "I'm rendering a list and using the array index as the key. When I delete an item from the middle, the wrong rows show stale input values and checkboxes get mixed up. Why?",
        "answer": """<thinking>
Using index as key means React identifies rows by position, not identity. When you remove a middle item, positions shift, React reuses DOM/state from a different logical item, and uncontrolled inputs/component state stick to the wrong row. Fix: use a stable unique id as key.
</thinking>

**Problem**
A list with `key={index}` shows corrupted UI after deleting a middle element: text inputs keep the previous row's text, checkbox states swap, and focus jumps.

**Root Cause**
Keys tell React how to match elements across renders. With the array index as key, item at position 2 is "key 2" regardless of which logical record it is. Removing a middle item shifts everything up by one, so React thinks the *positions* persisted and reuses the previous component instance and its internal/DOM state (input value, checkbox) for a now-different record.

**Solution with Code**
```jsx
// ❌ BEFORE: identity tied to position
{items.map((item, index) => (
  <Row key={index} item={item} />
))}

// ✅ AFTER: identity tied to the data
{items.map(item => (
  <Row key={item.id} item={item} /> // stable, unique, from the data model
))}

// If items truly lack an id, generate one once when created:
const withIds = rawItems.map(it => ({ ...it, id: it.id ?? crypto.randomUUID() }));
```

**Prevention**
- Always use a stable, unique key derived from the data (a database id), not the array index.
- Index keys are only acceptable for static lists that never reorder, insert, or delete.
- If your data has no natural id, assign one at creation time and persist it, not on every render."""
    },

    # ====================== JAVASCRIPT (5) ======================
    {
        "prompt": "I use array.forEach with an async callback to fetch and save records, then log 'done', but 'done' prints before the saves finish and errors aren't caught. How do I fix sequential/parallel async over an array?",
        "answer": """<thinking>
forEach ignores the promises returned by its async callback, so it doesn't await anything. The loop finishes synchronously and 'done' logs immediately; rejections become unhandled. Fix with for...of + await (sequential) or Promise.all(map) (parallel).
</thinking>

**Problem**
```js
items.forEach(async (item) => {
  await save(item);
});
console.log('done'); // prints before any save completes
```
"done" logs first, and a rejected `save` produces an unhandled rejection.

**Root Cause**
`Array.prototype.forEach` is not promise-aware. It calls the async callback (which returns a promise) and throws that promise away, moving to the next iteration immediately. There is nothing to `await`, so execution continues to the log, and rejected promises have no handler.

**Solution with Code**
```js
// ✅ SEQUENTIAL: one after another, awaited
for (const item of items) {
  await save(item);
}
console.log('done'); // now truly after all saves

// ✅ PARALLEL: fire all, await together (faster, preserves errors)
await Promise.all(items.map(item => save(item)));
console.log('done');

// ✅ PARALLEL but don't fail-fast: collect outcomes
const results = await Promise.allSettled(items.map(item => save(item)));
const failures = results.filter(r => r.status === 'rejected');
```

**Prevention**
- Never use `forEach` with `async` when you need to wait; use `for...of` (sequential) or `Promise.all`/`allSettled` (parallel).
- Remember `map` returns the promises, which is why `Promise.all(items.map(...))` works while `forEach` doesn't.
- Lint with rules like `no-await-in-loop` (when parallel is intended) and always handle rejections."""
    },
    {
        "prompt": "I pass a class method as a callback (button.addEventListener('click', this.handleClick)) and inside it `this` is undefined. Why does `this` get lost and how do I fix it?",
        "answer": """<thinking>
Passing a method detaches it from its object; `this` is determined by how it's called, not where it's defined. As a bare callback it's called with `this` = undefined (strict mode) or the element. Fix: bind in constructor, use an arrow class field, or wrap in an arrow at the call site.
</thinking>

**Problem**
```js
class Counter {
  count = 0;
  handleClick() { this.count++; } // `this` is undefined when called as a callback
}
const c = new Counter();
button.addEventListener('click', c.handleClick); // TypeError: cannot read 'count' of undefined
```

**Root Cause**
In JavaScript, `this` is bound at *call time* based on the call site, not lexically. `c.handleClick` extracts the function reference and detaches it from `c`. When the event system invokes it as a plain function, `this` is `undefined` in strict mode (class bodies are strict), so `this.count` throws.

**Solution with Code**
```js
// ✅ FIX A: arrow function as a class field — lexically binds `this`
class Counter {
  count = 0;
  handleClick = () => { this.count++; }; // `this` always the instance
}
button.addEventListener('click', c.handleClick);

// ✅ FIX B: bind in the constructor
class Counter {
  constructor() { this.handleClick = this.handleClick.bind(this); }
  handleClick() { this.count++; }
}

// ✅ FIX C: wrap at the call site
button.addEventListener('click', () => c.handleClick());
```

**Prevention**
- Prefer arrow class fields for handlers that get passed around — they capture `this` lexically.
- Be wary any time you pass `obj.method` without calling it; you've detached the receiver.
- In React class components this is the same trap; in function components, hooks sidestep it entirely."""
    },
    {
        "prompt": "I build click handlers in a `for (var i = 0; ...)` loop and every button alerts the final value of i instead of its own index. Why, and how do I fix it without rewriting much?",
        "answer": """<thinking>
Classic var-in-loop closure bug. `var` is function-scoped, so all closures share one `i` that ends at the loop's final value. `let` creates a fresh binding per iteration. Alternatively an IIFE captures the value.
</thinking>

**Problem**
```js
for (var i = 0; i < 3; i++) {
  buttons[i].addEventListener('click', () => alert(i));
}
// every button alerts 3
```

**Root Cause**
`var` is function-scoped, not block-scoped, so there is a single `i` shared by all three closures. The handlers don't run until clicked, by which time the loop has finished and `i === 3`. All closures read that same final value.

**Solution with Code**
```js
// ✅ FIX A (minimal): use `let` — block-scoped, fresh binding each iteration
for (let i = 0; i < 3; i++) {
  buttons[i].addEventListener('click', () => alert(i)); // 0, 1, 2
}

// ✅ FIX B (pre-ES6): capture the value with an IIFE
for (var i = 0; i < 3; i++) {
  (function (j) {
    buttons[j].addEventListener('click', () => alert(j));
  })(i);
}
```

**Prevention**
- Default to `let`/`const`; reserve `var` for legacy code. `let` in a `for` header creates a new binding per iteration.
- Understand that closures capture *variables*, not values — share a variable and you share its latest value.
- ESLint's `no-var` and `block-scoped-var` catch this class of bug."""
    },
    {
        "prompt": "My Node process crashes intermittently with 'UnhandledPromiseRejection' and sometimes exits silently. I do call functions that return promises but don't always await them. How do I track this down?",
        "answer": """<thinking>
Un-awaited promises whose rejection isn't handled become unhandled rejections; in modern Node this can terminate the process. Need to await/catch, add a global handler for diagnostics, and avoid floating promises. Show the fire-and-forget bug and the fix.
</thinking>

**Problem**
A server occasionally logs `UnhandledPromiseRejection` and, on newer Node versions, exits non-zero. The crashes are intermittent and hard to reproduce because they depend on which async call happens to reject.

**Root Cause**
A promise is created but never `await`ed or `.catch()`ed — a "floating promise." When it rejects, there is no handler, so Node emits `unhandledRejection`. Since Node 15 the default behavior is to crash the process on unhandled rejections.

**Solution with Code**
```js
// ❌ BEFORE: fire-and-forget; rejection escapes
function handler(req, res) {
  logAnalytics(req); // returns a promise, not awaited, may reject
  res.send('ok');
}

// ✅ AFTER: await (or explicitly catch) every promise
async function handler(req, res) {
  try {
    await logAnalytics(req);
  } catch (err) {
    logger.error('analytics failed', err); // contained, doesn't crash
  }
  res.send('ok');
}

// ✅ Diagnostics: surface the origin of any stray rejection
process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled rejection', reason);
});
```

**Prevention**
- Never let a promise float: `await` it, `return` it, or `.catch()` it.
- Enable TypeScript's `no-floating-promises` (typescript-eslint) — it flags exactly these cases.
- Keep a global `unhandledRejection` handler for observability, but treat it as a backstop, not the fix."""
    },
    {
        "prompt": "Two async functions run and my code assumes the first 'await' finishes before the second starts, but logs show interleaving and a value is sometimes undefined. How do I reason about async ordering bugs?",
        "answer": """<thinking>
The bug is assuming sequential ordering when promises were started concurrently, or reading a shared variable that another async path hasn't set yet. Need to make data dependencies explicit: await in order, or pass values through return rather than shared mutable state.
</thinking>

**Problem**
```js
let token;
async function login() { token = await fetchToken(); }
async function loadData() { return fetch('/data', { headers: { token } }); }

login();
loadData(); // sometimes token is undefined
```
`loadData` occasionally runs before `login` has set `token`.

**Root Cause**
`login()` and `loadData()` are both started without awaiting, so they run concurrently. `loadData` reads the shared `token` variable, which `login` only sets *after* its own `await fetchToken()` resolves. The two async timelines interleave, so `token` may still be `undefined` when `loadData` reads it. The code relies on implicit ordering between independent promises.

**Solution with Code**
```js
// ✅ FIX A: make the dependency explicit by awaiting in order
async function main() {
  const token = await fetchToken();          // finishes first
  return fetch('/data', { headers: { token } });
}

// ✅ FIX B: pass data through return values, not shared mutable state
async function login() { return fetchToken(); }
async function loadData(token) { return fetch('/data', { headers: { token } }); }

async function main() {
  const token = await login();
  return loadData(token);
}
```

**Prevention**
- Sequence dependent steps with `await`; only run things concurrently when they're truly independent.
- Prefer returning values over mutating shared variables across async boundaries — it makes ordering explicit.
- When you *do* want concurrency with dependencies, combine with `Promise.all` only for the independent parts."""
    },

    # ====================== PYTHON (4) ======================
    {
        "prompt": "I have two Python modules that import each other and I get 'ImportError: cannot import name X' or partially initialized module errors. How do I diagnose and fix a circular import?",
        "answer": """<thinking>
Circular import: module A imports B at top level while B imports A; whichever loads first sees the other half-initialized. Fixes: move the import inside the function (lazy), restructure to break the cycle, or import the module rather than the name. Explain the partially-initialized module mechanism.
</thinking>

**Problem**
```
ImportError: cannot import name 'helper' from partially initialized module 'b'
(most likely due to a circular import)
```
`a.py` does `from b import helper` and `b.py` does `from a import thing`.

**Root Cause**
When Python imports `a`, it starts executing `a` and registers a *partially built* module object. At the `from b import helper` line it switches to executing `b`, which hits `from a import thing` — but `a` isn't finished, so `thing` may not be defined yet. The name lookup fails on the half-initialized module.

**Solution with Code**
```python
# ❌ BEFORE: top-level mutual imports
# a.py
from b import helper
def thing(): return helper()
# b.py
from a import thing
def helper(): return thing()

# ✅ FIX A: defer the import to call time (lazy import)
# b.py
def helper():
    from a import thing   # imported when called, after modules finish loading
    return thing()

# ✅ FIX B: import the module, not the name (resolves attribute at use time)
# a.py
import b
def thing(): return b.helper()

# ✅ FIX C (best long-term): extract shared code into a third module
# common.py holds the shared dependency; a and b both import common, not each other.
```

**Prevention**
- Design module dependencies as a DAG; if A and B need each other, factor the shared piece into a third module.
- Prefer `import module` over `from module import name` when cycles are a risk — attributes resolve lazily.
- Keep import-time side effects minimal so partial initialization is less harmful."""
    },
    {
        "prompt": "My Python function uses a list as a default argument and it 'remembers' values across calls — appending to it accumulates results from previous calls. Why and how do I fix it?",
        "answer": """<thinking>
Mutable default argument evaluated once at def time, shared across calls. Classic gotcha. Fix: default to None and create inside. Explain that the default object persists in the function's __defaults__.
</thinking>

**Problem**
```python
def add_item(item, bucket=[]):
    bucket.append(item)
    return bucket

add_item(1)  # [1]
add_item(2)  # [1, 2]  <- unexpectedly remembers
```

**Root Cause**
Default argument values are evaluated **once**, when the `def` statement executes — not on each call. So `bucket=[]` creates a single list stored on the function object (`add_item.__defaults__`). Every call that omits `bucket` mutates that same shared list, accumulating values across calls.

**Solution with Code**
```python
# ✅ FIX: use a sentinel (None) and build a fresh object per call
def add_item(item, bucket=None):
    if bucket is None:
        bucket = []          # new list every call
    bucket.append(item)
    return bucket

add_item(1)  # [1]
add_item(2)  # [2]   <- independent

# Same pattern for dicts/sets and any mutable default:
def configure(opts=None):
    opts = {} if opts is None else opts
    ...
```

**Prevention**
- Never use a mutable object (`[]`, `{}`, `set()`) as a default argument; default to `None` and construct inside.
- Remember defaults are bound at definition time — this also applies to calls like `datetime.now()` as a default.
- Linters (pylint `W0102`, ruff `B006`) flag mutable default arguments automatically."""
    },
    {
        "prompt": "Our service swallows errors — something fails but we just see 'something went wrong' with no traceback, and even KeyboardInterrupt seems caught. We use a bare `except:` somewhere. How do I fix exception handling?",
        "answer": """<thinking>
Bare except catches everything including SystemExit/KeyboardInterrupt and hides the real error and traceback. Fix: catch specific exceptions, log with traceback, re-raise or chain. Show the dangers and the corrected pattern.
</thinking>

**Problem**
A failure produces only "something went wrong" with no traceback. Even Ctrl-C (KeyboardInterrupt) doesn't stop the program. The real exception type and location are invisible.

**Root Cause**
A bare `except:` (or `except Exception:` with a swallowed message) catches *every* exception, including control-flow ones like `KeyboardInterrupt` and `SystemExit`, and discards the original error object and traceback. Debugging is impossible because the diagnostic information is thrown away.

**Solution with Code**
```python
import logging
logger = logging.getLogger(__name__)

# ❌ BEFORE: catches everything, hides the cause
try:
    result = process(data)
except:
    print("something went wrong")

# ✅ AFTER: catch what you can handle, log the traceback, re-raise the rest
try:
    result = process(data)
except (ValueError, KeyError) as exc:
    logger.exception("process failed for input %r", data)  # logs full traceback
    raise ProcessingError("could not process input") from exc  # chain, preserve cause
```
`logger.exception` records the traceback; `raise ... from exc` keeps the original cause; specific exception types let `KeyboardInterrupt` propagate normally.

**Prevention**
- Catch the narrowest exception types you can actually handle; never use bare `except:`.
- Use `logger.exception(...)` (inside an except block) to capture the traceback.
- Re-raise with `raise NewError(...) from original` to preserve the cause chain.
- ruff/flake8 `E722` flags bare excepts; enable it."""
    },
    {
        "prompt": "I build a list of functions in a Python loop (using a comprehension/loop with lambdas) and they all return the last loop value instead of their own. Why does Python behave like this?",
        "answer": """<thinking>
Python late binding closures: the lambda captures the variable, not its value, and the loop variable is looked up when the lambda runs — by then it's the final value. Same family as JS var bug but for a different reason (late binding, not scoping). Fix: default-argument capture or functools.partial.
</thinking>

**Problem**
```python
funcs = [lambda: i for i in range(3)]
print([f() for f in funcs])  # [2, 2, 2], not [0, 1, 2]
```

**Root Cause**
Python closures use **late binding**: the lambda doesn't store the current value of `i`, it stores a reference to the variable `i`. The lambda bodies don't run until called, and by then the loop has completed and `i` holds its final value (`2`). All three lambdas look up the same `i`.

**Solution with Code**
```python
# ✅ FIX A: bind the current value via a default argument (evaluated at def time)
funcs = [lambda i=i: i for i in range(3)]
print([f() for f in funcs])  # [0, 1, 2]

# ✅ FIX B: use functools.partial to bind the argument now
from functools import partial
def identity(x): return x
funcs = [partial(identity, i) for i in range(3)]
print([f() for f in funcs])  # [0, 1, 2]

# ✅ FIX C: a factory function that closes over its own parameter
def make(i):
    return lambda: i
funcs = [make(i) for i in range(3)]  # [0, 1, 2]
```

**Prevention**
- To capture a loop value, bind it explicitly: default argument, `partial`, or a factory function.
- Remember Python closures capture variables by reference (late binding), unlike some languages.
- Be especially careful generating callbacks/handlers inside loops."""
    },

    # ====================== PERFORMANCE (3) ======================
    {
        "prompt": "An API endpoint that lists orders with their customer names is extremely slow as data grows. The query itself looks simple. How do I diagnose and fix this performance problem?",
        "answer": """<thinking>
Classic N+1 query problem: fetch list, then one query per row for the related entity. Slowness scales with row count. Diagnose by counting queries / using the ORM's query log; fix with eager loading / a join. Generic enough to show in an ORM-agnostic way.
</thinking>

**Problem**
Listing orders is fast with 10 rows but crawls with thousands. The endpoint loads orders, then displays each order's customer name. Query logs show hundreds of nearly identical `SELECT * FROM customers WHERE id = ?` statements per request.

**Root Cause**
This is the **N+1 query** problem: 1 query loads N orders, then accessing `order.customer` lazily issues 1 query per order — N additional round trips. Total queries grow linearly with rows, and per-query latency dominates.

**Solution with Code**
```python
# ❌ BEFORE: 1 + N queries (lazy load per order)
orders = session.query(Order).all()
for o in orders:
    print(o.customer.name)   # one SELECT per order

# ✅ AFTER: eager-load the relationship in a single round trip
from sqlalchemy.orm import joinedload
orders = (session.query(Order)
          .options(joinedload(Order.customer))  # JOIN, one query
          .all())
for o in orders:
    print(o.customer.name)   # already loaded
```
```sql
-- Equivalent raw SQL: one JOIN instead of 1 + N selects
SELECT o.*, c.name FROM orders o JOIN customers c ON c.id = o.customer_id;
```

**Prevention**
- Watch your ORM's query log/count in tests; a query count that scales with rows signals N+1.
- Use eager loading (`joinedload`/`selectinload`, Django `select_related`/`prefetch_related`) for relationships you'll access.
- Add an assertion or tooling (e.g., django `nplusone`) in CI to catch regressions."""
    },
    {
        "prompt": "A React table with 10,000 rows freezes the browser when it mounts and scrolling is janky. Memoization didn't help much. How do I make it performant?",
        "answer": """<thinking>
Rendering 10k DOM nodes is the problem; memoization avoids re-renders but you still mount 10k nodes. The real fix is virtualization (windowing) so only visible rows are in the DOM. Show react-window usage.
</thinking>

**Problem**
Mounting a 10,000-row table blocks the main thread for seconds and scrolling stutters. `React.memo` on rows reduced re-renders but the initial mount is still slow.

**Root Cause**
The browser has to create and lay out ~10,000 DOM nodes (plus their children) at once. Memoization only prevents *re-rendering* existing nodes; it doesn't reduce how many nodes exist. The DOM size itself — not React diffing — is the bottleneck.

**Solution with Code**
```jsx
// ❌ BEFORE: all rows in the DOM
function Table({ rows }) {
  return <div>{rows.map(r => <Row key={r.id} row={r} />)}</div>;
}

// ✅ AFTER: virtualize — render only visible rows + a small buffer
import { FixedSizeList } from 'react-window';

function Table({ rows }) {
  const Item = ({ index, style }) => (
    <div style={style}><Row row={rows[index]} /></div>
  );
  return (
    <FixedSizeList height={600} itemCount={rows.length} itemSize={40} width="100%">
      {Item}
    </FixedSizeList>
  );
}
```
Now only ~15–20 rows exist in the DOM at any time regardless of dataset size.

**Prevention**
- Virtualize any list/table/grid that can grow large (`react-window`, `react-virtualized`, TanStack Virtual).
- Distinguish "too many re-renders" (fix with memo) from "too many DOM nodes" (fix with virtualization).
- Paginate or lazy-load on the server when datasets are huge; don't ship 10k rows to the client at once."""
    },
    {
        "prompt": "A scroll handler that updates a progress bar makes scrolling feel laggy and the CPU spikes. The handler does layout reads and some state updates. How do I profile and fix it?",
        "answer": """<thinking>
Scroll fires very frequently; a heavy handler running on each event floods the main thread and may force layout thrashing (reading layout after writing). Fixes: throttle/rAF, batch reads, use passive listeners. Show requestAnimationFrame throttle.
</thinking>

**Problem**
Scrolling is laggy; the Performance profile shows the scroll handler firing dozens of times per frame, each doing `getBoundingClientRect()` and a state update, with long "Recalculate Style/Layout" bars.

**Root Cause**
`scroll` fires at very high frequency. Running expensive work (layout reads + state updates) on *every* event saturates the main thread. Reading layout (`getBoundingClientRect`) right after a write also causes forced synchronous reflow ("layout thrashing"), multiplying the cost.

**Solution with Code**
```js
// ❌ BEFORE: heavy work on every scroll event
window.addEventListener('scroll', () => {
  const pct = window.scrollY / document.body.scrollHeight;
  setProgress(pct); // many times per frame
});

// ✅ AFTER: throttle to one update per animation frame, passive listener
let ticking = false;
window.addEventListener('scroll', () => {
  if (ticking) return;
  ticking = true;
  requestAnimationFrame(() => {
    const pct = window.scrollY / document.body.scrollHeight;
    setProgress(pct);   // at most once per frame (~60fps)
    ticking = false;
  });
}, { passive: true });   // tells the browser we won't preventDefault
```

**Prevention**
- Throttle/debounce high-frequency events (`scroll`, `resize`, `mousemove`); `requestAnimationFrame` aligns work to frames.
- Add `{ passive: true }` to scroll/touch listeners so the browser can scroll without waiting on your handler.
- Avoid layout thrashing: batch all reads, then all writes; profile with DevTools Performance to spot forced reflows."""
    },

    # ====================== MEMORY LEAKS (2) ======================
    {
        "prompt": "A long-running single-page app's memory keeps climbing. I add global event listeners and timers when widgets initialize but never see them go away. How do I find and fix the leak?",
        "answer": """<thinking>
Leak from listeners/timers added but not removed; closures keep references to detached DOM/objects. Diagnose with heap snapshots (detached nodes, growing closures) and Performance memory timeline. Fix by removing listeners and clearing timers on teardown.
</thinking>

**Problem**
Over hours of use, the SPA's heap grows without bound until it gets sluggish or crashes. Each widget adds `window`/`document` listeners and a `setInterval` on init, but nothing is removed when widgets are destroyed.

**Root Cause**
Every `addEventListener`/`setInterval` keeps a strong reference to its callback, which closes over the widget (and its DOM). Without `removeEventListener`/`clearInterval` on teardown, those callbacks — and everything they reference — can never be garbage collected. Re-initializing widgets stacks more uncollectable closures, so the heap only grows.

**Solution with Code**
```js
class Widget {
  constructor(el) {
    this.el = el;
    this.onResize = () => this.layout();
    window.addEventListener('resize', this.onResize);
    this.timer = setInterval(() => this.poll(), 5000);
  }

  // ✅ explicit teardown that releases every retained reference
  destroy() {
    window.removeEventListener('resize', this.onResize); // same reference!
    clearInterval(this.timer);
    this.el = null;
  }
}

// Using AbortController to remove many listeners at once:
const ac = new AbortController();
window.addEventListener('resize', onResize, { signal: ac.signal });
document.addEventListener('keydown', onKey, { signal: ac.signal });
// teardown:
ac.abort(); // removes both listeners
```

**Prevention**
- Pair every `addEventListener` with `removeEventListener` (using the *same* function reference) and every timer with `clear*`.
- Use `AbortController` signals to tear down groups of listeners in one call.
- Confirm with Chrome DevTools Memory: take heap snapshots before/after repeated init+destroy; "Detached" nodes and growing closure counts reveal leaks."""
    },
    {
        "prompt": "We added an in-memory cache (a plain object/Map) to speed up lookups, and now the server's memory grows until it OOMs. Entries never seem to be removed. How do I fix this unbounded cache?",
        "answer": """<thinking>
Unbounded cache leak: every unique key adds an entry that is never evicted, so memory grows with cardinality. Fix: bound the cache with an LRU and/or TTL. Show a simple LRU using Map insertion order, mention WeakMap when keys are objects.
</thinking>

**Problem**
After adding a `Map`-based cache keyed by request parameters, RSS climbs steadily until the process is killed with an out-of-memory error. The cache is read on every request but entries are only ever added, never removed.

**Root Cause**
The cache has no eviction policy. Each distinct key inserts a permanent entry, and the `Map` holds strong references to keys and values forever. With high-cardinality keys (user ids, query strings), the cache grows unbounded — it's effectively a memory leak disguised as an optimization.

**Solution with Code**
```js
// ❌ BEFORE: grows forever
const cache = new Map();
function get(key) {
  if (!cache.has(key)) cache.set(key, compute(key));
  return cache.get(key);
}

// ✅ AFTER: bounded LRU using Map insertion order
class LRU {
  constructor(max = 1000) { this.max = max; this.map = new Map(); }
  get(key) {
    if (!this.map.has(key)) return undefined;
    const v = this.map.get(key);
    this.map.delete(key); this.map.set(key, v); // mark as recently used
    return v;
  }
  set(key, v) {
    if (this.map.has(key)) this.map.delete(key);
    this.map.set(key, v);
    if (this.map.size > this.max) {
      this.map.delete(this.map.keys().next().value); // evict oldest
    }
  }
}
```

**Prevention**
- Always bound caches: max size (LRU) and/or TTL expiry; never cache into an unbounded structure.
- Use a vetted library (`lru-cache`, `quick-lru`) instead of hand-rolling in production.
- When keys are objects and you want GC to reclaim them, consider `WeakMap`; monitor memory with metrics/alerts."""
    },

    # ====================== RACE CONDITIONS (2) ======================
    {
        "prompt": "A search-as-you-type component sometimes shows results for an earlier query — I type 'react' but briefly see results for 'rea'. Slow responses arrive out of order. How do I fix this race condition?",
        "answer": """<thinking>
Out-of-order async responses: multiple fetches in flight, a slower earlier one resolves after a faster later one and overwrites state. Fix: ignore stale responses via a cleanup flag in useEffect, or AbortController to cancel previous requests.
</thinking>

**Problem**
In a search box, fast typing fires several requests. Occasionally a response for an older query (`rea`) arrives *after* the newer one (`react`) and overwrites the displayed results, so the UI shows stale matches.

**Root Cause**
Each keystroke starts an independent fetch; there's no guarantee responses return in request order. A slow earlier request can resolve last and call `setResults`, clobbering the correct newer results. This is a classic out-of-order async race.

**Solution with Code**
```jsx
// ✅ FIX A: ignore stale responses with a per-effect flag
useEffect(() => {
  let active = true;
  fetchResults(query).then(data => {
    if (active) setResults(data); // only the latest effect's flag is true
  });
  return () => { active = false; }; // marks previous effect stale on re-run
}, [query]);

// ✅ FIX B: cancel the previous request with AbortController
useEffect(() => {
  const controller = new AbortController();
  fetchResults(query, { signal: controller.signal })
    .then(setResults)
    .catch(err => { if (err.name !== 'AbortError') throw err; });
  return () => controller.abort(); // abort the in-flight request
}, [query]);
```

**Prevention**
- For "latest wins" async UI, either cancel prior requests (`AbortController`) or guard setState with a freshness flag.
- Debounce input so you fire fewer requests in the first place.
- Never assume async responses arrive in the order you sent them."""
    },
    {
        "prompt": "Under concurrent requests, our 'reserve the last item if available' logic oversells inventory — two requests both think stock is available and both decrement it. How do I fix this check-then-act race?",
        "answer": """<thinking>
Check-then-act (TOCTOU) race: read stock, decide, then write, with a gap where another transaction interleaves. Fix: make the read-modify-write atomic — conditional UPDATE, DB row lock (SELECT ... FOR UPDATE), or an atomic decrement guarded by a constraint.
</thinking>

**Problem**
Two concurrent requests both read `stock = 1`, both pass the "in stock?" check, and both decrement, leaving `stock = -1`: the item is oversold.

**Root Cause**
A **check-then-act** (time-of-check to time-of-use) race. The read, the decision, and the write are separate steps. Between the check and the write, another transaction runs the same check on the same stale value. Without atomicity or locking, both succeed.

**Solution with Code**
```python
# ❌ BEFORE: read, decide, write — interleavable
row = db.execute("SELECT stock FROM items WHERE id=:id", id=item_id)
if row.stock > 0:
    db.execute("UPDATE items SET stock = stock - 1 WHERE id=:id", id=item_id)

# ✅ FIX A: atomic conditional update — the DB enforces the guard
result = db.execute(
    "UPDATE items SET stock = stock - 1 WHERE id=:id AND stock > 0",
    id=item_id,
)
if result.rowcount == 0:
    raise OutOfStock()   # the guarded update affected 0 rows

# ✅ FIX B: lock the row for the duration of the transaction
with db.begin():
    row = db.execute(
        "SELECT stock FROM items WHERE id=:id FOR UPDATE", id=item_id
    )
    if row.stock <= 0:
        raise OutOfStock()
    db.execute("UPDATE items SET stock = stock - 1 WHERE id=:id", id=item_id)
```

**Prevention**
- Collapse read-modify-write into a single atomic operation (conditional `UPDATE`, atomic counters) whenever possible.
- Use database locking (`SELECT ... FOR UPDATE`) or serializable transactions when logic can't be one statement.
- Add a `CHECK (stock >= 0)` constraint as a backstop so the invariant can never be violated.
- Load-test with concurrency; single-threaded tests will never reveal these races."""
    },

    # ====================== TYPE ERRORS (3) ======================
    {
        "prompt": "I get 'TypeError: Cannot read properties of undefined (reading 'name')' when rendering user data fetched from an API. It works sometimes but crashes on load. How do I debug and fix it?",
        "answer": """<thinking>
Accessing a nested property before async data arrives or when it's missing. Initial render has user = undefined/null. Fix: optional chaining, default values, loading guards, and validate the shape. Explain the timing.
</thinking>

**Problem**
`user.profile.name` throws `TypeError: Cannot read properties of undefined (reading 'name')` on initial load, but works after a refresh or once data is present.

**Root Cause**
The component renders before the async fetch resolves, so `user` (or `user.profile`) is `undefined`. Reading `.name` off `undefined` throws. It "works sometimes" because of caching or timing where data happens to be ready on first render.

**Solution with Code**
```jsx
// ❌ BEFORE: assumes the whole chain exists immediately
function Profile({ user }) {
  return <h1>{user.profile.name}</h1>; // throws when user/profile is undefined
}

// ✅ AFTER: guard for loading, then access safely
function Profile({ user, loading }) {
  if (loading || !user) return <Spinner />;

  // optional chaining + fallback for missing fields
  const name = user.profile?.name ?? 'Anonymous';
  return <h1>{name}</h1>;
}
```
```js
// At the data layer, normalize/validate the shape so the UI can trust it:
const safeUser = { profile: {}, ...apiUser };
```

**Prevention**
- Use optional chaining (`?.`) and nullish coalescing (`??`) for data that may be absent.
- Always model loading/empty/error states explicitly; never assume async data is present on first render.
- Validate API responses at the boundary (e.g., zod) and adopt TypeScript so `user.profile.name` is a compile-time error when nullable."""
    },
    {
        "prompt": "In JavaScript, '5' + 3 gives '53' but '5' - 3 gives 2, and my totals are sometimes string-concatenated instead of summed. How do I debug these implicit type coercion bugs?",
        "answer": """<thinking>
Implicit coercion: + is overloaded for string concat, so a string operand makes the whole thing a string; other arithmetic operators coerce to number. Values from inputs/JSON are strings. Fix: explicitly convert with Number()/parseFloat and validate. Show NaN pitfalls.
</thinking>

**Problem**
A running total sometimes becomes a concatenated string: `total = total + price` yields `"010.5"` instead of `10.5`. Subtraction "works" (`'5' - 3 === 2`) which masks the issue.

**Root Cause**
JavaScript's `+` is overloaded: if either operand is a string, it does string concatenation, not addition. Values coming from form inputs, `dataset`, query strings, or some JSON are strings. So `'5' + 3` is `'53'`, while `-`, `*`, `/` coerce operands to numbers (`'5' - 3 === 2`), which is why only addition misbehaves.

**Solution with Code**
```js
// ❌ BEFORE: price is a string from an input -> concatenation
let total = 0;
for (const item of items) {
  total = total + item.price; // '0' + '10.5' style concatenation
}

// ✅ AFTER: convert to number explicitly and validate
let total = 0;
for (const item of items) {
  const price = Number(item.price);
  if (Number.isNaN(price)) throw new Error(`Invalid price: ${item.price}`);
  total += price;
}

// Parsing at the boundary:
const qty = parseInt(input.value, 10);          // base 10 explicit
const amount = Number.parseFloat(input.value);  // floats
```

**Prevention**
- Convert external values to numbers explicitly (`Number()`, `parseInt(x, 10)`, `parseFloat`) at the input boundary.
- Guard against `NaN` with `Number.isNaN`; `NaN` propagates silently through arithmetic.
- Use TypeScript and validation (zod) so number fields can't arrive as strings unnoticed."""
    },
    {
        "prompt": "My TypeScript build fails with 'Object is possibly null' / 'possibly undefined' on values I'm sure exist (like document.getElementById or an array .find result). Are these false positives and how should I handle them?",
        "answer": """<thinking>
Not false positives: these APIs are genuinely nullable (getElementById -> HTMLElement | null, find -> T | undefined). Strict null checks force handling. Fix: narrow with guards, throw if invariant, or assert only when truly safe. Discourage blanket `!`.
</thinking>

**Problem**
With `strictNullChecks` on, TypeScript reports "Object is possibly 'null'" for `document.getElementById('app').innerHTML` and "possibly 'undefined'" for `users.find(u => u.id === id).name`.

**Root Cause**
These are not false positives. `getElementById` returns `HTMLElement | null` (the element may not exist), and `Array.prototype.find` returns `T | undefined` (no match). Under strict null checks, TypeScript correctly forces you to handle the absent case before dereferencing, because at runtime these *can* be null/undefined.

**Solution with Code**
```ts
// ❌ BEFORE: dereferencing a possibly-null/undefined value
document.getElementById('app').innerHTML = 'hi';     // error
const name = users.find(u => u.id === id).name;      // error

// ✅ FIX A: narrow with a guard (handles the real case)
const app = document.getElementById('app');
if (app) app.innerHTML = 'hi';

const user = users.find(u => u.id === id);
if (!user) throw new Error(`No user ${id}`); // make the invariant explicit
const name = user.name;                       // now safely string

// ✅ FIX B: non-null assertion ONLY when you can prove it's present
const root = document.getElementById('root')!; // use sparingly, document why
```

**Prevention**
- Treat strict-null errors as real bugs to handle, not noise to silence; narrow with `if`/early return/throw.
- Reserve the `!` non-null assertion for cases you can genuinely guarantee, and comment the justification.
- Keep `strict: true` on — it surfaces null-deref bugs at compile time instead of in production."""
    },
]

if __name__ == "__main__":
    assert len(examples) == 25, f"expected 25, got {len(examples)}"
    for i, ex in enumerate(examples):
        assert set(ex) == {"prompt", "answer"}, f"example {i} has wrong keys"
        assert "<thinking>" in ex["answer"] and "</thinking>" in ex["answer"]
        for header in ("**Problem**", "**Root Cause**", "**Prevention**"):
            assert header in ex["answer"], f"example {i} missing {header}"
    print(f"OK: {len(examples)} debugging examples validated")
