#!/usr/bin/env python3
"""
25 high-quality React training examples for model distillation.

Each example is a dict with:
  - "prompt": a specific, realistic React question
  - "answer": a detailed response with a <thinking> section, production-grade
              code, and an explanation of best practices.

Coverage: hooks, GSAP animations, state management, performance, patterns,
context, refs, lifecycle, async, error handling, and testing patterns.

Use directly or fold into SYNTHETIC_ANSWERS in generate_synthetic_data.py.
"""

examples = [
    # 1 -- Custom hook: useDebounce
    {
        "prompt": "Write a reusable useDebounce hook in React that debounces a rapidly changing value (like a search input), and show how to use it to fetch search results without firing a request on every keystroke.",
        "answer": """<thinking>
The user wants a generic value-debouncing hook plus a realistic usage. Design decisions:
1. Debounce the *value*, not the callback -- this composes better with effects and is easier to test.
2. Use useEffect with a setTimeout, and clean up the timer on every change so only the last update after `delay` ms survives.
3. Generic over the value type so it works for strings, numbers, objects.
4. In the consumer, drive the fetch off the debounced value inside an effect, and guard against stale responses with an AbortController so an earlier in-flight request can't overwrite a newer one.
</thinking>

```tsx
import { useState, useEffect } from 'react';

export function useDebounce<T>(value: T, delay = 300): T {
  const [debounced, setDebounced] = useState<T>(value);

  useEffect(() => {
    const id = setTimeout(() => setDebounced(value), delay);
    return () => clearTimeout(id); // cancel if value changes before delay elapses
  }, [value, delay]);

  return debounced;
}

// Usage: debounced search
function SearchBox() {
  const [query, setQuery] = useState('');
  const debouncedQuery = useDebounce(query, 400);
  const [results, setResults] = useState<string[]>([]);

  useEffect(() => {
    if (!debouncedQuery) {
      setResults([]);
      return;
    }
    const controller = new AbortController();
    fetch(`/api/search?q=${encodeURIComponent(debouncedQuery)}`, {
      signal: controller.signal,
    })
      .then((r) => r.json())
      .then(setResults)
      .catch((err) => {
        if (err.name !== 'AbortError') console.error(err);
      });
    return () => controller.abort(); // cancel stale request
  }, [debouncedQuery]);

  return (
    <>
      <input value={query} onChange={(e) => setQuery(e.target.value)} />
      <ul>{results.map((r) => <li key={r}>{r}</li>)}</ul>
    </>
  );
}
```

**Explanation**
- Debouncing the value (not a callback) keeps the hook stateless about *what* you do with the result, so it's trivially reusable.
- The cleanup `clearTimeout` is what makes it a debounce rather than a throttle: each new keystroke discards the pending update.
- The `AbortController` in the consumer prevents a race where a slow earlier response resolves after a fast later one, which is the classic search-autocomplete bug.""",
    },

    # 2 -- useReducer for complex state
    {
        "prompt": "Refactor a multi-field form with interdependent validation from many useState calls to a single useReducer. Explain why useReducer is the better fit here.",
        "answer": """<thinking>
Many useState calls become unmanageable when fields depend on each other and updates need to be atomic (e.g. clearing an error when a field changes). useReducer centralizes transitions into one pure function:
1. A single state object holds values, touched flags, and errors.
2. Actions describe intent ("CHANGE_FIELD", "BLUR_FIELD", "SUBMIT"), and the reducer computes derived validation in one place.
3. Validation lives in the reducer so values and errors can never drift out of sync across renders.
</thinking>

```tsx
import { useReducer } from 'react';

type State = {
  values: { email: string; password: string; confirm: string };
  errors: Partial<Record<keyof State['values'], string>>;
  touched: Partial<Record<keyof State['values'], boolean>>;
};

type Action =
  | { type: 'CHANGE'; field: keyof State['values']; value: string }
  | { type: 'BLUR'; field: keyof State['values'] };

function validate(values: State['values']): State['errors'] {
  const errors: State['errors'] = {};
  if (!/^[^@]+@[^@]+$/.test(values.email)) errors.email = 'Invalid email';
  if (values.password.length < 8) errors.password = 'Min 8 characters';
  if (values.confirm !== values.password) errors.confirm = 'Passwords differ';
  return errors;
}

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'CHANGE': {
      const values = { ...state.values, [action.field]: action.value };
      return { ...state, values, errors: validate(values) };
    }
    case 'BLUR':
      return { ...state, touched: { ...state.touched, [action.field]: true } };
    default:
      return state;
  }
}

const initial: State = {
  values: { email: '', password: '', confirm: '' },
  errors: {},
  touched: {},
};

function SignupForm() {
  const [state, dispatch] = useReducer(reducer, initial);
  const field = (name: keyof State['values']) => ({
    value: state.values[name],
    onChange: (e: React.ChangeEvent<HTMLInputElement>) =>
      dispatch({ type: 'CHANGE', field: name, value: e.target.value }),
    onBlur: () => dispatch({ type: 'BLUR', field: name }),
  });

  return (
    <form>
      <input {...field('email')} />
      {state.touched.email && state.errors.email && <span>{state.errors.email}</span>}
      <input type="password" {...field('password')} />
      {state.touched.password && state.errors.password && <span>{state.errors.password}</span>}
      <input type="password" {...field('confirm')} />
      {state.touched.confirm && state.errors.confirm && <span>{state.errors.confirm}</span>}
    </form>
  );
}
```

**Explanation**
- With interdependent fields (confirm depends on password), recomputing all errors on every change in one reducer guarantees consistency -- there's no "I updated password but the confirm error is stale" bug.
- Reducers are pure and serializable, so the state logic is unit-testable in isolation without rendering.
- `touched` separates "is this invalid" from "should I show the error yet", a standard UX pattern.""",
    },

    # 3 -- GSAP timeline with cleanup via gsap.context
    {
        "prompt": "Build a React hero-section component that animates a heading, subheading, and CTA button in sequence on mount using a GSAP timeline. Make sure it cleans up correctly and works with React 18 Strict Mode double-invocation.",
        "answer": """<thinking>
React 18 Strict Mode mounts, unmounts, and remounts components in dev, so any animation setup must be fully reversible. The idiomatic GSAP-in-React tool for this is `gsap.context()`, which scopes all animations created inside it and reverts them with a single `ctx.revert()` in cleanup. I'll:
1. Scope selectors to a container ref so the component is self-contained.
2. Build one timeline with staggered, eased steps.
3. Revert the context on cleanup so a remount re-runs from a clean slate.
</thinking>

```jsx
import { useRef, useLayoutEffect } from 'react';
import gsap from 'gsap';

export function Hero() {
  const root = useRef(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      const tl = gsap.timeline({ defaults: { ease: 'power3.out', duration: 0.6 } });
      tl.from('.hero-title', { y: 40, opacity: 0 })
        .from('.hero-sub', { y: 30, opacity: 0 }, '-=0.3')
        .from('.hero-cta', { scale: 0.8, opacity: 0 }, '-=0.2');
    }, root); // scope selectors to this element

    return () => ctx.revert(); // undo everything on unmount / Strict Mode remount
  }, []);

  return (
    <section ref={root} className="hero">
      <h1 className="hero-title">Ship faster.</h1>
      <p className="hero-sub">Animations that don't fight React.</p>
      <button className="hero-cta">Get started</button>
    </section>
  );
}
```

**Explanation**
- `gsap.context(fn, root)` records every animation created inside `fn` and confines selector strings like `.hero-title` to descendants of `root`, so two `<Hero>` instances never collide.
- `ctx.revert()` returns the DOM to its pre-animation state, which is exactly what Strict Mode's mount/unmount/remount cycle needs -- without it the second mount would animate from already-final values and appear frozen.
- `useLayoutEffect` runs before paint, so `.from()` (which sets the start state) avoids a flash of the final layout.""",
    },

    # 4 -- React.memo + useCallback to stop re-renders
    {
        "prompt": "A parent component re-renders a long list of expensive child items on every keystroke in an unrelated input. Diagnose why and fix it with React.memo, useCallback, and useMemo. Explain when memoization actually helps.",
        "answer": """<thinking>
The chain is: typing updates parent state -> parent re-renders -> every child re-renders because (a) children aren't memoized and (b) inline callbacks/objects passed as props are new references each render. Fix:
1. Wrap each child in React.memo so it only re-renders when its props change by reference.
2. Stabilize callback props with useCallback so memo's shallow compare succeeds.
3. Stabilize any derived data passed down with useMemo.
And I should be honest that memoization has a cost and only pays off when render is expensive and props are genuinely stable.
</thinking>

```jsx
import { memo, useCallback, useMemo, useState } from 'react';

const Row = memo(function Row({ item, onToggle }) {
  // pretend this is expensive (formatting, charts, etc.)
  return (
    <li>
      <label>
        <input type="checkbox" checked={item.done} onChange={() => onToggle(item.id)} />
        {item.label}
      </label>
    </li>
  );
});

export function List({ items }) {
  const [filter, setFilter] = useState('');
  const [state, setState] = useState(items);

  // stable identity across renders -> memo on Row holds
  const onToggle = useCallback((id) => {
    setState((prev) =>
      prev.map((it) => (it.id === id ? { ...it, done: !it.done } : it))
    );
  }, []);

  // recomputed only when inputs change, not on every keystroke elsewhere
  const visible = useMemo(
    () => state.filter((it) => it.label.includes(filter)),
    [state, filter]
  );

  return (
    <>
      <input value={filter} onChange={(e) => setFilter(e.target.value)} />
      <ul>
        {visible.map((it) => (
          <Row key={it.id} item={it} onToggle={onToggle} />
        ))}
      </ul>
    </>
  );
}
```

**Explanation**
- `React.memo` does a shallow prop comparison; it's useless if you hand it a fresh `onToggle={() => ...}` each render, so `useCallback([])` is what makes the memo effective.
- `useMemo` here recomputes `visible` only when `state` or `filter` change -- but note: typing in the filter still updates `visible`, which is correct; the win is that toggling an item no longer re-runs unrelated work.
- When NOT to bother: if `Row` is cheap, memoization adds comparison + memory overhead for no gain. Reach for it only after profiling shows wasted renders of genuinely expensive components.""",
    },

    # 5 -- Context + useReducer store pattern, split contexts
    {
        "prompt": "Design a global theme + auth store using Context and useReducer, and explain how to avoid the common pitfall of unnecessary re-renders by splitting state and dispatch into separate contexts.",
        "answer": """<thinking>
A single context whose value is `{ state, dispatch }` re-renders every consumer whenever state changes -- even consumers that only call dispatch. The fix is to split into two providers:
1. A StateContext that changes when state changes.
2. A DispatchContext whose value (the dispatch function) is referentially stable forever.
Components that only dispatch subscribe to DispatchContext and never re-render on state changes. I'll also memoize state value to keep it stable when unrelated parts of a parent re-render.
</thinking>

```tsx
import { createContext, useContext, useReducer, useMemo, Dispatch } from 'react';

type State = { theme: 'light' | 'dark'; user: { name: string } | null };
type Action =
  | { type: 'TOGGLE_THEME' }
  | { type: 'LOGIN'; user: { name: string } }
  | { type: 'LOGOUT' };

function reducer(s: State, a: Action): State {
  switch (a.type) {
    case 'TOGGLE_THEME': return { ...s, theme: s.theme === 'light' ? 'dark' : 'light' };
    case 'LOGIN': return { ...s, user: a.user };
    case 'LOGOUT': return { ...s, user: null };
  }
}

const StateCtx = createContext<State | null>(null);
const DispatchCtx = createContext<Dispatch<Action> | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, dispatch] = useReducer(reducer, { theme: 'light', user: null });
  // dispatch is stable for the lifetime of the component; state is the only changing value
  return (
    <DispatchCtx.Provider value={dispatch}>
      <StateCtx.Provider value={state}>{children}</StateCtx.Provider>
    </DispatchCtx.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(StateCtx);
  if (!ctx) throw new Error('useAppState must be used within AppProvider');
  return ctx;
}
export function useAppDispatch() {
  const ctx = useContext(DispatchCtx);
  if (!ctx) throw new Error('useAppDispatch must be used within AppProvider');
  return ctx;
}
```

**Explanation**
- `dispatch` from `useReducer` is guaranteed stable, so a component using only `useAppDispatch()` (e.g. a logout button) never re-renders when `theme` or `user` change.
- Throwing in the hooks when the context is null gives a clear error instead of a confusing null-deref deep in a child.
- For very large stores where even state-consumers over-render, the next step is a selector-based library (Zustand, Redux Toolkit) or `use-context-selector`; this split is the zero-dependency version that handles most apps.""",
    },

    # 6 -- useRef for mutable values + previous value
    {
        "prompt": "Show two correct uses of useRef that are NOT about DOM nodes: (1) storing a mutable value that survives renders without triggering them, and (2) tracking the previous value of a prop or state. Explain why useState would be wrong for these.",
        "answer": """<thinking>
useRef is commonly taught only as "grab a DOM node", but its more general role is "a mutable box that persists across renders and does not cause a re-render when mutated". Two canonical non-DOM uses:
1. An interval/timer id, or a render counter -- mutating it should not re-render.
2. A `usePrevious` hook that captures the value from the last render via an effect that runs after render.
useState would be wrong because writing to it triggers a render, causing loops or unnecessary work.
</thinking>

```jsx
import { useRef, useEffect, useState } from 'react';

// (1) Mutable value that doesn't trigger renders: a stable interval id
function Stopwatch() {
  const [seconds, setSeconds] = useState(0);
  const intervalId = useRef(null);

  const start = () => {
    if (intervalId.current) return; // already running
    intervalId.current = setInterval(() => setSeconds((s) => s + 1), 1000);
  };
  const stop = () => {
    clearInterval(intervalId.current);
    intervalId.current = null;
  };

  useEffect(() => stop, []); // clear on unmount
  return (
    <>
      <span>{seconds}s</span>
      <button onClick={start}>Start</button>
      <button onClick={stop}>Stop</button>
    </>
  );
}

// (2) Track the previous value
function usePrevious(value) {
  const ref = useRef();
  useEffect(() => {
    ref.current = value; // runs AFTER render, so during render ref holds the prior value
  }, [value]);
  return ref.current;
}
```

**Explanation**
- Storing `intervalId` in state would re-render on every start/stop for no visual reason and risk stale closures; a ref is the right "instance variable" for React function components.
- In `usePrevious`, the effect updates the ref *after* the render commits, so the value read during the current render is the one from the previous render -- exactly the previous-value semantics you want.
- Rule of thumb: if changing the value should update the UI, use state; if it's bookkeeping that the UI doesn't directly display, use a ref.""",
    },

    # 7 -- useEffect cleanup and dependency correctness
    {
        "prompt": "Explain the most common useEffect mistakes (missing cleanup, wrong dependency array, stale closures) and rewrite a buggy subscription effect correctly. Include the case where you intentionally want a 'run once' effect.",
        "answer": """<thinking>
The classic bugs:
1. Subscribing without unsubscribing -> memory leaks / duplicate handlers.
2. Wrong deps -> either stale closures (deps too narrow) or re-subscribing every render (deps too wide / new object identities).
3. Reading state in an interval without using the functional updater -> stale value frozen at subscription time.
I'll show a buggy version, then a corrected one, and address the legitimate empty-deps "run once" case plus how to handle values you read but don't want to re-subscribe on (via the functional updater or a ref).
</thinking>

```jsx
// BUGGY: leaks listeners, stale count
function Buggy({ channel }) {
  const [count, setCount] = useState(0);
  useEffect(() => {
    socket.on(channel, () => setCount(count + 1)); // stale `count`, no cleanup
  }); // no deps -> runs every render
  return <p>{count}</p>;
}

// CORRECT
function Counter({ channel }) {
  const [count, setCount] = useState(0);

  useEffect(() => {
    const handler = () => setCount((c) => c + 1); // functional updater: no stale closure
    socket.on(channel, handler);
    return () => socket.off(channel, handler); // cleanup on channel change / unmount
  }, [channel]); // re-subscribe only when channel changes

  return <p>{count}</p>;
}
```

**Explanation**
- Using `setCount((c) => c + 1)` removes `count` from the dependency list legitimately, so the effect only re-runs when `channel` actually changes -- avoiding both the stale-closure bug and needless re-subscription.
- Always return a cleanup that mirrors the setup (`on`/`off`, `addEventListener`/`removeEventListener`, `subscribe`/`unsubscribe`). React calls it before re-running the effect and on unmount.
- A truly "run once on mount" effect with `[]` is fine for things like an initial fetch or a one-time analytics ping -- but be honest that under Strict Mode it runs twice in dev, so the effect must be idempotent (use an AbortController for fetches).""",
    },

    # 8 -- Error boundary (class) + fallback + reset
    {
        "prompt": "Implement a reusable React error boundary that catches render errors, shows a fallback UI, logs the error, and lets the user retry. Explain what error boundaries can and cannot catch.",
        "answer": """<thinking>
Error boundaries must be class components (there is no hook equivalent for `componentDidCatch` / `getDerivedStateFromError`). Requirements:
1. `getDerivedStateFromError` to flip into the error state and render a fallback.
2. `componentDidCatch` to log to an error reporting service.
3. A reset mechanism so the user can retry without a full reload, plus a `resetKeys` pattern to auto-reset when inputs change.
4. Be clear about limits: boundaries do NOT catch errors in event handlers, async code, SSR, or the boundary itself.
</thinking>

```tsx
import { Component, ReactNode, ErrorInfo } from 'react';

type Props = {
  children: ReactNode;
  fallback: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
};
type State = { error: Error | null };

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error }; // render fallback on next render
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    this.props.onError?.(error, info); // e.g. Sentry.captureException(error)
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) return this.props.fallback(this.state.error, this.reset);
    return this.props.children;
  }
}

// Usage
<ErrorBoundary
  onError={(e) => reportToSentry(e)}
  fallback={(error, reset) => (
    <div role="alert">
      <p>Something went wrong: {error.message}</p>
      <button onClick={reset}>Try again</button>
    </div>
  )}
>
  <Dashboard />
</ErrorBoundary>
```

**Explanation**
- Error boundaries catch errors during *rendering*, in lifecycle methods, and in constructors of the tree below them.
- They do NOT catch: event-handler errors (use try/catch there), asynchronous code like setTimeout/fetch callbacks, server-side rendering, or errors thrown in the boundary component itself -- wrap such code manually.
- Pair boundaries with React Suspense for a complete fallback strategy: Suspense handles loading, the boundary handles failure. Granular boundaries (per widget) keep one broken panel from taking down the whole page.""",
    },

    # 9 -- Data fetching hook with loading/error/refetch
    {
        "prompt": "Write a useFetch hook with proper loading/error/data states, request cancellation, and a refetch function. Discuss why you might prefer a library like React Query in production.",
        "answer": """<thinking>
A robust fetch hook needs: a discriminated state machine (idle/loading/success/error), cancellation on unmount or url change, and a manual refetch. I'll model state with a reducer-ish object, use AbortController, and track a mounted flag implicitly via the abort signal. Then I'll be candid that caching, dedup, retries, and background refetch are why React Query usually wins in production.
</thinking>

```tsx
import { useState, useEffect, useCallback } from 'react';

type FetchState<T> =
  | { status: 'loading' }
  | { status: 'error'; error: Error }
  | { status: 'success'; data: T };

export function useFetch<T>(url: string) {
  const [state, setState] = useState<FetchState<T>>({ status: 'loading' });
  const [nonce, setNonce] = useState(0); // bump to refetch

  const refetch = useCallback(() => setNonce((n) => n + 1), []);

  useEffect(() => {
    const controller = new AbortController();
    setState({ status: 'loading' });

    fetch(url, { signal: controller.signal })
      .then((res) => {
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        return res.json() as Promise<T>;
      })
      .then((data) => setState({ status: 'success', data }))
      .catch((error: Error) => {
        if (error.name !== 'AbortError') setState({ status: 'error', error });
      });

    return () => controller.abort(); // cancel on url change / unmount
  }, [url, nonce]);

  return { ...state, refetch };
}

// Usage
function User({ id }: { id: string }) {
  const res = useFetch<{ name: string }>(`/api/users/${id}`);
  if (res.status === 'loading') return <Spinner />;
  if (res.status === 'error') return <button onClick={res.refetch}>Retry</button>;
  return <h1>{res.data.name}</h1>;
}
```

**Explanation**
- The discriminated union makes illegal states unrepresentable: you can't have `data` and an `error` at the same time, and TypeScript narrows `res.data` only inside the `'success'` branch.
- `AbortController` cancels in-flight requests when `url`/`nonce` changes, preventing the stale-response race and React's "state update on unmounted component" warnings.
- In production prefer React Query / SWR: they add request deduplication, caching with stale-while-revalidate, automatic retries, window-focus refetch, and pagination -- all of which this hook would have to reinvent. Hand-rolled fetch hooks are great for learning and for the rare case you can't add a dependency.""",
    },

    # 10 -- GSAP ScrollTrigger reveal with React + cleanup
    {
        "prompt": "Create a React component that uses GSAP ScrollTrigger to fade-and-slide multiple cards into view as the user scrolls, with a stagger. Ensure ScrollTriggers are killed on unmount and refreshed on layout changes.",
        "answer": """<thinking>
ScrollTrigger needs registering once, and each trigger must be killed on cleanup or they accumulate. Using gsap.context scoped to a ref handles the kill automatically via revert(). For a staggered reveal I'll batch the cards with ScrollTrigger.batch so they animate in groups as they enter, and call ScrollTrigger.refresh() if content can change size after load (images, fonts).
</thinking>

```jsx
import { useRef, useLayoutEffect } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export function CardGrid({ cards }) {
  const root = useRef(null);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      gsap.set('.card', { opacity: 0, y: 40 });
      ScrollTrigger.batch('.card', {
        start: 'top 85%',
        onEnter: (batch) =>
          gsap.to(batch, {
            opacity: 1,
            y: 0,
            duration: 0.6,
            stagger: 0.12,
            ease: 'power2.out',
            overwrite: true,
          }),
      });
    }, root);

    // images/fonts can shift positions after first paint
    const onLoad = () => ScrollTrigger.refresh();
    window.addEventListener('load', onLoad);

    return () => {
      window.removeEventListener('load', onLoad);
      ctx.revert(); // kills all ScrollTriggers created in this context
    };
  }, [cards.length]);

  return (
    <div ref={root} className="grid">
      {cards.map((c) => (
        <div key={c.id} className="card">{c.title}</div>
      ))}
    </div>
  );
}
```

**Explanation**
- `ScrollTrigger.batch` groups elements that enter the viewport together, so you get a clean staggered cascade instead of each card firing its own animation -- much smoother and cheaper.
- `ctx.revert()` is the single cleanup call that kills every ScrollTrigger and animation created inside the context, which is essential because orphaned ScrollTriggers keep listening to scroll events and leak.
- `ScrollTrigger.refresh()` after `load` recomputes start/end positions once images and web fonts have changed the layout; without it triggers fire at the wrong scroll offsets.""",
    },

    # 11 -- Compound component pattern
    {
        "prompt": "Implement an accessible Tabs component using the compound component pattern (Tabs, TabList, Tab, TabPanels, TabPanel) sharing state via context. Explain the benefits over a monolithic props-based API.",
        "answer": """<thinking>
The compound component pattern lets consumers compose subcomponents declaratively while internal state (active index) is shared implicitly through context. I'll:
1. Create a TabsContext with the active index and a setter.
2. Build Tab (button) and TabPanel that read context and self-determine selection by index.
3. Add ARIA roles/attributes for accessibility and keyboard support.
This beats a monolithic `<Tabs items={[...]} />` because consumers fully control markup and ordering.
</thinking>

```tsx
import { createContext, useContext, useId, useState, ReactNode } from 'react';

const TabsCtx = createContext<{ active: number; setActive: (i: number) => void; baseId: string } | null>(null);
const useTabs = () => {
  const c = useContext(TabsCtx);
  if (!c) throw new Error('Tabs.* must be used inside <Tabs>');
  return c;
};

export function Tabs({ children, defaultIndex = 0 }: { children: ReactNode; defaultIndex?: number }) {
  const [active, setActive] = useState(defaultIndex);
  const baseId = useId();
  return <TabsCtx.Provider value={{ active, setActive, baseId }}>{children}</TabsCtx.Provider>;
}

export function TabList({ children }: { children: ReactNode }) {
  return <div role="tablist">{children}</div>;
}

export function Tab({ index, children }: { index: number; children: ReactNode }) {
  const { active, setActive, baseId } = useTabs();
  const selected = active === index;
  return (
    <button
      role="tab"
      id={`${baseId}-tab-${index}`}
      aria-selected={selected}
      aria-controls={`${baseId}-panel-${index}`}
      tabIndex={selected ? 0 : -1}
      onClick={() => setActive(index)}
    >
      {children}
    </button>
  );
}

export function TabPanel({ index, children }: { index: number; children: ReactNode }) {
  const { active, baseId } = useTabs();
  if (active !== index) return null;
  return (
    <div role="tabpanel" id={`${baseId}-panel-${index}`} aria-labelledby={`${baseId}-tab-${index}`}>
      {children}
    </div>
  );
}

// Usage
<Tabs>
  <TabList>
    <Tab index={0}>Profile</Tab>
    <Tab index={1}>Settings</Tab>
  </TabList>
  <TabPanel index={0}>Profile content</TabPanel>
  <TabPanel index={1}>Settings content</TabPanel>
</Tabs>
```

**Explanation**
- Context shares the active index without prop-drilling, so `<Tab>` and `<TabPanel>` stay decoupled yet coordinated -- the hallmark of the compound pattern.
- Consumers control the markup completely (wrap tabs in a flexbox, add icons, reorder) instead of being constrained by a fixed `items` schema.
- `useId` generates stable, collision-free ids so `aria-controls`/`aria-labelledby` wire the tab to its panel for screen readers; `tabIndex` roving keeps keyboard focus on the active tab.""",
    },

    # 12 -- useImperativeHandle + forwardRef
    {
        "prompt": "Build a custom <FancyInput> that exposes focus() and clear() methods to its parent via forwardRef and useImperativeHandle, instead of leaking the raw DOM node. Explain when this pattern is appropriate.",
        "answer": """<thinking>
Sometimes a parent needs to imperatively command a child (focus a field, scroll, play a video) without owning its internals. forwardRef + useImperativeHandle lets the child expose a curated API rather than the bare DOM node, which preserves encapsulation. I'll expose only focus and clear, keep the real input ref private, and type the handle.
</thinking>

```tsx
import { forwardRef, useImperativeHandle, useRef, useState } from 'react';

export type FancyInputHandle = { focus: () => void; clear: () => void };

export const FancyInput = forwardRef<FancyInputHandle, { placeholder?: string }>(
  function FancyInput({ placeholder }, ref) {
    const inputRef = useRef<HTMLInputElement>(null);
    const [value, setValue] = useState('');

    useImperativeHandle(ref, () => ({
      focus: () => inputRef.current?.focus(),
      clear: () => {
        setValue('');
        inputRef.current?.focus();
      },
    }), []); // empty deps: handle identity is stable

    return (
      <input
        ref={inputRef}
        value={value}
        placeholder={placeholder}
        onChange={(e) => setValue(e.target.value)}
      />
    );
  }
);

// Usage
function Form() {
  const inputRef = useRef<FancyInputHandle>(null);
  return (
    <>
      <FancyInput ref={inputRef} placeholder="Email" />
      <button onClick={() => inputRef.current?.focus()}>Focus</button>
      <button onClick={() => inputRef.current?.clear()}>Clear</button>
    </>
  );
}
```

**Explanation**
- Exposing `{ focus, clear }` instead of the DOM node keeps the component's internals private -- the parent can't accidentally reach in and mutate `value` or styles, so you can refactor the child freely.
- Use this pattern only for genuinely imperative actions (focus, scroll, media controls, animations). For data, stick with props/state -- imperative handles that mutate state are a smell.
- The empty dependency array keeps the handle object stable across renders; include deps if the methods close over changing values.""",
    },

    # 13 -- Virtualized long list performance
    {
        "prompt": "A list of 50,000 rows freezes the browser. Explain windowing/virtualization and implement a minimal virtualized list from scratch (fixed row height) without a library. Then note when to reach for react-window.",
        "answer": """<thinking>
Rendering 50k DOM nodes is the problem; the fix is to render only the rows visible in the scroll viewport plus a small overscan buffer, and use a spacer to preserve scrollbar size. With fixed row height the math is simple: startIndex = scrollTop / rowHeight. I'll implement it with a scroll handler, a total-height spacer, and absolute/translateY positioning of the visible window.
</thinking>

```tsx
import { useRef, useState } from 'react';

function VirtualList({ items, rowHeight = 32, height = 400, overscan = 5 }: {
  items: string[]; rowHeight?: number; height?: number; overscan?: number;
}) {
  const [scrollTop, setScrollTop] = useState(0);
  const total = items.length * rowHeight;

  const startIndex = Math.max(0, Math.floor(scrollTop / rowHeight) - overscan);
  const visibleCount = Math.ceil(height / rowHeight) + overscan * 2;
  const endIndex = Math.min(items.length, startIndex + visibleCount);
  const slice = items.slice(startIndex, endIndex);

  return (
    <div
      style={{ height, overflowY: 'auto', position: 'relative' }}
      onScroll={(e) => setScrollTop(e.currentTarget.scrollTop)}
    >
      <div style={{ height: total }}>{/* spacer preserves scroll range */}
        <div style={{ transform: `translateY(${startIndex * rowHeight}px)` }}>
          {slice.map((item, i) => (
            <div key={startIndex + i} style={{ height: rowHeight }}>
              {item}
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

**Explanation**
- Only ~`visibleCount` rows exist in the DOM at any time regardless of list size, so render and memory stay constant -- that's what unfreezes the page.
- The full-height spacer (`total`) makes the scrollbar behave as if all rows were present; the `translateY` offset slides the rendered window to the correct position.
- `overscan` renders a few extra rows above/below so fast scrolling doesn't flash blank gaps. For variable row heights, sticky headers, or horizontal lists, use `react-window`/`react-virtual` -- measuring dynamic heights correctly is fiddly and they solve it well.""",
    },

    # 14 -- useSyncExternalStore for external state
    {
        "prompt": "Use useSyncExternalStore to subscribe a React component to an external store (e.g. browser online/offline status or a tiny custom store). Explain why this hook exists and how it prevents tearing in concurrent rendering.",
        "answer": """<thinking>
Before useSyncExternalStore, subscribing to external mutable sources via useEffect+useState risked "tearing" in concurrent React (different components rendering with different values of the same store mid-render). useSyncExternalStore takes a subscribe function and a getSnapshot function and guarantees a consistent, tear-free read. I'll show the canonical online-status example plus a tiny store.
</thinking>

```tsx
import { useSyncExternalStore } from 'react';

// (1) Browser online/offline
function subscribeOnline(callback: () => void) {
  window.addEventListener('online', callback);
  window.addEventListener('offline', callback);
  return () => {
    window.removeEventListener('online', callback);
    window.removeEventListener('offline', callback);
  };
}

export function useOnlineStatus() {
  return useSyncExternalStore(
    subscribeOnline,
    () => navigator.onLine,       // client snapshot
    () => true                    // server snapshot (SSR fallback)
  );
}

// (2) A tiny custom store
function createStore<T>(initial: T) {
  let state = initial;
  const listeners = new Set<() => void>();
  return {
    getState: () => state,
    setState: (next: T) => { state = next; listeners.forEach((l) => l()); },
    subscribe: (l: () => void) => { listeners.add(l); return () => listeners.delete(l); },
  };
}

const counterStore = createStore(0);
export function useCounter() {
  return useSyncExternalStore(counterStore.subscribe, counterStore.getState);
}
```

**Explanation**
- "Tearing" is when two components read different values of the same source during a single concurrent render pass, producing visually inconsistent UI. useSyncExternalStore forces a synchronous, consistent snapshot, eliminating it.
- It takes three args: `subscribe` (returns an unsubscribe), `getSnapshot` (current value), and an optional `getServerSnapshot` for SSR so hydration doesn't mismatch.
- This is the officially recommended primitive for any external/mutable data source and is exactly what state libraries like Zustand and Redux use under the hood now.""",
    },

    # 15 -- Controlled vs uncontrolled, lifting state
    {
        "prompt": "Explain controlled vs uncontrolled components in React forms with concrete examples, and show how to build a component that works in both modes (like many UI libraries do). Cover the defaultValue vs value gotcha.",
        "answer": """<thinking>
Controlled = React state is the single source of truth (value + onChange). Uncontrolled = the DOM holds the value, read via ref/defaultValue. A flexible component supports both: if a `value` prop is provided it's controlled, otherwise it self-manages with internal state seeded by `defaultValue`. The gotcha is passing `value` without `onChange` (read-only warning) or switching between defined/undefined value (controlled<->uncontrolled warning).
</thinking>

```tsx
import { useState, useRef } from 'react';

// Uncontrolled: DOM owns the value
function UncontrolledForm() {
  const ref = useRef<HTMLInputElement>(null);
  return (
    <form onSubmit={(e) => { e.preventDefault(); alert(ref.current?.value); }}>
      <input ref={ref} defaultValue="hello" />
    </form>
  );
}

// Dual-mode input (controlled if `value` provided, else uncontrolled internally)
function Input({
  value: controlledValue,
  defaultValue = '',
  onChange,
}: {
  value?: string;
  defaultValue?: string;
  onChange?: (v: string) => void;
}) {
  const isControlled = controlledValue !== undefined;
  const [internal, setInternal] = useState(defaultValue);
  const value = isControlled ? controlledValue : internal;

  return (
    <input
      value={value}
      onChange={(e) => {
        if (!isControlled) setInternal(e.target.value);
        onChange?.(e.target.value);
      }}
    />
  );
}
```

**Explanation**
- Controlled components give you validation, conditional disabling, and formatting because every keystroke flows through React state -- prefer them for anything interactive.
- Uncontrolled components (`defaultValue` + ref) are simpler and slightly faster for fire-and-forget forms or when integrating non-React widgets.
- The dual-mode trick keys off whether `value` is `undefined`. Never flip a single input between controlled and uncontrolled at runtime (passing `value={x ?? undefined}`) -- React warns and resets cursor position; decide the mode once based on whether the prop is supplied.""",
    },

    # 16 -- GSAP Flip layout animation
    {
        "prompt": "Use the GSAP Flip plugin in React to smoothly animate list items when they reorder (e.g. sorting), so elements glide to their new positions instead of jumping. Explain the FLIP technique.",
        "answer": """<thinking>
FLIP = First, Last, Invert, Play: record positions before a DOM change (First), let React reorder (Last), compute the delta and apply an inverse transform so it looks unmoved (Invert), then animate the transform to zero (Play). GSAP's Flip plugin automates this. In React the trick is to capture state with Flip.getState before the state update and run Flip.from in a layout effect after the reorder commits.
</thinking>

```jsx
import { useState, useRef, useLayoutEffect } from 'react';
import gsap from 'gsap';
import { Flip } from 'gsap/Flip';

gsap.registerPlugin(Flip);

export function SortableList({ initial }) {
  const [items, setItems] = useState(initial);
  const root = useRef(null);
  const stateRef = useRef(null);

  const shuffle = () => {
    stateRef.current = Flip.getState(root.current.querySelectorAll('.item')); // First
    setItems((prev) => [...prev].sort(() => Math.random() - 0.5));            // triggers Last
  };

  useLayoutEffect(() => {
    if (!stateRef.current) return;
    Flip.from(stateRef.current, { duration: 0.6, ease: 'power2.inOut', absolute: true }); // Invert + Play
    stateRef.current = null;
  }, [items]);

  return (
    <div ref={root}>
      <button onClick={shuffle}>Shuffle</button>
      {items.map((it) => (
        <div key={it.id} className="item">{it.label}</div>
      ))}
    </div>
  );
}
```

**Explanation**
- `Flip.getState` snapshots the current bounding boxes *before* you change state; capturing it inside the click handler (just before `setItems`) is critical so it reflects the pre-reorder layout.
- After React commits the new order, the `useLayoutEffect` runs before paint and `Flip.from` computes each element's delta, applies an inverting transform, and animates it away -- the eye sees a smooth glide rather than an instant jump.
- `absolute: true` takes moving items out of flow during the tween so siblings don't shove each other mid-animation; using stable `key`s ensures React reuses the same DOM nodes that Flip is tracking.""",
    },

    # 17 -- Render props pattern
    {
        "prompt": "Implement a <MouseTracker> component using the render-props pattern that supplies the current mouse position to its children. Compare render props with custom hooks and say which to prefer today.",
        "answer": """<thinking>
Render props share behavior by taking a function child and calling it with internal state. I'll build a mouse tracker that attaches a mousemove listener and calls `children({ x, y })`. Then I'll note that custom hooks supersede render props for most cases (less nesting), while render props still shine when you need to control *where* in the tree the shared value is consumed or to share JSX.
</thinking>

```tsx
import { useState, useEffect, ReactNode } from 'react';

type Pos = { x: number; y: number };

function MouseTracker({ children }: { children: (pos: Pos) => ReactNode }) {
  const [pos, setPos] = useState<Pos>({ x: 0, y: 0 });

  useEffect(() => {
    const onMove = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, []);

  return <>{children(pos)}</>;
}

// Usage
<MouseTracker>
  {({ x, y }) => <p>Mouse at {x}, {y}</p>}
</MouseTracker>

// The modern equivalent as a hook
function useMousePosition(): Pos {
  const [pos, setPos] = useState<Pos>({ x: 0, y: 0 });
  useEffect(() => {
    const onMove = (e: MouseEvent) => setPos({ x: e.clientX, y: e.clientY });
    window.addEventListener('mousemove', onMove);
    return () => window.removeEventListener('mousemove', onMove);
  }, []);
  return pos;
}
```

**Explanation**
- Render props were the pre-hooks way to share stateful logic without HOCs; the function-as-child gets called with the live value, letting the consumer render whatever it wants.
- Custom hooks (`useMousePosition`) do the same logic-sharing with far less nesting and no "wrapper hell", so prefer hooks by default in modern React.
- Render props are still useful when the shared thing is *JSX* (e.g. a virtualized list passing each row's props) or when you genuinely need different rendering at different positions in the same subtree -- something a hook's flat return can't express as ergonomically.""",
    },

    # 18 -- useTransition / deferred value for responsiveness
    {
        "prompt": "Use React 18's useTransition (and useDeferredValue) to keep a UI responsive while filtering a large list. Explain the difference between urgent and non-urgent updates.",
        "answer": """<thinking>
React 18 concurrency lets you mark some state updates as non-urgent (transitions) so React can interrupt them to keep urgent updates (like typing) snappy. useTransition gives you startTransition + isPending. useDeferredValue is the prop-driven equivalent: it returns a lagging copy of a value that updates at non-urgent priority. I'll show both: the input update stays urgent, the expensive filtered list update is deferred.
</thinking>

```tsx
import { useState, useTransition, useDeferredValue, useMemo } from 'react';

function FilterableList({ items }: { items: string[] }) {
  const [query, setQuery] = useState('');
  const [isPending, startTransition] = useTransition();
  const [list, setList] = useState(items);

  const onChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setQuery(e.target.value); // urgent: input must feel instant
    startTransition(() => {
      // non-urgent: heavy filtering can be interrupted
      setList(items.filter((i) => i.includes(e.target.value)));
    });
  };

  return (
    <>
      <input value={query} onChange={onChange} />
      <ul style={{ opacity: isPending ? 0.6 : 1 }}>
        {list.map((i) => <li key={i}>{i}</li>)}
      </ul>
    </>
  );
}

// Alternative with useDeferredValue (no manual startTransition)
function FilterableList2({ items }: { items: string[] }) {
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query); // lags behind during heavy work
  const filtered = useMemo(
    () => items.filter((i) => i.includes(deferredQuery)),
    [items, deferredQuery]
  );
  return (
    <>
      <input value={query} onChange={(e) => setQuery(e.target.value)} />
      <ul>{filtered.map((i) => <li key={i}>{i}</li>)}</ul>
    </>
  );
}
```

**Explanation**
- Urgent updates (typing, clicks, hovers) must reflect immediately; non-urgent updates (re-filtering 10k items, re-rendering a chart) can lag a frame or two without the user noticing.
- `startTransition` wraps the expensive state update so React can interrupt and re-prioritize it; `isPending` lets you show a subtle "updating" hint.
- `useDeferredValue` is the cleaner choice when the expensive work derives from a single value and you don't control the setter directly -- the deferred value trails the latest input, and `useMemo` ensures the heavy filter only re-runs when that deferred value settles.""",
    },

    # 19 -- Portal for modal + focus trap basics
    {
        "prompt": "Build an accessible Modal using createPortal that renders outside the DOM hierarchy, closes on Escape and backdrop click, and restores focus to the trigger on close. Explain why a portal is needed.",
        "answer": """<thinking>
Modals must escape overflow:hidden / z-index / transform stacking contexts of ancestors, which is why we render them into document.body via createPortal. Accessibility essentials: close on Escape, close on backdrop click (but not content click), lock body scroll, set role=dialog + aria-modal, move focus into the dialog on open and restore it to the trigger on close.
</thinking>

```tsx
import { useEffect, useRef, ReactNode } from 'react';
import { createPortal } from 'react-dom';

export function Modal({ open, onClose, children }: {
  open: boolean; onClose: () => void; children: ReactNode;
}) {
  const dialogRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<Element | null>(null);

  useEffect(() => {
    if (!open) return;
    triggerRef.current = document.activeElement;          // remember focus
    dialogRef.current?.focus();                           // move focus in
    document.body.style.overflow = 'hidden';              // lock scroll

    const onKey = (e: KeyboardEvent) => e.key === 'Escape' && onClose();
    document.addEventListener('keydown', onKey);

    return () => {
      document.removeEventListener('keydown', onKey);
      document.body.style.overflow = '';
      (triggerRef.current as HTMLElement | null)?.focus(); // restore focus
    };
  }, [open, onClose]);

  if (!open) return null;

  return createPortal(
    <div className="backdrop" onClick={onClose}>
      <div
        ref={dialogRef}
        role="dialog"
        aria-modal="true"
        tabIndex={-1}
        className="dialog"
        onClick={(e) => e.stopPropagation()} // don't close when clicking inside
      >
        {children}
      </div>
    </div>,
    document.body
  );
}
```

**Explanation**
- A portal renders the modal into `document.body` so it isn't clipped by an ancestor's `overflow: hidden` or trapped under a low `z-index`, while still participating in React's event system and context tree.
- `stopPropagation` on the dialog separates "clicked backdrop" (close) from "clicked content" (keep open); Escape handling is a baseline accessibility expectation.
- Saving `document.activeElement` on open and refocusing it on close preserves keyboard users' place in the page -- losing focus to `<body>` after closing a dialog is a common and frustrating a11y bug. A production version should also trap Tab focus within the dialog.""",
    },

    # 20 -- Higher-order component with proper ref/displayName
    {
        "prompt": "Write a higher-order component withLoading that shows a spinner while a `loading` prop is true and otherwise renders the wrapped component. Forward refs and props correctly and set a useful displayName. Note why hooks usually replace HOCs.",
        "answer": """<thinking>
A correct HOC must: pass through all props, forward refs (so the parent can still ref the inner component), and set displayName for good DevTools output. I'll strip the `loading` prop and spread the rest. Then I'll acknowledge that custom hooks usually replace HOCs to avoid wrapper nesting and prop-name collisions, but HOCs remain useful for cross-cutting concerns applied at the component boundary.
</thinking>

```tsx
import { ComponentType, forwardRef } from 'react';

type WithLoadingProps = { loading?: boolean };

export function withLoading<P extends object>(Wrapped: ComponentType<P>) {
  const Hoc = forwardRef<unknown, P & WithLoadingProps>(({ loading, ...rest }, ref) => {
    if (loading) return <div role="status" aria-live="polite">Loading...</div>;
    return <Wrapped ref={ref} {...(rest as P)} />;
  });

  Hoc.displayName = `withLoading(${Wrapped.displayName || Wrapped.name || 'Component'})`;
  return Hoc;
}

// Usage
const UserCard = ({ name }: { name: string }) => <div>{name}</div>;
const UserCardWithLoading = withLoading(UserCard);
// <UserCardWithLoading loading={isLoading} name="Ada" />
```

**Explanation**
- Spreading `...rest` after destructuring out `loading` keeps the HOC transparent -- the wrapped component receives every prop it expects and nothing it doesn't.
- `forwardRef` ensures `ref` reaches the inner component rather than the HOC wrapper, which matters when the parent needs imperative access; setting `displayName` keeps React DevTools readable (`withLoading(UserCard)` instead of `ForwardRef`).
- Modern React prefers a `useLoading`-style hook or simply conditional rendering at the call site, because HOCs add wrapper depth and risk prop-name collisions. Use HOCs when you need to uniformly augment many components at their boundary (e.g. analytics, error boundaries).""",
    },

    # 21 -- Testing a component with React Testing Library
    {
        "prompt": "Write React Testing Library tests for a Counter component, covering rendering, user interaction, and async behavior. Explain the guiding philosophy ('test behavior, not implementation') and why you query by role/text rather than test ids.",
        "answer": """<thinking>
RTL's philosophy is to test what the user sees and does, not internal state. I'll show: render + assert initial UI, simulate a click with userEvent and assert the result, and an async case using findBy for an element that appears after a delay. Prefer getByRole/getByText (accessible queries) over test ids so tests double as accessibility checks and survive refactors.
</thinking>

```tsx
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { Counter } from './Counter';

test('renders with initial count of 0', () => {
  render(<Counter />);
  expect(screen.getByRole('button', { name: /count: 0/i })).toBeInTheDocument();
});

test('increments when clicked', async () => {
  const user = userEvent.setup();
  render(<Counter />);
  const button = screen.getByRole('button', { name: /count/i });

  await user.click(button);
  await user.click(button);

  expect(button).toHaveTextContent('Count: 2');
});

test('shows a saved message after async save', async () => {
  const user = userEvent.setup();
  render(<Counter />);

  await user.click(screen.getByRole('button', { name: /save/i }));

  // findBy* retries until the async element appears (or times out)
  expect(await screen.findByText(/saved!/i)).toBeInTheDocument();
});
```

**Explanation**
- "Test behavior, not implementation" means assert on rendered output and user-visible effects, so tests don't break when you refactor internal state from useState to useReducer -- the contract with the user is unchanged.
- Prefer `getByRole` / `getByText`: these mirror how assistive tech finds elements, so passing tests also indicate decent accessibility. Reserve `data-testid` for things with no accessible handle.
- Use `userEvent` (not `fireEvent`) because it simulates real interaction sequences (focus, keydown, input), and use `findBy*` for anything asynchronous -- it polls and waits, removing flaky manual `act`/`waitFor` boilerplate.""",
    },

    # 22 -- Optimistic UI update with rollback
    {
        "prompt": "Implement an optimistic UI update for a 'like' button: update the UI immediately, send the request, and roll back if it fails. Show both a plain useState version and mention React 19's useOptimistic.",
        "answer": """<thinking>
Optimistic UI assumes the server will succeed and updates immediately for snappiness, then reconciles. The manual pattern: snapshot previous state, apply the optimistic change, fire the request, and on error revert to the snapshot. I'll guard against double-clicks and show the React 19 useOptimistic primitive that formalizes this.
</thinking>

```tsx
import { useState } from 'react';

function LikeButton({ postId, initialLikes, initiallyLiked }: {
  postId: string; initialLikes: number; initiallyLiked: boolean;
}) {
  const [liked, setLiked] = useState(initiallyLiked);
  const [likes, setLikes] = useState(initialLikes);
  const [pending, setPending] = useState(false);

  const toggle = async () => {
    if (pending) return;
    const prev = { liked, likes };

    // optimistic update
    setLiked(!liked);
    setLikes((n) => n + (liked ? -1 : 1));
    setPending(true);

    try {
      const res = await fetch(`/api/posts/${postId}/like`, { method: 'POST' });
      if (!res.ok) throw new Error('Request failed');
    } catch {
      setLiked(prev.liked);   // rollback
      setLikes(prev.likes);
    } finally {
      setPending(false);
    }
  };

  return (
    <button onClick={toggle} aria-pressed={liked}>
      {liked ? '♥' : '♡'} {likes}
    </button>
  );
}
```

**Explanation**
- Snapshotting `prev` before mutating lets you restore the exact prior state on failure, so a flaky network never leaves the UI in a wrong, "stuck liked" state.
- Guarding with `pending` prevents rapid double-clicks from sending contradictory requests and corrupting the count.
- In React 19, `useOptimistic(state, reducer)` formalizes this: you render the optimistic value during the async action and React automatically reverts to the real state when the action settles -- less manual snapshot/rollback bookkeeping. The manual version above is the portable equivalent.""",
    },

    # 23 -- useLayoutEffect vs useEffect (measure DOM)
    {
        "prompt": "Explain when to use useLayoutEffect instead of useEffect, with a concrete example: measuring an element and positioning a tooltip so it never flickers. Note the SSR caveat.",
        "answer": """<thinking>
useEffect runs after paint; useLayoutEffect runs synchronously after DOM mutations but before the browser paints. So for read-then-mutate-layout work (measure an element, then position something based on it), useLayoutEffect avoids a visible flicker of the wrong position. The SSR caveat: useLayoutEffect warns on the server because there's no layout; guard or use useEffect for SSR. Example: a tooltip that measures its size to flip above/below the trigger.
</thinking>

```tsx
import { useLayoutEffect, useRef, useState } from 'react';

function Tooltip({ targetRect, children }: {
  targetRect: DOMRect; children: React.ReactNode;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [style, setStyle] = useState<React.CSSProperties>({ visibility: 'hidden' });

  useLayoutEffect(() => {
    const tip = ref.current!;
    const { height } = tip.getBoundingClientRect();
    const fitsAbove = targetRect.top - height > 8;
    // synchronously set final position BEFORE paint -> no flicker
    setStyle({
      position: 'fixed',
      left: targetRect.left,
      top: fitsAbove ? targetRect.top - height - 8 : targetRect.bottom + 8,
      visibility: 'visible',
    });
  }, [targetRect]);

  return <div ref={ref} style={style} role="tooltip">{children}</div>;
}
```

**Explanation**
- The tooltip must be measured (`getBoundingClientRect`) and then repositioned. With `useEffect` the browser would paint the tooltip at its initial position first, then jump -- a visible flicker. `useLayoutEffect` runs before paint, so the user only ever sees the corrected position.
- Starting with `visibility: 'hidden'` lets it be measured without showing the unpositioned state.
- Caveat: `useLayoutEffect` warns during SSR because the server can't lay out the DOM. If the component renders on the server, either guard it (`typeof window !== 'undefined'`) or use the `useIsomorphicLayoutEffect` pattern (alias to `useEffect` on the server). Default to `useEffect` unless you specifically need pre-paint DOM reads.""",
    },

    # 24 -- Cancelable async + race-safe effect
    {
        "prompt": "Show the correct pattern for handling async work in useEffect that depends on a prop, avoiding the 'set state after unmount' warning and the out-of-order response race. Cover both AbortController and the ignore-flag fallback.",
        "answer": """<thinking>
Two related hazards: (1) component unmounts (or dep changes) before the promise resolves, so the setState lands on stale work; (2) responses arrive out of order when the dep changes faster than requests resolve. The canonical fixes are AbortController for fetch (cancels the network request) and an `ignore` boolean captured in the effect closure for non-cancelable promises. I'll show both.
</thinking>

```tsx
import { useState, useEffect } from 'react';

// Preferred: AbortController (cancels the actual request)
function Profile({ userId }: { userId: string }) {
  const [data, setData] = useState<unknown>(null);

  useEffect(() => {
    const controller = new AbortController();
    fetch(`/api/users/${userId}`, { signal: controller.signal })
      .then((r) => r.json())
      .then(setData)
      .catch((e) => { if (e.name !== 'AbortError') throw e; });
    return () => controller.abort();
  }, [userId]);

  return <pre>{JSON.stringify(data)}</pre>;
}

// Fallback: ignore flag for non-cancelable promises
function Profile2({ userId }: { userId: string }) {
  const [data, setData] = useState<unknown>(null);

  useEffect(() => {
    let ignore = false;
    someUncancelableApi(userId).then((result) => {
      if (!ignore) setData(result); // skip stale resolutions
    });
    return () => { ignore = true; };
  }, [userId]);

  return <pre>{JSON.stringify(data)}</pre>;
}
```

**Explanation**
- AbortController is the best tool when the work is a `fetch`: aborting both prevents the stale `setState` and actually cancels the in-flight network request, saving bandwidth.
- The `ignore` flag is the universal fallback for promises you can't cancel (a third-party SDK): the cleanup flips `ignore` so the resolved-but-stale callback becomes a no-op, eliminating both the unmount warning and the out-of-order race.
- Because the cleanup runs whenever `userId` changes (not just on unmount), rapid prop changes can't let an earlier slow response overwrite a newer one -- only the latest effect's resolution is allowed to set state.""",
    },

    # 25 -- GSAP + React: timeline controls bound to state
    {
        "prompt": "Create a React component that builds a GSAP timeline once and exposes play/pause/reverse controls plus a scrubber slider bound to the timeline progress. Explain how to keep the timeline instance stable across renders.",
        "answer": """<thinking>
The timeline should be created once (not rebuilt every render) and stored in a ref so its identity is stable; controls call methods on it. A scrubber reads/writes timeline.progress(). To reflect playback in the slider I subscribe to the timeline's onUpdate. Cleanup kills the timeline. I'll use gsap.context for scoped, revertible setup.
</thinking>

```jsx
import { useRef, useState, useLayoutEffect } from 'react';
import gsap from 'gsap';

export function TimelinePlayer() {
  const root = useRef(null);
  const tl = useRef(null);
  const [progress, setProgress] = useState(0);

  useLayoutEffect(() => {
    const ctx = gsap.context(() => {
      tl.current = gsap.timeline({
        paused: true,
        onUpdate: () => setProgress(tl.current.progress()),
      });
      tl.current
        .to('.box', { x: 300, duration: 1, ease: 'power2.inOut' })
        .to('.box', { rotation: 360, duration: 1 })
        .to('.box', { backgroundColor: '#e11', duration: 0.5 });
    }, root);
    return () => ctx.revert(); // kills timeline + reverts DOM
  }, []);

  return (
    <div ref={root}>
      <div className="box" style={{ width: 60, height: 60, background: '#06f' }} />
      <button onClick={() => tl.current.play()}>Play</button>
      <button onClick={() => tl.current.pause()}>Pause</button>
      <button onClick={() => tl.current.reverse()}>Reverse</button>
      <input
        type="range" min={0} max={1} step={0.001} value={progress}
        onChange={(e) => {
          tl.current.pause();
          tl.current.progress(Number(e.target.value)); // scrub
        }}
      />
    </div>
  );
}
```

**Explanation**
- Storing the timeline in `tl.current` (a ref) keeps its identity stable across re-renders, so building it inside a `[]` layout effect guarantees it's constructed exactly once -- rebuilding it every render would restart animations and leak instances.
- `onUpdate` -> `setProgress` is a one-way binding from timeline to slider so the scrubber tracks playback; the slider's `onChange` writes back via `progress()` for two-way control, pausing first so it doesn't fight the playhead.
- `gsap.context(...).revert()` is the clean teardown: it kills the timeline and undoes its DOM effects, which is exactly right for unmount and React 18 Strict Mode's double-mount.""",
    },
]


if __name__ == "__main__":
    print(f"{len(examples)} React training examples")
    for i, ex in enumerate(examples, 1):
        assert "prompt" in ex and "answer" in ex, f"example {i} malformed"
        assert "<thinking>" in ex["answer"], f"example {i} missing <thinking>"
        assert "```" in ex["answer"], f"example {i} missing code block"
    print("All examples validated (prompt, answer, <thinking>, code block present).")
