#!/usr/bin/env python3
"""
Dataset 2: ~300 high-quality frontend training examples for Qwen3-8B distillation.
8 categories: React Advanced, CSS/Animations, TypeScript, JavaScript Advanced,
Build Tools/Perf, Web APIs, Frontend Architecture, Frontend Testing.
Non-overlapping with dataset 1 (train.jsonl / val.jsonl).
Run: python generate_dataset2.py
Outputs: data/train2.jsonl, data/val2.jsonl
"""
import json
import random
from pathlib import Path

SYSTEM = """You are an expert frontend engineer and senior software architect with deep knowledge of React, TypeScript, JavaScript, CSS animations, web performance, and modern browser APIs.
Think through problems step by step inside <thinking>...</thinking> tags, showing your reasoning, tradeoffs, and approach.
Then give your final answer with clean, production-quality code and precise explanations.
Be thorough, technically accurate, and explain subtle nuances that separate good code from great code."""

EXAMPLES = []

# ─────────────────────────────────────────────
# CATEGORY 1: REACT ADVANCED
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "What are React Server Components and how do I decide which components should be Server Components vs Client Components in Next.js App Router?",
        "thinking": "This is a foundational App Router concept. RSCs run only on the server — no JS shipped to client, can await data directly, can't use hooks or browser APIs. Client Components run on both server (for SSR) and client. The key decision rule: if it needs interactivity (onClick, useState, useEffect, browser APIs) it must be a Client Component. Otherwise prefer Server. The common mistake is making entire trees client-side unnecessarily. I should explain the mental model, the 'use client' boundary, and how to push client boundaries as low as possible.",
        "answer": """**The mental model**

Server Components (SC) and Client Components (CC) are not about *where* they render — both render on the server for the initial HTML. The real difference is:
- **SC**: runs only on server. Zero JS bundle cost. Can `await` databases/APIs directly. Cannot use hooks, event handlers, or browser APIs.
- **CC**: runs on server for SSR *and* ships JS to the client for interactivity. Marked with `'use client'`.

**Decision rule**

| Need | Component type |
|---|---|
| `useState`, `useEffect`, hooks | Client |
| `onClick`, `onChange`, event handlers | Client |
| `localStorage`, `window`, browser APIs | Client |
| `async/await` data fetching | Server |
| Access to environment secrets | Server |
| Reduce bundle size | Prefer Server |
| Static markup, no interactivity | Server |

**Push `'use client'` as low as possible**

```tsx
// ❌ Makes the whole page a Client Component
'use client'
export default async function Page() {
  const data = await fetchData(); // async forbidden in CC!
  return <main><InteractiveChart data={data} /></main>;
}

// ✅ SC fetches data, CC only wraps the interactive part
// app/page.tsx  (Server Component — no directive)
export default async function Page() {
  const data = await fetchData(); // fine — SC
  return (
    <main>
      <StaticHeader />           {/* SC — stays server */}
      <InteractiveChart data={data} /> {/* CC — gets 'use client' */}
    </main>
  );
}

// components/InteractiveChart.tsx
'use client'
import { useState } from 'react';
export function InteractiveChart({ data }: { data: ChartData }) {
  const [filter, setFilter] = useState('all');
  // ...
}
```

**SCs can pass SCs as children to CCs**

```tsx
// ✅ This works — children is serialized as a prop
'use client'
export function Modal({ children }: { children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return open ? <dialog>{children}</dialog> : null;
}

// Server Component passes a Server Component as children
<Modal>
  <ServerRenderedContent />  {/* stays SC! */}
</Modal>
```

**What can't cross the SC→CC boundary**: non-serializable values (functions, class instances, Date objects raw). Only JSON-serializable props can be passed from SC to CC.

**Practical rule**: start everything as SC, add `'use client'` only when the compiler or runtime tells you you need a browser API or hook. Your bundle will be dramatically smaller."""
    },
    {
        "question": "How do I implement a proper Error Boundary in React that supports retry, shows a fallback UI, and logs errors to a monitoring service?",
        "thinking": "Error Boundaries must be class components — there's no hook equivalent that catches render errors. The key lifecycle is getDerivedStateFromError (for rendering fallback) and componentDidCatch (for side effects like logging). I need to show: the class component, a retry mechanism (resetting state), passing through to Sentry/equivalent, and a clean usage pattern. Also worth noting react-error-boundary package which wraps this nicely.",
        "answer": """**Class-based Error Boundary with retry and logging**

```tsx
import { Component, ErrorInfo, ReactNode } from 'react';

interface Props {
  children: ReactNode;
  fallback?: (error: Error, reset: () => void) => ReactNode;
  onError?: (error: Error, info: ErrorInfo) => void;
}

interface State {
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { error: null };

  static getDerivedStateFromError(error: Error): State {
    return { error };
  }

  componentDidCatch(error: Error, info: ErrorInfo) {
    // Log to monitoring (Sentry, Datadog, etc.)
    this.props.onError?.(error, info);
    console.error('ErrorBoundary caught:', error, info.componentStack);
  }

  reset = () => this.setState({ error: null });

  render() {
    if (this.state.error) {
      return this.props.fallback
        ? this.props.fallback(this.state.error, this.reset)
        : <DefaultFallback error={this.state.error} onRetry={this.reset} />;
    }
    return this.props.children;
  }
}

function DefaultFallback({ error, onRetry }: { error: Error; onRetry: () => void }) {
  return (
    <div role="alert" style={{ padding: 24, textAlign: 'center' }}>
      <h2>Something went wrong</h2>
      <pre style={{ color: 'red', fontSize: 12 }}>{error.message}</pre>
      <button onClick={onRetry}>Try again</button>
    </div>
  );
}
```

**Usage with Sentry**

```tsx
import * as Sentry from '@sentry/react';

<ErrorBoundary
  onError={(error, info) => {
    Sentry.captureException(error, {
      extra: { componentStack: info.componentStack },
    });
  }}
  fallback={(error, reset) => (
    <div>
      <p>Failed to load dashboard. Error ID logged.</p>
      <button onClick={reset}>Retry</button>
    </div>
  )}
>
  <Dashboard />
</ErrorBoundary>
```

**Using react-error-boundary (recommended shortcut)**

```tsx
import { ErrorBoundary, useErrorBoundary } from 'react-error-boundary';

// Throw from within a component to trigger the nearest boundary
function DataLoader() {
  const { showBoundary } = useErrorBoundary();
  useEffect(() => {
    fetchData().catch(showBoundary); // promotes async errors into boundary
  }, []);
}
```

**Critical caveat**: Error Boundaries only catch errors in the *render* phase and lifecycle methods. They do NOT catch:
- Errors in event handlers (use try/catch)
- Async errors in useEffect (use `.catch(showBoundary)`)
- Errors in the boundary itself

Wrap each major page section with its own boundary so one broken widget doesn't take down the whole page."""
    },
    {
        "question": "Implement a compound component pattern in TypeScript for a fully accessible `<Select>` dropdown — the parent manages state, children are `<Select.Option>` and `<Select.Trigger>`.",
        "thinking": "Compound components use React Context to share state between a parent and its co-designed children. I need: a context for the open state and selected value, the parent Select that provides context, Select.Trigger that opens/closes, Select.Option that reads/writes the value. Accessibility is critical: role='listbox', role='option', aria-selected, keyboard navigation. TypeScript: generic value type, namespace pattern for Select.Trigger etc.",
        "answer": """```tsx
import {
  createContext, useContext, useState, useRef,
  useEffect, KeyboardEvent, ReactNode
} from 'react';

interface SelectCtx<T> {
  value: T;
  open: boolean;
  select: (v: T) => void;
  toggle: () => void;
  close: () => void;
}

const SelectContext = createContext<SelectCtx<unknown> | null>(null);

function useSelectCtx<T>() {
  const ctx = useContext(SelectContext);
  if (!ctx) throw new Error('Must be used inside <Select>');
  return ctx as SelectCtx<T>;
}

// ── Parent ───────────────────────────────────────
interface SelectProps<T> {
  value: T;
  onChange: (v: T) => void;
  children: ReactNode;
}

function Select<T>({ value, onChange, children }: SelectProps<T>) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Close on outside click
  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!ref.current?.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', handler);
    return () => document.removeEventListener('mousedown', handler);
  }, []);

  return (
    <SelectContext.Provider value={{
      value,
      open,
      select: (v) => { onChange(v as T); setOpen(false); },
      toggle: () => setOpen(o => !o),
      close: () => setOpen(false),
    }}>
      <div ref={ref} style={{ position: 'relative', display: 'inline-block' }}>
        {children}
      </div>
    </SelectContext.Provider>
  );
}

// ── Trigger ──────────────────────────────────────
Select.Trigger = function Trigger({ children }: { children: ReactNode }) {
  const { toggle, open } = useSelectCtx();
  return (
    <button
      type="button"
      aria-haspopup="listbox"
      aria-expanded={open}
      onClick={toggle}
      onKeyDown={(e: KeyboardEvent) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); toggle(); }
      }}
    >
      {children}
    </button>
  );
};

// ── Options container ─────────────────────────────
Select.List = function List({ children }: { children: ReactNode }) {
  const { open } = useSelectCtx();
  if (!open) return null;
  return (
    <ul role="listbox" style={{
      position: 'absolute', top: '100%', left: 0,
      background: '#fff', border: '1px solid #ccc',
      listStyle: 'none', margin: 0, padding: 4, minWidth: 120, zIndex: 10,
    }}>
      {children}
    </ul>
  );
};

// ── Option ───────────────────────────────────────
interface OptionProps<T> { value: T; children: ReactNode; }
Select.Option = function Option<T>({ value, children }: OptionProps<T>) {
  const ctx = useSelectCtx<T>();
  const selected = ctx.value === value;
  return (
    <li
      role="option"
      aria-selected={selected}
      onClick={() => ctx.select(value)}
      style={{ padding: '6px 12px', cursor: 'pointer',
               background: selected ? '#e0f0ff' : 'transparent' }}
    >
      {children}
    </li>
  );
};

// ── Usage ────────────────────────────────────────
function App() {
  const [fruit, setFruit] = useState('apple');
  return (
    <Select value={fruit} onChange={setFruit}>
      <Select.Trigger>{fruit}</Select.Trigger>
      <Select.List>
        <Select.Option value="apple">Apple</Select.Option>
        <Select.Option value="banana">Banana</Select.Option>
        <Select.Option value="cherry">Cherry</Select.Option>
      </Select.List>
    </Select>
  );
}
```

The context is the glue — Trigger and Option don't need explicit props passed from the parent; they read from context. The generic `<T>` means this works for `string`, `number`, or any object value."""
    },
    {
        "question": "I have a React context that holds a large object (user, theme, cart). Any update to it re-renders every consumer even if they only care about one field. How do I fix this without replacing Context?",
        "thinking": "This is the classic Context performance problem. Solutions in order of complexity: (1) split the context into smaller ones, (2) use a selector pattern by wrapping context + useSyncExternalStore or using a library like use-context-selector, (3) memoize children with React.memo or useMemo. The cleanest built-in solution for large apps is splitting contexts. I'll show splitting first, then the useSyncExternalStore selector pattern for when splitting isn't practical.",
        "answer": """**Root cause**: `createContext` + `useContext` has no built-in selector support — any value change notifies every consumer.

**Solution 1: Split the context (recommended for most apps)**

```tsx
// ❌ One fat context — everything re-renders on cart update
const AppContext = createContext({ user, theme, cart });

// ✅ Three focused contexts
const UserContext  = createContext<User | null>(null);
const ThemeContext = createContext<Theme>('light');
const CartContext  = createContext<CartState>({ items: [] });

// Components subscribe only to what they need
function Avatar() {
  const user = useContext(UserContext); // only re-renders on user change
}
function CartBadge() {
  const cart = useContext(CartContext); // only re-renders on cart change
}
```

**Solution 2: Selector pattern with useSyncExternalStore**

When splitting isn't practical (context is provided by a library, etc.):

```tsx
import { useRef, useSyncExternalStore, createContext, useContext } from 'react';

function createStoreContext<T>(initialState: T) {
  type Listener = () => void;
  const Context = createContext<{
    get: () => T;
    set: (partial: Partial<T>) => void;
    subscribe: (l: Listener) => () => void;
  } | null>(null);

  function Provider({ children }: { children: React.ReactNode }) {
    const storeRef = useRef(initialState);
    const listeners = useRef(new Set<Listener>());

    const api = useRef({
      get: () => storeRef.current,
      set: (partial: Partial<T>) => {
        storeRef.current = { ...storeRef.current, ...partial };
        listeners.current.forEach(l => l());
      },
      subscribe: (l: Listener) => {
        listeners.current.add(l);
        return () => listeners.current.delete(l);
      },
    });

    return <Context.Provider value={api.current}>{children}</Context.Provider>;
  }

  function useStore<S>(selector: (state: T) => S): S {
    const store = useContext(Context)!;
    return useSyncExternalStore(
      store.subscribe,
      () => selector(store.get()),
    );
  }

  return { Provider, useStore };
}

// Usage
const { Provider, useStore } = createStoreContext({ user: null, theme: 'light', cart: [] });

function CartBadge() {
  const cartCount = useStore(s => s.cart.length); // only re-renders when cart.length changes
  return <span>{cartCount}</span>;
}
```

**Solution 3: memo the children (quick fix)**

```tsx
const MemoizedExpensiveChild = React.memo(ExpensiveChild);
// Works only if the child's props don't change when irrelevant context updates
```

**Recommendation**: split first (simple, zero overhead), reach for the store pattern only when the context is shared across 3+ domains and splitting creates too many providers."""
    },
    {
        "question": "Build a polymorphic `<Box>` component in TypeScript that can render as any HTML element or component, with full prop type safety — `<Box as='button'>` gets button props, `<Box as={Link}>` gets Link props.",
        "thinking": "Polymorphic components are a TypeScript challenge. I need the `as` prop to determine which element/component to render and merge that element's native props with my own. The key is the AsProp + PolymorphicProps pattern using generics. `React.ComponentPropsWithoutRef<C>` gives me the native props for element C. I need to exclude my own prop names from the native props to avoid conflicts, and ensure forwardRef works too.",
        "answer": """```tsx
import { ElementType, ComponentPropsWithoutRef, forwardRef, ForwardedRef } from 'react';

// Own props that Box always accepts
type BoxOwnProps<E extends ElementType> = {
  as?: E;
  children?: React.ReactNode;
};

// Merge own props with native element props, own props win on conflicts
type BoxProps<E extends ElementType> = BoxOwnProps<E> &
  Omit<ComponentPropsWithoutRef<E>, keyof BoxOwnProps<E>>;

// Allows ref type to track the element
type PolymorphicRef<E extends ElementType> =
  React.ComponentPropsWithRef<E>['ref'];

// Full signature with forwardRef support
type BoxComponent = <E extends ElementType = 'div'>(
  props: BoxProps<E> & { ref?: PolymorphicRef<E> }
) => React.ReactElement | null;

const Box: BoxComponent = forwardRef(
  <E extends ElementType = 'div'>(
    { as, children, ...rest }: BoxProps<E>,
    ref: ForwardedRef<unknown>
  ) => {
    const Component = as ?? 'div';
    return <Component ref={ref} {...rest}>{children}</Component>;
  }
) as BoxComponent;

// ── Usage (all fully type-checked) ───────────────
<Box>Default div</Box>
<Box as="button" onClick={() => {}} type="submit">Button</Box>
<Box as="a" href="https://example.com" target="_blank">Link</Box>
<Box as="input" type="email" required placeholder="Email" />

// With a component — gets that component's props
import { Link } from 'react-router-dom';
<Box as={Link} to="/home">Router Link</Box>

// ❌ Type error — 'href' is not a button prop
<Box as="button" href="/bad" />
```

**Why `Omit<ComponentPropsWithoutRef<E>, keyof BoxOwnProps<E>>`**: without the Omit, if `as` is both your prop and an HTML attribute (it's not, but `className`, `style` etc. are in both), TypeScript throws union conflicts. The Omit ensures your props always win.

**Practical extension** — add variant props:

```tsx
type BoxProps<E extends ElementType> = BoxOwnProps<E> &
  Omit<ComponentPropsWithoutRef<E>, keyof BoxOwnProps<E>> & {
    padding?: 'sm' | 'md' | 'lg';
    radius?: 'none' | 'sm' | 'full';
  };
```

This pattern is used internally by Radix UI, Chakra UI, and Mantine for their base primitive components."""
    },
    {
        "question": "Implement a `useKeyboardShortcut` hook that listens for keyboard combos (e.g. `Cmd+K`, `Ctrl+Shift+P`) and fires a callback. It must handle cleanup, ignore shortcuts when focused inside inputs, and support dynamic enable/disable.",
        "thinking": "I need to parse modifier keys (meta, ctrl, shift, alt) plus a key, attach a keydown listener to document, and clean it up on unmount. Ignoring inputs is important — check e.target for input/textarea/select/contenteditable. The hook needs to be stable (not re-register the listener on every render) so I'll use useRef for the callback to avoid stale closures. Dynamic enable/disable means a boolean param that conditionally adds/removes the listener.",
        "answer": """```ts
import { useEffect, useRef } from 'react';

interface ShortcutOptions {
  enabled?: boolean;
  ignoreInputs?: boolean; // default true
}

type Modifier = 'meta' | 'ctrl' | 'shift' | 'alt';

interface Shortcut {
  key: string;          // e.g. 'k', 'p', 'Escape'
  modifiers?: Modifier[];
}

function parseShortcut(combo: string): Shortcut {
  const parts = combo.toLowerCase().split('+');
  const key = parts[parts.length - 1];
  const modifiers = parts.slice(0, -1) as Modifier[];
  return { key, modifiers };
}

function isInputTarget(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  const tag = target.tagName;
  return (
    tag === 'INPUT' ||
    tag === 'TEXTAREA' ||
    tag === 'SELECT' ||
    target.isContentEditable
  );
}

export function useKeyboardShortcut(
  combo: string,            // e.g. 'meta+k' | 'ctrl+shift+p'
  callback: () => void,
  { enabled = true, ignoreInputs = true }: ShortcutOptions = {}
) {
  const callbackRef = useRef(callback);
  callbackRef.current = callback; // always fresh, no stale closure

  useEffect(() => {
    if (!enabled) return;

    const { key, modifiers = [] } = parseShortcut(combo);

    const handler = (e: KeyboardEvent) => {
      if (ignoreInputs && isInputTarget(e.target)) return;

      const match =
        e.key.toLowerCase() === key &&
        modifiers.every(mod => {
          if (mod === 'meta')  return e.metaKey;
          if (mod === 'ctrl')  return e.ctrlKey;
          if (mod === 'shift') return e.shiftKey;
          if (mod === 'alt')   return e.altKey;
          return false;
        }) &&
        // Ensure no extra modifiers are pressed
        (modifiers.includes('meta')  || !e.metaKey)  &&
        (modifiers.includes('ctrl')  || !e.ctrlKey)  &&
        (modifiers.includes('shift') || !e.shiftKey) &&
        (modifiers.includes('alt')   || !e.altKey);

      if (match) {
        e.preventDefault();
        callbackRef.current();
      }
    };

    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [combo, enabled, ignoreInputs]);
}

// ── Usage ──────────────────────────────────────
function CommandPalette() {
  const [open, setOpen] = useState(false);

  useKeyboardShortcut('meta+k', () => setOpen(o => !o));
  useKeyboardShortcut('Escape', () => setOpen(false), { enabled: open });

  return open ? <Palette onClose={() => setOpen(false)} /> : null;
}
```

**Why `useRef` for the callback**: putting `callback` in the `useEffect` dependency array would re-register the listener on every render (every time the parent re-renders). The ref pattern keeps the listener stable while always calling the latest version of the callback."""
    },
    {
        "question": "What is `useId` for and when do I actually need it? Show a concrete accessible form example where `Math.random()` or index-based IDs break things.",
        "thinking": "useId was added in React 18 specifically for generating stable IDs that are consistent between server and client, solving the hydration mismatch problem that Math.random() causes. The ID is also unique per component instance, solving the problem of using the same component twice on a page with hardcoded IDs. Key use cases: label-for-input association, aria-describedby, aria-labelledby, aria-controls.",
        "answer": """**The problem with naive IDs**

```tsx
// ❌ Math.random() — different on server vs client = hydration mismatch
function Field({ label }: { label: string }) {
  const id = `field-${Math.random()}`; // SSR generates X, client generates Y → React error
  return <>
    <label htmlFor={id}>{label}</label>
    <input id={id} />
  </>;
}

// ❌ Hardcoded — breaks when used twice on a page
function EmailField() {
  return <>
    <label htmlFor="email">Email</label> {/* duplicate id in DOM */}
    <input id="email" />
  </>;
}

// ❌ Index-based — fragile, breaks with reordering
{fields.map((f, i) => <Field key={i} id={`field-${i}`} />)}
```

**The fix: `useId`**

```tsx
import { useId } from 'react';

function FormField({
  label,
  hint,
  error,
  ...inputProps
}: {
  label: string;
  hint?: string;
  error?: string;
} & React.InputHTMLAttributes<HTMLInputElement>) {
  const id = useId();
  const hintId  = `${id}-hint`;
  const errorId = `${id}-error`;

  return (
    <div>
      <label htmlFor={id}>{label}</label>
      <input
        id={id}
        aria-describedby={[
          hint  ? hintId  : '',
          error ? errorId : '',
        ].filter(Boolean).join(' ') || undefined}
        aria-invalid={!!error}
        {...inputProps}
      />
      {hint  && <p id={hintId}  style={{ color: 'gray'}}>{hint}</p>}
      {error && <p id={errorId} role="alert" style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}

// Use the same component multiple times — IDs are automatically unique
<FormField label="Email"    type="email"    hint="We won't share your email" />
<FormField label="Username" type="text"     error="Username taken" />
<FormField label="Password" type="password" hint="Min 8 characters" />
```

Each instance gets a React-generated ID like `:r0:`, `:r1:`, `:r2:` that matches between SSR and hydration.

**Rule of thumb**: use `useId` whenever you need an ID to associate a label with a control (`htmlFor`/`id`), or to wire `aria-describedby`, `aria-labelledby`, or `aria-controls`. Never use it as a data ID (e.g. as a React `key` or a database record ID — those should come from your data)."""
    },
    {
        "question": "Show how to implement `useOptimistic` (React 19) for a like button — instant UI update, revert on failure, with the correct TypeScript types.",
        "thinking": "useOptimistic is a React 19 hook for optimistic UI. It takes the real state and an update function, returns [optimisticState, addOptimistic]. When you call addOptimistic, the UI immediately shows the optimistic value. Once the server action/async operation completes, React reverts to the real state (or keeps it if the server confirmed). I need to show the full pattern: the hook, the server action, the async update, and error reversion.",
        "answer": """```tsx
'use client';
import { useOptimistic, useTransition } from 'react';

interface Post {
  id: string;
  likes: number;
  likedByMe: boolean;
}

async function toggleLikeAction(postId: string, liked: boolean): Promise<void> {
  const res = await fetch(`/api/posts/${postId}/like`, {
    method: liked ? 'POST' : 'DELETE',
  });
  if (!res.ok) throw new Error('Failed to update like');
}

function LikeButton({ post }: { post: Post }) {
  const [isPending, startTransition] = useTransition();

  const [optimisticPost, updateOptimistic] = useOptimistic(
    post,
    // Updater: applies the optimistic change on top of current state
    (currentPost: Post, liked: boolean): Post => ({
      ...currentPost,
      likedByMe: liked,
      likes: liked ? currentPost.likes + 1 : currentPost.likes - 1,
    })
  );

  const handleClick = () => {
    const newLiked = !optimisticPost.likedByMe;

    startTransition(async () => {
      // 1. Apply optimistic update immediately
      updateOptimistic(newLiked);

      // 2. Hit the server
      try {
        await toggleLikeAction(post.id, newLiked);
        // Server confirmed — optimistic state is now the real state
      } catch {
        // React automatically reverts to `post` (the real state prop)
        // No manual rollback needed
      }
    });
  };

  return (
    <button
      onClick={handleClick}
      disabled={isPending}
      aria-pressed={optimisticPost.likedByMe}
      aria-label={`${optimisticPost.likedByMe ? 'Unlike' : 'Like'} post`}
    >
      {optimisticPost.likedByMe ? '❤️' : '🤍'} {optimisticPost.likes}
    </button>
  );
}
```

**Key mechanics**:
- `useOptimistic` is tied to a `useTransition` — it only "activates" during a transition.
- When the transition ends (async work done), React reconciles: if the real `post` prop hasn't changed, the optimistic value sticks; if the server invalidated/re-fetched and updated the prop, the real value wins.
- **Automatic rollback**: if you throw inside `startTransition`, React reverts `optimisticPost` to the current `post` — no need to track original values manually.
- `isPending` gives a loading indicator during the network call."""
    },
    {
        "question": "I need to code-split a heavy chart library (recharts, ~400KB) so it only loads when the chart is visible in the viewport, not on page load. Implement this with React.lazy + Suspense + IntersectionObserver.",
        "thinking": "This combines two techniques: lazy loading the component via React.lazy (so the chunk is only fetched when React needs to render it) and delaying that render until the element is visible via IntersectionObserver. The tricky part is that React.lazy triggers the import as soon as the component is rendered — so I need to control *when* I render the lazy component. A simple approach: use IntersectionObserver to set a boolean state, render a placeholder div until visible, then render the lazy chart (which triggers the import). Also need Suspense for the loading state.",
        "answer": """```tsx
import { lazy, Suspense, useRef, useState, useEffect } from 'react';

// Chunk only fetched when the component is actually rendered
const Chart = lazy(() => import('./HeavyChart'));

function useIntersected(rootMargin = '200px') {
  const ref = useRef<HTMLDivElement>(null);
  const [intersected, setIntersected] = useState(false);

  useEffect(() => {
    const el = ref.current;
    if (!el || intersected) return;

    const observer = new IntersectionObserver(
      ([entry]) => {
        if (entry.isIntersecting) {
          setIntersected(true);
          observer.disconnect(); // one-shot
        }
      },
      { rootMargin } // start loading 200px before it enters viewport
    );

    observer.observe(el);
    return () => observer.disconnect();
  }, [intersected, rootMargin]);

  return { ref, intersected };
}

function LazyChart({ data }: { data: ChartData }) {
  const { ref, intersected } = useIntersected('200px');

  return (
    <div ref={ref} style={{ minHeight: 300 }}>
      {intersected ? (
        <Suspense fallback={<ChartSkeleton />}>
          <Chart data={data} />
        </Suspense>
      ) : (
        <ChartSkeleton />
      )}
    </div>
  );
}

function ChartSkeleton() {
  return (
    <div
      style={{ height: 300, background: '#f0f0f0', borderRadius: 8,
               display: 'flex', alignItems: 'center', justifyContent: 'center' }}
      aria-busy="true"
      aria-label="Chart loading"
    >
      <span style={{ color: '#999' }}>Loading chart...</span>
    </div>
  );
}
```

**Why `minHeight` on the wrapper**: prevents CLS (Cumulative Layout Shift) — the page doesn't reflow when the chart loads because the placeholder has the same height.

**Why `rootMargin: '200px'`**: start fetching the chunk 200px before the chart enters the viewport so it's ready by the time the user scrolls to it. Adjust based on scroll speed and chunk size.

**Production tip**: tell your bundler to give the chunk a meaningful name:
```tsx
const Chart = lazy(() =>
  import(/* webpackChunkName: "chart", webpackPrefetch: true */ './HeavyChart')
);
```
`webpackPrefetch` adds a `<link rel="prefetch">` — the browser downloads the chunk during idle time, before the user even scrolls."""
    },
    {
        "question": "Write a `useFocusTrap` hook for modals and drawers — focus must stay inside a container, Escape closes it, and focus returns to the trigger element when closed.",
        "thinking": "Focus trapping is critical for modal accessibility. I need to: find all focusable elements in the container, intercept Tab/Shift+Tab to wrap around, listen for Escape to close, and on mount focus the first element (or a designated element). On unmount, return focus to whatever element had focus before the modal opened. This is a well-defined pattern but the devil is in the details: dynamic content, portals, and `inert` attribute on the rest of the page.",
        "answer": """```ts
import { useEffect, useRef } from 'react';

const FOCUSABLE =
  'a[href], button:not([disabled]), textarea, input, select, ' +
  '[tabindex]:not([tabindex="-1"]), details > summary';

export function useFocusTrap(
  containerRef: React.RefObject<HTMLElement>,
  active: boolean,
  onClose: () => void
) {
  const previousFocusRef = useRef<HTMLElement | null>(null);

  useEffect(() => {
    if (!active) return;

    // Save element to restore focus to later
    previousFocusRef.current = document.activeElement as HTMLElement;

    const container = containerRef.current;
    if (!container) return;

    const getFocusable = () =>
      Array.from(container.querySelectorAll<HTMLElement>(FOCUSABLE))
        .filter(el => !el.closest('[inert]'));

    // Focus first element
    const focusable = getFocusable();
    (focusable[0] ?? container).focus();

    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.preventDefault();
        onClose();
        return;
      }

      if (e.key !== 'Tab') return;

      const els = getFocusable();
      if (els.length === 0) { e.preventDefault(); return; }

      const first = els[0];
      const last  = els[els.length - 1];

      if (e.shiftKey) {
        if (document.activeElement === first) {
          e.preventDefault();
          last.focus();
        }
      } else {
        if (document.activeElement === last) {
          e.preventDefault();
          first.focus();
        }
      }
    };

    document.addEventListener('keydown', handleKeyDown);

    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      // Restore focus when trap deactivates
      previousFocusRef.current?.focus();
    };
  }, [active, containerRef, onClose]);
}

// ── Usage ──────────────────────────────────────────
function Modal({ open, onClose, children }: {
  open: boolean; onClose: () => void; children: React.ReactNode;
}) {
  const containerRef = useRef<HTMLDivElement>(null);
  useFocusTrap(containerRef, open, onClose);

  if (!open) return null;

  return createPortal(
    <div role="dialog" aria-modal="true" style={overlay}>
      <div ref={containerRef} style={panel}>
        <button onClick={onClose} aria-label="Close">✕</button>
        {children}
      </div>
    </div>,
    document.body
  );
}
```

**Bonus: use the `inert` attribute** to prevent screen readers from reaching content outside the modal (modern browsers):

```tsx
useEffect(() => {
  if (!active) return;
  // Mark everything except the modal as inert
  document.querySelectorAll('body > *:not([data-modal])').forEach(el => {
    el.setAttribute('inert', '');
  });
  return () => {
    document.querySelectorAll('[inert]').forEach(el => el.removeAttribute('inert'));
  };
}, [active]);
```

`inert` is now baseline across Chrome/Firefox/Safari and is the correct semantic signal to AT that the content is non-interactive."""
    },
    {
        "question": "Implement `React.memo` correctly for a `<Row>` component inside a large list. Show why a naive object/array prop breaks memoization and how to fix it with useMemo and useCallback.",
        "thinking": "React.memo does a shallow comparison by default. If a parent passes a new object/array literal or inline function on every render, the comparison always fails and memo is useless. The fix is to stabilize references: useMemo for objects/arrays, useCallback for functions. I should show the broken pattern, the symptoms, and the correct fix.",
        "answer": """**Why your memo is broken**

```tsx
// ❌ Broken — new object and function created every parent render
function Table({ rows }: { rows: Row[] }) {
  return rows.map(row => (
    <MemoRow
      key={row.id}
      row={row}
      style={{ color: 'blue' }}       // new object every render ❌
      onSelect={() => select(row.id)} // new function every render ❌
    />
  ));
}
const MemoRow = React.memo(Row); // shallow-compare always fails
```

**The fix**

```tsx
function Table({ rows }: { rows: Row[] }) {
  const rowStyle = useMemo(() => ({ color: 'blue' }), []); // stable reference

  const handleSelect = useCallback((id: string) => {
    select(id);
  }, []); // stable unless `select` changes

  return rows.map(row => (
    <MemoRow key={row.id} row={row} style={rowStyle} onSelect={handleSelect} />
  ));
}

// Custom comparator: only re-render when meaningful data changes
const MemoRow = React.memo(Row, (prev, next) =>
  prev.row.id === next.row.id && prev.row.data === next.row.data
);
```

**React DevTools Profiler → "Why did this render?"** is the tool to confirm memo is working. If it says "Props changed", log `prev` vs `next` to find the unstable reference.

**Honest caveat**: for a 50-row list, memo is premature. Profile first. Reach for `react-window` (windowing) before extensive memoization — windowing removes DOM nodes entirely rather than just skipping re-renders."""
    },
    {
        "question": "How do React 18's automatic batching and `flushSync` work, and when would I actually need `flushSync`?",
        "thinking": "React 18 batches all state updates by default — including in setTimeout, Promises, and native event handlers. Before React 18, only React event handlers were batched. flushSync forces synchronous flushing, bypassing batching. Real use cases: measuring DOM after state update, autoscroll after adding items, View Transitions API.",
        "answer": """**React 17 (old batching)** — only React event handlers were batched:

```tsx
setTimeout(() => {
  setA(1); // triggers render
  setB(2); // triggers another render → 2 renders total ❌
}, 0);
```

**React 18 (automatic batching)** — everything batched:

```tsx
setTimeout(() => {
  setA(1);
  setB(2); // → 1 render total ✅
}, 0);
```

**When `flushSync` is needed**

```tsx
import { flushSync } from 'react-dom';

// Autoscroll: DOM must be updated before we measure/scroll
function addMessage(msg: Message) {
  flushSync(() => {
    setMessages(prev => [...prev, msg]); // DOM updated synchronously after this
  });
  bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); // sees new DOM
}

// View Transitions API needs React to finish before the transition
document.startViewTransition(() => {
  flushSync(() => setPage('about'));
});
```

**Warning**: `flushSync` is expensive — it blocks the browser with a synchronous render. Never use it inside a render or another `flushSync`. Use it only when the DOM must be measured or observed immediately after a state update."""
    },
    {
        "question": "Build a `useInfiniteScroll` hook with IntersectionObserver + React Query that appends the next page when a sentinel element enters the viewport.",
        "thinking": "Combine IntersectionObserver for scroll detection and useInfiniteQuery for paginated data. Observe a sentinel div at the bottom of the list, fire fetchNextPage when it's intersecting and hasNextPage is true. Guard against duplicate calls while isFetchingNextPage is true.",
        "answer": """```tsx
import { useInfiniteQuery } from '@tanstack/react-query';
import { useEffect, useRef } from 'react';

function useInfiniteScrollSentinel(onIntersect: () => void, enabled = true) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = ref.current;
    if (!el || !enabled) return;
    const observer = new IntersectionObserver(
      ([entry]) => { if (entry.isIntersecting) onIntersect(); },
      { rootMargin: '200px' }
    );
    observer.observe(el);
    return () => observer.disconnect();
  }, [onIntersect, enabled]);
  return ref;
}

async function fetchPosts({ pageParam = 0 }: { pageParam?: number }) {
  const res = await fetch(`/api/posts?cursor=${pageParam}&limit=20`);
  if (!res.ok) throw new Error('Failed');
  return res.json() as Promise<{ posts: Post[]; nextCursor: number | null }>;
}

export function PostFeed() {
  const { data, fetchNextPage, hasNextPage, isFetchingNextPage, status } =
    useInfiniteQuery({
      queryKey: ['posts'],
      queryFn: fetchPosts,
      getNextPageParam: (last) => last.nextCursor ?? undefined,
      initialPageParam: 0,
    });

  const sentinelRef = useInfiniteScrollSentinel(
    () => { if (hasNextPage && !isFetchingNextPage) fetchNextPage(); },
    hasNextPage && !isFetchingNextPage
  );

  if (status === 'pending') return <Spinner />;
  if (status === 'error')   return <p>Error</p>;

  const posts = data.pages.flatMap(p => p.posts);
  return (
    <div>
      {posts.map(post => <PostCard key={post.id} post={post} />)}
      <div ref={sentinelRef} style={{ height: 1 }} aria-hidden />
      {isFetchingNextPage && <Spinner />}
      {!hasNextPage && <p style={{ textAlign: 'center' }}>All posts loaded</p>}
    </div>
  );
}
```

`rootMargin: '200px'` starts fetching 200px before the user hits the bottom. Disabling the observer while `isFetchingNextPage` prevents duplicate fetches as the sentinel oscillates in/out of the expanded margin."""
    },
    {
        "question": "What is `useDeferredValue` and how is it different from debouncing? Show how it keeps filtering a 10,000-item list responsive.",
        "thinking": "useDeferredValue tells React to show a stale version of a value while a higher-priority update (the input) renders first. Unlike debouncing (artificial delay), it uses the concurrent scheduler to yield to urgent updates. The deferred value shows stale UI immediately — no blank/spinner — while the expensive re-render catches up in the background.",
        "answer": """```tsx
import { useState, useDeferredValue, useMemo } from 'react';

function FastSearch({ items }: { items: Item[] }) {
  const [query, setQuery] = useState('');
  const deferredQuery = useDeferredValue(query);

  const isStale = query !== deferredQuery;

  // Only re-runs at deferred (lower) priority
  const filtered = useMemo(
    () => items.filter(i => i.name.toLowerCase().includes(deferredQuery.toLowerCase())),
    [items, deferredQuery]
  );

  return (
    <>
      <input value={query} onChange={e => setQuery(e.target.value)} />
      {/* Input stays responsive; list shows stale results while catching up */}
      <div style={{ opacity: isStale ? 0.6 : 1, transition: 'opacity 0.15s' }}>
        <ItemList items={filtered} />
      </div>
    </>
  );
}
```

**`useDeferredValue` vs debouncing**

| | `useDeferredValue` | Debouncing |
|---|---|---|
| Delay | Dynamic (CPU-driven) | Fixed (e.g. 300ms) |
| Fast machine | Updates immediately | Always waits 300ms |
| Slow machine | Yields automatically | May still block |
| Stale UI | Shows old results | Shows nothing |
| Network calls | Not suited for this | Ideal for this |

Use `useDeferredValue` for CPU-bound rendering work. Use debouncing for network requests where you want to limit API calls."""
    },
    {
        "question": "Implement a multi-step form with `useReducer` in TypeScript using discriminated union actions, per-step validation, and back navigation that preserves entered data.",
        "thinking": "Multi-step forms are a classic useReducer use case. State tracks current step + all field values. Actions are a discriminated union: SET_FIELD, NEXT, BACK, SUBMIT. Validation runs on NEXT/SUBMIT and populates an errors map. Back navigation just decrements step without clearing data.",
        "answer": """```tsx
import { useReducer } from 'react';

interface Fields { name: string; email: string; plan: string; card: string; }
interface FormState { step: 1|2|3; fields: Fields; errors: Partial<Record<keyof Fields, string>>; done: boolean; }
type Action =
  | { type: 'SET'; field: keyof Fields; value: string }
  | { type: 'NEXT' } | { type: 'BACK' } | { type: 'SUBMIT' };

function validate(step: number, f: Fields) {
  const e: FormState['errors'] = {};
  if (step === 1) {
    if (!f.name.trim()) e.name = 'Required';
    if (!f.email.includes('@')) e.email = 'Invalid email';
  }
  if (step === 2 && !f.plan) e.plan = 'Choose a plan';
  if (step === 3 && f.plan !== 'free' && f.card.length < 16) e.card = 'Invalid card';
  return e;
}

const INIT: FormState = {
  step: 1, fields: { name: '', email: '', plan: '', card: '' }, errors: {}, done: false
};

function reducer(s: FormState, a: Action): FormState {
  switch (a.type) {
    case 'SET':
      return { ...s, fields: { ...s.fields, [a.field]: a.value }, errors: { ...s.errors, [a.field]: undefined } };
    case 'NEXT': {
      const e = validate(s.step, s.fields);
      return Object.keys(e).length ? { ...s, errors: e } : { ...s, step: Math.min(s.step + 1, 3) as 1|2|3, errors: {} };
    }
    case 'BACK': return { ...s, step: Math.max(s.step - 1, 1) as 1|2|3, errors: {} };
    case 'SUBMIT': {
      const e = validate(s.step, s.fields);
      return Object.keys(e).length ? { ...s, errors: e } : { ...s, done: true };
    }
  }
}

export function MultiStepForm() {
  const [s, dispatch] = useReducer(reducer, INIT);
  if (s.done) return <h2>Done! Welcome, {s.fields.name}</h2>;

  const bind = (f: keyof Fields) => ({
    value: s.fields[f],
    onChange: (e: React.ChangeEvent<HTMLInputElement|HTMLSelectElement>) =>
      dispatch({ type: 'SET', field: f, value: e.target.value })
  });

  return (
    <form onSubmit={e => { e.preventDefault(); dispatch({ type: 'SUBMIT' }); }}>
      <p>Step {s.step}/3</p>
      {s.step === 1 && <>
        <input placeholder="Name" {...bind('name')} />
        {s.errors.name && <em>{s.errors.name}</em>}
        <input placeholder="Email" {...bind('email')} />
        {s.errors.email && <em>{s.errors.email}</em>}
      </>}
      {s.step === 2 && <>
        <select {...bind('plan')}>
          <option value="">Plan...</option>
          <option value="free">Free</option>
          <option value="pro">Pro $9/mo</option>
        </select>
        {s.errors.plan && <em>{s.errors.plan}</em>}
      </>}
      {s.step === 3 && s.fields.plan !== 'free' && <>
        <input placeholder="Card number" {...bind('card')} />
        {s.errors.card && <em>{s.errors.card}</em>}
      </>}
      <div style={{ display: 'flex', gap: 8 }}>
        {s.step > 1 && <button type="button" onClick={() => dispatch({ type: 'BACK' })}>Back</button>}
        {s.step < 3
          ? <button type="button" onClick={() => dispatch({ type: 'NEXT' })}>Next</button>
          : <button type="submit">Submit</button>}
      </div>
    </form>
  );
}
```

Going Back preserves all field values because they live in reducer state, not per-step local state. The discriminated union means TypeScript errors on unhandled action types."""
    },
    {
        "question": "How do I use `useImperativeHandle` correctly? Show a real use case with a custom video player component that exposes `play()`, `pause()`, and `seek(seconds)` to its parent.",
        "thinking": "useImperativeHandle lets you customize what ref.current exposes. Without it, forwardRef gives the parent the raw DOM node. With it, you pick exactly which methods to expose, keeping internal implementation details private. I should show the full pattern: forwardRef wrapper, useImperativeHandle with the video element internally, and the parent calling methods imperatively.",
        "answer": """```tsx
import { forwardRef, useImperativeHandle, useRef } from 'react';

// ── What the parent can call ──────────────────
export interface VideoPlayerHandle {
  play: () => void;
  pause: () => void;
  seek: (seconds: number) => void;
  getDuration: () => number;
}

interface VideoPlayerProps {
  src: string;
  autoPlay?: boolean;
  onEnded?: () => void;
}

// ── Component ─────────────────────────────────
export const VideoPlayer = forwardRef<VideoPlayerHandle, VideoPlayerProps>(
  ({ src, autoPlay, onEnded }, ref) => {
    const videoRef = useRef<HTMLVideoElement>(null);

    useImperativeHandle(ref, () => ({
      play:        () => videoRef.current?.play(),
      pause:       () => videoRef.current?.pause(),
      seek:        (s) => { if (videoRef.current) videoRef.current.currentTime = s; },
      getDuration: () => videoRef.current?.duration ?? 0,
    }), []); // empty deps — methods close over videoRef (stable)

    return (
      <video
        ref={videoRef}
        src={src}
        autoPlay={autoPlay}
        onEnded={onEnded}
        controls
        style={{ width: '100%' }}
      />
    );
  }
);

// ── Parent ────────────────────────────────────
function VideoPage() {
  const playerRef = useRef<VideoPlayerHandle>(null);

  return (
    <>
      <VideoPlayer ref={playerRef} src="/intro.mp4" />

      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={() => playerRef.current?.play()}>Play</button>
        <button onClick={() => playerRef.current?.pause()}>Pause</button>
        <button onClick={() => playerRef.current?.seek(30)}>Skip to 0:30</button>
        <button onClick={() => alert(playerRef.current?.getDuration() + 's')}>Duration</button>
      </div>
    </>
  );
}
```

**Why not just expose the `<video>` element directly?** The parent could do anything — call `load()`, change `src`, manipulate `controls`. `useImperativeHandle` makes the API explicit: the parent can only do what you expose. This is the same reason class component APIs expose methods rather than `this`.

**When NOT to use it**: most of the time. Prefer props + callbacks for communication. `useImperativeHandle` is for genuinely imperative actions — play/pause/focus/scroll — where you can't model the trigger as state."""
    },
    {
        "question": "Set up a global theme + dark mode system in React using CSS custom properties and `prefers-color-scheme`, with a toggle that persists to localStorage.",
        "thinking": "The right approach is CSS custom properties (variables) for theming — they're inherited, performant, and don't require JS for the default theme. The JS layer only handles: reading localStorage, listening to prefers-color-scheme, and updating a data-theme attribute on the root element. No React context needed for the colors themselves — CSS handles the cascade. I should show the CSS variables, the hook, and the toggle.",
        "answer": """```css
/* globals.css */
:root {
  --color-bg:      #ffffff;
  --color-surface: #f5f5f5;
  --color-text:    #111111;
  --color-primary: #0066cc;
  --color-border:  #e0e0e0;
}

[data-theme="dark"] {
  --color-bg:      #0f0f0f;
  --color-surface: #1a1a1a;
  --color-text:    #eeeeee;
  --color-primary: #4da6ff;
  --color-border:  #333333;
}

@media (prefers-color-scheme: dark) {
  :root:not([data-theme]) {   /* only if user hasn't overridden */
    --color-bg:      #0f0f0f;
    --color-surface: #1a1a1a;
    --color-text:    #eeeeee;
    --color-primary: #4da6ff;
    --color-border:  #333333;
  }
}
```

```tsx
// hooks/useTheme.ts
import { useEffect, useState } from 'react';

type Theme = 'light' | 'dark' | 'system';

function getEffectiveTheme(theme: Theme): 'light' | 'dark' {
  if (theme !== 'system') return theme;
  return window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
}

export function useTheme() {
  const [theme, setTheme] = useState<Theme>(() => {
    if (typeof window === 'undefined') return 'system';
    return (localStorage.getItem('theme') as Theme) ?? 'system';
  });

  useEffect(() => {
    const root = document.documentElement;
    const effective = getEffectiveTheme(theme);

    if (theme === 'system') {
      root.removeAttribute('data-theme'); // let CSS media query take over
    } else {
      root.setAttribute('data-theme', effective);
    }

    localStorage.setItem('theme', theme);
  }, [theme]);

  // React to system changes when theme is 'system'
  useEffect(() => {
    if (theme !== 'system') return;
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const handler = () => {
      document.documentElement.removeAttribute('data-theme'); // CSS takes over
    };
    mq.addEventListener('change', handler);
    return () => mq.removeEventListener('change', handler);
  }, [theme]);

  return { theme, setTheme };
}

// ThemeToggle.tsx
export function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <select value={theme} onChange={e => setTheme(e.target.value as Theme)}>
      <option value="system">System</option>
      <option value="light">Light</option>
      <option value="dark">Dark</option>
    </select>
  );
}
```

**No flash of wrong theme (FOUC)**: add this inline script in `<head>` before any CSS loads:
```html
<script>
  const t = localStorage.getItem('theme');
  if (t && t !== 'system') document.documentElement.setAttribute('data-theme', t);
</script>
```

Components use variables directly in their CSS: `background: var(--color-bg)` — no prop drilling, no context overhead."""
    },
    {
        "question": "Explain React 19's `use()` hook — what it does, how it differs from `await` in Server Components, and show a real Suspense-integrated data fetching example.",
        "thinking": "The `use()` hook in React 19 can unwrap Promises and Context. For Promises: it's like await but inside a Client Component — if the promise isn't resolved, it suspends the component (throws a promise, which Suspense catches). Unlike async/await, you can call use() conditionally (inside if/loops). For Context: `use(MyContext)` replaces `useContext(MyContext)`. I should show the Promise use case with Suspense and contrast with Server Component async/await.",
        "answer": """```tsx
// ── Server Component (React 18+ App Router) ───
// Can use async/await directly — no use() needed
async function UserPage({ id }: { id: string }) {
  const user = await fetchUser(id); // awaits on server, no Suspense needed
  return <UserProfile user={user} />;
}

// ── Client Component with use() ───────────────
'use client';
import { use, Suspense } from 'react';

// Promise created OUTSIDE the component (stable reference)
const userPromise = fetchUser('123');

function UserProfile() {
  // use() suspends the component until the promise resolves
  const user = use(userPromise); // ← no await, no async function
  return <h1>{user.name}</h1>;
}

function Page() {
  return (
    <Suspense fallback={<Spinner />}>
      <UserProfile />
    </Suspense>
  );
}

// ── Practical pattern: pass promise as prop ───
async function ServerShell() {
  const promise = fetchUser('123'); // starts fetching on server
  return (
    <Suspense fallback={<Skeleton />}>
      <ClientProfile userPromise={promise} />
    </Suspense>
  );
}

'use client';
function ClientProfile({ userPromise }: { userPromise: Promise<User> }) {
  const user = use(userPromise); // client reads the (already-in-flight) promise
  return <div>{user.name}</div>;
}

// ── use() with Context (replaces useContext) ──
const ThemeContext = createContext('light');
function Button() {
  // Can be called conditionally — unlike useContext
  if (someCondition) {
    const theme = use(ThemeContext);
    return <button data-theme={theme}>Click</button>;
  }
  return null;
}
```

**Key distinctions**:
- `use(promise)` — Client Component only, triggers Suspense. Works conditionally (inside `if`).
- `await fetch()` in `async function` — Server Component only.
- The promise must be **stable** (created outside the component or memoized) — creating it inside the component recreates it on every render, causing infinite suspension.
- On rejection, `use()` propagates to the nearest Error Boundary."""
    },
    {
        "question": "Write a `useWindowSize` hook and explain how to SSR it safely — no hydration mismatch, correct initial value on server.",
        "thinking": "On the server, window doesn't exist. So the hook must return a safe default during SSR. The trick is initializing state with undefined or a static value, then reading the real window size only after mount in useEffect. This matches: server renders with the safe default, client hydrates with the same safe default, then useEffect fires and updates to the real size — no mismatch. I should also show how to use this in a component without layout shifts.",
        "answer": """```tsx
import { useState, useEffect } from 'react';

interface WindowSize {
  width:  number | undefined;
  height: number | undefined;
}

export function useWindowSize(): WindowSize {
  // undefined = "not measured yet" — safe for SSR
  const [size, setSize] = useState<WindowSize>({ width: undefined, height: undefined });

  useEffect(() => {
    function update() {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    }

    update(); // set on mount
    window.addEventListener('resize', update);
    return () => window.removeEventListener('resize', update);
  }, []);

  return size;
}

// ── Usage ─────────────────────────────────────
function ResponsivePanel() {
  const { width } = useWindowSize();

  // width is undefined on server and first render — handle it
  if (width === undefined) return <LoadingPlaceholder />;

  return width < 768 ? <MobileView /> : <DesktopView />;
}
```

**Why undefined instead of 0 or a guess**: `0` would cause a layout flash — the component briefly renders as if width is 0 (mobile), then snaps to the real size. `undefined` lets you render a neutral placeholder (skeleton/spinner) until the real value is known.

**Debounce resize for performance**:
```tsx
useEffect(() => {
  let timer: ReturnType<typeof setTimeout>;
  function update() {
    clearTimeout(timer);
    timer = setTimeout(() => {
      setSize({ width: window.innerWidth, height: window.innerHeight });
    }, 100);
  }
  update();
  window.addEventListener('resize', update);
  return () => { clearTimeout(timer); window.removeEventListener('resize', update); };
}, []);
```

**Alternative for purely CSS-based responsiveness**: use CSS `@media` or CSS `@container` instead — they don't need JS at all and work during SSR without any special handling."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 2: CSS & ANIMATIONS
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "Implement a text reveal animation using GSAP SplitText where each word flies in from below with a stagger effect, triggered by ScrollTrigger when the element enters the viewport.",
        "thinking": "SplitText splits text into chars/words/lines that can be individually animated. The pattern: split on mount, set initial state (y offset, opacity 0), animate with stagger on ScrollTrigger. I need to handle cleanup (revert the split on unmount) and ensure the animation runs after the DOM is ready. Also important: handle dynamic content and window resize which breaks line splits.",
        "answer": """```tsx
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { SplitText } from 'gsap/SplitText';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(SplitText, ScrollTrigger);

interface RevealTextProps {
  children: string;
  tag?: keyof JSX.IntrinsicElements;
  delay?: number;
}

export function RevealText({ children, tag: Tag = 'h2', delay = 0 }: RevealTextProps) {
  const ref = useRef<HTMLElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const split = new SplitText(el, { type: 'words,chars' });

    // Hide words initially
    gsap.set(split.words, { yPercent: 110, opacity: 0 });

    const tween = gsap.to(split.words, {
      yPercent: 0,
      opacity: 1,
      duration: 0.8,
      ease: 'power3.out',
      stagger: 0.06,
      delay,
      scrollTrigger: {
        trigger: el,
        start: 'top 85%',  // fire when top of element is 85% down the viewport
        toggleActions: 'play none none reverse',
      },
    });

    return () => {
      tween.kill();
      split.revert(); // restore original DOM — critical for cleanup
      ScrollTrigger.getAll()
        .filter(st => st.vars.trigger === el)
        .forEach(st => st.kill());
    };
  }, [children, delay]);

  return (
    <Tag
      ref={ref as React.RefObject<any>}
      style={{ overflow: 'hidden', lineHeight: 1.2 }}
    >
      {children}
    </Tag>
  );
}

// Usage
<RevealText tag="h1">
  The quick brown fox jumps over the lazy dog
</RevealText>
<RevealText tag="p" delay={0.2}>
  Every word enters from below with a staggered cascade.
</RevealText>
```

**Why `overflow: hidden`**: each word starts at `yPercent: 110` (below the line). Without clipping, words are briefly visible below the text block before animating in.

**Window resize**: SplitText's line splits break when the element reflows. For line-based animations, re-split inside a ResizeObserver:
```tsx
const ro = new ResizeObserver(() => {
  split.revert();
  split = new SplitText(el, { type: 'lines' });
});
ro.observe(el);
```"""
    },
    {
        "question": "Build a scroll-driven card stack animation using GSAP ScrollTrigger where cards pin to the top and the next card slides in over the previous one as you scroll.",
        "thinking": "Card stacking scroll experiences pin a container with ScrollTrigger and animate each card's transform as scroll progresses. The technique: pin the section, create one ScrollTrigger per card that animates it into position (scaling previous card slightly and sliding new one from below). I need to handle the timeline correctly so each card occupies an equal section of the scrollable distance.",
        "answer": """```tsx
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

const CARDS = [
  { title: 'Card 1', bg: '#ff6b6b' },
  { title: 'Card 2', bg: '#4ecdc4' },
  { title: 'Card 3', bg: '#45b7d1' },
  { title: 'Card 4', bg: '#96ceb4' },
];

export function CardStack() {
  const sectionRef = useRef<HTMLDivElement>(null);
  const cardsRef   = useRef<HTMLDivElement[]>([]);

  useEffect(() => {
    const section = sectionRef.current;
    const cards   = cardsRef.current;
    if (!section || cards.length === 0) return;

    // Set initial positions: all cards stacked at center
    gsap.set(cards, { yPercent: 100, scale: 1, transformOrigin: 'top center' });
    gsap.set(cards[0], { yPercent: 0 }); // first card visible

    const tl = gsap.timeline({
      scrollTrigger: {
        trigger: section,
        pin: true,                   // pin the section
        start: 'top top',
        end: `+=${cards.length * 600}`, // 600px scroll per card
        scrub: 1,                    // smooth scrub
      },
    });

    cards.forEach((card, i) => {
      if (i === 0) return;
      tl
        .to(cards[i - 1], { scale: 0.9, filter: 'brightness(0.7)' }, `card${i}`)
        .to(card, { yPercent: 0, duration: 1 }, `card${i}`);
    });

    return () => ScrollTrigger.getAll().forEach(st => st.kill());
  }, []);

  return (
    <div
      ref={sectionRef}
      style={{ height: '100vh', display: 'flex', alignItems: 'center',
               justifyContent: 'center', position: 'relative' }}
    >
      {CARDS.map((card, i) => (
        <div
          key={i}
          ref={el => { if (el) cardsRef.current[i] = el; }}
          style={{
            position: 'absolute',
            width: 340, height: 460,
            background: card.bg,
            borderRadius: 20,
            display: 'flex', alignItems: 'center', justifyContent: 'center',
            boxShadow: '0 20px 60px rgba(0,0,0,0.3)',
            zIndex: i + 1,
          }}
        >
          <h2 style={{ color: '#fff', fontSize: 32 }}>{card.title}</h2>
        </div>
      ))}
    </div>
  );
}
```

**Key settings**:
- `pin: true` keeps the section fixed while you scroll `cards.length * 600px` of distance.
- `scrub: 1` ties animation progress directly to scroll position (1 = 1s lag for smoothness).
- `scale: 0.9 + brightness(0.7)` on previous cards creates depth without 3D transforms."""
    },
    {
        "question": "Implement a Framer Motion layout animation where items reorder in a list and the transition is smooth — no position jumping.",
        "thinking": "Framer Motion's `layout` prop auto-detects position changes between renders and animates between them using the FLIP technique (First, Last, Invert, Play). I need to show: adding `layout` to each item, using `layoutId` for shared element transitions, and wrapping in `AnimatePresence` for enter/exit animations during reorder. The key gotcha is that layout animations need each item to have a stable `key`.",
        "answer": """```tsx
import { motion, AnimatePresence } from 'framer-motion';
import { useState } from 'react';

interface Item { id: number; label: string; color: string; }

const INITIAL: Item[] = [
  { id: 1, label: 'Alpha',   color: '#ff6b6b' },
  { id: 2, label: 'Beta',    color: '#4ecdc4' },
  { id: 3, label: 'Gamma',   color: '#45b7d1' },
  { id: 4, label: 'Delta',   color: '#96ceb4' },
];

export function SortableList() {
  const [items, setItems] = useState(INITIAL);

  const shuffle = () =>
    setItems(prev => [...prev].sort(() => Math.random() - 0.5));

  const remove = (id: number) =>
    setItems(prev => prev.filter(i => i.id !== id));

  return (
    <div>
      <button onClick={shuffle}>Shuffle</button>
      <ul style={{ listStyle: 'none', padding: 0 }}>
        <AnimatePresence>
          {items.map(item => (
            <motion.li
              key={item.id}          // stable key is essential
              layout                 // animate position changes
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.8, transition: { duration: 0.2 } }}
              transition={{ type: 'spring', stiffness: 350, damping: 30 }}
              style={{
                background: item.color,
                borderRadius: 12,
                padding: '12px 20px',
                marginBottom: 8,
                display: 'flex',
                justifyContent: 'space-between',
                cursor: 'grab',
              }}
            >
              <span style={{ color: '#fff', fontWeight: 600 }}>{item.label}</span>
              <button
                onClick={() => remove(item.id)}
                style={{ background: 'none', border: 'none', color: '#fff', cursor: 'pointer' }}
              >
                ✕
              </button>
            </motion.li>
          ))}
        </AnimatePresence>
      </ul>
    </div>
  );
}
```

**How it works**: when `items` reorders, React re-renders with items in new DOM positions. Framer Motion reads each element's position *before* the re-render (First), reads the *after* position, calculates the delta (Invert), then plays a spring from old to new position. This is FLIP — it's purely transform-based so it's compositor-thread only, no layout thrashing.

**`layoutId`**: for shared element transitions between different components (e.g. clicking a card to expand it into a modal), use `layoutId` on both the card and the modal — Framer Motion animates between them automatically."""
    },
    {
        "question": "How do CSS scroll-driven animations work with `animation-timeline: scroll()` and `animation-timeline: view()`? Show a progress bar and a card entrance animation.",
        "thinking": "Scroll-driven animations are a native CSS feature (now baseline in Chrome/Edge, Firefox behind flag). Two timeline types: scroll() links to the scroll container's scroll position, view() links to an element's position in the viewport. The animation-range property controls when it starts/ends. This replaces a lot of JavaScript IntersectionObserver + GSAP ScrollTrigger for simple animations.",
        "answer": """```css
/* ── 1. Reading progress bar ──────────────────── */
@keyframes grow-bar {
  from { transform: scaleX(0); }
  to   { transform: scaleX(1); }
}

.progress-bar {
  position: fixed;
  top: 0; left: 0;
  width: 100%; height: 4px;
  background: #0066cc;
  transform-origin: left;

  animation: grow-bar linear;
  animation-timeline: scroll(root block); /* root = <html>, block = vertical */
  animation-fill-mode: both;
}

/* ── 2. Card entrance on scroll ───────────────── */
@keyframes slide-up {
  from {
    opacity: 0;
    transform: translateY(40px);
  }
  to {
    opacity: 1;
    transform: translateY(0);
  }
}

.card {
  animation: slide-up 0.6s ease both;
  animation-timeline: view();           /* timeline = element's visibility in viewport */
  animation-range: entry 0% entry 40%; /* play as element enters, finish at 40% entry */
}

/* ── 3. Parallax image ────────────────────────── */
@keyframes parallax {
  from { transform: translateY(-20%); }
  to   { transform: translateY(20%); }
}

.hero-image {
  animation: parallax linear both;
  animation-timeline: view();
  animation-range: contain;  /* while element is fully in viewport */
}
```

```tsx
// React component
export function ScrollDemo() {
  return (
    <>
      <div className="progress-bar" />

      <main style={{ padding: '0 20px' }}>
        {Array.from({ length: 8 }).map((_, i) => (
          <div key={i} className="card" style={{
            margin: '60px auto', maxWidth: 600,
            padding: 32, borderRadius: 16,
            background: `hsl(${i * 45}, 70%, 95%)`,
            boxShadow: '0 4px 24px rgba(0,0,0,0.1)',
          }}>
            <h2>Card {i + 1}</h2>
            <p>This card animates in as it enters the viewport.</p>
          </div>
        ))}
      </main>
    </>
  );
}
```

**Browser support** (2025): Chrome/Edge full support, Firefox behind `layout.css.scroll-driven-animations.enabled`. For broad support, use `@supports`:

```css
@supports (animation-timeline: scroll()) {
  .progress-bar { animation: grow-bar linear; animation-timeline: scroll(); }
}
```

Otherwise fall back to a GSAP ScrollTrigger implementation."""
    },
    {
        "question": "Create a magnetic button effect in React where the button subtly follows the mouse cursor when hovered, using GSAP for smooth lerp animation.",
        "thinking": "The magnetic effect uses mouse position relative to the button center to compute an offset, then applies it via transform. The smoothness comes from GSAP's quickTo (or gsap.to with low duration + overwrite). I need to track mousemove, compute delta from button center, apply a strength multiplier, and reset on mouseleave. Key: use getBoundingClientRect to get the center, not stored positions that go stale.",
        "answer": """```tsx
import { useRef, useEffect } from 'react';
import gsap from 'gsap';

interface MagneticButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  strength?: number; // how far the button moves (0–1)
  children: React.ReactNode;
}

export function MagneticButton({ strength = 0.4, children, style, ...props }: MagneticButtonProps) {
  const btnRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    const btn = btnRef.current;
    if (!btn) return;

    // quickTo creates an optimized setter with built-in lerp
    const xTo = gsap.quickTo(btn, 'x', { duration: 0.6, ease: 'power3.out' });
    const yTo = gsap.quickTo(btn, 'y', { duration: 0.6, ease: 'power3.out' });

    const handleMove = (e: MouseEvent) => {
      const rect = btn.getBoundingClientRect();
      const cx = rect.left + rect.width  / 2;
      const cy = rect.top  + rect.height / 2;
      xTo((e.clientX - cx) * strength);
      yTo((e.clientY - cy) * strength);
    };

    const handleLeave = () => {
      xTo(0);
      yTo(0);
    };

    btn.addEventListener('mousemove', handleMove);
    btn.addEventListener('mouseleave', handleLeave);
    return () => {
      btn.removeEventListener('mousemove', handleMove);
      btn.removeEventListener('mouseleave', handleLeave);
      gsap.set(btn, { x: 0, y: 0 });
    };
  }, [strength]);

  return (
    <button
      ref={btnRef}
      style={{
        display: 'inline-flex',
        alignItems: 'center',
        justifyContent: 'center',
        padding: '14px 32px',
        borderRadius: 999,
        border: '2px solid currentColor',
        fontSize: 16,
        fontWeight: 600,
        cursor: 'pointer',
        willChange: 'transform', // hint compositor
        ...style,
      }}
      {...props}
    >
      {children}
    </button>
  );
}

// Usage
<MagneticButton strength={0.5} onClick={() => {}}>
  Get Started
</MagneticButton>
```

**`gsap.quickTo`** is the key — it creates a function that updates a specific property with a built-in duration/ease, so calling it on every `mousemove` doesn't create a new tween each time (which would be expensive). It smoothly interpolates from current value to target.

**`willChange: transform`** tells the browser to promote the button to its own compositor layer ahead of time, so the transform updates don't trigger main-thread layout."""
    },
    {
        "question": "Implement a smooth page transition system in Next.js App Router using Framer Motion where each route fades and slides in, and the outgoing page fades out.",
        "thinking": "App Router uses React Server Components and doesn't support Framer Motion's AnimatePresence out of the box at the layout level. The solution is to wrap page content in a client component with AnimatePresence + motion.div. The key trick: use pathname as the key so AnimatePresence knows when to trigger exit/enter. I need to show the layout wrapper, the page template component, and how to apply it.",
        "answer": """```tsx
// app/template.tsx  ← Next.js template (re-mounts on each navigation)
// Use template.tsx not layout.tsx — layout persists, template re-mounts
'use client';
import { motion } from 'framer-motion';

const variants = {
  hidden:  { opacity: 0, y: 20 },
  visible: { opacity: 1, y: 0 },
  exit:    { opacity: 0, y: -20 },
};

export default function Template({ children }: { children: React.ReactNode }) {
  return (
    <motion.div
      variants={variants}
      initial="hidden"
      animate="visible"
      exit="exit"
      transition={{ duration: 0.3, ease: 'easeInOut' }}
    >
      {children}
    </motion.div>
  );
}

// That's it — Next.js calls template.tsx on every navigation,
// and Framer Motion handles the enter/exit animation.
```

**Why `template.tsx` not `layout.tsx`**: `layout.tsx` is *persistent* — it doesn't unmount between navigations, so Framer Motion never sees a component exit. `template.tsx` re-mounts on every navigation, giving Framer Motion a fresh component to animate in and out.

**Cross-fade between pages** (for a smoother feel):

```tsx
// Slower exit, faster enter — feels snappier
const variants = {
  hidden:  { opacity: 0, filter: 'blur(4px)' },
  visible: { opacity: 1, filter: 'blur(0px)', transition: { duration: 0.25 } },
  exit:    { opacity: 0, filter: 'blur(4px)', transition: { duration: 0.15 } },
};
```

**With View Transitions API (native, no library)**:
```tsx
// app/layout.tsx — enable native transitions
import { useRouter } from 'next/navigation';
// In Next.js 14+, add to next.config.js:
// experimental: { viewTransition: true }
// Then use CSS:
// ::view-transition-old(root) { animation: fade-out 0.2s; }
// ::view-transition-new(root) { animation: fade-in 0.2s; }
```"""
    },
    {
        "question": "Build a `<AnimatedCounter>` component in React that smoothly counts up to a target number using GSAP when it enters the viewport.",
        "thinking": "This combines IntersectionObserver to trigger on viewport entry and GSAP to animate a number from 0 to the target. The key is animating a plain object's value property (not a DOM transform) and updating the text content in the onUpdate callback. I also need to handle formatting (commas, decimals) and cleanup.",
        "answer": """```tsx
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

interface AnimatedCounterProps {
  target: number;
  duration?: number;
  prefix?: string;
  suffix?: string;
  decimals?: number;
}

export function AnimatedCounter({
  target,
  duration = 2,
  prefix = '',
  suffix = '',
  decimals = 0,
}: AnimatedCounterProps) {
  const ref = useRef<HTMLSpanElement>(null);

  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const counter = { value: 0 };

    const tween = gsap.to(counter, {
      value: target,
      duration,
      ease: 'power2.out',
      onUpdate() {
        el.textContent =
          prefix +
          counter.value.toLocaleString('en-US', {
            minimumFractionDigits: decimals,
            maximumFractionDigits: decimals,
          }) +
          suffix;
      },
      scrollTrigger: {
        trigger: el,
        start: 'top 90%',
        once: true, // only animate once
      },
    });

    return () => tween.kill();
  }, [target, duration, prefix, suffix, decimals]);

  return (
    <span ref={ref} aria-live="polite">
      {prefix}0{suffix}
    </span>
  );
}

// Usage
<AnimatedCounter target={12500} prefix="$" suffix="+" duration={2.5} />
<AnimatedCounter target={99.9} suffix="%" decimals={1} />
<AnimatedCounter target={1000000} duration={3} />
```

**Why animate a plain object, not a state variable**: calling `setState` on every GSAP tick (60fps) would cause 120 React re-renders per 2-second animation — very expensive. Writing directly to `el.textContent` bypasses React's render cycle entirely, which is correct here since we're driving a visual effect, not application state.

**`once: true`**: prevents the counter from resetting if the user scrolls back up past it."""
    },
    {
        "question": "Implement CSS `@layer` to organize styles with explicit specificity control — show how it solves the specificity wars problem in a component library.",
        "thinking": "CSS @layer gives you an explicit stack of specificity layers. Styles in a later layer always win over styles in an earlier layer, regardless of selector specificity. This solves the classic problem where a user's `.my-button` (1 class) loses to the library's `.btn.primary` (2 classes). With layers, you put library styles in a lower layer and user overrides always win. I should show the layer declaration order, library styles, and user overrides.",
        "answer": """```css
/* Declare layer order upfront — later layers win */
@layer reset, base, components, utilities, overrides;

/* ── Reset (lowest priority) ─────────────────── */
@layer reset {
  *, *::before, *::after { box-sizing: border-box; margin: 0; }
}

/* ── Component library styles ─────────────────── */
@layer components {
  /* Even with 3 classes, this loses to a 1-class rule in a higher layer */
  .btn.btn--primary.btn--large {
    background: #0066cc;
    color: white;
    padding: 12px 24px;
    border-radius: 6px;
    font-weight: 600;
  }
}

/* ── Utilities (higher priority than components) ─ */
@layer utilities {
  .mt-4  { margin-top:  16px; }
  .text-center { text-align: center; }
}

/* ── User overrides (highest, always win) ──────── */
@layer overrides {
  /* 1 class here beats any number of classes in @layer components */
  .my-custom-button {
    background: hotpink; /* ✅ wins over .btn.btn--primary.btn--large */
  }
}
```

```tsx
// In a React component library (e.g. design system)
// styles/base.css — loaded first
@layer components {
  .button { /* all button styles */ }
  .button.variant-primary { /* primary variant */ }
}

// Consumer app can easily override without specificity tricks:
@layer overrides {
  .button { border-radius: 0; } /* square buttons — always works */
}
```

**Before `@layer`**: to override `.btn.btn--primary` you needed `.my-page .my-section .btn.btn--primary` or `!important`. With layers, a single class in a higher layer always wins.

**Import layered CSS**:
```css
@import url('library.css') layer(components);
/* All library styles are now in 'components' layer — safely overridable */
```

**Order matters**: declare all layer names at the top with `@layer reset, base, components, utilities, overrides;` — this fixes the priority regardless of where the actual rules appear in the file."""
    },
    {
        "question": "Implement a clip-path morphing animation in CSS and GSAP where a shape transitions smoothly between a circle, a hexagon, and a star on hover and scroll.",
        "thinking": "clip-path morphing requires both shapes to have the same number of points — otherwise the browser interpolates awkwardly. For CSS transitions, use polygon() with the same point count. For complex shapes (star), I need to manually define all polygon points. GSAP MorphSVG plugin handles arbitrary path morphing more gracefully. I'll show both: CSS polygon morphing and GSAP approach.",
        "answer": """```css
/* ── CSS polygon morphing ────────────────────── */
/* All shapes must have the same number of points */

.morphing-shape {
  width: 200px;
  height: 200px;
  background: linear-gradient(135deg, #6366f1, #8b5cf6);
  transition: clip-path 0.6s cubic-bezier(0.25, 0.46, 0.45, 0.94);

  /* Default: circle (approximated with 8 polygon points) */
  clip-path: polygon(
    50% 0%, 85% 15%, 100% 50%, 85% 85%,
    50% 100%, 15% 85%, 0% 50%, 15% 15%
  );
}

.morphing-shape:hover {
  /* Square */
  clip-path: polygon(
    0% 0%, 100% 0%, 100% 0%, 100% 50%,
    100% 100%, 100% 100%, 0% 100%, 0% 50%
  );
}

.morphing-shape.star {
  /* Star — same 8 points as above */
  clip-path: polygon(
    50% 0%, 61% 35%, 98% 35%, 68% 57%,
    79% 91%, 50% 70%, 21% 91%, 32% 57%
    /* Note: true star needs many points — this is a simplified 4-point star */
  );
}
```

```tsx
// ── GSAP MorphSVG (arbitrary path morphing) ────
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { MorphSVGPlugin } from 'gsap/MorphSVGPlugin';

gsap.registerPlugin(MorphSVGPlugin);

export function MorphingShape() {
  const pathRef = useRef<SVGPathElement>(null);

  const shapes = {
    circle: 'M 100,50 A 50,50 0 1,1 99.9,50 Z',
    star:   'M100,10 L120,80 L190,80 L130,120 L150,190 L100,150 L50,190 L70,120 L10,80 L80,80 Z',
    hex:    'M100,20 L160,55 L160,125 L100,160 L40,125 L40,55 Z',
  };

  const morphTo = (shape: keyof typeof shapes) => {
    gsap.to(pathRef.current, {
      morphSVG: shapes[shape],
      duration: 0.8,
      ease: 'power2.inOut',
    });
  };

  return (
    <div>
      <svg viewBox="0 0 200 200" width="200" height="200">
        <path
          ref={pathRef}
          d={shapes.circle}
          fill="url(#grad)"
        />
        <defs>
          <linearGradient id="grad" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#8b5cf6" />
          </linearGradient>
        </defs>
      </svg>
      <div style={{ display: 'flex', gap: 8 }}>
        <button onClick={() => morphTo('circle')}>Circle</button>
        <button onClick={() => morphTo('star')}>Star</button>
        <button onClick={() => morphTo('hex')}>Hex</button>
      </div>
    </div>
  );
}
```

**MorphSVG advantage**: handles paths with different point counts — it automatically subdivides the simpler path to match the complex one. CSS `clip-path: polygon()` requires identical point counts or the morphing looks broken."""
    },
    {
        "question": "Implement a CSS container query-based card component that switches from a horizontal to vertical layout based on its container width, not the viewport.",
        "thinking": "Container queries let components respond to their parent's size rather than the viewport. This solves the 'same component in a wide sidebar vs narrow main column' problem — the component adapts to where it's placed, not to the overall screen size. I need to show containment setup, the container query syntax, and a practical component that works differently in different containers.",
        "answer": """```css
/* ── 1. Establish a containment context ───────── */
.card-container {
  container-type: inline-size;  /* enables width-based queries */
  container-name: card;         /* optional name for targeting */
}

/* ── 2. Default (narrow) layout ──────────────── */
.card {
  display: grid;
  grid-template-rows: auto 1fr auto;  /* image / content / footer */
  border-radius: 12px;
  overflow: hidden;
  box-shadow: 0 2px 12px rgba(0,0,0,0.1);
}

.card__image { aspect-ratio: 16/9; object-fit: cover; width: 100%; }
.card__body  { padding: 16px; }

/* ── 3. Wide layout (when container ≥ 480px) ─── */
@container card (min-width: 480px) {
  .card {
    grid-template-rows: 1fr;
    grid-template-columns: 220px 1fr;  /* horizontal: image left, content right */
  }

  .card__image {
    aspect-ratio: auto;
    height: 100%;
  }
}

/* ── 4. Extra wide layout ─────────────────────── */
@container card (min-width: 720px) {
  .card__body { padding: 24px 32px; }
  .card__title { font-size: 1.5rem; }
}
```

```tsx
// The same <Card> component works in any context
export function Card({ image, title, description }: CardProps) {
  return (
    <article className="card">
      <img className="card__image" src={image} alt={title} />
      <div className="card__body">
        <h3 className="card__title">{title}</h3>
        <p>{description}</p>
      </div>
    </article>
  );
}

// In a narrow sidebar → vertical card
function Sidebar() {
  return (
    <aside style={{ width: 300 }}>
      <div className="card-container">  {/* container context */}
        <Card image="/img.jpg" title="Article" description="..." />
      </div>
    </aside>
  );
}

// In a wide main content area → horizontal card automatically
function MainFeed() {
  return (
    <main style={{ maxWidth: 900 }}>
      {articles.map(a => (
        <div key={a.id} className="card-container">
          <Card {...a} />
        </div>
      ))}
    </main>
  );
}
```

**Why this is better than media queries for components**: a media query on `.card` would always show the horizontal layout on a wide viewport, even if the card is in a narrow sidebar. Container queries make the component self-contained — it adapts to *its context*, not the page width.

**Browser support**: baseline in Chrome 105+, Firefox 110+, Safari 16+. Use `@supports (container-type: inline-size)` for a fallback."""
    },
    {
        "question": "Create a smooth infinite horizontal scroll marquee in CSS with `animation` and no JavaScript — it must pause on hover and work with dynamic content.",
        "thinking": "The CSS marquee technique uses a duplicated set of items and a translateX animation that moves one full copy width. No JS needed. The trick is to know the total width of one set — which we can make work by using percentage-based animation and the CSS animation-duration tied to content length. I'll show the pure CSS approach and a React wrapper that handles dynamic content.",
        "answer": """```css
/* ── Marquee container ───────────────────────── */
.marquee {
  overflow: hidden;
  white-space: nowrap;
  user-select: none;
}

/* Inner track — twice as wide, contains 2 copies */
.marquee__track {
  display: inline-flex;
  animation: marquee-scroll 20s linear infinite;
  width: max-content;
}

.marquee:hover .marquee__track {
  animation-play-state: paused;
}

/* Each item group (duplicated) */
.marquee__group {
  display: inline-flex;
  align-items: center;
  gap: 48px;
  padding-right: 48px; /* gap after last item before repeat */
  flex-shrink: 0;
}

@keyframes marquee-scroll {
  from { transform: translateX(0); }
  to   { transform: translateX(-50%); }  /* -50% = move one full copy */
}

/* ── Fade edges ───────────────────────────────── */
.marquee {
  -webkit-mask-image: linear-gradient(
    to right, transparent 0%, black 10%, black 90%, transparent 100%
  );
  mask-image: linear-gradient(
    to right, transparent 0%, black 10%, black 90%, transparent 100%
  );
}
```

```tsx
interface MarqueeProps {
  items: React.ReactNode[];
  speed?: number; // seconds for one full loop
  reverse?: boolean;
}

export function Marquee({ items, speed = 20, reverse = false }: MarqueeProps) {
  return (
    <div className="marquee">
      <div
        className="marquee__track"
        style={{
          animationDuration: `${speed}s`,
          animationDirection: reverse ? 'reverse' : 'normal',
        }}
      >
        {/* Duplicate items so the scroll is seamless */}
        <div className="marquee__group" aria-hidden="false">
          {items.map((item, i) => <div key={i}>{item}</div>)}
        </div>
        <div className="marquee__group" aria-hidden="true">
          {items.map((item, i) => <div key={`dup-${i}`}>{item}</div>)}
        </div>
      </div>
    </div>
  );
}

// Usage
<Marquee
  speed={15}
  items={logos.map(logo => <img src={logo.src} alt={logo.name} height={40} />)}
/>
```

**Why `-50%`**: the track contains 2 identical copies. Moving `-50%` of the total track width moves exactly one copy, at which point the animation loops — the user sees a seamless repeat.

**Speed proportional to content**: for a fair speed regardless of item count, compute `speed = items.length * 3` seconds in your JS."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 3: TYPESCRIPT ADVANCED
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "What are branded/opaque types in TypeScript and why should I use them for domain IDs? Show how to create `UserId`, `OrderId`, and `ProductId` that can't be mixed up at compile time.",
        "thinking": "TypeScript's structural type system means `string` === `string` — a UserId and an OrderId are interchangeable, which causes bugs when you pass the wrong ID to the wrong function. Branded types add a phantom property that makes the types nominally distinct without any runtime overhead. The pattern uses intersection types with a unique symbol or string literal brand.",
        "answer": """```ts
// ── Pattern 1: String literal brand ───────────
type Branded<T, Brand extends string> = T & { readonly __brand: Brand };

type UserId    = Branded<string, 'UserId'>;
type OrderId   = Branded<string, 'OrderId'>;
type ProductId = Branded<string, 'ProductId'>;

// Constructor functions that cast (only place where 'as' is used)
const UserId    = (id: string): UserId    => id as UserId;
const OrderId   = (id: string): OrderId   => id as OrderId;
const ProductId = (id: string): ProductId => id as ProductId;

// ── Usage ──────────────────────────────────────
function getUser(id: UserId): Promise<User> { /* ... */ }
function getOrder(id: OrderId): Promise<Order> { /* ... */ }

const uid = UserId('user-123');
const oid = OrderId('order-456');

getUser(uid);  // ✅
getOrder(oid); // ✅
getUser(oid);  // ❌ Type error: 'OrderId' is not assignable to 'UserId'
getOrder(uid); // ❌ Type error: 'UserId' is not assignable to 'OrderId'
getUser('raw-string'); // ❌ Type error: 'string' is not assignable to 'UserId'

// ── Pattern 2: Unique symbol brand (stronger) ──
declare const __brand: unique symbol;
type Brand<T, B> = T & { [__brand]: B };

type Milliseconds = Brand<number, 'Milliseconds'>;
type Seconds      = Brand<number, 'Seconds'>;

const toMs = (s: Seconds): Milliseconds => (s * 1000) as Milliseconds;
const sleep = (ms: Milliseconds) => new Promise(r => setTimeout(r, ms));

sleep(toMs(5 as Seconds));  // ✅
sleep(5000);                // ❌ 'number' not assignable to 'Milliseconds'

// ── With Zod for runtime validation ───────────
import { z } from 'zod';

const UserIdSchema = z.string().uuid().transform(s => s as UserId);
// Parses AND brands in one step
const uid2 = UserIdSchema.parse(req.params.id); // type: UserId
```

**Zero runtime cost**: the brand is a phantom type — `{ readonly __brand: 'UserId' }` never exists at runtime. It's erased during compilation. You pay only in type-checking time.

**When to use**: any domain primitive that has a specific identity — IDs, monetary amounts, temperatures, durations. Anywhere you've ever mixed up two `string` or `number` parameters in a function call."""
    },
    {
        "question": "Explain TypeScript's `infer` keyword with 3 practical utility types: extracting a function's return type, unwrapping a Promise, and getting array element types.",
        "thinking": "infer is used inside conditional types to capture a type at a specific position in a pattern match. It's like a type-level variable that TypeScript fills in when the condition is true. I should show 3 concrete examples that each demonstrate a different use of infer, then explain the mental model.",
        "answer": """**Mental model**: `infer T` says "if this pattern matches, call the matched type `T` and use it in the true branch."

```ts
// ── 1. Extract function return type ───────────
type ReturnType<T> = T extends (...args: any[]) => infer R ? R : never;

type A = ReturnType<() => string>;           // string
type B = ReturnType<(x: number) => boolean>; // boolean
type C = ReturnType<string>;                 // never (not a function)

// ── 2. Unwrap a Promise ────────────────────────
type Awaited<T> =
  T extends Promise<infer Value>
    ? Awaited<Value>  // recursive: unwrap nested promises
    : T;

type D = Awaited<Promise<string>>;                  // string
type E = Awaited<Promise<Promise<number>>>;         // number
type F = Awaited<string>;                           // string (not a promise)

// ── 3. Array element type ──────────────────────
type ElementOf<T> = T extends (infer E)[] ? E : never;

type G = ElementOf<string[]>;    // string
type H = ElementOf<number[][]>;  // number[] (one level)
type I = ElementOf<readonly boolean[]>; // boolean

// ── 4. Practical: function parameter types ─────
type FirstParam<T> = T extends (first: infer P, ...rest: any[]) => any ? P : never;

type J = FirstParam<(id: string, options?: object) => void>; // string

// ── 5. Extract object method return type ───────
type MethodReturn<T, M extends keyof T> =
  T[M] extends (...args: any[]) => infer R ? R : never;

interface Api {
  fetchUser: (id: string) => Promise<User>;
  fetchOrder: (id: string) => Promise<Order>;
}

type UserResult  = MethodReturn<Api, 'fetchUser'>;  // Promise<User>
type OrderResult = MethodReturn<Api, 'fetchOrder'>; // Promise<Order>

// ── Real use: unwrap API response ──────────────
type ApiData<T> = T extends Promise<infer Data> ? Data : T;

type FetchedUser = ApiData<ReturnType<Api['fetchUser']>>; // User
```

**Key rules**:
- `infer` only works inside `extends` conditional types.
- Multiple `infer` in one type extracts multiple positions: `infer H extends infer T[]` extracts both first and rest.
- `infer` in a covariant position (return type) gives the union of all matching types; in a contravariant position (parameters) gives the intersection."""
    },
    {
        "question": "Create a type-safe event emitter in TypeScript where each event name is mapped to its payload type — `emit('userCreated', user)` must match `on('userCreated', (user: User) => void)`.",
        "thinking": "The key is a generic Events map that maps event names to their payload types. The on/emit methods should use mapped types and template literal types to enforce that the listener signature matches the event's payload. I need to use generic constraints and lookup types: `Events[K]` gives the payload for event K.",
        "answer": """```ts
type EventMap = Record<string, unknown>;
type Listener<T> = (payload: T) => void;

class TypedEventEmitter<Events extends EventMap> {
  private listeners = new Map<keyof Events, Set<Listener<any>>>();

  on<K extends keyof Events>(event: K, listener: Listener<Events[K]>): () => void {
    if (!this.listeners.has(event)) this.listeners.set(event, new Set());
    this.listeners.get(event)!.add(listener);
    return () => this.off(event, listener); // returns unsubscribe function
  }

  once<K extends keyof Events>(event: K, listener: Listener<Events[K]>): void {
    const wrapper: Listener<Events[K]> = (payload) => {
      listener(payload);
      this.off(event, wrapper);
    };
    this.on(event, wrapper);
  }

  off<K extends keyof Events>(event: K, listener: Listener<Events[K]>): void {
    this.listeners.get(event)?.delete(listener);
  }

  emit<K extends keyof Events>(event: K, payload: Events[K]): void {
    this.listeners.get(event)?.forEach(l => l(payload));
  }
}

// ── Usage ──────────────────────────────────────
interface AppEvents {
  userCreated:  { id: string; name: string; email: string };
  orderPlaced:  { orderId: string; total: number; items: string[] };
  pageView:     { path: string; referrer?: string };
  error:        Error;
}

const bus = new TypedEventEmitter<AppEvents>();

// ✅ All correct
bus.on('userCreated', (user) => {
  console.log(user.name); // user: { id, name, email } — fully typed
});

bus.emit('userCreated', { id: '1', name: 'Alice', email: 'a@example.com' });

// ❌ Type errors
bus.emit('userCreated', { id: '1' }); // missing name, email
bus.emit('pageView', { orderId: '1' }); // wrong payload shape
bus.on('orderPlaced', (order) => {
  order.nonExistentField; // Property does not exist
});

// Unsubscribe
const unsub = bus.on('pageView', (e) => console.log(e.path));
unsub(); // clean up
```

**The magic**: `Events[K]` is a lookup type — when TypeScript infers `K = 'userCreated'`, `Events[K]` resolves to `{ id: string; name: string; email: string }`. The `listener` and `payload` parameters are then checked against that exact type."""
    },
    {
        "question": "Show 5 TypeScript mapped type patterns that every frontend developer should know: making fields optional, readonly, required, remapping keys, and filtering by value type.",
        "thinking": "Mapped types iterate over a union of keys and transform each property. The key patterns are: Partial (all optional), Required (all required), Readonly (immutable), key remapping with `as` clause, and filtering properties by their value type using conditional types. I'll show all 5 with practical examples.",
        "answer": """```ts
// ── 1. Partial — all properties optional ───────
type Partial<T> = { [K in keyof T]?: T[K] };

// Deep partial (recursive)
type DeepPartial<T> = {
  [K in keyof T]?: T[K] extends object ? DeepPartial<T[K]> : T[K];
};

interface Config { db: { host: string; port: number }; debug: boolean; }
type PatchConfig = DeepPartial<Config>;
// { db?: { host?: string; port?: number }; debug?: boolean }

// ── 2. Readonly — immutable ────────────────────
type Readonly<T> = { readonly [K in keyof T]: T[K] };

// Deep readonly (freeze-like at type level)
type DeepReadonly<T> = {
  readonly [K in keyof T]: T[K] extends object ? DeepReadonly<T[K]> : T[K];
};

// ── 3. Required — remove all optionals ─────────
type Required<T> = { [K in keyof T]-?: T[K] }; // -? removes optional modifier

// ── 4. Key remapping with `as` clause ──────────
// Prefix all keys with 'get'
type Getters<T> = {
  [K in keyof T as `get${Capitalize<string & K>}`]: () => T[K];
};

interface User { name: string; age: number; }
type UserGetters = Getters<User>;
// { getName: () => string; getAge: () => number }

// Create form field names
type FormFields<T> = {
  [K in keyof T as `${string & K}Field`]: string;
};
// { nameField: string; ageField: string }

// ── 5. Filter properties by value type ─────────
type PickByValue<T, V> = {
  [K in keyof T as T[K] extends V ? K : never]: T[K];
};

interface Mixed {
  name:    string;
  age:     number;
  active:  boolean;
  email:   string;
  score:   number;
}

type StringOnly  = PickByValue<Mixed, string>;  // { name: string; email: string }
type NumberOnly  = PickByValue<Mixed, number>;  // { age: number; score: number }
type Primitives  = PickByValue<Mixed, string | number | boolean>; // all

// ── Practical combo: form update handler ───────
type FormUpdate<T> = {
  [K in keyof T]: (value: T[K]) => void;
};
// Each field gets a setter with the correct type:
// { name: (value: string) => void; age: (value: number) => void; ... }
```

The `-?` modifier in `Required` removes optional markers. Similarly `+?` adds them (same as `?`). The `as` clause in mapped types (TS 4.1+) enables key renaming and filtering (returning `never` from `as` removes the key entirely)."""
    },
    {
        "question": "What is the TypeScript `satisfies` operator and when does it give you something that `as` and explicit annotation don't?",
        "thinking": "satisfies validates that an expression matches a type without widening it. The difference from annotation: annotation widens the type to the declared type (losing specific literal types). The difference from as: as silences errors and overrides type inference. satisfies validates AND keeps the narrower inferred type. Key use case: const objects where you want to validate shape but keep literal types for autocomplete and safety.",
        "answer": """```ts
// ── The problem with explicit annotation ───────
type Route = { path: string; component: React.FC };
const ROUTES: Record<string, Route> = {
  home:    { path: '/', component: Home },
  about:   { path: '/about', component: About },
};
// ROUTES.home.path is typed as `string` — lost the literal '/'.
// Object.keys(ROUTES) returns string[] — no autocomplete.

// ── The problem with `as` ──────────────────────
const ROUTES2 = {
  home: { path: '/', component: Home },
  bad:  { path: 42, component: About }, // ❌ wrong, but as won't catch it
} as Record<string, Route>; // silences the error — dangerous

// ── `satisfies` — validates AND preserves ──────
const ROUTES3 = {
  home:  { path: '/',      component: Home  },
  about: { path: '/about', component: About },
} satisfies Record<string, Route>;

// Validation: shape is checked against Record<string, Route>
// Preservation: ROUTES3.home.path is still typed as '/' (literal string)
// Autocomplete: ROUTES3.hom → IDE suggests 'home', 'about'

ROUTES3.home.path;   // type: '/'      (literal!) ← not 'string'
ROUTES3.about.path;  // type: '/about' (literal!)

// ── Practical: palette/theme object ───────────
type Color = `#${string}` | `rgb(${string})`;

const PALETTE = {
  primary:   '#0066cc',
  secondary: '#6b7280',
  danger:    '#dc2626',
  success:   '#16a34a',
  invalid:   'not-a-color', // ❌ caught by satisfies
} satisfies Record<string, Color>;

// PALETTE.primary is '#0066cc' (literal), not 'string'
// You get autocomplete AND a type error on 'not-a-color'

// ── Practical: i18n messages ───────────────────
type Messages = typeof en; // English is the source of truth
const en = { greeting: 'Hello', farewell: 'Goodbye' };
const fr = {
  greeting: 'Bonjour',
  farewell: 'Au revoir',
  extra: 'Oops', // ❌ caught — fr must have exactly the keys en has
} satisfies Messages;
```

**Rule of thumb**: replace `as const` with `satisfies YourType as const` when you want both narrowing and validation. Replace explicit type annotations on object literals with `satisfies` when you need the literal types for downstream inference."""
    },
    {
        "question": "Show how to use TypeScript template literal types to build a type-safe CSS-in-JS utility that only accepts valid CSS property names and rejects typos.",
        "thinking": "TypeScript 4.1 introduced template literal types which can concatenate string literals. Combined with mapped types and Extract, I can build a type that accepts only valid CSS properties. React's CSSProperties is the source of truth. I can also build utility types that generate CSS variable names, class names, and more from template literals.",
        "answer": """```ts
import type { CSSProperties } from 'react';

// ── 1. Only valid CSS property names ──────────
type CSSProp = keyof CSSProperties;

function css<T extends Partial<CSSProperties>>(styles: T): T {
  return styles;
}

css({ color: 'red', backgroundColor: '#fff' });  // ✅
css({ colour: 'red' });  // ❌ Object literal may only specify known properties

// ── 2. Template literal CSS variable names ─────
type CSSVar<Name extends string> = `--${Name}`;
type CSSVarValue<Name extends string> = `var(--${Name})`;

function cssVar<N extends string>(name: N): CSSVarValue<N> {
  return `var(--${name})` as CSSVarValue<N>;
}

const colorPrimary = cssVar('color-primary');
// type: 'var(--color-primary)' — not just string

// ── 3. Type-safe breakpoint utilities ──────────
type Breakpoint = 'sm' | 'md' | 'lg' | 'xl';
type ResponsiveProp<T> = T | Partial<Record<Breakpoint, T>>;

interface TypographyProps {
  fontSize: ResponsiveProp<CSSProperties['fontSize']>;
  fontWeight: ResponsiveProp<CSSProperties['fontWeight']>;
}

// ── 4. CSS property getter type ─────────────────
type CSSGetter<Prop extends CSSProp> =
  `get${Capitalize<string & Prop>}`;

type FontSizeGetter = CSSGetter<'fontSize'>;    // 'getFontSize'
type ColorGetter    = CSSGetter<'color'>;       // 'getColor'

// ── 5. Atomic class name builder ────────────────
type SpacingKey = 'p' | 'px' | 'py' | 'pt' | 'pb' | 'pl' | 'pr' | 'm' | 'mx' | 'my';
type SpacingScale = 0 | 1 | 2 | 3 | 4 | 5 | 6 | 8 | 10 | 12 | 16;
type SpacingClass = `${SpacingKey}-${SpacingScale}`;

function cls(...classes: SpacingClass[]): string {
  return classes.join(' ');
}

cls('p-4', 'mx-2', 'py-8');   // ✅
cls('p-4', 'padding-4');       // ❌ 'padding-4' not assignable to SpacingClass
cls('p-4', 'm-99');            // ❌ '99' not in SpacingScale

// ── 6. Event handler type narrowing ────────────
type EventHandler<T extends keyof HTMLElementEventMap> =
  (event: HTMLElementEventMap[T]) => void;

function listen<T extends keyof HTMLElementEventMap>(
  el: HTMLElement, event: T, handler: EventHandler<T>
) {
  el.addEventListener(event, handler);
}

listen(button, 'click', (e) => {
  e.clientX; // e is MouseEvent — fully typed
});
listen(input, 'keydown', (e) => {
  e.key; // e is KeyboardEvent
});
```

Template literal types shine at **compile-time string validation** — you get autocomplete and typo prevention for string-based APIs at zero runtime cost."""
    },
    {
        "question": "Implement a fully type-safe fetch wrapper in TypeScript with typed routes, request bodies, and response types — no `any`, no casting.",
        "thinking": "The key is to define a route map that maps path strings to their request body type and response type, then write a fetch function that looks up the route and infers the correct types. I'll use overloads or a generic interface to make this work. The goal is that calling apiFetch('/users/123') returns User without any casting.",
        "answer": """```ts
// ── Route definitions ──────────────────────────
interface ApiRoutes {
  'GET /users':                { body: never;               response: User[] };
  'GET /users/:id':            { body: never;               response: User };
  'POST /users':               { body: CreateUserDto;       response: User };
  'PUT /users/:id':            { body: UpdateUserDto;       response: User };
  'DELETE /users/:id':         { body: never;               response: { ok: true } };
  'GET /orders':               { body: never;               response: Order[] };
  'POST /orders':              { body: CreateOrderDto;      response: Order };
}

type Method = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE';
type RouteKey = keyof ApiRoutes;
type RouteBody<R extends RouteKey>     = ApiRoutes[R]['body'];
type RouteResponse<R extends RouteKey> = ApiRoutes[R]['response'];

// ── Overloaded fetch wrapper ────────────────────
async function apiFetch<R extends RouteKey>(
  route: R,
  ...args: RouteBody<R> extends never
    ? []                          // no body needed (GET/DELETE)
    : [body: RouteBody<R>]        // body required
): Promise<RouteResponse<R>> {
  const [method, path] = (route as string).split(' ');
  const body = args[0];

  const res = await fetch(`/api${path}`, {
    method,
    headers: body ? { 'Content-Type': 'application/json' } : undefined,
    body: body ? JSON.stringify(body) : undefined,
  });

  if (!res.ok) {
    const error = await res.json().catch(() => ({}));
    throw new ApiError(res.status, error.message ?? 'Request failed');
  }

  return res.json() as Promise<RouteResponse<R>>;
}

class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
    this.name = 'ApiError';
  }
}

// ── Usage — fully typed, zero casts ────────────
const users = await apiFetch('GET /users');
// users: User[] ✅

const user = await apiFetch('GET /users/:id');
// user: User ✅

const newUser = await apiFetch('POST /users', {
  name: 'Alice', email: 'alice@example.com'
});
// newUser: User ✅

// ❌ Type errors caught at compile time:
await apiFetch('GET /users', { name: 'Alice' }); // No body allowed for GET
await apiFetch('POST /users');                   // Body is required
await apiFetch('POST /users', { wrong: 'shape' }); // Wrong body type
```

**The conditional type trick**: `RouteBody<R> extends never ? [] : [body: RouteBody<R>]` makes the body argument optional when the route body is `never` (GET routes), and required when it has a type. This is the "rest args" overload pattern."""
    },
    {
        "question": "Explain TypeScript discriminated unions with exhaustiveness checking — build a `Shape` type and a `getArea` function that TypeScript guarantees handles every case.",
        "thinking": "Discriminated unions use a shared literal property (the discriminant) to narrow the type. The key to exhaustiveness checking is the `never` trick in the default case — if a case is unhandled, the type is not `never`, and TypeScript errors. I should show the shape example, the exhaustive switch, and how to use assertNever.",
        "answer": """```ts
// ── Discriminated union ────────────────────────
type Circle    = { kind: 'circle';    radius: number };
type Rectangle = { kind: 'rectangle'; width: number; height: number };
type Triangle  = { kind: 'triangle';  base: number;  height: number };

type Shape = Circle | Rectangle | Triangle;

// ── Exhaustiveness helper ──────────────────────
function assertNever(x: never, message?: string): never {
  throw new Error(message ?? `Unhandled case: ${JSON.stringify(x)}`);
}

// ── Exhaustive switch ──────────────────────────
function getArea(shape: Shape): number {
  switch (shape.kind) {
    case 'circle':
      return Math.PI * shape.radius ** 2;
    case 'rectangle':
      return shape.width * shape.height;
    case 'triangle':
      return (shape.base * shape.height) / 2;
    default:
      return assertNever(shape); // ← TypeScript error if any case is missing
  }
}

// Add a new shape without updating getArea — TypeScript catches it immediately:
type Pentagon = { kind: 'pentagon'; side: number };
type Shape2 = Shape | Pentagon;

function getArea2(shape: Shape2): number {
  switch (shape.kind) {
    case 'circle':    return Math.PI * shape.radius ** 2;
    case 'rectangle': return shape.width * shape.height;
    case 'triangle':  return (shape.base * shape.height) / 2;
    // ❌ TypeScript error: 'Pentagon' is not assignable to 'never'
    // Compiler forces you to handle the new case
    default: return assertNever(shape);
  }
}

// ── Discriminated unions for API responses ─────
type ApiResult<T> =
  | { status: 'success'; data: T }
  | { status: 'error';   error: string; code: number }
  | { status: 'loading' };

function renderResult<T>(result: ApiResult<T>, render: (data: T) => JSX.Element) {
  switch (result.status) {
    case 'success': return render(result.data);
    case 'error':   return <ErrorMessage msg={result.error} code={result.code} />;
    case 'loading': return <Spinner />;
    default: return assertNever(result);
  }
}

// ── Discriminated unions for Redux/useReducer ──
type Action =
  | { type: 'INCREMENT'; amount: number }
  | { type: 'RESET' }
  | { type: 'SET_USER'; user: User };

function reducer(state: State, action: Action): State {
  switch (action.type) {
    case 'INCREMENT': return { ...state, count: state.count + action.amount };
    case 'RESET':     return initialState;
    case 'SET_USER':  return { ...state, user: action.user };
    default: return assertNever(action);
  }
}
```

The `kind`/`type`/`status` discriminant must be a **literal type** — not just `string`. TypeScript uses the literal to narrow the union. `assertNever` converts a compile-time guarantee into a runtime safety net for code that bypasses TypeScript (dynamic imports, `JSON.parse`)."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 4: JAVASCRIPT ADVANCED
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "How do JavaScript Proxies work? Build a reactive state object where reading any property and then modifying it automatically triggers a re-render callback — similar to how Vue 3's reactivity works.",
        "thinking": "Proxy wraps an object and intercepts operations via traps: get, set, deleteProperty, etc. For reactivity I need: a get trap that tracks which properties are read (dependency tracking), a set trap that notifies listeners when a tracked property changes. This is the foundation of Vue 3's reactivity. I should show the Proxy, a simple effect/tracking system, and demonstrate it working.",
        "answer": """```js
// ── Dependency tracking ────────────────────────
let currentEffect = null;
const deps = new WeakMap(); // object → Map<key, Set<effect>>

function getDeps(target, key) {
  if (!deps.has(target)) deps.set(target, new Map());
  const keyMap = deps.get(target);
  if (!keyMap.has(key)) keyMap.set(key, new Set());
  return keyMap.get(key);
}

// ── Proxy-based reactive object ────────────────
function reactive(obj) {
  return new Proxy(obj, {
    get(target, key, receiver) {
      // Track this property access if we're inside an effect
      if (currentEffect) {
        getDeps(target, key).add(currentEffect);
      }
      const value = Reflect.get(target, key, receiver);
      // Recursively proxy nested objects
      return value !== null && typeof value === 'object'
        ? reactive(value)
        : value;
    },

    set(target, key, value, receiver) {
      const oldValue = target[key];
      const result = Reflect.set(target, key, value, receiver);
      // Notify all effects that read this property
      if (oldValue !== value) {
        getDeps(target, key).forEach(effect => effect());
      }
      return result;
    },

    deleteProperty(target, key) {
      const result = Reflect.deleteProperty(target, key);
      getDeps(target, key).forEach(effect => effect());
      return result;
    },
  });
}

// ── Effect system ──────────────────────────────
function effect(fn) {
  const run = () => {
    currentEffect = run;
    try { fn(); }
    finally { currentEffect = null; }
  };
  run(); // run immediately to collect dependencies
  return run;
}

// ── Usage ──────────────────────────────────────
const state = reactive({ count: 0, user: { name: 'Alice' } });

// This effect re-runs whenever state.count changes
effect(() => {
  console.log('Count:', state.count); // logs immediately: "Count: 0"
});

state.count = 1;  // logs: "Count: 1"
state.count = 2;  // logs: "Count: 2"
state.user.name = 'Bob'; // no log — not tracked by this effect

// Second effect
effect(() => {
  console.log('Name:', state.user.name); // logs: "Name: Alice"
});

state.user.name = 'Bob'; // logs: "Name: Bob"
state.count = 3;          // logs: "Count: 3" (first effect, not second)
```

**`Reflect.get/set`** instead of direct property access: Reflect correctly handles inherited properties, getters/setters on prototypes, and the `receiver` (this context) — important for class-based objects.

**This is exactly Vue 3's core**: `reactive()` uses this Proxy pattern, `effect()` is their `watchEffect`, and the dep tracking system is their `track/trigger` API."""
    },
    {
        "question": "Explain `WeakRef` and `FinalizationRegistry` — build a memory-sensitive image cache where cached images are automatically released under memory pressure.",
        "thinking": "WeakRef holds a weak reference to an object — the GC can collect it if there are no strong references. FinalizationRegistry fires a callback when a weakly-held object is collected. This is useful for caches: you hold the value weakly so memory pressure evicts it automatically, and FinalizationRegistry cleans up the cache entry. I should show a practical image cache with these APIs.",
        "answer": """```js
class WeakImageCache {
  #cache = new Map();        // url → WeakRef<HTMLImageElement>
  #registry = new FinalizationRegistry((url) => {
    // Called when an image is garbage collected
    this.#cache.delete(url);
    console.log(`Cache entry evicted: ${url}`);
  });

  get(url) {
    const ref = this.#cache.get(url);
    if (!ref) return null;

    const img = ref.deref(); // returns the image or undefined if GC'd
    if (!img) {
      this.#cache.delete(url); // stale entry — clean up eagerly
      return null;
    }
    return img;
  }

  set(url, img) {
    const ref = new WeakRef(img);
    this.#cache.set(url, ref);
    this.#registry.register(img, url, ref); // register for cleanup notification
  }

  async load(url) {
    const cached = this.get(url);
    if (cached) return cached;

    const img = new Image();
    await new Promise((resolve, reject) => {
      img.onload = resolve;
      img.onerror = reject;
      img.src = url;
    });

    this.set(url, img);
    return img;
  }

  get size() { return this.#cache.size; }
}

// Usage
const cache = new WeakImageCache();

async function loadImage(url) {
  const img = await cache.load(url);
  document.body.appendChild(img.cloneNode());
}

// When there are no strong references to the img element,
// the GC may collect it, and FinalizationRegistry removes the cache entry.
```

**Critical caveats**:
1. **GC timing is non-deterministic** — `WeakRef.deref()` may return the object for a long time even if there are no other references. Never rely on it for correctness, only for optimization.
2. **`FinalizationRegistry` callbacks fire on GC thread** in some engines — don't do heavy work in them.
3. This is for *truly* memory-sensitive caches (image/blob data). For most caches, use an LRU with a size limit instead — it's predictable and testable."""
    },
    {
        "question": "Show how to use Web Workers to offload a CPU-intensive computation (image processing / large array sort) off the main thread, with a clean TypeScript-typed message protocol.",
        "thinking": "Web Workers run on a separate thread — postMessage and onmessage for communication. The key challenges: type-safe message protocol (discriminated union), transferable objects for zero-copy large data transfer, and proper cleanup. I should show the worker file, the main thread wrapper, and the typed protocol.",
        "answer": """```ts
// ── worker.ts ─────────────────────────────────
type WorkerRequest =
  | { type: 'SORT'; id: string; data: number[] }
  | { type: 'FILTER'; id: string; data: number[]; threshold: number };

type WorkerResponse =
  | { type: 'SORT_DONE'; id: string; result: number[] }
  | { type: 'FILTER_DONE'; id: string; result: number[] }
  | { type: 'ERROR'; id: string; error: string };

self.onmessage = (e: MessageEvent<WorkerRequest>) => {
  const msg = e.data;
  try {
    switch (msg.type) {
      case 'SORT': {
        const result = [...msg.data].sort((a, b) => a - b);
        const response: WorkerResponse = { type: 'SORT_DONE', id: msg.id, result };
        // Transfer the buffer for zero-copy (avoids cloning large arrays)
        const buf = new Float64Array(result).buffer;
        self.postMessage({ type: 'SORT_DONE', id: msg.id, result }, [buf]);
        break;
      }
      case 'FILTER': {
        const result = msg.data.filter(n => n > msg.threshold);
        self.postMessage({ type: 'FILTER_DONE', id: msg.id, result } satisfies WorkerResponse);
        break;
      }
    }
  } catch (err) {
    self.postMessage({ type: 'ERROR', id: msg.id, error: String(err) } satisfies WorkerResponse);
  }
};

// ── useWorker.ts — main thread wrapper ─────────
import { useEffect, useRef, useCallback } from 'react';

export function useWorker() {
  const workerRef = useRef<Worker | null>(null);
  const pendingRef = useRef(new Map<string, {
    resolve: (r: WorkerResponse) => void;
    reject: (e: Error) => void;
  }>());

  useEffect(() => {
    const worker = new Worker(new URL('./worker.ts', import.meta.url), { type: 'module' });
    workerRef.current = worker;

    worker.onmessage = (e: MessageEvent<WorkerResponse>) => {
      const { id } = e.data;
      const pending = pendingRef.current.get(id);
      if (pending) {
        pendingRef.current.delete(id);
        if (e.data.type === 'ERROR') pending.reject(new Error(e.data.error));
        else pending.resolve(e.data);
      }
    };

    return () => worker.terminate();
  }, []);

  const send = useCallback(<R extends WorkerResponse>(
    msg: WorkerRequest
  ): Promise<R> => {
    return new Promise((resolve, reject) => {
      pendingRef.current.set(msg.id, { resolve: resolve as any, reject });
      workerRef.current?.postMessage(msg);
    });
  }, []);

  return { send };
}

// ── Usage ──────────────────────────────────────
function DataProcessor() {
  const { send } = useWorker();

  const sortData = async (data: number[]) => {
    const result = await send<WorkerResponse & { type: 'SORT_DONE' }>({
      type: 'SORT',
      id: crypto.randomUUID(),
      data,
    });
    console.log('Sorted:', result.result);
  };
}
```

**Transferable objects**: passing a large `ArrayBuffer` or `ImageBitmap` with `postMessage(data, [buffer])` transfers ownership to the worker — zero memory copy. The original buffer becomes detached (unusable) in the sender."""
    },
    {
        "question": "Implement a Service Worker with a cache-first strategy for static assets and a network-first strategy for API calls, using the Workbox-style approach manually.",
        "thinking": "Service Workers intercept fetch events. I need two strategies: cache-first (check cache, fall back to network, update cache) for static assets like JS/CSS/images, and network-first (try network, fall back to cache) for API calls. I should show the install/activate/fetch lifecycle, cache naming with versions, and cleanup of old caches.",
        "answer": """```js
// service-worker.js
const CACHE_NAME = 'app-v1';
const STATIC_ASSETS = [
  '/',
  '/index.html',
  '/assets/main.js',
  '/assets/main.css',
];

// ── Install: pre-cache static assets ──────────
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then(cache => cache.addAll(STATIC_ASSETS))
      .then(() => self.skipWaiting()) // activate immediately
  );
});

// ── Activate: clean up old caches ─────────────
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then(keys =>
      Promise.all(
        keys
          .filter(key => key !== CACHE_NAME)
          .map(key => caches.delete(key))
      )
    ).then(() => self.clients.claim()) // take control of all tabs
  );
});

// ── Fetch: route requests to strategies ────────
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);

  // API calls → network first, cache as fallback
  if (url.pathname.startsWith('/api/')) {
    event.respondWith(networkFirst(request));
    return;
  }

  // Static assets → cache first, network as fallback
  if (request.destination === 'script' ||
      request.destination === 'style'  ||
      request.destination === 'image') {
    event.respondWith(cacheFirst(request));
    return;
  }

  // Navigation → stale-while-revalidate for HTML
  if (request.mode === 'navigate') {
    event.respondWith(staleWhileRevalidate(request));
  }
});

// ── Strategies ─────────────────────────────────
async function cacheFirst(request) {
  const cached = await caches.match(request);
  if (cached) return cached;

  const response = await fetch(request);
  if (response.ok) {
    const cache = await caches.open(CACHE_NAME);
    cache.put(request, response.clone()); // clone: response can only be consumed once
  }
  return response;
}

async function networkFirst(request) {
  try {
    const response = await fetch(request);
    if (response.ok) {
      const cache = await caches.open(CACHE_NAME);
      cache.put(request, response.clone());
    }
    return response;
  } catch {
    const cached = await caches.match(request);
    return cached ?? new Response('Offline', { status: 503 });
  }
}

async function staleWhileRevalidate(request) {
  const cached = await caches.match(request);
  const fetchPromise = fetch(request).then(response => {
    if (response.ok) {
      caches.open(CACHE_NAME).then(cache => cache.put(request, response.clone()));
    }
    return response;
  });
  return cached ?? fetchPromise;
}
```

**Register in your app**:
```ts
if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    navigator.serviceWorker.register('/service-worker.js');
  });
}
```

**Why `response.clone()`**: a `Response` body is a stream that can only be read once. You must clone it before `cache.put()` so you can still return the original to the browser."""
    },
    {
        "question": "Demonstrate the Streams API — use `ReadableStream` to process a large CSV file in chunks without loading the whole file into memory, and pipe it through a `TransformStream`.",
        "thinking": "The Streams API lets you process data incrementally. ReadableStream produces data chunks, TransformStream processes each chunk, WritableStream consumes the result. For CSV: fetch creates a readable stream from the response body, pipe through a TextDecoder stream, then through a custom transform that splits on newlines and parses CSV rows. Never loads the full file into memory.",
        "answer": """```ts
// ── CSV line parser as a TransformStream ───────
function csvLineParser(): TransformStream<string, string[]> {
  let buffer = '';

  return new TransformStream({
    transform(chunk, controller) {
      buffer += chunk;
      const lines = buffer.split('\\n');
      buffer = lines.pop() ?? ''; // keep incomplete last line

      for (const line of lines) {
        if (line.trim()) {
          controller.enqueue(parseCSVLine(line));
        }
      }
    },
    flush(controller) {
      if (buffer.trim()) {
        controller.enqueue(parseCSVLine(buffer));
      }
    },
  });
}

function parseCSVLine(line: string): string[] {
  // Simple CSV parse (handles quoted commas)
  const result: string[] = [];
  let field = '';
  let inQuotes = false;

  for (const char of line) {
    if (char === '"') { inQuotes = !inQuotes; continue; }
    if (char === ',' && !inQuotes) { result.push(field); field = ''; continue; }
    field += char;
  }
  result.push(field);
  return result;
}

// ── Process a large CSV from URL ───────────────
async function processLargeCSV(url: string, onRow: (row: Record<string, string>) => void) {
  const response = await fetch(url);
  if (!response.body) throw new Error('No response body');

  let headers: string[] | null = null;
  let rowCount = 0;

  const writable = new WritableStream<string[]>({
    write(row) {
      if (!headers) {
        headers = row; // first row is headers
        return;
      }
      const record: Record<string, string> = {};
      headers.forEach((h, i) => { record[h] = row[i] ?? ''; });
      onRow(record);
      rowCount++;
    },
    close() {
      console.log(`Processed ${rowCount} rows`);
    },
  });

  await response.body
    .pipeThrough(new TextDecoderStream())  // bytes → string chunks
    .pipeThrough(csvLineParser())          // string chunks → string[] rows
    .pipeTo(writable);                     // consume row by row
}

// Usage
await processLargeCSV('/data/million-rows.csv', (row) => {
  if (parseFloat(row.revenue) > 10000) {
    console.log('High revenue:', row.store, row.revenue);
  }
});
```

**Memory profile**: a 500MB CSV never fully loads into RAM. Each chunk (~16KB default) flows through the transform pipeline and is GC'd after processing.

**Backpressure**: `.pipeTo()` automatically applies backpressure — if the `WritableStream` is slow (e.g. writing to IndexedDB), the pipeline pauses reading upstream. No manual throttling needed."""
    },
    {
        "question": "Show how `BroadcastChannel` and `Web Locks API` work together to synchronize state across browser tabs — implement a 'single leader' pattern where only one tab processes background tasks.",
        "thinking": "BroadcastChannel lets tabs communicate. Web Locks API provides mutual exclusion across tabs/workers — only one context can hold a lock at a time. Combining them: tabs compete for a lock (only one wins = 'leader'), the leader broadcasts its status, others listen and take over if the leader tab closes (lock is released). This is how you implement tab-aware singletons for background sync, WebSocket connections, etc.",
        "answer": """```ts
class TabLeader {
  private channel: BroadcastChannel;
  private isLeader = false;
  private lockController: AbortController | null = null;
  onBecomeLeader: (() => void) | null = null;
  onLoseLeadership: (() => void) | null = null;

  constructor(private name: string) {
    this.channel = new BroadcastChannel(`tab-leader-${name}`);
    this.channel.onmessage = (e) => this.handleMessage(e.data);
  }

  async start() {
    this.lockController = new AbortController();

    // Attempt to acquire the lock — only one tab can hold it
    await navigator.locks.request(
      `leader-${this.name}`,
      { signal: this.lockController.signal },
      async () => {
        // We are now the leader
        this.isLeader = true;
        this.broadcast({ type: 'LEADER_ELECTED', tabId: this.tabId });
        this.onBecomeLeader?.();

        // Hold the lock until stop() is called
        await new Promise<void>((resolve) => {
          this.lockController!.signal.addEventListener('abort', resolve);
        });

        this.isLeader = false;
        this.onLoseLeadership?.();
      }
    );
  }

  stop() {
    this.lockController?.abort();
    this.channel.close();
  }

  broadcast(msg: object) {
    this.channel.postMessage(msg);
  }

  private handleMessage(msg: { type: string; tabId: string }) {
    if (msg.type === 'LEADER_ELECTED' && !this.isLeader) {
      console.log(`Tab ${msg.tabId} is now leader`);
    }
  }

  private get tabId() {
    return sessionStorage.getItem('tabId') ?? (() => {
      const id = crypto.randomUUID();
      sessionStorage.setItem('tabId', id);
      return id;
    })();
  }
}

// ── Usage: single background WebSocket per browser session
const leader = new TabLeader('websocket-manager');

leader.onBecomeLeader = () => {
  console.log('This tab will manage the WebSocket');
  const ws = new WebSocket('wss://api.example.com/live');

  ws.onmessage = (e) => {
    // Broadcast server messages to all tabs
    leader.broadcast({ type: 'WS_MESSAGE', data: JSON.parse(e.data) });
  };
};

// All tabs listen for broadcast WS messages
const channel = new BroadcastChannel('tab-leader-websocket-manager');
channel.onmessage = (e) => {
  if (e.data.type === 'WS_MESSAGE') {
    updateUI(e.data.data);
  }
};

await leader.start();
```

When the leader tab is closed, the lock is released, and another tab's `navigator.locks.request` call (which was queued) resolves — automatically electing a new leader."""
    },
    {
        "question": "Explain the JavaScript event loop in depth — macrotasks, microtasks, and the rendering pipeline — and show why `queueMicrotask` beats `setTimeout(fn, 0)` for certain tasks.",
        "thinking": "The event loop processes: one macrotask, then drains the entire microtask queue, then (optionally) paints. This means microtasks run before the browser gets a chance to render. setTimeout creates a macrotask (minimum delay ~4ms, not 0). queueMicrotask adds to the microtask queue. Promise.then/.catch/.finally are microtasks. I should show the execution order with examples and practical use cases for queueMicrotask.",
        "answer": """```js
// ── Execution order demonstration ──────────────
console.log('1: synchronous');

setTimeout(() => console.log('5: setTimeout (macrotask)'), 0);

Promise.resolve()
  .then(() => console.log('3: Promise.then (microtask)'))
  .then(() => console.log('4: chained then (microtask)'));

queueMicrotask(() => console.log('2: queueMicrotask'));

console.log('1b: still synchronous');

// Output order:
// 1: synchronous
// 1b: still synchronous
// 2: queueMicrotask         ← microtask, before setTimeout
// 3: Promise.then           ← microtask
// 4: chained then           ← microtask
// 5: setTimeout             ← macrotask (next event loop tick, after painting)
```

**The full loop**:
```
┌─────────────────────────────────┐
│ 1. Dequeue one macrotask        │  ← setTimeout, setInterval, I/O, user events
│ 2. Drain microtask queue        │  ← Promise.then, queueMicrotask, MutationObserver
│    (until empty — including new │
│     microtasks added mid-drain) │
│ 3. Render (if needed)           │  ← requestAnimationFrame runs here
│ 4. Go to 1                      │
└─────────────────────────────────┘
```

**When to use `queueMicrotask`**

```js
// ✅ Batch multiple synchronous writes, flush once as a microtask
class BatchedStore {
  #pending = false;
  #subscribers = new Set();

  update(key, value) {
    this.#data[key] = value;
    if (!this.#pending) {
      this.#pending = true;
      queueMicrotask(() => {
        this.#pending = false;
        this.#subscribers.forEach(fn => fn(this.#data)); // one notification
      });
    }
  }
}

// Multiple synchronous updates → single subscriber notification
store.update('a', 1);
store.update('b', 2);
store.update('c', 3);
// Subscribers called ONCE with all 3 changes, before browser renders

// vs setTimeout: would delay notification until AFTER next render
```

**Why `setTimeout(fn, 0)` is 4ms minimum**: HTML spec mandates a 4ms floor after several nested calls. Microtasks have no such delay.

**Danger of infinite microtask loops**: unlike a `while(true)`, a microtask that queues another microtask will starve the render:
```js
// ❌ Browser freezes — microtask queue never empties
function badLoop() {
  queueMicrotask(badLoop); // queues another before render can run
}
```
Use `requestAnimationFrame` or `setTimeout` for loops that should yield to rendering."""
    },
    {
        "question": "Build a `structuredClone` deep copy alternative that also handles functions, class instances with methods, and circular references — explain where `structuredClone` falls short.",
        "thinking": "structuredClone handles: plain objects, arrays, Date, Map, Set, ArrayBuffer, RegExp. It does NOT handle: functions, class instances (loses prototype/methods), DOM nodes, Errors fully. For a complete clone utility I need: circular reference tracking (WeakMap), prototype chain preservation, and function reference copying. I should show what structuredClone fails at and the custom implementation.",
        "answer": """```js
// ── What structuredClone fails at ─────────────
class User {
  constructor(name, age) { this.name = name; this.age = age; }
  greet() { return `Hi, I'm ${this.name}`; }
}

const original = new User('Alice', 30);
const cloned = structuredClone(original);

console.log(cloned.name);  // 'Alice' ✅
console.log(cloned.greet); // undefined ❌ — methods lost (prototype not copied)
console.log(cloned instanceof User); // false ❌

// Functions — completely fail
structuredClone(() => 'hello'); // ❌ DataCloneError: function cannot be cloned

// ── Full deep clone with class + function support
function deepClone(value, seen = new WeakMap()) {
  // Primitives
  if (value === null || typeof value !== 'object' && typeof value !== 'function') {
    return value;
  }

  // Functions: copy reference (functions are usually shared)
  if (typeof value === 'function') return value;

  // Circular reference
  if (seen.has(value)) return seen.get(value);

  // Date
  if (value instanceof Date) return new Date(value);

  // RegExp
  if (value instanceof RegExp) return new RegExp(value.source, value.flags);

  // Map
  if (value instanceof Map) {
    const map = new Map();
    seen.set(value, map);
    value.forEach((v, k) => map.set(deepClone(k, seen), deepClone(v, seen)));
    return map;
  }

  // Set
  if (value instanceof Set) {
    const set = new Set();
    seen.set(value, set);
    value.forEach(v => set.add(deepClone(v, seen)));
    return set;
  }

  // Array
  if (Array.isArray(value)) {
    const arr = [];
    seen.set(value, arr);
    value.forEach((item, i) => { arr[i] = deepClone(item, seen); });
    return arr;
  }

  // Class instance or plain object — preserve prototype
  const proto = Object.getPrototypeOf(value);
  const clone = Object.create(proto); // preserves methods!
  seen.set(value, clone);

  for (const key of Object.getOwnPropertyNames(value)) {
    const desc = Object.getOwnPropertyDescriptor(value, key);
    if (desc.value !== undefined) {
      Object.defineProperty(clone, key, {
        ...desc,
        value: deepClone(desc.value, seen),
      });
    } else {
      Object.defineProperty(clone, key, desc); // preserve getters/setters
    }
  }

  return clone;
}

// Test
const user = new User('Alice', 30);
user.self = user; // circular reference

const cloned2 = deepClone(user);
console.log(cloned2.greet());          // 'Hi, I'm Alice' ✅
console.log(cloned2 instanceof User);  // true ✅
console.log(cloned2.self === cloned2); // true ✅ circular preserved
```

**Practical advice**: use `structuredClone` for plain data (JSON-like). Use `deepClone` above for class instances. For React state, use `JSON.parse(JSON.stringify(obj))` only for simple serializable data — it's fast but loses `Date`, `undefined`, and functions."""
    },
]

# ─────────────────────────────────────────────
# MORE REACT: Search, Tailwind, Server Actions
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "Build a fuzzy search component in React with Fuse.js: debounced query, keyboard navigation (ArrowUp/Down/Enter/Escape), and character-level match highlighting with proper aria-combobox.",
        "thinking": "Three concerns to juggle: (1) the search engine — Fuse.js instance memoized so the index doesn't rebuild every render, debounce the query ~220ms; (2) keyboard nav — track active index, wrap with modulo; (3) a11y — role=combobox on input, role=listbox on the dropdown, role=option on each result, aria-activedescendant pointing to the focused option. Fuse's includeMatches:true returns indices as [start, end] pairs used to wrap chars in <mark>.",
        "answer": """```tsx
import Fuse from 'fuse.js';
import { useState, useMemo, useCallback, useEffect } from 'react';

interface Item { id: string; title: string; category: string; }

function useDebounce<T>(v: T, ms: number) {
  const [val, set] = useState(v);
  useEffect(() => { const t = setTimeout(() => set(v), ms); return () => clearTimeout(t); }, [v, ms]);
  return val;
}

function highlight(text: string, indices: [number, number][]): React.ReactNode {
  const out: React.ReactNode[] = [];
  let cursor = 0;
  for (const [s, e] of indices) {
    if (cursor < s) out.push(text.slice(cursor, s));
    out.push(<mark key={s} style={{ background: '#fef08a', borderRadius: 2 }}>{text.slice(s, e + 1)}</mark>);
    cursor = e + 1;
  }
  out.push(text.slice(cursor));
  return <>{out}</>;
}

export function FuzzySearch({ items, onSelect }: { items: Item[]; onSelect: (item: Item) => void }) {
  const [q, setQ]       = useState('');
  const [active, setAct] = useState(-1);
  const dq = useDebounce(q, 220);

  const fuse = useMemo(() => new Fuse(items, {
    keys: ['title', 'category'], includeMatches: true, threshold: 0.4,
  }), [items]);

  const results = useMemo(() => (dq ? fuse.search(dq).slice(0, 8) : []), [fuse, dq]);

  const confirm = useCallback((item: Item) => {
    onSelect(item); setQ(''); setAct(-1);
  }, [onSelect]);

  const onKey = useCallback((e: React.KeyboardEvent) => {
    const len = results.length;
    if (!len) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); setAct(a => (a + 1) % len); }
    if (e.key === 'ArrowUp')   { e.preventDefault(); setAct(a => (a - 1 + len) % len); }
    if (e.key === 'Escape')    { setQ(''); setAct(-1); }
    if (e.key === 'Enter' && active >= 0) confirm(results[active].item);
  }, [results, active, confirm]);

  const open = q.length > 0 && results.length > 0;
  return (
    <div style={{ position: 'relative', width: 480 }}>
      <input
        role="combobox" aria-expanded={open} aria-autocomplete="list"
        aria-controls="fl" aria-activedescendant={active >= 0 ? `fo-${active}` : undefined}
        value={q} onChange={e => { setQ(e.target.value); setAct(-1); }}
        onKeyDown={onKey} placeholder="Search..."
        style={{ width: '100%', padding: '10px 14px', borderRadius: 8, border: '1.5px solid #e2e8f0', fontSize: 15 }}
      />
      {open && (
        <ul id="fl" role="listbox" style={{
          position: 'absolute', top: '110%', left: 0, right: 0, background: '#fff',
          border: '1px solid #e2e8f0', borderRadius: 8, boxShadow: '0 8px 24px rgba(0,0,0,.1)',
          listStyle: 'none', margin: 0, padding: 4, zIndex: 50,
        }}>
          {results.map(({ item, matches }, i) => {
            const tm = matches?.find(m => m.key === 'title');
            return (
              <li key={item.id} id={`fo-${i}`} role="option" aria-selected={i === active}
                onMouseDown={() => confirm(item)}
                style={{ padding: '8px 12px', borderRadius: 6, cursor: 'pointer',
                         background: i === active ? '#eff6ff' : 'transparent' }}
              >
                <div style={{ fontWeight: 500, fontSize: 14 }}>
                  {tm ? highlight(item.title, tm.indices as [number, number][]) : item.title}
                </div>
                <div style={{ fontSize: 12, color: '#64748b', marginTop: 2 }}>{item.category}</div>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
```

Use `onMouseDown` not `onClick` on options — `onClick` fires after `onBlur`, which can close the dropdown before the click registers. `threshold: 0.4` balances typo tolerance vs false positives; lower = stricter."""
    },
    {
        "question": "How do I configure Tailwind CSS v4 in Next.js 15 with custom design tokens, define component variants using `cva`, and wire up dark mode via `data-theme`?",
        "thinking": "Tailwind v4 is CSS-first: config moves into @theme {} blocks in your CSS file instead of tailwind.config.js. @variant dark lets you point dark mode at any selector — great for data-theme. cva (class-variance-authority) is the standard for variant-driven component APIs. I'll show the full setup: PostCSS config, globals.css with @theme, dark mode @variant, and a Button with cva + TypeScript.",
        "answer": """**Install**
```bash
npm install tailwindcss@next @tailwindcss/postcss
npm install class-variance-authority clsx tailwind-merge
```

**postcss.config.mjs**
```js
export default { plugins: { '@tailwindcss/postcss': {} } };
```

**app/globals.css**
```css
@import "tailwindcss";

@theme {
  --color-brand-50:  oklch(0.97 0.01 265);
  --color-brand-500: oklch(0.58 0.22 265);
  --color-brand-600: oklch(0.50 0.22 265);
  --color-brand-900: oklch(0.22 0.10 265);

  --font-sans: 'Inter Variable', system-ui, sans-serif;
  --radius-card: 1rem;
}

/* Dark mode triggered by data-theme="dark" on any ancestor */
@variant dark (&:where([data-theme="dark"], [data-theme="dark"] *));
```

**lib/utils.ts**
```ts
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';
export const cn = (...inputs: ClassValue[]) => twMerge(clsx(inputs));
```

**components/Button.tsx**
```tsx
import { cva, type VariantProps } from 'class-variance-authority';
import { cn } from '@/lib/utils';

const btn = cva(
  'inline-flex items-center justify-center font-semibold transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-brand-500 disabled:opacity-50 disabled:pointer-events-none',
  {
    variants: {
      variant: {
        solid:   'bg-brand-500 text-white hover:bg-brand-600',
        outline: 'border-2 border-brand-500 text-brand-500 hover:bg-brand-50 dark:hover:bg-brand-900',
        ghost:   'text-brand-500 hover:bg-brand-50 dark:hover:bg-brand-900',
        danger:  'bg-red-500 text-white hover:bg-red-600',
      },
      size: {
        sm: 'h-8  px-3 rounded-lg  text-sm',
        md: 'h-10 px-4 rounded-xl  text-sm',
        lg: 'h-12 px-6 rounded-xl  text-base',
        icon: 'h-10 w-10 rounded-xl',
      },
    },
    defaultVariants: { variant: 'solid', size: 'md' },
  }
);

export interface ButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement>,
    VariantProps<typeof btn> {}

export function Button({ variant, size, className, ...props }: ButtonProps) {
  return <button className={cn(btn({ variant, size }), className)} {...props} />;
}

// Usage
<Button>Save</Button>
<Button variant="outline" size="lg">Cancel</Button>
<Button variant="danger">Delete account</Button>
<Button variant="ghost" size="icon" aria-label="Settings">⚙️</Button>
```

**Theme toggle (sets cookie for SSR)**
```tsx
'use client';
export function ThemeToggle() {
  const toggle = () => {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    document.cookie = `theme=${next}; path=/; max-age=31536000`;
  };
  return <button onClick={toggle}>Toggle theme</button>;
}
```

In Tailwind v4 `@theme` tokens become CSS custom properties (`--color-brand-500`) **and** utility classes (`bg-brand-500`, `text-brand-500`) automatically — no plugin config needed."""
    },
    {
        "question": "Implement a Next.js 15 Server Action for a contact form — Zod validation on the server, useActionState hook on the client, and progressive enhancement (works without JS).",
        "thinking": "Server Actions run on the server and can be called from a form's action attribute (works without JS) or from useActionState hook (React 19). The pattern: define an async function with 'use server', validate with Zod, return state with errors or success. On the client useActionState manages the pending state. The form should work as a plain HTML form for progressive enhancement.",
        "answer": """```tsx
// app/actions/contact.ts
'use server';
import { z } from 'zod';

const schema = z.object({
  name:    z.string().min(2, 'Name must be at least 2 characters'),
  email:   z.string().email('Invalid email address'),
  message: z.string().min(10, 'Message must be at least 10 characters'),
});

export type ContactState = {
  success: boolean;
  errors?: Record<string, string[]>;
  message?: string;
};

export async function submitContact(
  _prev: ContactState,
  formData: FormData
): Promise<ContactState> {
  const raw = {
    name:    formData.get('name'),
    email:   formData.get('email'),
    message: formData.get('message'),
  };

  const result = schema.safeParse(raw);
  if (!result.success) {
    return { success: false, errors: result.error.flatten().fieldErrors };
  }

  // Send email, save to DB, etc.
  await sendEmail(result.data);

  return { success: true, message: 'Message sent! We'll reply within 24 hours.' };
}

// app/contact/page.tsx
'use client';
import { useActionState } from 'react';
import { submitContact, type ContactState } from '../actions/contact';

const init: ContactState = { success: false };

export default function ContactPage() {
  const [state, action, pending] = useActionState(submitContact, init);

  if (state.success) {
    return <p role="status" style={{ color: 'green' }}>{state.message}</p>;
  }

  return (
    <form action={action} noValidate>
      <div>
        <label htmlFor="name">Name</label>
        <input id="name" name="name" required aria-describedby={state.errors?.name ? 'name-err' : undefined} />
        {state.errors?.name && <p id="name-err" role="alert" style={{ color: 'red' }}>{state.errors.name[0]}</p>}
      </div>
      <div>
        <label htmlFor="email">Email</label>
        <input id="email" name="email" type="email" required />
        {state.errors?.email && <p role="alert" style={{ color: 'red' }}>{state.errors.email[0]}</p>}
      </div>
      <div>
        <label htmlFor="message">Message</label>
        <textarea id="message" name="message" rows={5} required />
        {state.errors?.message && <p role="alert" style={{ color: 'red' }}>{state.errors.message[0]}</p>}
      </div>
      <button type="submit" disabled={pending}>
        {pending ? 'Sending...' : 'Send message'}
      </button>
    </form>
  );
}
```

**Progressive enhancement**: the `action={action}` on the `<form>` means the form submits to the server even without JavaScript — the server action runs directly via POST. `useActionState` enhances this with client-side pending state and no-reload updates when JS is available."""
    },
]

# ─────────────────────────────────────────────
# MORE CSS: Tailwind, Anchor Positioning, Framer
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "Implement CSS anchor positioning to attach a tooltip to a button without JavaScript position calculations — the tooltip stays attached even on scroll.",
        "thinking": "CSS anchor positioning (@position-try, anchor(), position-anchor) is a new native CSS feature that lets an element position itself relative to another element (its anchor) declaratively. The pattern: set anchor-name on the trigger, set position-anchor on the tooltip, then use anchor() function to reference the anchor's edges for positioning. No JS getBoundingClientRect needed. Must handle fallback with @position-try for when it would go off-screen.",
        "answer": """```css
/* The trigger element gets a named anchor */
.tooltip-trigger {
  anchor-name: --my-tooltip;
  position: relative; /* not required, just for context */
}

/* The tooltip positions itself relative to the anchor */
.tooltip {
  position: absolute;    /* or fixed */
  position-anchor: --my-tooltip;

  /* Position below the anchor, horizontally centered */
  top:  anchor(bottom);
  left: anchor(center);
  transform: translateX(-50%);
  margin-top: 8px;

  background: #1e293b;
  color: #fff;
  padding: 6px 12px;
  border-radius: 6px;
  font-size: 13px;
  white-space: nowrap;
  width: max-content;

  /* Hide until shown */
  display: none;

  /* Flip to above if it would overflow bottom */
  @position-try --above {
    top: auto;
    bottom: anchor(top);
    margin-top: 0;
    margin-bottom: 8px;
  }
  position-try-fallbacks: --above;
}

.tooltip-trigger:hover + .tooltip,
.tooltip-trigger:focus + .tooltip {
  display: block;
}
```

```html
<button class="tooltip-trigger" aria-describedby="tt">Save</button>
<div id="tt" class="tooltip" role="tooltip">Ctrl+S</div>
```

```tsx
// React version
function Tooltip({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <span style={{ position: 'relative', display: 'inline-block' }}>
      <span className="tooltip-trigger">{children}</span>
      <div className="tooltip" role="tooltip">{label}</div>
    </span>
  );
}
```

**Browser support** (2025): Chrome/Edge 125+, Safari 18.2+. Not yet in Firefox stable. Add `@supports (anchor-name: --x)` guard and fall back to the classic JS-based approach for Firefox. The key advantage over JS positioning: works during scroll without `scroll` listeners, no layout thrashing, declarative fallback with `@position-try`."""
    },
    {
        "question": "Build a Framer Motion AnimatePresence exit animation for a notification toast stack where each toast flies out to the right when dismissed.",
        "thinking": "AnimatePresence allows components to animate when they're removed from the React tree. Without it, React deletes the DOM node immediately and exit animations can't run. The pattern: wrap the list in AnimatePresence, give each toast a stable key, set exit prop. For a stack of toasts, I also need to handle positioning — each toast should shift up when one below it is removed (layout animation). I'll use motion.div with layout + AnimatePresence.",
        "answer": """```tsx
import { motion, AnimatePresence } from 'framer-motion';
import { useState, useCallback } from 'react';

interface Toast {
  id: string;
  message: string;
  type: 'success' | 'error' | 'info';
}

const colors = { success: '#22c55e', error: '#ef4444', info: '#3b82f6' };

export function useToasts() {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const add = useCallback((message: string, type: Toast['type'] = 'info') => {
    const id = crypto.randomUUID();
    setToasts(p => [...p, { id, message, type }]);
    setTimeout(() => remove(id), 4000); // auto-dismiss
  }, []);
  const remove = useCallback((id: string) => {
    setToasts(p => p.filter(t => t.id !== id));
  }, []);
  return { toasts, add, remove };
}

export function ToastStack() {
  const { toasts, add, remove } = useToasts();
  return (
    <>
      {/* Demo buttons */}
      <button onClick={() => add('Saved!', 'success')}>Success toast</button>
      <button onClick={() => add('Something failed', 'error')}>Error toast</button>

      {/* Portal would be better in production */}
      <div style={{
        position: 'fixed', bottom: 16, right: 16,
        display: 'flex', flexDirection: 'column-reverse', gap: 8, zIndex: 9999,
      }}>
        <AnimatePresence initial={false}>
          {toasts.map(toast => (
            <motion.div
              key={toast.id}
              layout                           // shift position smoothly when stack changes
              initial={{ opacity: 0, x: 60, scale: 0.9 }}
              animate={{ opacity: 1, x: 0,  scale: 1 }}
              exit={{   opacity: 0, x: 120, scale: 0.9 }}
              transition={{ type: 'spring', stiffness: 400, damping: 30 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                background: '#fff', borderLeft: `4px solid ${colors[toast.type]}`,
                borderRadius: 8, padding: '12px 16px',
                boxShadow: '0 4px 16px rgba(0,0,0,0.12)',
                minWidth: 280, maxWidth: 380,
              }}
            >
              <span style={{ flex: 1, fontSize: 14 }}>{toast.message}</span>
              <button
                onClick={() => remove(toast.id)}
                aria-label="Dismiss"
                style={{ background: 'none', border: 'none', cursor: 'pointer', color: '#94a3b8', fontSize: 18 }}
              >
                ✕
              </button>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    </>
  );
}
```

**`layout` prop**: when a toast is removed, the remaining toasts shift smoothly into their new positions via Framer Motion's FLIP technique — no JS position math needed. **`initial={false}` on AnimatePresence**: prevents the initial load of existing toasts from playing the entrance animation."""
    },
    {
        "question": "How do I implement the GSAP Flip plugin in React to animate an element between two different DOM positions — for example, a card that moves from a grid into a modal/expanded state?",
        "thinking": "GSAP Flip captures an element's position/size before a DOM change, then animates from the old state to the new state using FLIP technique. The React pattern: call Flip.getState() before the DOM mutation (i.e., before the state change), trigger the state change (setState, reorder, move DOM), then call Flip.from(state) in a layout effect to animate from captured to current. The tricky part in React is getting the state BEFORE the re-render — use a ref to capture it synchronously.",
        "answer": """```tsx
import { useRef, useState, useLayoutEffect } from 'react';
import gsap from 'gsap';
import { Flip } from 'gsap/Flip';

gsap.registerPlugin(Flip);

interface CardItem { id: string; title: string; img: string; }

export function FlipGallery({ cards }: { cards: CardItem[] }) {
  const [selected, setSelected] = useState<string | null>(null);
  const stateRef = useRef<ReturnType<typeof Flip.getState> | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  const select = (id: string | null) => {
    // 1. Capture BEFORE the DOM changes
    stateRef.current = Flip.getState('.flip-card');
    setSelected(id);
  };

  useLayoutEffect(() => {
    if (!stateRef.current) return;
    // 2. DOM has updated — animate from captured state to current
    Flip.from(stateRef.current, {
      duration: 0.6,
      ease: 'power2.inOut',
      absolute: true,    // pull elements out of flow during animation to prevent jank
      scale: true,       // animate scale, not width/height (compositor-friendly)
      onComplete: () => { stateRef.current = null; },
    });
  });

  return (
    <div ref={containerRef}>
      {/* Expanded card */}
      {selected && (
        <div style={{
          position: 'fixed', inset: 0, background: 'rgba(0,0,0,.5)',
          display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 10,
        }} onClick={() => select(null)}>
          {cards.filter(c => c.id === selected).map(c => (
            <div key={c.id} className="flip-card" data-flip-id={c.id} style={{
              width: 600, background: '#fff', borderRadius: 16, overflow: 'hidden',
            }} onClick={e => e.stopPropagation()}>
              <img src={c.img} style={{ width: '100%', height: 300, objectFit: 'cover' }} />
              <div style={{ padding: 24 }}><h2>{c.title}</h2></div>
            </div>
          ))}
        </div>
      )}

      {/* Grid */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: 16, padding: 24 }}>
        {cards.filter(c => c.id !== selected).map(c => (
          <div key={c.id} className="flip-card" data-flip-id={c.id}
            onClick={() => select(c.id)}
            style={{ borderRadius: 12, overflow: 'hidden', cursor: 'pointer',
                     boxShadow: '0 2px 8px rgba(0,0,0,.1)' }}
          >
            <img src={c.img} style={{ width: '100%', height: 200, objectFit: 'cover' }} />
            <div style={{ padding: 12 }}><h3 style={{ margin: 0 }}>{c.title}</h3></div>
          </div>
        ))}
      </div>
    </div>
  );
}
```

**Key mechanic**: `data-flip-id={c.id}` links the grid card and modal card as the "same" element to GSAP Flip. When the card moves from grid → modal DOM location, Flip animates the positional delta. `useLayoutEffect` (not `useEffect`) fires synchronously after DOM paint — ensuring Flip runs before the browser draws the new state."""
    },
]

# ─────────────────────────────────────────────
# MORE JAVASCRIPT: Abort, IndexedDB, Grouping
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "How do I build cancellable fetch chains using AbortController — cancel a previous request when a new one starts, and handle cleanup in React useEffect?",
        "thinking": "AbortController creates a signal that can abort fetch calls mid-flight. Common patterns: (1) cancel on component unmount — return cleanup from useEffect; (2) cancel the previous call when a new one starts — store the controller in a ref and abort before creating a new one. Promise.race with AbortError handling needs careful filtering — DOMException name==='AbortError' means it was intentionally cancelled, not an error to report.",
        "answer": """```ts
// ── Basic cancellable fetch ────────────────────
async function fetchUser(id: string, signal: AbortSignal) {
  const res = await fetch(`/api/users/${id}`, { signal });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return res.json();
}

// ── React: cancel on unmount ───────────────────
function useUser(id: string) {
  const [data, setData]   = useState<User | null>(null);
  const [error, setError] = useState<Error | null>(null);

  useEffect(() => {
    const ctrl = new AbortController();

    fetchUser(id, ctrl.signal)
      .then(setData)
      .catch(err => {
        if (err.name !== 'AbortError') setError(err); // ignore intentional cancel
      });

    return () => ctrl.abort(); // fires on unmount or id change
  }, [id]);

  return { data, error };
}

// ── Cancel previous request on rapid changes ──
function useSearch(query: string) {
  const [results, setResults] = useState([]);
  const ctrlRef = useRef<AbortController | null>(null);

  useEffect(() => {
    if (!query) return;

    // Cancel in-flight request
    ctrlRef.current?.abort();
    ctrlRef.current = new AbortController();

    fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: ctrlRef.current.signal })
      .then(r => r.json())
      .then(setResults)
      .catch(err => { if (err.name !== 'AbortError') console.error(err); });

    return () => ctrlRef.current?.abort();
  }, [query]);

  return results;
}

// ── Abort after timeout ────────────────────────
async function fetchWithTimeout(url: string, ms = 5000) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(new DOMException('Timeout', 'AbortError')), ms);

  try {
    const res = await fetch(url, { signal: ctrl.signal });
    return await res.json();
  } finally {
    clearTimeout(timer);
  }
}

// ── Combine multiple signals ───────────────────
// AbortSignal.any() — abort when ANY signal fires (not yet universally available)
// Polyfill:
function anySignal(signals: AbortSignal[]): AbortSignal {
  const ctrl = new AbortController();
  for (const sig of signals) {
    if (sig.aborted) { ctrl.abort(sig.reason); break; }
    sig.addEventListener('abort', () => ctrl.abort(sig.reason), { once: true });
  }
  return ctrl.signal;
}
```

`ctrl.abort()` without an argument sets `signal.reason` to a generic `DOMException('signal is aborted without reason')`. Pass a reason for better debugging: `ctrl.abort(new Error('User navigated away'))`."""
    },
    {
        "question": "Use the `idb` library to build a type-safe IndexedDB store for offline-first notes — create, read, update, delete, and list with a cursor.",
        "thinking": "Raw IndexedDB is callback-heavy and hard to type. The `idb` library wraps it in Promises. Pattern: open the DB with a version migration that creates object stores, export typed get/put/delete/getAll helpers. The key is the versioned schema: when you add stores or indexes, bump the version and handle the upgrade callback. For React, wrap in a module-level singleton so the connection is shared.",
        "answer": """```ts
import { openDB, type IDBPDatabase } from 'idb';

interface Note {
  id:        string;
  title:     string;
  content:   string;
  updatedAt: number;
}

interface NoteDB {
  notes: { key: string; value: Note; indexes: { byUpdated: number } };
}

let _db: IDBPDatabase<NoteDB> | null = null;

async function getDB() {
  if (_db) return _db;
  _db = await openDB<NoteDB>('notes-db', 1, {
    upgrade(db) {
      const store = db.createObjectStore('notes', { keyPath: 'id' });
      store.createIndex('byUpdated', 'updatedAt');
    },
  });
  return _db;
}

// ── CRUD helpers ──────────────────────────────
export async function saveNote(note: Note): Promise<void> {
  const db = await getDB();
  await db.put('notes', { ...note, updatedAt: Date.now() });
}

export async function getNote(id: string): Promise<Note | undefined> {
  return (await getDB()).get('notes', id);
}

export async function deleteNote(id: string): Promise<void> {
  await (await getDB()).delete('notes', id);
}

export async function listNotes(): Promise<Note[]> {
  const db = await getDB();
  // Get all notes sorted by updatedAt descending
  const all = await db.getAllFromIndex('notes', 'byUpdated');
  return all.reverse();
}

export async function searchNotes(query: string): Promise<Note[]> {
  const all = await listNotes();
  const q   = query.toLowerCase();
  return all.filter(n =>
    n.title.toLowerCase().includes(q) ||
    n.content.toLowerCase().includes(q)
  );
}

// ── React hook ────────────────────────────────
export function useNotes() {
  const [notes, setNotes] = useState<Note[]>([]);

  const refresh = useCallback(async () => setNotes(await listNotes()), []);

  useEffect(() => { refresh(); }, [refresh]);

  const save = async (note: Note) => { await saveNote(note); refresh(); };
  const remove = async (id: string) => { await deleteNote(id); refresh(); };

  return { notes, save, remove };
}
```

`idb` infers types from the `NoteDB` schema interface — `db.get('notes', id)` returns `Note | undefined`, `db.put('notes', x)` requires a `Note`. Install: `npm install idb`. For sync across tabs, listen to `window` `storage` events or use `BroadcastChannel` to notify other tabs when data changes."""
    },
    {
        "question": "Show how to use `Object.groupBy` and `Map.groupBy` (ES2024) to replace manual reduce-based grouping, and explain the difference between the two.",
        "thinking": "Object.groupBy and Map.groupBy are ES2024 methods that replace the common pattern of reduce({}, ...) for grouping. Object.groupBy creates a plain object keyed by the group name (coerced to string). Map.groupBy creates a Map with the actual key value (preserving type). Map.groupBy is needed when keys are non-string (objects, numbers, Symbols). I should show common use cases: grouping by a string prop, grouping by computed key, and when to use Map vs Object.",
        "answer": """```js
const products = [
  { id: 1, name: 'MacBook',  category: 'laptops', price: 1299 },
  { id: 2, name: 'iPad',     category: 'tablets', price: 799  },
  { id: 3, name: 'ThinkPad', category: 'laptops', price: 999  },
  { id: 4, name: 'Surface',  category: 'tablets', price: 899  },
];

// ── Object.groupBy (string keys) ──────────────
const byCategory = Object.groupBy(products, p => p.category);
// {
//   laptops: [{id:1,...}, {id:3,...}],
//   tablets: [{id:2,...}, {id:4,...}]
// }
byCategory.laptops; // ✅ TypeScript: Product[]

// ── Map.groupBy (any key type) ─────────────────
const byPriceRange = Map.groupBy(products, p =>
  p.price < 900 ? 'budget' : p.price < 1200 ? 'mid' : 'premium'
);
byPriceRange.get('mid'); // [{id:3,name:'ThinkPad',...}]

// ── Grouping by object key (only Map supports) ─
const categories = [
  { id: 'lap', label: 'Laptops' },
  { id: 'tab', label: 'Tablets' },
];
const byCategoryObj = Map.groupBy(products, p =>
  categories.find(c => c.id === p.category.slice(0, 3)) ?? null
);
// Key is the category object itself, not its id string

// ── vs old reduce pattern ─────────────────────
// ❌ Verbose manual pattern replaced by groupBy
const old = products.reduce<Record<string, typeof products>>((acc, p) => {
  (acc[p.category] ??= []).push(p);
  return acc;
}, {});

// ── Practical: group + transform ──────────────
const categorySummary = Object.fromEntries(
  Object.entries(Object.groupBy(products, p => p.category))
    .map(([cat, items]) => [cat, {
      count: items!.length,
      avgPrice: items!.reduce((s, p) => s + p.price, 0) / items!.length,
    }])
);
// { laptops: { count: 2, avgPrice: 1149 }, tablets: { count: 2, avgPrice: 849 } }
```

**Object.groupBy vs Map.groupBy**:
| | `Object.groupBy` | `Map.groupBy` |
|---|---|---|
| Key type | string (coerced) | any value |
| Access | `.laptops` | `.get('laptops')` |
| Ordering | insertion order | insertion order |
| Use case | string props | complex/object keys |

**Browser support**: Chrome 117+, Safari 17.4+, Firefox 119+. Polyfill with `core-js` for older targets."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 5: BUILD TOOLS & PERFORMANCE
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "Write a custom Vite plugin that transforms `.svg` imports into React components automatically, similar to how SVGR works but as a minimal Vite plugin.",
        "thinking": "A Vite plugin implements hooks from Rollup's plugin interface. For SVG-to-React, I need the `transform` hook that fires for each module. Check if the id ends in .svg and the importer wants a component (suffix like ?react or based on config). Read the SVG file, transform it to a React component string, return it. I should also handle the load hook for virtual modules. Key: Vite plugins run in both dev and build, so the transform must be synchronous or return a Promise.",
        "answer": """```ts
// vite-plugin-svgr.ts
import { readFileSync } from 'fs';
import { type Plugin }  from 'vite';
import { optimize }     from 'svgo';

export function svgrPlugin(): Plugin {
  return {
    name: 'vite-plugin-svgr',

    // Transform runs on every resolved module
    async transform(code, id) {
      // Only handle .svg?react imports
      if (!id.endsWith('.svg?react') && !id.endsWith('.svg?component')) return null;

      const filePath = id.replace(/\?(react|component)$/, '');
      const svgRaw   = readFileSync(filePath, 'utf-8');

      // Optimize with SVGO
      const { data: optimized } = optimize(svgRaw, {
        plugins: ['removeViewBox', { name: 'removeAttrs', params: { attrs: '(stroke|fill)' } }],
      });

      // Convert SVG to React component source
      const componentCode = svgToReactComponent(optimized);

      return { code: componentCode, map: null };
    },
  };
}

function svgToReactComponent(svg: string): string {
  // Replace SVG attributes to React equivalents
  const reactSvg = svg
    .replace(/class=/g, 'className=')
    .replace(/for=/g, 'htmlFor=')
    .replace(/(stroke|fill)="currentColor"/g, '$1={$1}')  // make color prop-driven
    // Inject ...props spread into root <svg>
    .replace('<svg ', '<svg {...props} ');

  return `
import React from 'react';
export default function SvgComponent(props) {
  return (${reactSvg});
}
`;
}

// vite.config.ts
import { defineConfig } from 'vite';
import react            from '@vitejs/plugin-react';
import { svgrPlugin }   from './vite-plugin-svgr';

export default defineConfig({
  plugins: [react(), svgrPlugin()],
});

// Usage in React
import Logo from './logo.svg?react';
<Logo className="w-8 h-8" fill="currentColor" aria-label="Logo" />
```

**Plugin hook order**: `resolveId` → `load` → `transform`. If you need to handle the `import Logo from './logo.svg'` form (no suffix), use `resolveId` to rewrite the id and `load` to return the code. The `transform` hook is simpler when you control the import suffix. For production, use `@svgr/rollup` which handles edge cases (titles, IDs, accessibility)."""
    },
    {
        "question": "How do I analyze and fix bundle bloat in a Vite/React app? Walk through using `rollup-plugin-visualizer`, identifying problems, and fixing them with dynamic imports and tree-shaking.",
        "thinking": "Bundle analysis starts with visualization. rollup-plugin-visualizer generates an interactive treemap. Common problems to look for: (1) libraries imported whole instead of tree-shaken (lodash, moment, date-fns); (2) duplicate dependencies from different package versions; (3) large libraries that should be lazy-loaded (chart libs, rich text editors, PDFs); (4) dev-only code leaking into prod bundle. For each, there's a specific fix.",
        "answer": """**Step 1: Visualize the bundle**
```ts
// vite.config.ts
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    react(),
    visualizer({
      filename: 'dist/stats.html',
      open: true,         // auto-open after build
      gzipSize: true,     // show gzip sizes
      brotliSize: true,
      template: 'treemap', // or 'sunburst' | 'network'
    }),
  ],
});
```
```bash
npm run build  # opens stats.html automatically
```

**Common problems and fixes**

**1. Lodash (600KB raw) — import only what you use**
```ts
// ❌ imports entire lodash
import _ from 'lodash';
const sorted = _.sortBy(items, 'name');

// ✅ import specific function (tree-shaken)
import sortBy from 'lodash/sortBy';

// ✅ Even better: use native equivalents
const sorted = [...items].sort((a, b) => a.name.localeCompare(b.name));
```

**2. date-fns — already tree-shakeable, but avoid namespace import**
```ts
// ❌ may import everything in some bundlers
import * as dateFns from 'date-fns';

// ✅
import { format, parseISO } from 'date-fns';
```

**3. Chart library (400KB) — lazy load on demand**
```ts
// ❌ chart lib in main bundle
import { BarChart } from 'recharts';

// ✅ dynamic import — only loads when component renders
const BarChart = lazy(() =>
  import('recharts').then(mod => ({ default: mod.BarChart }))
);
```

**4. Route-based code splitting**
```ts
// router.tsx
const Dashboard = lazy(() => import('./pages/Dashboard'));
const Settings  = lazy(() => import('./pages/Settings'));

// Each page becomes its own chunk
<Suspense fallback={<PageSkeleton />}>
  <Routes>
    <Route path="/dashboard" element={<Dashboard />} />
    <Route path="/settings"  element={<Settings />} />
  </Routes>
</Suspense>
```

**5. Mark packages as sideEffect-free in package.json**
```json
{ "sideEffects": false }
// or list only files with side effects:
{ "sideEffects": ["./src/polyfills.js", "*.css"] }
```

**Key metrics to target**: initial JS < 150KB gzip, total per-route chunk < 50KB. Use `vite-bundle-visualizer` to find the top-10 heaviest modules and tackle them in order."""
    },
    {
        "question": "Set up a Progressive Web App with Vite using Workbox for offline support — precache shell assets, cache API responses with stale-while-revalidate, and show an update prompt when a new SW is installed.",
        "thinking": "Workbox (via vite-plugin-pwa) generates a service worker that handles precaching and runtime caching strategies. Three things to configure: (1) precache the shell (HTML, JS, CSS chunks) so the app loads offline; (2) runtime cache API calls with stale-while-revalidate so users see data immediately while fresh data loads; (3) detect SW updates and prompt users to reload. The update detection uses the Registration API's 'updatefound' event.",
        "answer": """```ts
// vite.config.ts
import { defineConfig } from 'vite';
import react            from '@vitejs/plugin-react';
import { VitePWA }      from 'vite-plugin-pwa';

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'prompt', // don't auto-update; let user decide
      workbox: {
        // Precache shell: all JS/CSS/HTML chunks Vite generates
        globPatterns: ['**/*.{js,css,html,ico,png,woff2}'],
        // Runtime caching strategies
        runtimeCaching: [
          {
            urlPattern: /^https:\/\/api\.myapp\.com\//,
            handler: 'StaleWhileRevalidate', // serve cache, update in background
            options: {
              cacheName: 'api-cache',
              expiration: { maxEntries: 100, maxAgeSeconds: 60 * 60 * 24 },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
          {
            urlPattern: /^https:\/\/fonts\.googleapis\.com\//,
            handler: 'CacheFirst',   // fonts rarely change
            options: { cacheName: 'google-fonts', expiration: { maxAgeSeconds: 60 * 60 * 24 * 365 } },
          },
        ],
      },
      manifest: {
        name: 'My App',
        short_name: 'MyApp',
        theme_color: '#0066cc',
        icons: [
          { src: '/icon-192.png', sizes: '192x192', type: 'image/png' },
          { src: '/icon-512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
    }),
  ],
});
```

```tsx
// components/UpdatePrompt.tsx
import { useRegisterSW } from 'virtual:pwa-register/react';

export function UpdatePrompt() {
  const { needRefresh: [needs], updateServiceWorker } = useRegisterSW({
    onRegistered(r) { r && setInterval(() => r.update(), 60 * 60 * 1000); }, // check hourly
  });

  if (!needs) return null;

  return (
    <div style={{
      position: 'fixed', bottom: 16, left: '50%', transform: 'translateX(-50%)',
      background: '#1e293b', color: '#fff', padding: '12px 20px',
      borderRadius: 10, display: 'flex', gap: 12, alignItems: 'center',
      boxShadow: '0 4px 20px rgba(0,0,0,.3)', zIndex: 9999,
    }}>
      <span>A new version is available</span>
      <button
        onClick={() => updateServiceWorker(true)}
        style={{ background: '#0066cc', color: '#fff', border: 'none',
                 padding: '6px 16px', borderRadius: 6, cursor: 'pointer' }}
      >
        Update
      </button>
    </div>
  );
}
```

**`StaleWhileRevalidate`**: serves cached response immediately (no wait), then fetches fresh data and updates the cache for next time. Users see stale data briefly but the UX feels instant. Use `CacheFirst` only for truly static assets (fonts, immutable images). Never use `CacheFirst` for API data that changes."""
    },
    {
        "question": "Set up a Turborepo monorepo with a shared `ui` package, a Next.js app, and a Vite library — configure TypeScript project references, shared ESLint config, and run dev for both apps in parallel.",
        "thinking": "Turborepo orchestrates tasks across packages using a dependency graph defined in turbo.json. Key parts: (1) workspace setup with pnpm/npm workspaces pointing to apps/* and packages/*; (2) a packages/ui with its own package.json that exports components; (3) TypeScript project references so the app TS server doesn't re-compile the ui package; (4) turbo.json pipeline that defines task dependencies so build runs ui before apps.",
        "answer": """**Directory structure**
```
my-monorepo/
├── apps/
│   ├── web/           (Next.js 15)
│   └── docs/          (Vite React)
├── packages/
│   ├── ui/            (shared components)
│   ├── eslint-config/ (shared ESLint)
│   └── tsconfig/      (shared TS configs)
├── turbo.json
└── pnpm-workspace.yaml
```

**pnpm-workspace.yaml**
```yaml
packages:
  - "apps/*"
  - "packages/*"
```

**turbo.json**
```json
{
  "$schema": "https://turbo.build/schema.json",
  "tasks": {
    "build": {
      "dependsOn": ["^build"],   // build deps before self
      "outputs": [".next/**", "dist/**"]
    },
    "dev": {
      "cache": false,
      "persistent": true
    },
    "lint": { "dependsOn": ["^build"] },
    "typecheck": { "dependsOn": ["^build"] }
  }
}
```

**packages/ui/package.json**
```json
{
  "name": "@repo/ui",
  "exports": {
    ".": { "import": "./src/index.ts", "types": "./src/index.ts" }
  },
  "devDependencies": { "typescript": "^5.5", "react": "^19" },
  "peerDependencies": { "react": "^19" }
}
```

**packages/ui/src/index.ts**
```ts
export { Button }  from './Button';
export { Card }    from './Card';
export type { ButtonProps } from './Button';
```

**apps/web/package.json** (consumer)
```json
{
  "dependencies": { "@repo/ui": "workspace:*" }
}
```

**Shared tsconfig** (packages/tsconfig/base.json)
```json
{
  "compilerOptions": {
    "strict": true, "moduleResolution": "bundler",
    "jsx": "react-jsx", "esModuleInterop": true
  }
}
```

**Run all apps in parallel**
```bash
# Dev (persistent, runs both apps simultaneously)
pnpm turbo dev

# Build (builds ui first, then apps in parallel)
pnpm turbo build

# Lint only changed packages since main
pnpm turbo lint --filter="...[origin/main]"
```

**Key advantage**: Turborepo caches task outputs. If `packages/ui` hasn't changed, `turbo build` uses the cached output instead of rebuilding it — dramatically faster CI on large monorepos."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 6: WEB APIs, TOOL CALLING & AI
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "Build a streaming AI chat interface using the Anthropic Claude API with tool calling — the assistant can call a `get_weather` tool and display intermediate tool use steps in the UI.",
        "thinking": "The Anthropic SDK supports streaming with tool use. The pattern: send a message, stream the response, handle text_delta events for text chunks, input_json_delta for tool input accumulation, and tool_use events to show what tool is being called. On the frontend, use Server-Sent Events or a streaming fetch. I need to handle the multi-turn pattern: stream → stop_reason=tool_use → process tool results → send results back → stream again.",
        "answer": """```ts
// server/chat.ts (Next.js App Router or Node server)
import Anthropic from '@anthropic-ai/sdk';

const claude = new Anthropic(); // reads ANTHROPIC_API_KEY from env

const tools: Anthropic.Tool[] = [
  {
    name: 'get_weather',
    description: 'Get current weather for a city',
    input_schema: {
      type: 'object',
      properties: {
        city:    { type: 'string', description: 'City name' },
        country: { type: 'string', description: 'ISO country code' },
      },
      required: ['city'],
    },
  },
];

async function getWeather(city: string, country?: string) {
  // Real impl would call a weather API
  return { temp: 22, condition: 'Sunny', city, country };
}

// POST /api/chat — returns streaming SSE
export async function POST(req: Request) {
  const { messages }: { messages: Anthropic.MessageParam[] } = await req.json();
  const encoder = new TextEncoder();

  const stream = new ReadableStream({
    async start(controller) {
      const send = (data: object) =>
        controller.enqueue(encoder.encode(`data: ${JSON.stringify(data)}\n\n`));

      let msgs = [...messages];

      // Agentic loop — may take multiple turns if tools are used
      while (true) {
        const response = await claude.messages.create({
          model: 'claude-sonnet-4-6',
          max_tokens: 1024,
          tools,
          messages: msgs,
          stream: true,
        });

        let fullText = '';
        const toolUses: Anthropic.ToolUseBlock[] = [];

        for await (const event of response) {
          if (event.type === 'content_block_delta') {
            if (event.delta.type === 'text_delta') {
              fullText += event.delta.text;
              send({ type: 'text', delta: event.delta.text });
            }
            if (event.delta.type === 'input_json_delta') {
              send({ type: 'tool_input', delta: event.delta.partial_json });
            }
          }
          if (event.type === 'content_block_start' && event.content_block.type === 'tool_use') {
            send({ type: 'tool_start', tool: event.content_block.name, id: event.content_block.id });
          }
          if (event.type === 'message_stop') {
            const final = await response.finalMessage();
            if (final.stop_reason !== 'tool_use') {
              controller.close();
              return; // done
            }
            // Collect tool use blocks
            for (const block of final.content) {
              if (block.type === 'tool_use') toolUses.push(block);
            }
          }
        }

        // Process tool calls
        const toolResults: Anthropic.ToolResultBlockParam[] = [];
        for (const tu of toolUses) {
          const input = tu.input as Record<string, string>;
          const result = tu.name === 'get_weather'
            ? await getWeather(input.city, input.country)
            : { error: 'Unknown tool' };
          send({ type: 'tool_result', tool: tu.name, result });
          toolResults.push({ type: 'tool_result', tool_use_id: tu.id, content: JSON.stringify(result) });
        }

        // Continue conversation with tool results
        msgs = [...msgs,
          { role: 'assistant', content: toolUses.map(tu => ({ type: 'tool_use' as const, ...tu })) },
          { role: 'user',      content: toolResults },
        ];
      }
    },
  });

  return new Response(stream, { headers: { 'Content-Type': 'text/event-stream', 'Cache-Control': 'no-cache' } });
}
```

```tsx
// client: ChatUI.tsx
'use client';
import { useState, useRef } from 'react';

interface ChatEvent {
  type: 'text' | 'tool_start' | 'tool_input' | 'tool_result';
  delta?: string; tool?: string; result?: object;
}

export function ChatUI() {
  const [messages, setMessages] = useState<{ role: string; content: string }[]>([]);
  const [events, setEvents]     = useState<ChatEvent[]>([]);
  const [input, setInput]       = useState('');
  const [streaming, setStream]  = useState(false);

  const send = async () => {
    if (!input.trim() || streaming) return;
    const userMsg = { role: 'user', content: input };
    setMessages(m => [...m, userMsg]);
    setInput('');
    setStream(true);
    setEvents([]);

    const res = await fetch('/api/chat', {
      method: 'POST',
      body: JSON.stringify({ messages: [...messages, userMsg] }),
      headers: { 'Content-Type': 'application/json' },
    });

    const reader = res.body!.getReader();
    const dec    = new TextDecoder();
    let text     = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      const chunk = dec.decode(value);
      for (const line of chunk.split('\n')) {
        if (!line.startsWith('data: ')) continue;
        const event: ChatEvent = JSON.parse(line.slice(6));
        setEvents(e => [...e, event]);
        if (event.type === 'text') text += event.delta ?? '';
      }
    }

    setMessages(m => [...m, { role: 'assistant', content: text }]);
    setStream(false);
  };

  return (
    <div style={{ maxWidth: 680, margin: '0 auto', padding: 24 }}>
      {messages.map((m, i) => (
        <div key={i} style={{ marginBottom: 12, textAlign: m.role === 'user' ? 'right' : 'left' }}>
          <span style={{ background: m.role === 'user' ? '#0066cc' : '#f1f5f9',
            color: m.role === 'user' ? '#fff' : '#000',
            padding: '8px 14px', borderRadius: 12, display: 'inline-block' }}>
            {m.content}
          </span>
        </div>
      ))}
      {streaming && events.map((e, i) => (
        <div key={i} style={{ fontSize: 12, color: '#64748b', marginBottom: 4 }}>
          {e.type === 'tool_start' && `🔧 Calling ${e.tool}...`}
          {e.type === 'tool_result' && `✅ Got result: ${JSON.stringify(e.result)}`}
          {e.type === 'text' && e.delta}
        </div>
      ))}
      <div style={{ display: 'flex', gap: 8, marginTop: 16 }}>
        <input value={input} onChange={e => setInput(e.target.value)}
          onKeyDown={e => e.key === 'Enter' && send()}
          placeholder="Ask about weather..."
          style={{ flex: 1, padding: '10px 14px', borderRadius: 8, border: '1.5px solid #e2e8f0' }} />
        <button onClick={send} disabled={streaming} style={{ padding: '10px 20px', borderRadius: 8,
          background: '#0066cc', color: '#fff', border: 'none', cursor: 'pointer' }}>
          Send
        </button>
      </div>
    </div>
  );
}
```

The agentic loop continues until `stop_reason !== 'tool_use'` — Claude may call multiple tools in sequence. Tool results are sent back as `user` role messages with `type: 'tool_result'`; Claude sees the results and either calls another tool or writes its final answer."""
    },
    {
        "question": "Implement a WebSocket client in React with automatic exponential backoff reconnection, connection status indicator, and message queue that flushes on reconnect.",
        "thinking": "A robust WebSocket client needs: exponential backoff (start at 1s, double each attempt, cap at 30s) with jitter to prevent thundering herd; a message queue for messages sent while disconnected; cleanup on unmount to prevent memory leaks; and a connection state machine (connecting, connected, disconnected, reconnecting). The retry logic should be in a ref to survive re-renders without resetting state.",
        "answer": """```tsx
import { useEffect, useRef, useCallback, useState } from 'react';

type ConnState = 'connecting' | 'connected' | 'reconnecting' | 'disconnected';

interface UseWebSocketOptions {
  onMessage?: (data: unknown) => void;
  onConnect?:  () => void;
  maxRetries?: number;
}

export function useWebSocket(url: string, opts: UseWebSocketOptions = {}) {
  const [state, setState] = useState<ConnState>('connecting');
  const wsRef       = useRef<WebSocket | null>(null);
  const queueRef    = useRef<string[]>([]);
  const retriesRef  = useRef(0);
  const timerRef    = useRef<ReturnType<typeof setTimeout>>();
  const unmountedRef = useRef(false);

  const connect = useCallback(() => {
    if (unmountedRef.current) return;
    setState(retriesRef.current > 0 ? 'reconnecting' : 'connecting');

    const ws = new WebSocket(url);
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmountedRef.current) { ws.close(); return; }
      retriesRef.current = 0;
      setState('connected');
      opts.onConnect?.();
      // Flush queued messages
      queueRef.current.forEach(msg => ws.send(msg));
      queueRef.current = [];
    };

    ws.onmessage = (e) => {
      try { opts.onMessage?.(JSON.parse(e.data)); }
      catch { opts.onMessage?.(e.data); }
    };

    ws.onclose = (e) => {
      if (unmountedRef.current || e.wasClean) {
        setState('disconnected'); return;
      }
      const maxRetries = opts.maxRetries ?? 10;
      if (retriesRef.current >= maxRetries) { setState('disconnected'); return; }

      // Exponential backoff with jitter: 1s, 2s, 4s, 8s… capped at 30s
      const base  = Math.min(1000 * 2 ** retriesRef.current, 30_000);
      const jitter = Math.random() * 1000;
      retriesRef.current++;

      timerRef.current = setTimeout(connect, base + jitter);
    };

    ws.onerror = () => ws.close(); // trigger onclose handler
  }, [url, opts.onMessage, opts.onConnect, opts.maxRetries]);

  const send = useCallback((data: unknown) => {
    const msg = typeof data === 'string' ? data : JSON.stringify(data);
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(msg);
    } else {
      queueRef.current.push(msg); // queue for later
    }
  }, []);

  useEffect(() => {
    unmountedRef.current = false;
    connect();
    return () => {
      unmountedRef.current = true;
      clearTimeout(timerRef.current);
      wsRef.current?.close(1000, 'Component unmounted');
    };
  }, [connect]);

  return { state, send };
}

// ── Usage ──────────────────────────────────────
const STATUS_COLOR: Record<ConnState, string> = {
  connected: '#22c55e', connecting: '#eab308',
  reconnecting: '#f97316', disconnected: '#ef4444',
};

function LiveFeed() {
  const [messages, setMessages] = useState<string[]>([]);
  const { state, send } = useWebSocket('wss://api.example.com/ws', {
    onMessage: (data) => setMessages(m => [String(data), ...m].slice(0, 100)),
  });

  return (
    <div>
      <span style={{ display: 'inline-flex', alignItems: 'center', gap: 6 }}>
        <span style={{ width: 8, height: 8, borderRadius: '50%', background: STATUS_COLOR[state] }} />
        {state}
      </span>
      <button onClick={() => send({ type: 'ping' })}>Ping</button>
      <ul>{messages.map((m, i) => <li key={i}>{m}</li>)}</ul>
    </div>
  );
}
```

The message queue ensures no messages are lost during reconnection. Jitter (`Math.random() * 1000`) prevents multiple clients from reconnecting simultaneously — "thundering herd" on a restarted server."""
    },
    {
        "question": "Implement Server-Sent Events (SSE) for real-time streaming in Next.js — a long-running AI generation that streams tokens to the browser as they're produced.",
        "thinking": "SSE is one-directional (server → client), text-based, and automatically reconnects. It's ideal for streaming AI responses. On the server: set Content-Type: text/event-stream, write 'data: ...\n\n' chunks, flush immediately. On the client: EventSource API or a custom fetch-based reader for POST requests (EventSource only supports GET). For Next.js App Router, return a ReadableStream as Response.",
        "answer": """```ts
// app/api/generate/route.ts
import Anthropic from '@anthropic-ai/sdk';

const client = new Anthropic();

export async function POST(req: Request) {
  const { prompt }: { prompt: string } = await req.json();
  const enc = new TextEncoder();

  const stream = new ReadableStream({
    async start(ctrl) {
      const write = (data: object) =>
        ctrl.enqueue(enc.encode(`data: ${JSON.stringify(data)}\n\n`));

      try {
        const response = await client.messages.create({
          model: 'claude-sonnet-4-6',
          max_tokens: 2048,
          messages: [{ role: 'user', content: prompt }],
          stream: true,
        });

        for await (const event of response) {
          if (event.type === 'content_block_delta' && event.delta.type === 'text_delta') {
            write({ type: 'delta', text: event.delta.text });
          }
          if (event.type === 'message_stop') {
            write({ type: 'done' });
          }
        }
      } catch (err) {
        write({ type: 'error', message: String(err) });
      } finally {
        ctrl.close();
      }
    },
  });

  return new Response(stream, {
    headers: {
      'Content-Type':  'text/event-stream',
      'Cache-Control': 'no-cache, no-transform',
      'Connection':    'keep-alive',
    },
  });
}
```

```tsx
// components/StreamingOutput.tsx
'use client';
import { useState } from 'react';

export function StreamingOutput() {
  const [text, setText]     = useState('');
  const [loading, setLoad]  = useState(false);

  const generate = async (prompt: string) => {
    setText('');
    setLoad(true);

    const res = await fetch('/api/generate', {
      method: 'POST',
      body: JSON.stringify({ prompt }),
      headers: { 'Content-Type': 'application/json' },
    });

    const reader = res.body!.getReader();
    const dec    = new TextDecoder();
    let buffer   = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += dec.decode(value, { stream: true });
      const lines = buffer.split('\n\n');
      buffer = lines.pop() ?? '';   // keep incomplete line in buffer

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const event = JSON.parse(line.slice(6));
        if (event.type === 'delta') setText(t => t + event.text);
        if (event.type === 'done')  setLoad(false);
        if (event.type === 'error') { console.error(event.message); setLoad(false); }
      }
    }
  };

  return (
    <div>
      <button onClick={() => generate('Write a haiku about TypeScript')}>
        {loading ? 'Generating...' : 'Generate'}
      </button>
      <pre style={{
        whiteSpace: 'pre-wrap', fontFamily: 'inherit',
        background: '#f8fafc', padding: 16, borderRadius: 8, marginTop: 12,
        minHeight: 80,
      }}>
        {text}{loading && <span style={{ animation: 'blink 1s step-end infinite' }}>▋</span>}
      </pre>
    </div>
  );
}
```

**Why not native `EventSource`?** It only supports GET requests. For POST-based AI APIs, use `fetch` + `ReadableStream` as shown. The buffer logic handles the case where a chunk boundary falls in the middle of a `data:` line — always accumulate and split on `\n\n`."""
    },
    {
        "question": "Build a generic type-safe API client using the Anthropic SDK to call Claude with structured output — use `zod` to parse the response into a typed object and retry on schema validation failures.",
        "thinking": "Structured output with Claude means asking Claude to return JSON, then validating it with Zod. If the JSON doesn't match the schema (e.g. Claude hallucinated a field), retry with the validation error in the next message. I'll build a generic wrapper that takes a Zod schema, builds the appropriate system prompt instructing Claude to respond in that format, and retries up to 3 times.",
        "answer": """```ts
import Anthropic from '@anthropic-ai/sdk';
import { z, type ZodSchema } from 'zod';

const claude = new Anthropic();

interface StructuredCallOptions<T> {
  schema:       ZodSchema<T>;
  prompt:       string;
  systemPrompt?: string;
  model?:        string;
  maxRetries?:   number;
}

export async function callClaude<T>(opts: StructuredCallOptions<T>): Promise<T> {
  const {
    schema, prompt, maxRetries = 3,
    model = 'claude-sonnet-4-6',
  } = opts;

  const schemaDescription = JSON.stringify(
    (schema as z.ZodObject<z.ZodRawShape>).shape
      ? Object.entries((schema as z.ZodObject<z.ZodRawShape>).shape)
          .reduce<Record<string, string>>((acc, [k, v]) => {
            acc[k] = (v as z.ZodTypeAny).description ?? (v as z.ZodTypeAny)._def.typeName;
            return acc;
          }, {})
      : schema.description,
    null, 2
  );

  const system = opts.systemPrompt ??
    `You are a helpful assistant. Always respond with valid JSON matching this schema:\n${schemaDescription}\n\nRespond with ONLY the JSON object, no explanation.`;

  const messages: Anthropic.MessageParam[] = [
    { role: 'user', content: prompt },
  ];

  for (let attempt = 0; attempt < maxRetries; attempt++) {
    const response = await claude.messages.create({
      model, max_tokens: 1024, system, messages,
    });

    const raw = response.content[0].type === 'text' ? response.content[0].text : '';

    // Extract JSON from possible markdown code fence
    const jsonMatch = raw.match(/```(?:json)?\n?([\s\S]*?)\n?```/) ?? [null, raw];
    const jsonStr   = jsonMatch[1]?.trim() ?? raw.trim();

    try {
      const parsed = JSON.parse(jsonStr);
      return schema.parse(parsed); // throws ZodError if invalid
    } catch (err) {
      if (attempt === maxRetries - 1) throw err;

      // Ask Claude to fix its output
      messages.push(
        { role: 'assistant', content: raw },
        { role: 'user',      content: `Your response failed validation: ${err}. Please correct it and respond with valid JSON only.` }
      );
    }
  }

  throw new Error('Max retries exceeded');
}

// ── Usage ──────────────────────────────────────
const ProductSchema = z.object({
  name:        z.string().describe('Product name'),
  price:       z.number().describe('Price in USD'),
  categories:  z.array(z.string()).describe('Product categories'),
  inStock:     z.boolean().describe('Whether product is in stock'),
  description: z.string().max(200).describe('Short product description'),
});

type Product = z.infer<typeof ProductSchema>;

async function extractProduct(userInput: string): Promise<Product> {
  return callClaude({
    schema: ProductSchema,
    prompt: `Extract product information from this text: "${userInput}"`,
  });
}

// Fully typed result
const product = await extractProduct('Nike Air Max 90, $130, running shoes, available in size 10');
console.log(product.name);   // string ✅
console.log(product.price);  // number ✅
console.log(product.inStock); // boolean ✅
```

The retry loop feeds the validation error back to Claude so it can self-correct. For production: add exponential backoff between retries and log validation failures for prompt improvement. The Anthropic SDK also supports `response_format` in newer API versions for guaranteed JSON output."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 7: FRONTEND ARCHITECTURE
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "Design a Zustand store with the slice pattern for a large e-commerce app — separate slices for cart, user, and UI state, combined into one store with TypeScript types.",
        "thinking": "Zustand's slice pattern splits a large store into separate files, each defining its own state shape and actions. They're combined in a root store file using a TypeScript intersection type. The key: each slice factory receives the full set/get from Zustand so slices can read/update other slices' state. Use immer middleware for nested state mutations without spreading.",
        "answer": """```ts
// store/cartSlice.ts
import type { StateCreator } from 'zustand';
import type { RootStore } from './store';

interface CartItem { id: string; name: string; price: number; qty: number; }

export interface CartSlice {
  cart: {
    items:    CartItem[];
    total:    number;
    addItem:  (item: Omit<CartItem, 'qty'>) => void;
    removeItem: (id: string) => void;
    updateQty:  (id: string, qty: number) => void;
    clear:    () => void;
  };
}

export const createCartSlice: StateCreator<RootStore, [['zustand/immer', never]], [], CartSlice> = (set) => ({
  cart: {
    items: [],
    total: 0,
    addItem: (item) => set(state => {
      const existing = state.cart.items.find(i => i.id === item.id);
      if (existing) {
        existing.qty += 1;
      } else {
        state.cart.items.push({ ...item, qty: 1 });
      }
      state.cart.total = state.cart.items.reduce((s, i) => s + i.price * i.qty, 0);
    }),
    removeItem: (id) => set(state => {
      state.cart.items = state.cart.items.filter(i => i.id !== id);
      state.cart.total = state.cart.items.reduce((s, i) => s + i.price * i.qty, 0);
    }),
    updateQty: (id, qty) => set(state => {
      const item = state.cart.items.find(i => i.id === id);
      if (item) { item.qty = qty; }
      state.cart.total = state.cart.items.reduce((s, i) => s + i.price * i.qty, 0);
    }),
    clear: () => set(state => { state.cart.items = []; state.cart.total = 0; }),
  },
});

// store/userSlice.ts
export interface UserSlice {
  user: {
    profile: { id: string; name: string; email: string } | null;
    loading: boolean;
    login:  (email: string, password: string) => Promise<void>;
    logout: () => void;
  };
}

export const createUserSlice: StateCreator<RootStore, [['zustand/immer', never]], [], UserSlice> = (set) => ({
  user: {
    profile: null,
    loading: false,
    login: async (email, password) => {
      set(s => { s.user.loading = true; });
      const profile = await authApi.login(email, password);
      set(s => { s.user.profile = profile; s.user.loading = false; });
    },
    logout: () => set(s => {
      s.user.profile = null;
      s.cart.clear(); // cross-slice: clear cart on logout
    }),
  },
});

// store/store.ts
import { create } from 'zustand';
import { immer } from 'zustand/middleware/immer';
import { devtools, persist } from 'zustand/middleware';
import { createCartSlice, type CartSlice } from './cartSlice';
import { createUserSlice, type UserSlice } from './userSlice';

export type RootStore = CartSlice & UserSlice;

export const useStore = create<RootStore>()(
  devtools(
    persist(
      immer((...a) => ({
        ...createCartSlice(...a),
        ...createUserSlice(...a),
      })),
      {
        name: 'app-store',
        partialize: (s) => ({ cart: { items: s.cart.items } }), // only persist cart items
      }
    )
  )
);

// ── Usage with selectors (avoid re-renders) ────
const cartTotal = useStore(s => s.cart.total);
const cartCount = useStore(s => s.cart.items.reduce((n, i) => n + i.qty, 0));
const { addItem } = useStore(s => s.cart);
```

Cross-slice access (logout clearing the cart) works because both slices share the same `set`/`get`. `partialize` in persist middleware lets you save only cart items to localStorage, excluding loading states and functions."""
    },
    {
        "question": "Implement a type-safe form with `react-hook-form` and `zod` validation — nested fields, dynamic field arrays, conditional validation, and server error injection.",
        "thinking": "react-hook-form + zod is the standard pairing: zodResolver bridges them. Key patterns: nested fields use dot-notation ('address.city'), dynamic arrays use useFieldArray, conditional validation uses z.discriminatedUnion or .superRefine, and server errors are injected with setError after a failed submission. TypeScript: infer the form type from the schema with z.infer.",
        "answer": """```tsx
import { useForm, useFieldArray, SubmitHandler } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';

const schema = z.object({
  name:     z.string().min(2, 'Name required'),
  email:    z.string().email('Invalid email'),
  role:     z.enum(['user', 'admin']),
  address: z.object({
    street: z.string().min(1, 'Street required'),
    city:   z.string().min(1, 'City required'),
  }),
  // Dynamic array of contacts
  contacts: z.array(z.object({
    name:  z.string().min(1, 'Required'),
    email: z.string().email('Invalid'),
  })).min(1, 'At least one contact required'),
}).superRefine((data, ctx) => {
  // Conditional: admins must have @company.com email
  if (data.role === 'admin' && !data.email.endsWith('@company.com')) {
    ctx.addIssue({ code: 'custom', path: ['email'], message: 'Admin must use @company.com email' });
  }
});

type FormData = z.infer<typeof schema>;

async function createUser(data: FormData): Promise<{ errors?: Record<string, string> }> {
  const res = await fetch('/api/users', { method: 'POST', body: JSON.stringify(data) });
  return res.json();
}

export function UserForm() {
  const { register, control, handleSubmit, setError, watch,
          formState: { errors, isSubmitting } } = useForm<FormData>({
    resolver: zodResolver(schema),
    defaultValues: { role: 'user', contacts: [{ name: '', email: '' }] },
  });

  const { fields, append, remove } = useFieldArray({ control, name: 'contacts' });
  const role = watch('role');

  const onSubmit: SubmitHandler<FormData> = async (data) => {
    const result = await createUser(data);
    if (result.errors) {
      // Inject server-side errors into specific fields
      Object.entries(result.errors).forEach(([field, msg]) =>
        setError(field as keyof FormData, { message: msg })
      );
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate>
      <div>
        <label>Name</label>
        <input {...register('name')} aria-invalid={!!errors.name} />
        {errors.name && <p role="alert">{errors.name.message}</p>}
      </div>

      <div>
        <label>Email {role === 'admin' && '(must be @company.com)'}</label>
        <input {...register('email')} type="email" />
        {errors.email && <p role="alert">{errors.email.message}</p>}
      </div>

      <div>
        <label>Role</label>
        <select {...register('role')}>
          <option value="user">User</option>
          <option value="admin">Admin</option>
        </select>
      </div>

      {/* Nested fields */}
      <fieldset>
        <legend>Address</legend>
        <input {...register('address.street')} placeholder="Street" />
        {errors.address?.street && <p role="alert">{errors.address.street.message}</p>}
        <input {...register('address.city')} placeholder="City" />
        {errors.address?.city && <p role="alert">{errors.address.city.message}</p>}
      </fieldset>

      {/* Dynamic field array */}
      <fieldset>
        <legend>Contacts</legend>
        {fields.map((field, i) => (
          <div key={field.id} style={{ display: 'flex', gap: 8, marginBottom: 8 }}>
            <input {...register(`contacts.${i}.name`)}  placeholder="Name"  />
            <input {...register(`contacts.${i}.email`)} placeholder="Email" />
            <button type="button" onClick={() => remove(i)}>✕</button>
            {errors.contacts?.[i]?.email && <p role="alert">{errors.contacts[i]?.email?.message}</p>}
          </div>
        ))}
        <button type="button" onClick={() => append({ name: '', email: '' })}>+ Add contact</button>
        {errors.contacts?.root && <p role="alert">{errors.contacts.root.message}</p>}
      </fieldset>

      <button type="submit" disabled={isSubmitting}>
        {isSubmitting ? 'Saving...' : 'Save'}
      </button>
    </form>
  );
}
```

`setError(field, { message })` after a failed server call marks fields as invalid and triggers the same error display UI. `noValidate` on the form suppresses browser built-in validation so Zod's messages are the only ones shown."""
    },
    {
        "question": "Build a feature flag system in React without a third-party SDK — flags defined in a JSON config, evaluated server-side per user, and consumed via a React hook with TypeScript types.",
        "thinking": "Feature flags need: a store of flag definitions (name, enabled, rollout %, user overrides), an evaluation function that checks if a given user sees the flag, and a React hook for consuming flags in components. Server-side evaluation is important — the client should receive already-evaluated flags, not the raw config. I'll show: the flag config shape, server evaluation in Next.js, and the useFlag hook.",
        "answer": """```ts
// lib/flags/types.ts
export interface FlagDefinition {
  name:         string;
  enabled:      boolean;
  rolloutPct?:  number;   // 0-100, deterministic per userId
  allowedUsers?: string[]; // specific user IDs that always get the flag
  blockedUsers?: string[]; // users who never get the flag
}

export type EvaluatedFlags = Record<string, boolean>;

// lib/flags/evaluate.ts
import { createHash } from 'crypto';

function hashRollout(userId: string, flagName: string): number {
  const hash = createHash('sha256')
    .update(`${userId}:${flagName}`)
    .digest('hex')
    .slice(0, 8);
  return (parseInt(hash, 16) / 0xffffffff) * 100; // 0-100
}

export function evaluateFlags(
  flags: FlagDefinition[],
  userId: string
): EvaluatedFlags {
  return Object.fromEntries(
    flags.map(flag => {
      if (!flag.enabled)                        return [flag.name, false];
      if (flag.blockedUsers?.includes(userId))  return [flag.name, false];
      if (flag.allowedUsers?.includes(userId))  return [flag.name, true];
      if (flag.rolloutPct !== undefined) {
        return [flag.name, hashRollout(userId, flag.name) < flag.rolloutPct];
      }
      return [flag.name, true];
    })
  );
}

// flags.json (could come from DB or CMS)
const FLAGS: FlagDefinition[] = [
  { name: 'new_checkout',   enabled: true, rolloutPct: 20 },
  { name: 'ai_suggestions', enabled: true, allowedUsers: ['user_123', 'user_456'] },
  { name: 'dark_mode_v2',   enabled: false },
];

// app/layout.tsx — evaluate on server, pass to client
import { cookies } from 'next/headers';
import { evaluateFlags } from '@/lib/flags/evaluate';

export default async function Layout({ children }) {
  const userId = (await cookies()).get('userId')?.value ?? 'anonymous';
  const flags  = evaluateFlags(FLAGS, userId);
  return (
    <html>
      <body>
        <FlagProvider flags={flags}>{children}</FlagProvider>
      </body>
    </html>
  );
}

// providers/FlagProvider.tsx
'use client';
import { createContext, useContext } from 'react';
import type { EvaluatedFlags } from '@/lib/flags/types';

const FlagCtx = createContext<EvaluatedFlags>({});

export function FlagProvider({ flags, children }: { flags: EvaluatedFlags; children: React.ReactNode }) {
  return <FlagCtx.Provider value={flags}>{children}</FlagCtx.Provider>;
}

export function useFlag(name: string): boolean {
  return useContext(FlagCtx)[name] ?? false;
}

export function useFlags(): EvaluatedFlags {
  return useContext(FlagCtx);
}

// ── Usage ──────────────────────────────────────
function CheckoutButton() {
  const newCheckout = useFlag('new_checkout');
  return newCheckout ? <NewCheckoutFlow /> : <LegacyCheckout />;
}

function FeatureGate({ flag, children, fallback = null }: {
  flag: string; children: React.ReactNode; fallback?: React.ReactNode;
}) {
  const enabled = useFlag(flag);
  return enabled ? <>{children}</> : <>{fallback}</>;
}

<FeatureGate flag="ai_suggestions" fallback={<StandardSearch />}>
  <AiSearch />
</FeatureGate>
```

The `hashRollout` function is deterministic — the same user always gets or doesn't get a feature (consistent experience across sessions). The SHA256 hash ensures even distribution across users, so `rolloutPct: 20` exposes exactly ~20% of users."""
    },
]

# ─────────────────────────────────────────────
# CATEGORY 8: FRONTEND TESTING
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "What are the best practices for querying with React Testing Library? Explain the query priority order and why `getByRole` is almost always the right choice.",
        "thinking": "RTL's testing philosophy is to test the way users interact with the app — so queries should reflect what users see and interact with, not implementation details like CSS classes or test IDs. The priority order: getByRole (ARIA roles) > getByLabelText (form labels) > getByPlaceholderText > getByText > getByDisplayValue > getByAltText > getByTitle > getByTestId. I should explain why each is in this order and show concrete examples of each with accessible elements.",
        "answer": """**RTL query priority (highest → lowest confidence in a11y)**

```tsx
// Component under test
function LoginForm({ onSubmit }: { onSubmit: (data: { email: string; pass: string }) => void }) {
  return (
    <form onSubmit={...}>
      <label htmlFor="email">Email address</label>
      <input id="email" type="email" placeholder="you@example.com" />

      <label htmlFor="pass">Password</label>
      <input id="pass" type="password" />

      <button type="submit">Sign in</button>
      <img src="/logo.png" alt="Company logo" />
    </form>
  );
}

// ── Tests ──────────────────────────────────────
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

test('submits login form', async () => {
  const user = userEvent.setup();
  const onSubmit = vi.fn();
  render(<LoginForm onSubmit={onSubmit} />);

  // 1st priority: getByRole — tests accessibility directly
  // The role is inferred from HTML semantics, or explicit role=""
  const emailInput = screen.getByRole('textbox', { name: /email address/i });
  const submitBtn  = screen.getByRole('button',  { name: /sign in/i });

  // 2nd priority: getByLabelText — works with htmlFor, aria-label, aria-labelledby
  const passInput = screen.getByLabelText(/password/i);

  await user.type(emailInput, 'user@example.com');
  await user.type(passInput, 'secret123');
  await user.click(submitBtn);

  expect(onSubmit).toHaveBeenCalledWith({ email: 'user@example.com', pass: 'secret123' });
});

// Role reference sheet
screen.getByRole('button')           // <button>, <input type="submit">
screen.getByRole('textbox')          // <input type="text|email">, <textarea>
screen.getByRole('checkbox')         // <input type="checkbox">
screen.getByRole('combobox')         // <select>, aria-role="combobox"
screen.getByRole('heading', { level: 2 }) // <h2>
screen.getByRole('img', { name: /logo/i }) // <img alt="logo">
screen.getByRole('link', { name: /home/i }) // <a>
screen.getByRole('dialog')           // aria-role="dialog"
screen.getByRole('alert')            // role="alert"
screen.getByRole('listitem')         // <li>
screen.getByRole('tab')              // role="tab"

// When to use lower-priority queries:
screen.getByPlaceholderText('Search...'); // only if no label (bad a11y — fix the component)
screen.getByText('Error occurred');       // for non-interactive text nodes
screen.getByTestId('submit-btn');         // last resort; means your component isn't accessible
```

**`{ name: /regex/i }` option**: the "accessible name" computed from the element's label, `aria-label`, `aria-labelledby`, or button text. Using this makes your test also verify the accessible name is correct — a two-for-one a11y + behavior test.

**Golden rule**: if you can't query for it by role or label, the component probably isn't accessible. Fix the component, not the test."""
    },
    {
        "question": "Set up Mock Service Worker (MSW v2) for both unit tests (Vitest) and browser dev mode — intercept API calls without changing component code.",
        "thinking": "MSW v2 uses handlers with the http helper (replacing the old rest helper). For Vitest: set up a server with setupServer(), call server.listen() in beforeAll, server.resetHandlers() in afterEach, server.close() in afterAll. For the browser: register a service worker with msw/browser setupWorker(). The key advantage: the same handlers work in both environments. I should show the handler definition, Vitest setup, and browser setup.",
        "answer": """```ts
// src/mocks/handlers.ts
import { http, HttpResponse } from 'msw';

interface User { id: string; name: string; email: string; }

export const handlers = [
  // GET users
  http.get('/api/users', () =>
    HttpResponse.json<User[]>([
      { id: '1', name: 'Alice', email: 'alice@example.com' },
      { id: '2', name: 'Bob',   email: 'bob@example.com' },
    ])
  ),

  // GET single user
  http.get('/api/users/:id', ({ params }) =>
    HttpResponse.json<User>({ id: String(params.id), name: 'Alice', email: 'alice@example.com' })
  ),

  // POST create user — validates body
  http.post('/api/users', async ({ request }) => {
    const body = await request.json() as Partial<User>;
    if (!body.email) {
      return HttpResponse.json({ error: 'Email required' }, { status: 422 });
    }
    return HttpResponse.json({ ...body, id: crypto.randomUUID() }, { status: 201 });
  }),
];

// src/mocks/server.ts (Vitest / Node)
import { setupServer } from 'msw/node';
export const server = setupServer(...handlers);

// src/mocks/browser.ts (browser dev mode)
import { setupWorker } from 'msw/browser';
export const worker = setupWorker(...handlers);
```

```ts
// vitest.setup.ts
import { beforeAll, afterEach, afterAll } from 'vitest';
import { server } from './mocks/server';

beforeAll(() => server.listen({ onUnhandledRequest: 'error' }));
afterEach(() => server.resetHandlers()); // reset per-test overrides
afterAll(() => server.close());

// vitest.config.ts
export default defineConfig({
  test: { setupFiles: ['./src/vitest.setup.ts'] },
});
```

```tsx
// ── In tests — override handlers per test ─────
import { http, HttpResponse } from 'msw';
import { server } from '../mocks/server';

test('shows error state when API fails', async () => {
  // Override just for this test
  server.use(
    http.get('/api/users', () => HttpResponse.json({ error: 'Server error' }, { status: 500 }))
  );

  render(<UserList />);
  expect(await screen.findByRole('alert')).toHaveTextContent(/server error/i);
});

// ── Browser dev mode (src/main.tsx) ───────────
async function enableMocking() {
  if (process.env.NODE_ENV !== 'development') return;
  const { worker } = await import('./mocks/browser');
  return worker.start({ onUnhandledRequest: 'bypass' });
}

enableMocking().then(() => {
  ReactDOM.createRoot(document.getElementById('root')!).render(<App />);
});
```

`onUnhandledRequest: 'error'` in tests fails the test if your component makes an API call you haven't mocked — a safety net that prevents silent failures. `'bypass'` in browser dev mode lets real API calls through for endpoints you haven't mocked yet."""
    },
    {
        "question": "Write a Playwright test suite for a shopping cart — use Page Object Model, test adding items, updating quantities, and checkout flow with network mocking.",
        "thinking": "Playwright Page Object Model (POM) encapsulates page-specific locators and actions into a class, making tests readable and maintainable. For a cart, I need: ProductPage to add items, CartPage to verify and update the cart, CheckoutPage for the form. Network mocking with page.route() intercepts API calls. Key: use web-first assertions (toBeVisible, toHaveText) which auto-wait, not expect(await locator.text()).",
        "answer": """```ts
// tests/pages/CartPage.ts
import { type Page, type Locator } from '@playwright/test';

export class CartPage {
  readonly page:        Page;
  readonly items:       Locator;
  readonly totalPrice:  Locator;
  readonly checkoutBtn: Locator;

  constructor(page: Page) {
    this.page        = page;
    this.items       = page.getByRole('listitem').filter({ has: page.getByTestId('cart-item') });
    this.totalPrice  = page.getByTestId('cart-total');
    this.checkoutBtn = page.getByRole('button', { name: /checkout/i });
  }

  async open() { await this.page.goto('/cart'); }

  async getItemCount() { return this.items.count(); }

  async updateQuantity(productName: string, qty: number) {
    const item = this.items.filter({ hasText: productName });
    await item.getByRole('spinbutton', { name: /quantity/i }).fill(String(qty));
    await item.getByRole('spinbutton').press('Tab'); // trigger update
  }

  async removeItem(productName: string) {
    const item = this.items.filter({ hasText: productName });
    await item.getByRole('button', { name: /remove/i }).click();
  }
}

// tests/pages/CheckoutPage.ts
export class CheckoutPage {
  constructor(private page: Page) {}

  async fillShipping({ name, address, city }: { name: string; address: string; city: string }) {
    await this.page.getByLabel('Full name').fill(name);
    await this.page.getByLabel('Address').fill(address);
    await this.page.getByLabel('City').fill(city);
  }

  async placeOrder() {
    await this.page.getByRole('button', { name: /place order/i }).click();
    return this.page.waitForURL('**/order-confirmation/**');
  }
}

// tests/cart.spec.ts
import { test, expect } from '@playwright/test';
import { CartPage }     from './pages/CartPage';
import { CheckoutPage } from './pages/CheckoutPage';

test.beforeEach(async ({ page }) => {
  // Mock product API
  await page.route('/api/products*', route =>
    route.fulfill({
      status: 200,
      body: JSON.stringify([
        { id: 'p1', name: 'Widget Pro', price: 29.99, stock: 5 },
        { id: 'p2', name: 'Gadget Max', price: 49.99, stock: 2 },
      ]),
    })
  );
  // Mock cart API
  await page.route('/api/cart', route => route.fulfill({ status: 200, body: '{"items":[]}' }));
});

test('add items to cart and verify total', async ({ page }) => {
  const cart = new CartPage(page);

  // Add product via product page
  await page.goto('/products/p1');
  await page.getByRole('button', { name: /add to cart/i }).click();

  // Verify cart badge updates
  await expect(page.getByTestId('cart-badge')).toHaveText('1');

  await cart.open();
  await expect(cart.items).toHaveCount(1);
  await expect(cart.totalPrice).toContainText('29.99');
});

test('update quantity recalculates total', async ({ page }) => {
  const cart = new CartPage(page);
  // Pre-seed cart via localStorage
  await page.goto('/');
  await page.evaluate(() => {
    localStorage.setItem('cart', JSON.stringify([{ id: 'p1', name: 'Widget Pro', price: 29.99, qty: 1 }]));
  });

  await cart.open();
  await cart.updateQuantity('Widget Pro', 3);
  await expect(cart.totalPrice).toContainText('89.97');
});

test('complete checkout flow', async ({ page }) => {
  // Mock order submission
  await page.route('/api/orders', route =>
    route.fulfill({ status: 201, body: JSON.stringify({ orderId: 'ord_123' }) })
  );

  const cart     = new CartPage(page);
  const checkout = new CheckoutPage(page);

  await cart.open();
  await cart.checkoutBtn.click();

  await checkout.fillShipping({ name: 'Alice', address: '123 Main St', city: 'Portland' });
  await checkout.placeOrder();

  await expect(page).toHaveURL(/order-confirmation/);
  await expect(page.getByRole('heading')).toContainText('Order confirmed');
});
```

`page.route()` intercepts network calls before they leave the browser — no test server needed. POM classes use Playwright `Locator` (lazy, auto-waiting) not `ElementHandle` (immediately resolved, stale-prone)."""
    },
    {
        "question": "How do I test a custom React hook with `renderHook` from RTL — show testing a `useLocalStorage` hook including initial value, updates, and cross-tab sync.",
        "thinking": "renderHook renders a component that only renders the hook, giving access to the returned value and act() for triggering updates. For useLocalStorage: test initial value from localStorage, test setting a new value (both in state and in localStorage), test the storage event listener for cross-tab sync. I need to mock localStorage and dispatch StorageEvent manually in tests.",
        "answer": """```ts
// hooks/useLocalStorage.ts
import { useState, useEffect, useCallback } from 'react';

export function useLocalStorage<T>(key: string, initial: T) {
  const [value, setValue] = useState<T>(() => {
    try {
      const stored = localStorage.getItem(key);
      return stored ? JSON.parse(stored) : initial;
    } catch { return initial; }
  });

  const set = useCallback((v: T | ((prev: T) => T)) => {
    setValue(prev => {
      const next = typeof v === 'function' ? (v as (p: T) => T)(prev) : v;
      localStorage.setItem(key, JSON.stringify(next));
      return next;
    });
  }, [key]);

  const remove = useCallback(() => {
    localStorage.removeItem(key);
    setValue(initial);
  }, [key, initial]);

  // Cross-tab sync
  useEffect(() => {
    const handler = (e: StorageEvent) => {
      if (e.key === key && e.newValue !== null) {
        try { setValue(JSON.parse(e.newValue)); } catch { /* ignore */ }
      }
    };
    window.addEventListener('storage', handler);
    return () => window.removeEventListener('storage', handler);
  }, [key]);

  return [value, set, remove] as const;
}

// hooks/useLocalStorage.test.ts
import { renderHook, act } from '@testing-library/react';
import { useLocalStorage } from './useLocalStorage';

// Mock localStorage
const store: Record<string, string> = {};
beforeEach(() => {
  Object.defineProperty(window, 'localStorage', {
    value: {
      getItem:    (k: string) => store[k] ?? null,
      setItem:    (k: string, v: string) => { store[k] = v; },
      removeItem: (k: string) => { delete store[k]; },
    },
    writable: true,
  });
});
afterEach(() => { Object.keys(store).forEach(k => delete store[k]); });

test('returns initial value when key not in localStorage', () => {
  const { result } = renderHook(() => useLocalStorage('theme', 'light'));
  expect(result.current[0]).toBe('light');
});

test('reads existing value from localStorage', () => {
  store['theme'] = JSON.stringify('dark');
  const { result } = renderHook(() => useLocalStorage('theme', 'light'));
  expect(result.current[0]).toBe('dark');
});

test('updates state and localStorage when set is called', () => {
  const { result } = renderHook(() => useLocalStorage('count', 0));

  act(() => { result.current[1](42); });

  expect(result.current[0]).toBe(42);
  expect(JSON.parse(store['count'])).toBe(42);
});

test('supports functional updates', () => {
  const { result } = renderHook(() => useLocalStorage('count', 0));

  act(() => { result.current[1](c => c + 1); });
  act(() => { result.current[1](c => c + 1); });

  expect(result.current[0]).toBe(2);
});

test('syncs across tabs via storage event', () => {
  const { result } = renderHook(() => useLocalStorage('theme', 'light'));

  act(() => {
    window.dispatchEvent(new StorageEvent('storage', {
      key: 'theme', newValue: JSON.stringify('dark'), storageArea: localStorage,
    }));
  });

  expect(result.current[0]).toBe('dark');
});

test('remove() resets to initial value', () => {
  store['theme'] = JSON.stringify('dark');
  const { result } = renderHook(() => useLocalStorage('theme', 'light'));

  act(() => { result.current[2](); }); // remove()

  expect(result.current[0]).toBe('light');
  expect(store['theme']).toBeUndefined();
});
```

`act()` is required whenever the hook update involves state changes — it ensures all queued React updates are flushed before the assertion. For async hooks, use `await act(async () => { ... })`."""
    },
]

# ─────────────────────────────────────────────
# PYTHON DATA VISUALIZATION: matplotlib, pandas
# ─────────────────────────────────────────────
EXAMPLES += [
    {
        "question": "How do I create a multi-panel matplotlib figure with subplots — a line chart, bar chart, scatter plot, and histogram — with a shared title and consistent styling?",
        "thinking": "plt.subplots() returns a figure and array of axes. For a 2x2 grid, axes is a 2D array. Key styling concerns: consistent color palette, readable font sizes, proper axis labels, tight_layout() to prevent overlap. I should also show how to share axes (sharex/sharey) and how to customize individual subplots. Using the object-oriented API (ax.plot) is preferred over pyplot state machine (plt.plot) for multi-subplot figures.",
        "answer": """```python
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd

# Sample data
np.random.seed(42)
x    = np.linspace(0, 10, 100)
y    = np.sin(x) + np.random.normal(0, 0.1, 100)
cats = ['Q1', 'Q2', 'Q3', 'Q4']
vals = [23, 45, 31, 58]

df = pd.DataFrame({
    'height': np.random.normal(170, 10, 200),
    'weight': np.random.normal(70, 12, 200),
    'gender': np.random.choice(['M', 'F'], 200),
})

# ── Create 2×2 subplot grid ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
fig.suptitle('Sales Dashboard 2024', fontsize=16, fontweight='bold', y=1.02)

PALETTE = ['#0066cc', '#00a651', '#e63946', '#f4a261']

# ── 1. Line chart ──────────────────────────────
ax1 = axes[0, 0]
ax1.plot(x, y, color=PALETTE[0], linewidth=2, label='Revenue')
ax1.fill_between(x, y - 0.2, y + 0.2, alpha=0.15, color=PALETTE[0])
ax1.set_title('Monthly Revenue Trend')
ax1.set_xlabel('Month')
ax1.set_ylabel('Revenue ($M)')
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.spines[['top', 'right']].set_visible(False)

# ── 2. Bar chart ───────────────────────────────
ax2 = axes[0, 1]
bars = ax2.bar(cats, vals, color=PALETTE, edgecolor='white', linewidth=1.5, zorder=2)
ax2.bar_label(bars, padding=3, fontsize=10)
ax2.set_title('Quarterly Sales')
ax2.set_ylabel('Units sold')
ax2.set_ylim(0, max(vals) * 1.2)
ax2.grid(axis='y', alpha=0.3, zorder=0)
ax2.spines[['top', 'right', 'left']].set_visible(False)
ax2.tick_params(left=False)

# ── 3. Scatter plot ────────────────────────────
ax3 = axes[1, 0]
for gender, color, label in [('M', PALETTE[0], 'Male'), ('F', PALETTE[2], 'Female')]:
    subset = df[df.gender == gender]
    ax3.scatter(subset.height, subset.weight, c=color, alpha=0.5, s=30, label=label)

# Add trend line
z = np.polyfit(df.height, df.weight, 1)
p = np.poly1d(z)
ax3.plot(sorted(df.height), p(sorted(df.height)), '--', color='gray', linewidth=1)
ax3.set_title('Height vs Weight')
ax3.set_xlabel('Height (cm)')
ax3.set_ylabel('Weight (kg)')
ax3.legend()
ax3.spines[['top', 'right']].set_visible(False)

# ── 4. Histogram ───────────────────────────────
ax4 = axes[1, 1]
ax4.hist(df.height, bins=25, color=PALETTE[1], edgecolor='white', alpha=0.8, density=True)
# Overlay KDE
from scipy.stats import gaussian_kde
kde = gaussian_kde(df.height)
x_range = np.linspace(df.height.min(), df.height.max(), 200)
ax4.plot(x_range, kde(x_range), color='#004d00', linewidth=2)
ax4.set_title('Height Distribution')
ax4.set_xlabel('Height (cm)')
ax4.set_ylabel('Density')
ax4.spines[['top', 'right']].set_visible(False)

# ── Final layout ───────────────────────────────
plt.tight_layout()
plt.savefig('dashboard.png', dpi=150, bbox_inches='tight', facecolor='white')
plt.show()
```

**Key practices**:
- `ax.spines[['top', 'right']].set_visible(False)` — removes chart junk for a cleaner look
- `ax.grid(True, alpha=0.3)` — subtle grid that doesn't dominate
- `zorder` on bar + grid ensures grid lines render behind bars
- `bbox_inches='tight'` prevents label clipping on save
- Always use OO API (`ax.plot()`) over `plt.plot()` in multi-subplot figures — no hidden state confusion"""
    },
    {
        "question": "Build a pandas + matplotlib analysis pipeline — read a CSV, clean data, compute group aggregations, and produce a publication-quality chart with annotations.",
        "thinking": "A typical data pipeline: load CSV with pd.read_csv, clean nulls and types, groupby + agg for statistics, then plot with matplotlib. Publication quality means: sensible figure size (double-column = ~7 inches), no chart junk, proper error bars if showing means, axis labels with units, and annotations for key data points. I'll use a sales dataset as the example.",
        "answer": """```python
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── Load & clean ───────────────────────────────
df = pd.read_csv('sales.csv', parse_dates=['date'])
df = df.dropna(subset=['revenue', 'product', 'region'])
df['revenue'] = pd.to_numeric(df['revenue'], errors='coerce')
df['date']    = pd.to_datetime(df['date'], errors='coerce')
df = df[df['revenue'] > 0]  # remove returns/negatives
df['month'] = df['date'].dt.to_period('M')

# ── Aggregate ──────────────────────────────────
monthly = (
    df.groupby(['month', 'region'])['revenue']
      .agg(['sum', 'mean', 'count'])
      .reset_index()
      .rename(columns={'sum': 'total', 'mean': 'avg', 'count': 'orders'})
)
monthly['month_dt'] = monthly['month'].dt.to_timestamp()

# Pivot for easy plotting
pivot = monthly.pivot(index='month_dt', columns='region', values='total').fillna(0)

# ── Plot ───────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
})

REGIONS = pivot.columns.tolist()
COLORS  = plt.cm.Set2(np.linspace(0, 1, len(REGIONS)))

# Stacked area chart
ax1.stackplot(pivot.index, [pivot[r] for r in REGIONS],
              labels=REGIONS, colors=COLORS, alpha=0.85)
ax1.set_title('Monthly Revenue by Region')
ax1.set_ylabel('Revenue (USD)')
ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.1f}M'))
ax1.legend(loc='upper left', framealpha=0.9)
ax1.spines[['top', 'right']].set_visible(False)

# Annotate peak month
total_monthly = pivot.sum(axis=1)
peak_month    = total_monthly.idxmax()
peak_value    = total_monthly.max()
ax1.annotate(
    f'Peak: ${peak_value/1e6:.1f}M\n{peak_month.strftime("%b %Y")}',
    xy=(peak_month, peak_value),
    xytext=(peak_month, peak_value * 1.1),
    arrowprops=dict(arrowstyle='->', color='black'),
    fontsize=9, ha='center',
)

# Bar chart: top 5 regions by total revenue
region_totals = df.groupby('region')['revenue'].sum().nlargest(5)
bars = ax2.barh(region_totals.index[::-1], region_totals.values[::-1],
                color=COLORS[:5], edgecolor='white')
ax2.bar_label(bars, fmt=lambda x: f'${x/1e6:.2f}M', padding=4)
ax2.set_title('Top 5 Regions by Revenue')
ax2.set_xlabel('Total Revenue (USD)')
ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'${x/1e6:.0f}M'))
ax2.spines[['top', 'right', 'left']].set_visible(False)
ax2.tick_params(left=False)

# ── Export ─────────────────────────────────────
plt.tight_layout()
fig.savefig('sales_analysis.pdf', dpi=300, bbox_inches='tight', facecolor='white')
fig.savefig('sales_analysis.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f'Saved. Peak revenue month: {peak_month.strftime("%B %Y")} (${peak_value:,.0f})')
```

**`parse_dates` in `read_csv`** parses date columns directly without a separate `pd.to_datetime` call — use when date column format is standard ISO. **`Period`** vs **`Timestamp`**: periods are better for grouping (no time-of-day ambiguity), convert back to Timestamp for matplotlib's date axis."""
    },
    {
        "question": "Create an interactive matplotlib figure with dropdowns and sliders using `matplotlib.widgets` — a sine wave where users control frequency, amplitude, and phase in real time.",
        "thinking": "matplotlib.widgets provides Slider and Button widgets that attach to axes. The pattern: create a figure with extra space at the bottom for controls (plt.subplots_adjust), add axes for each widget, create Slider/Button on those axes, define update functions that call fig.canvas.draw_idle() to redraw without full repaint. For production interactive viz, use Plotly or Bokeh, but for quick data exploration matplotlib.widgets is built-in.",
        "answer": """```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons

# ── Initial parameters ─────────────────────────
INIT = {'freq': 2.0, 'amp': 1.0, 'phase': 0.0}
t = np.linspace(0, 2 * np.pi, 500)

def compute(freq, amp, phase):
    return amp * np.sin(freq * t + phase)

# ── Layout ─────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
plt.subplots_adjust(left=0.1, bottom=0.38, right=0.75)
fig.patch.set_facecolor('#f8fafc')
ax.set_facecolor('#f8fafc')

y = compute(**INIT)
line,  = ax.plot(t, y, lw=2.5, color='#0066cc')
ax.set_xlim(0, 2 * np.pi)
ax.set_ylim(-3, 3)
ax.set_xlabel('Time (radians)', fontsize=12)
ax.set_ylabel('Amplitude',      fontsize=12)
ax.set_title('Interactive Sine Wave', fontsize=14, fontweight='bold')
ax.axhline(0, color='gray', lw=0.8, ls='--')
ax.grid(True, alpha=0.3)
ax.spines[['top', 'right']].set_visible(False)

# ── Sliders ────────────────────────────────────
ax_freq  = plt.axes([0.10, 0.22, 0.60, 0.03], facecolor='#e2e8f0')
ax_amp   = plt.axes([0.10, 0.15, 0.60, 0.03], facecolor='#e2e8f0')
ax_phase = plt.axes([0.10, 0.08, 0.60, 0.03], facecolor='#e2e8f0')

s_freq  = Slider(ax_freq,  'Frequency', 0.1, 10.0, valinit=INIT['freq'],  valstep=0.1,  color='#0066cc')
s_amp   = Slider(ax_amp,   'Amplitude', 0.1,  3.0, valinit=INIT['amp'],   valstep=0.05, color='#00a651')
s_phase = Slider(ax_phase, 'Phase',     0.0, 2 * np.pi, valinit=INIT['phase'], color='#e63946')

def update(_):
    line.set_ydata(compute(s_freq.val, s_amp.val, s_phase.val))
    ax.set_title(f'y = {s_amp.val:.2f} sin({s_freq.val:.1f}t + {s_phase.val:.2f})', fontsize=13)
    fig.canvas.draw_idle()  # efficient partial redraw

s_freq.on_changed(update)
s_amp.on_changed(update)
s_phase.on_changed(update)

# ── Radio buttons for wave type ────────────────
ax_radio = plt.axes([0.78, 0.25, 0.18, 0.20], facecolor='#e2e8f0')
radio    = RadioButtons(ax_radio, ('Sine', 'Cosine', 'Square'), active=0)

WAVES = {
    'Sine':   lambda f, a, p: a * np.sin(f * t + p),
    'Cosine': lambda f, a, p: a * np.cos(f * t + p),
    'Square': lambda f, a, p: a * np.sign(np.sin(f * t + p)),
}

def on_wave_change(label):
    line.set_ydata(WAVES[label](s_freq.val, s_amp.val, s_phase.val))
    fig.canvas.draw_idle()

radio.on_clicked(on_wave_change)

# ── Reset button ───────────────────────────────
ax_btn = plt.axes([0.78, 0.10, 0.15, 0.06])
btn    = Button(ax_btn, 'Reset', color='#e2e8f0', hovercolor='#cbd5e1')

def reset(_):
    s_freq.reset(); s_amp.reset(); s_phase.reset()

btn.on_clicked(reset)

plt.show()
```

`fig.canvas.draw_idle()` redraws only what's changed (not the full figure), making slider updates smooth. For sharing interactive charts as HTML, use **Plotly** (`fig = px.line(...); fig.show()`) or **Bokeh** (`output_file(); show()`) which generate self-contained HTML files without needing a running Python kernel."""
    },
]

# ── Output ──────────────────────────────────────
if __name__ == '__main__':
    import random
    from pathlib import Path

    random.seed(42)
    random.shuffle(EXAMPLES)
    split = int(len(EXAMPLES) * 0.9)
    train, val = EXAMPLES[:split], EXAMPLES[split:]

    out = Path('data')
    out.mkdir(exist_ok=True)

    def write_jsonl(path, examples):
        with open(path, 'w') as f:
            for ex in examples:
                record = {
                    'messages': [
                        {'role': 'system',    'content': SYSTEM},
                        {'role': 'user',      'content': ex['question']},
                        {'role': 'assistant', 'content': f'<thinking>\n{ex["thinking"]}\n</thinking>\n\n{ex["answer"]}'},
                    ]
                }
                f.write(json.dumps(record) + '\n')

    write_jsonl(out / 'train2.jsonl', train)
    write_jsonl(out / 'val2.jsonl', val)
    print(f'Wrote {len(train)} train + {len(val)} val examples')
    print(f'Total: {len(EXAMPLES)} examples')
