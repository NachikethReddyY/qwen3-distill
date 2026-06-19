#!/usr/bin/env python3
"""Dataset 3 — memory, context, agentic, Python advanced, JS/animations."""
import json, random
from pathlib import Path

SYSTEM = """You are an expert software engineer with deep knowledge of Python, JavaScript, React, TypeScript, animations, AI/LLM systems, and distributed architectures.
Think through problems step by step inside <thinking>...</thinking> tags, showing your reasoning, tradeoffs, and approach.
Then give your final answer with clean, production-quality code and precise explanations.
Be thorough, technically accurate, and explain subtle nuances that separate good code from great code."""

EXAMPLES = []

# ── BATCH 1: Memory & Context Management ─────────────────────
EXAMPLES += [
    {
        "question": "Implement an LRU cache in Python. When should you use a hand-rolled one versus `functools.lru_cache`?",
        "thinking": "LRU evicts the least-recently-used entry when capacity is full. OrderedDict gives O(1) move-to-end and popitem(last=False) for the oldest. functools.lru_cache is perfect for memoizing pure functions but it never evicts by size in a custom way and can't be inspected/cleared per-key. Roll your own when you need: explicit capacity, per-key TTL, thread-safe eviction callbacks, or shared cache across instances.",
        "answer": """```python
from collections import OrderedDict
from threading import Lock
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class LRUCache(Generic[K, V]):
    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("capacity must be >= 1")
        self._cap = capacity
        self._store: OrderedDict[K, V] = OrderedDict()
        self._lock = Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            if key not in self._store:
                return None
            self._store.move_to_end(key)   # mark as recently used
            return self._store[key]

    def put(self, key: K, value: V) -> None:
        with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = value
            if len(self._store) > self._cap:
                self._store.popitem(last=False)  # evict LRU

    def __len__(self) -> int:
        return len(self._store)


# Usage
cache: LRUCache[str, int] = LRUCache(3)
cache.put("a", 1); cache.put("b", 2); cache.put("c", 3)
cache.get("a")          # promotes "a" to MRU
cache.put("d", 4)       # evicts "b" (LRU), not "a"
assert cache.get("b") is None
```

**When to use `functools.lru_cache` instead**

```python
from functools import lru_cache

@lru_cache(maxsize=256)
def fib(n: int) -> int:
    return n if n < 2 else fib(n-1) + fib(n-2)
```

Use `lru_cache` for: pure functions, memoizing expensive computations, module-level caches.
Roll your own when you need: TTL expiry, per-key invalidation, shared state across objects, or eviction callbacks.""",
    },
    {
        "question": "Build a TTL cache in Python where entries expire after N seconds. Show thread-safe get/set and lazy expiry.",
        "thinking": "Lazy expiry checks TTL on read rather than running a background sweeper — simpler, no threads needed. Store (value, expire_at) tuples. On get, if time.monotonic() > expire_at, delete and return None. Monotonic clock avoids clock-skew issues from wall time. Thread safety needs a lock around the dict operations.",
        "answer": """```python
import time
from threading import Lock
from typing import Generic, Hashable, TypeVar

K = TypeVar("K", bound=Hashable)
V = TypeVar("V")


class TTLCache(Generic[K, V]):
    def __init__(self, ttl: float, maxsize: int = 1000) -> None:
        self._ttl = ttl
        self._maxsize = maxsize
        self._store: dict[K, tuple[V, float]] = {}
        self._lock = Lock()

    def get(self, key: K) -> V | None:
        with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            value, expire_at = entry
            if time.monotonic() > expire_at:
                del self._store[key]
                return None
            return value

    def set(self, key: K, value: V, ttl: float | None = None) -> None:
        expire_at = time.monotonic() + (ttl if ttl is not None else self._ttl)
        with self._lock:
            if len(self._store) >= self._maxsize and key not in self._store:
                # Evict one expired entry or the oldest key
                self._evict_one()
            self._store[key] = (value, expire_at)

    def _evict_one(self) -> None:
        now = time.monotonic()
        for k, (_, exp) in list(self._store.items()):
            if now > exp:
                del self._store[k]
                return
        # No expired entries — remove arbitrary oldest
        self._store.pop(next(iter(self._store)), None)

    def delete(self, key: K) -> None:
        with self._lock:
            self._store.pop(key, None)


# Usage
cache: TTLCache[str, str] = TTLCache(ttl=5.0)
cache.set("session:abc", '{"user": 42}')
data = cache.get("session:abc")   # returns value
time.sleep(6)
data = cache.get("session:abc")   # returns None — expired
```

**Production note**: for distributed TTL caches, use Redis `SET key value EX seconds` — it handles expiry atomically across processes.""",
    },
    {
        "question": "Implement a Zustand store for a shopping cart in React. How does it avoid Context re-render overhead?",
        "thinking": "Zustand uses a pub-sub model outside React's rendering tree. Components subscribe with a selector; only re-render when the selected slice changes. Context re-renders every consumer on any value change. Zustand's store lives in a JS closure — React only knows about the parts each component opts into via the selector. This is the key difference.",
        "answer": """```ts
// store/cart.ts
import { create } from 'zustand';
import { persist } from 'zustand/middleware';

interface CartItem { id: string; name: string; price: number; qty: number; }

interface CartState {
  items: CartItem[];
  add: (item: Omit<CartItem, 'qty'>) => void;
  remove: (id: string) => void;
  updateQty: (id: string, qty: number) => void;
  clear: () => void;
  total: () => number;
}

export const useCartStore = create<CartState>()(
  persist(
    (set, get) => ({
      items: [],

      add: (item) =>
        set((s) => {
          const existing = s.items.find((i) => i.id === item.id);
          if (existing) {
            return { items: s.items.map((i) => i.id === item.id ? { ...i, qty: i.qty + 1 } : i) };
          }
          return { items: [...s.items, { ...item, qty: 1 }] };
        }),

      remove: (id) => set((s) => ({ items: s.items.filter((i) => i.id !== id) })),

      updateQty: (id, qty) =>
        set((s) => ({
          items: qty <= 0
            ? s.items.filter((i) => i.id !== id)
            : s.items.map((i) => i.id === id ? { ...i, qty } : i),
        })),

      clear: () => set({ items: [] }),

      total: () => get().items.reduce((sum, i) => sum + i.price * i.qty, 0),
    }),
    { name: 'cart-storage' }   // persists to localStorage automatically
  )
);

// ── Components — each subscribes only to what it needs ──────
function CartBadge() {
  // Only re-renders when item count changes
  const count = useCartStore((s) => s.items.reduce((n, i) => n + i.qty, 0));
  return <span className="badge">{count}</span>;
}

function CartTotal() {
  // Only re-renders when total changes
  const total = useCartStore((s) => s.total());
  return <p>${total.toFixed(2)}</p>;
}

function AddToCart({ product }: { product: { id: string; name: string; price: number } }) {
  const add = useCartStore((s) => s.add);  // stable reference — never re-renders
  return <button onClick={() => add(product)}>Add to cart</button>;
}
```

**Why no Context overhead**: Zustand's store is a singleton module-level object. React only re-renders a component when its *selector* returns a new value (by reference equality). `CartBadge` never re-renders when prices change — only when count changes.""",
    },
    {
        "question": "Show how to store and retrieve structured data in IndexedDB for offline persistence — a notes app that works without a network.",
        "thinking": "IndexedDB is async, key-value with indexes. The API is callback-based but wraps cleanly in Promises. I need: open DB, create object store on upgrade, then CRUD operations. For a notes app, each note is a JS object with an auto-increment key. The tricky part is that the connection must be opened before any transaction — store the db reference in a module-level promise to avoid repeated opens.",
        "answer": """```ts
// lib/notesDB.ts
const DB_NAME = 'notes-app';
const STORE   = 'notes';
const VERSION = 1;

export interface Note {
  id?: number;    // auto-assigned by IndexedDB
  title: string;
  body: string;
  updatedAt: number;
}

function openDB(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, VERSION);
    req.onupgradeneeded = (e) => {
      const db = (e.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(STORE)) {
        const store = db.createObjectStore(STORE, { keyPath: 'id', autoIncrement: true });
        store.createIndex('updatedAt', 'updatedAt');
      }
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror   = () => reject(req.error);
  });
}

// Cache the connection — don't re-open on every operation
const dbPromise = openDB();

async function tx(mode: IDBTransactionMode) {
  const db = await dbPromise;
  return db.transaction(STORE, mode).objectStore(STORE);
}

function wrap<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((res, rej) => {
    req.onsuccess = () => res(req.result);
    req.onerror   = () => rej(req.error);
  });
}

export const notesDB = {
  async getAll(): Promise<Note[]> {
    return wrap((await tx('readonly')).getAll());
  },
  async get(id: number): Promise<Note | undefined> {
    return wrap((await tx('readonly')).get(id));
  },
  async save(note: Note): Promise<number> {
    return wrap((await tx('readwrite')).put({ ...note, updatedAt: Date.now() })) as Promise<number>;
  },
  async delete(id: number): Promise<void> {
    return wrap((await tx('readwrite')).delete(id));
  },
};

// ── Usage in React ──────────────────────────────────────────
import { useEffect, useState } from 'react';

function Notes() {
  const [notes, setNotes] = useState<Note[]>([]);

  useEffect(() => {
    notesDB.getAll().then(setNotes);
  }, []);

  const addNote = async () => {
    const id = await notesDB.save({ title: 'New note', body: '', updatedAt: 0 });
    setNotes(await notesDB.getAll());
  };

  return (
    <div>
      <button onClick={addNote}>+ New</button>
      {notes.map(n => <div key={n.id}>{n.title}</div>)}
    </div>
  );
}
```

Works fully offline — IndexedDB is local, survives page refresh, and has no size limit (unlike localStorage's 5MB cap).""",
    },
    {
        "question": "Design a rolling-window conversation memory for an AI agent. Old messages must be summarized when the window exceeds a token budget.",
        "thinking": "The key constraint is the LLM's context window. Naive approach: drop oldest messages — loses information. Better: keep a running summary of old context plus the last N full messages. When total tokens exceed budget, summarize the oldest half and replace them with a single summary message. I need a token counter (use tiktoken or estimate at 4 chars/token) and a summarization call.",
        "answer": """```python
import tiktoken
from dataclasses import dataclass, field
from typing import Callable

@dataclass
class Message:
    role: str   # "system" | "user" | "assistant"
    content: str

@dataclass
class RollingMemory:
    max_tokens: int = 4096
    summary_buffer: int = 512     # reserve for summary overhead
    messages: list[Message] = field(default_factory=list)
    _enc: tiktoken.Encoding = field(default_factory=lambda: tiktoken.get_encoding("cl100k_base"), repr=False)
    _summarize: Callable[[list[Message]], str] | None = None

    def count_tokens(self, msgs: list[Message]) -> int:
        return sum(4 + len(self._enc.encode(m.content)) for m in msgs)

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role, content))
        self._maybe_compress()

    def _maybe_compress(self) -> None:
        if self.count_tokens(self.messages) <= self.max_tokens:
            return
        if self._summarize is None:
            # fallback: just drop oldest non-system messages
            non_system = [m for m in self.messages if m.role != "system"]
            keep = non_system[len(non_system)//2:]
            system = [m for m in self.messages if m.role == "system"]
            self.messages = system + keep
            return

        # Summarize the oldest half of non-system messages
        system_msgs = [m for m in self.messages if m.role == "system"]
        non_system  = [m for m in self.messages if m.role != "system"]
        split       = len(non_system) // 2
        to_compress = non_system[:split]
        to_keep     = non_system[split:]

        summary_text = self._summarize(to_compress)
        summary_msg  = Message("assistant", f"[Summary of earlier conversation]: {summary_text}")
        self.messages = system_msgs + [summary_msg] + to_keep

    def get_context(self) -> list[dict]:
        return [{"role": m.role, "content": m.content} for m in self.messages]


# Wire up with an LLM summarizer
def make_summarizer(llm_call):
    def summarize(msgs: list[Message]) -> str:
        prompt = "Summarize the following conversation segment concisely:\n" + \
                 "\n".join(f"{m.role}: {m.content}" for m in msgs)
        return llm_call([{"role": "user", "content": prompt}])
    return summarize

# Usage
memory = RollingMemory(max_tokens=2000, _summarize=make_summarizer(my_llm))
memory.add("user", "Tell me about quantum computing.")
memory.add("assistant", "Quantum computing uses qubits...")
# ... more messages — auto-compresses when budget exceeded
```

**The tradeoff**: summarization loses detail but preserves semantic gist. For tasks needing exact recall (code written 10 turns ago), prefer retrieval from a vector store over summarization.""",
    },
    {
        "question": "How do you count tokens and stay within an LLM's context budget when building prompts programmatically?",
        "thinking": "Token counting is model-specific — GPT-4 uses cl100k_base, Qwen models use their own BPE. The safe approach: use the model's tokenizer to count exactly, then build the prompt in priority order: system + task description first, then retrieved context, then conversation history, trimming from lowest priority until it fits. Reserve headroom for the response.",
        "answer": """```python
import tiktoken
from dataclasses import dataclass

@dataclass
class PromptBudget:
    model: str = "gpt-4o"
    context_limit: int = 128_000
    response_reserve: int = 2048   # tokens to leave for the reply

    def __post_init__(self):
        self._enc = tiktoken.encoding_for_model(self.model)

    @property
    def available(self) -> int:
        return self.context_limit - self.response_reserve

    def count(self, text: str) -> int:
        return len(self._enc.encode(text))

    def count_messages(self, messages: list[dict]) -> int:
        # Each message has ~4 tokens of overhead in the chat format
        return sum(4 + self.count(m["content"]) for m in messages) + 2

    def fit_messages(
        self,
        required: list[dict],        # always included (system, task)
        optional: list[dict],        # trimmed from END if over budget
    ) -> list[dict]:
        used = self.count_messages(required)
        result = list(required)

        for msg in optional:
            cost = 4 + self.count(msg["content"])
            if used + cost > self.available:
                break
            result.append(msg)
            used += cost

        return result


# Example: RAG prompt builder
def build_rag_prompt(
    question: str,
    retrieved_chunks: list[str],
    history: list[dict],
    budget: PromptBudget,
) -> list[dict]:
    system = {"role": "system", "content": "You are a helpful assistant. Answer using only the provided context."}
    task   = {"role": "user",   "content": f"Context:\n{'---'.join(retrieved_chunks)}\n\nQuestion: {question}"}

    # history is lowest priority — trim if needed
    return budget.fit_messages(
        required=[system, task],
        optional=list(reversed(history)),  # keep most recent history
    )


budget = PromptBudget(model="gpt-4o", context_limit=128_000, response_reserve=2048)
messages = build_rag_prompt(
    question="What is RAG?",
    retrieved_chunks=["RAG stands for Retrieval Augmented Generation..."],
    history=[{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}],
    budget=budget,
)
```

**Key insight**: always trim from the *least important* content first. System prompt and current task are sacred; conversation history and retrieved chunks are expendable.""",
    },
    {
        "question": "Implement a sliding window context manager for long AI conversations — keep the last K messages but always include the system prompt.",
        "thinking": "The naive slice `messages[-k:]` drops the system prompt if it's index 0. The correct pattern: partition into system messages (always keep) and non-system messages (sliding window). Then reassemble. Edge case: if K is so small it would drop needed context, enforce a minimum.",
        "answer": """```python
from typing import TypedDict

class Message(TypedDict):
    role: str
    content: str

def sliding_window(
    messages: list[Message],
    max_messages: int = 20,
) -> list[Message]:
    \"\"\"Return the last max_messages turns, always keeping system messages.\"\"\"
    system     = [m for m in messages if m["role"] == "system"]
    non_system = [m for m in messages if m["role"] != "system"]

    # Keep the last max_messages non-system messages
    window = non_system[-max_messages:] if len(non_system) > max_messages else non_system
    return system + window


class ConversationManager:
    def __init__(self, system_prompt: str, window_size: int = 20):
        self._messages: list[Message] = [{"role": "system", "content": system_prompt}]
        self._window = window_size

    def add_user(self, content: str) -> None:
        self._messages.append({"role": "user", "content": content})

    def add_assistant(self, content: str) -> None:
        self._messages.append({"role": "assistant", "content": content})

    def get_context(self) -> list[Message]:
        return sliding_window(self._messages, self._window)

    def __len__(self) -> int:
        return len(self._messages)


# Usage
conv = ConversationManager("You are a helpful assistant.", window_size=6)
for i in range(20):
    conv.add_user(f"Message {i}")
    conv.add_assistant(f"Reply {i}")

ctx = conv.get_context()
print(len(ctx))  # 7 — 1 system + 6 recent messages (not 41)
assert ctx[0]["role"] == "system"   # system always present
```

**When window_size is not enough**: pair this with the rolling-window summarization pattern — once messages are pushed out of the window, they get summarized rather than dropped entirely.""",
    },
    {
        "question": "Explain fixed-size, sentence-based, and semantic chunking for RAG. Show Python implementations of each.",
        "thinking": "Chunking strategy directly affects retrieval quality. Fixed-size is fast but splits mid-sentence. Sentence-based respects natural boundaries but produces variable-size chunks. Semantic chunking groups sentences until cosine similarity drops — most expensive but best for coherent retrieval. I should show all three with their tradeoffs clearly.",
        "answer": """```python
from typing import Iterator

# ── 1. Fixed-size chunking ───────────────────────────────────
def fixed_chunks(text: str, size: int = 512, overlap: int = 64) -> list[str]:
    \"\"\"Split by character count with overlap to avoid cutting context.\"\"\"
    chunks, start = [], 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks


# ── 2. Sentence-based chunking ───────────────────────────────
import re

def sentence_chunks(text: str, max_sentences: int = 5) -> list[str]:
    \"\"\"Group sentences into chunks of max_sentences each.\"\"\"
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    chunks = []
    for i in range(0, len(sentences), max_sentences):
        chunk = ' '.join(sentences[i:i + max_sentences])
        if chunk:
            chunks.append(chunk)
    return chunks


# ── 3. Semantic chunking ─────────────────────────────────────
import numpy as np

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def semantic_chunks(
    text: str,
    embed_fn,                   # callable: str -> np.ndarray
    threshold: float = 0.75,   # split when similarity drops below this
    max_sentences: int = 10,
) -> list[str]:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    if not sentences:
        return []

    embeddings = [embed_fn(s) for s in sentences]
    chunks, current = [], [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i-1], embeddings[i])
        if sim < threshold or len(current) >= max_sentences:
            chunks.append(' '.join(current))
            current = [sentences[i]]
        else:
            current.append(sentences[i])

    if current:
        chunks.append(' '.join(current))
    return chunks
```

| Strategy | Speed | Coherence | Best for |
|---|---|---|---|
| Fixed-size | Fast | Low | Large docs, quick prototype |
| Sentence | Medium | Good | General purpose |
| Semantic | Slow | Best | Technical docs, precision retrieval |

**Practical choice**: start with sentence-based chunking (size 3–5 sentences, small overlap). Switch to semantic only if retrieval quality is noticeably poor.""",
    },
    {
        "question": "Build a minimal RAG pipeline in Python: embed documents, store in a vector DB, retrieve relevant chunks, and pass them to an LLM.",
        "thinking": "RAG = embed corpus offline, embed query at runtime, find nearest vectors, stuff chunks into prompt. For a minimal implementation I'll use sentence-transformers for embeddings and a simple numpy cosine search instead of a full vector DB — makes it self-contained. Then show how to swap in Chroma or Qdrant for production.",
        "answer": """```python
import numpy as np
from dataclasses import dataclass, field
from sentence_transformers import SentenceTransformer

@dataclass
class RAGPipeline:
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 3
    _model: SentenceTransformer = field(init=False)
    _docs: list[str] = field(default_factory=list)
    _embeddings: np.ndarray | None = field(default=None, repr=False)

    def __post_init__(self):
        self._model = SentenceTransformer(self.model_name)

    # ── Indexing ────────────────────────────────────────────
    def index(self, docs: list[str]) -> None:
        self._docs = docs
        vecs = self._model.encode(docs, normalize_embeddings=True)
        self._embeddings = np.array(vecs)

    # ── Retrieval ───────────────────────────────────────────
    def retrieve(self, query: str) -> list[str]:
        assert self._embeddings is not None, "Call .index() first"
        q_vec = self._model.encode([query], normalize_embeddings=True)[0]
        scores = self._embeddings @ q_vec          # cosine sim (vecs are normalized)
        top_idx = np.argsort(scores)[::-1][:self.top_k]
        return [self._docs[i] for i in top_idx]

    # ── Generate ────────────────────────────────────────────
    def answer(self, query: str, llm_fn) -> str:
        chunks = self.retrieve(query)
        context = "\n\n---\n\n".join(chunks)
        prompt = (
            f"Answer the question using only the context below.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\n\nAnswer:"
        )
        return llm_fn(prompt)


# Usage
rag = RAGPipeline(top_k=3)
rag.index([
    "Python was created by Guido van Rossum in 1991.",
    "JavaScript was created by Brendan Eich in 1995.",
    "RAG stands for Retrieval Augmented Generation.",
    "Vector databases store high-dimensional embeddings.",
])

def my_llm(prompt: str) -> str:
    # Replace with your LLM call
    return f"LLM response to: {prompt[:80]}..."

answer = rag.answer("Who created Python?", my_llm)
```

**Swapping in Chroma for production**:
```python
import chromadb
client = chromadb.Client()
col = client.create_collection("docs")
col.add(documents=docs, ids=[str(i) for i in range(len(docs))])
results = col.query(query_texts=[query], n_results=3)
chunks = results["documents"][0]
```""",
    },
    {
        "question": "Implement asyncio.gather vs asyncio.create_task — show when each is right and how errors propagate differently.",
        "thinking": "Both run coroutines concurrently. gather collects all results in one await and by default cancels all if one fails. create_task schedules a task independently — it runs even if you don't await it, and errors are silently dropped unless you add a done callback or await the task. gather is for fan-out-collect patterns; create_task is for fire-and-forget or when you need independent lifetimes.",
        "answer": """```python
import asyncio

async def fetch(url: str, delay: float) -> str:
    await asyncio.sleep(delay)
    if "bad" in url:
        raise ValueError(f"Bad URL: {url}")
    return f"data from {url}"

# ── gather: fan-out, collect all results ────────────────────
async def gather_example():
    results = await asyncio.gather(
        fetch("api/users", 0.1),
        fetch("api/posts", 0.2),
        fetch("api/comments", 0.15),
    )
    return results  # ["data from api/users", ...]

# gather with error isolation (return_exceptions=True)
async def gather_safe():
    results = await asyncio.gather(
        fetch("api/users", 0.1),
        fetch("bad/endpoint", 0.1),
        fetch("api/posts", 0.2),
        return_exceptions=True,   # errors become values, not exceptions
    )
    for r in results:
        if isinstance(r, Exception):
            print(f"One request failed: {r}")
        else:
            print(f"Got: {r}")

# ── create_task: independent lifetime ───────────────────────
async def task_example():
    # Fire background work — doesn't block
    log_task = asyncio.create_task(fire_and_forget_log("event=start"))

    # Do other work concurrently
    result = await fetch("api/users", 0.1)

    # Optionally await the background task before exit
    await log_task
    return result

async def fire_and_forget_log(msg: str):
    await asyncio.sleep(0.05)
    print(f"[log] {msg}")

# ── Choosing between them ────────────────────────────────────
# gather  → you need ALL results before proceeding
# gather(return_exceptions=True) → partial results are OK
# create_task → background work with independent lifetime
#             → you want tasks to outlive the caller scope
#             → use TaskGroup (3.11+) for structured concurrency

# Python 3.11+ TaskGroup — best practice
async def task_group_example():
    async with asyncio.TaskGroup() as tg:
        t1 = tg.create_task(fetch("api/users", 0.1))
        t2 = tg.create_task(fetch("api/posts", 0.2))
    # Both done here; any exception is re-raised
    return t1.result(), t2.result()

asyncio.run(gather_safe())
```

**Critical gotcha with create_task**: if you create a task and never await it, exceptions are silently swallowed until the GC collects the task — a very hard bug to find. Always either await tasks or attach a done callback: `task.add_done_callback(lambda t: t.exception())`.""",
    },
    {
        "question": "Show how to write a Python context manager using both `__enter__`/`__exit__` and `@contextmanager`. When do you need the class form?",
        "thinking": "The generator-based @contextmanager is shorter for simple cases. The class form is needed when: the context manager stores state across multiple calls, has __aenter__/__aexit__ for async, needs to be reentrant, or must implement custom __repr__. Both must handle exceptions — the class form via __exit__'s exc_type/value/tb, the generator form via try/finally or try/except around yield.",
        "answer": """```python
from contextlib import contextmanager
import sqlite3, time

# ── Generator form — concise, covers 80% of use cases ───────
@contextmanager
def timer(label: str):
    start = time.perf_counter()
    try:
        yield                            # caller's block runs here
    finally:
        elapsed = time.perf_counter() - start
        print(f"{label}: {elapsed:.3f}s")

with timer("db query"):
    time.sleep(0.1)   # 0.100s


# ── Class form — needed for stateful / reentrant managers ────
class ManagedConnection:
    \"\"\"Reusable connection that auto-commits on success, rolls back on error.\"\"\"
    def __init__(self, db_path: str):
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    def __enter__(self) -> sqlite3.Connection:
        self._conn = sqlite3.connect(self._path)
        return self._conn

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._conn is None:
            return False
        if exc_type is None:
            self._conn.commit()
        else:
            self._conn.rollback()
        self._conn.close()
        self._conn = None
        return False   # don't suppress exceptions

db = ManagedConnection(":memory:")
with db as conn:
    conn.execute("CREATE TABLE t (x INT)")
    conn.execute("INSERT INTO t VALUES (1)")
# auto-committed

# ── Async context manager ────────────────────────────────────
from contextlib import asynccontextmanager
import aiohttp

@asynccontextmanager
async def http_session():
    async with aiohttp.ClientSession() as session:
        yield session

async def main():
    async with http_session() as s:
        resp = await s.get("https://httpbin.org/get")
```

**When class form is required**:
- Storing mutable state that persists across nested uses
- Implementing `__aenter__`/`__aexit__` on the same object as `__enter__`/`__exit__`
- Reentrant context managers (where the same instance is entered multiple times)""",
    },
    {
        "question": "Explain Python TypeVar and Generic — show a typed Stack and a typed retry decorator that preserves the wrapped function's signature.",
        "thinking": "TypeVar lets you write functions/classes that work on multiple types while preserving type information. Generic[T] makes a class parameterized. The tricky part for decorators is preserving the signature — use ParamSpec (Python 3.10+) to capture arbitrary argument signatures, otherwise the decorator's return type is just Callable which loses specificity.",
        "answer": """```python
from typing import TypeVar, Generic
from collections.abc import Callable
from typing import ParamSpec
import functools, time, asyncio

T = TypeVar("T")
P = ParamSpec("P")   # captures *args/**kwargs types
R = TypeVar("R")

# ── Generic Stack ────────────────────────────────────────────
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: list[T] = []

    def push(self, item: T) -> None:
        self._items.append(item)

    def pop(self) -> T:
        if not self._items:
            raise IndexError("pop from empty stack")
        return self._items.pop()

    def peek(self) -> T:
        return self._items[-1]

    def __len__(self) -> int:
        return len(self._items)

s: Stack[int] = Stack()
s.push(1)         # ✅
s.push("x")      # ❌ type error — Stack[int] rejects str


# ── Retry decorator preserving signature ─────────────────────
def retry(
    times: int = 3,
    delay: float = 1.0,
    exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            last_exc: Exception | None = None
            for attempt in range(times):
                try:
                    return fn(*args, **kwargs)
                except exceptions as e:
                    last_exc = e
                    if attempt < times - 1:
                        time.sleep(delay)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator


@retry(times=3, delay=0.5, exceptions=(ConnectionError, TimeoutError))
def fetch_data(url: str, timeout: int = 10) -> dict:
    # mypy knows this returns dict, not Any
    ...


# ── Async variant ────────────────────────────────────────────
def async_retry(times: int = 3, delay: float = 1.0):
    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            for attempt in range(times):
                try:
                    return await fn(*args, **kwargs)  # type: ignore[misc]
                except Exception:
                    if attempt == times - 1:
                        raise
                    await asyncio.sleep(delay)
        return wrapper  # type: ignore[return-value]
    return decorator
```

`ParamSpec` is the key — without it, `wrapper(*args, **kwargs)` has type `(*Any, **Any) -> Any`, which loses all argument type checking at call sites.""",
    },
]

# ── Output ───────────────────────────────────────────────────
def to_chat(ex: dict) -> dict:
    return {"messages": [
        {"role": "system", "content": SYSTEM},
        {"role": "user",   "content": ex["question"]},
        {"role": "assistant", "content": f"<thinking>\n{ex['thinking']}\n</thinking>\n\n{ex['answer']}"},
    ]}

if __name__ == "__main__":
    random.seed(3)
    random.shuffle(EXAMPLES)
    n_val     = max(1, len(EXAMPLES) // 10)
    val_exs   = EXAMPLES[:n_val]
    train_exs = EXAMPLES[n_val:]
    Path("data").mkdir(exist_ok=True)
    with open("data/train3.jsonl", "w") as f:
        for ex in train_exs:
            f.write(json.dumps(to_chat(ex)) + "\n")
    with open("data/val3.jsonl", "w") as f:
        for ex in val_exs:
            f.write(json.dumps(to_chat(ex)) + "\n")
    print(f"✅ Dataset 3 batch 1: {len(train_exs)} train + {len(val_exs)} val")
    print(f"   train3.jsonl + val3.jsonl written to data/")
