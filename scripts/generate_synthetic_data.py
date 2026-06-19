#!/usr/bin/env python3
"""
Generate synthetic training data from prompts.
This creates realistic but synthetic examples for testing the training pipeline.
"""

import json
from pathlib import Path

SYSTEM_PROMPT = """You are an expert software engineer with deep knowledge of React, JavaScript, Python, animations, and data visualization.
Think through problems step by step inside <thinking>...</thinking> tags, explaining your reasoning and approach.
Then give your final answer with clean, well-structured code and detailed explanations.
Be thorough, correct, and explain complex concepts clearly."""

# Synthetic answers for each category
SYNTHETIC_ANSWERS = {
    "react_animations_gsap.txt": [
        {
            "prompt": "Create a React component with GSAP timeline animations for a carousel that transitions between images smoothly with stagger effects.",
            "answer": """<thinking>
The user wants a carousel with GSAP animations. I need to:
1. Create a React component with useState for slide index
2. Use GSAP timeline for smooth transitions
3. Implement stagger effects for multiple images
4. Add navigation controls
</thinking>

```jsx
import React, { useEffect, useRef, useState } from 'react';
import gsap from 'gsap';

const GSAPCarousel = ({ images }) => {
  const [currentIndex, setCurrentIndex] = useState(0);
  const slideRefs = useRef([]);
  const tlRef = useRef(null);

  useEffect(() => {
    // Kill previous timeline
    if (tlRef.current) tlRef.current.kill();

    // Create new timeline
    const tl = gsap.timeline();

    // Reset all slides
    gsap.set(slideRefs.current, { opacity: 0, x: 100 });

    // Animate current slide and adjacent ones with stagger
    tl.to(slideRefs.current[currentIndex], {
      opacity: 1,
      x: 0,
      duration: 0.8,
      ease: 'power2.out'
    })
    .to(slideRefs.current.map((_, i) =>
      (i + 1) % slideRefs.current.length === currentIndex ? slideRefs.current[i] : null
    ).filter(Boolean), {
      opacity: 0.5,
      duration: 0.6
    }, '<')

    tlRef.current = tl;
  }, [currentIndex]);

  const nextSlide = () => setCurrentIndex((prev) => (prev + 1) % images.length);
  const prevSlide = () => setCurrentIndex((prev) => (prev - 1 + images.length) % images.length);

  return (
    <div className="carousel-container">
      <div className="slides">
        {images.map((img, i) => (
          <img
            key={i}
            ref={(el) => slideRefs.current[i] = el}
            src={img}
            className="slide"
          />
        ))}
      </div>
      <button onClick={prevSlide}>Prev</button>
      <button onClick={nextSlide}>Next</button>
    </div>
  );
};

export default GSAPCarousel;
```

This component uses GSAP timelines to create smooth transitions. The stagger effect is achieved by animating slides in sequence."""
        },
        {
            "prompt": "Build a React component using GSAP ScrollTrigger to animate elements as they enter the viewport during scroll.",
            "answer": """<thinking>
ScrollTrigger requires:
1. Registering ScrollTrigger plugin
2. Creating triggers based on element position
3. Animating when element enters viewport
4. Cleaning up on unmount
</thinking>

```jsx
import { useEffect, useRef } from 'react';
import gsap from 'gsap';
import { ScrollTrigger } from 'gsap/ScrollTrigger';

gsap.registerPlugin(ScrollTrigger);

export const ScrollAnimations = () => {
  const elementsRef = useRef([]);

  useEffect(() => {
    elementsRef.current.forEach((el) => {
      gsap.to(el, {
        scrollTrigger: {
          trigger: el,
          start: 'top 80%',
          onEnter: () => gsap.to(el, { opacity: 1, y: 0, duration: 0.8 })
        }
      });
    });

    return () => ScrollTrigger.getAll().forEach(trigger => trigger.kill());
  }, []);

  return (
    <div>
      {[...Array(5)].map((_, i) => (
        <div
          key={i}
          ref={(el) => elementsRef.current[i] = el}
          className="animate-box"
          style={{ opacity: 0, transform: 'translateY(50px)' }}
        >
          Box {i + 1}
        </div>
      ))}
    </div>
  );
};
```"""
        }
    ],
    "hard_javascript.txt": [
        {
            "prompt": "Implement a function to find all permutations of a string.",
            "answer": """<thinking>
Finding permutations requires:
1. Base case: string of length 1
2. Recursive approach: fix each character and permute the rest
3. Handle duplicates with a Set
4. Return sorted unique permutations
</thinking>

```javascript
function getPermutations(str) {
  if (str.length <= 1) return [str];

  const permutations = new Set();

  for (let i = 0; i < str.length; i++) {
    const char = str[i];
    const remaining = str.slice(0, i) + str.slice(i + 1);

    for (const perm of getPermutations(remaining)) {
      permutations.add(char + perm);
    }
  }

  return Array.from(permutations).sort();
}

// Time: O(n × n!)
// Space: O(n × n!)
```"""
        },
        {
            "prompt": "Implement a debounce function that delays execution until N milliseconds pass without calls.",
            "answer": """<thinking>
Debounce needs to:
1. Return a function that can be called multiple times
2. Cancel previous timeout on each call
3. Only execute after wait period passes without new calls
4. Preserve 'this' context and arguments
</thinking>

```javascript
function debounce(func, wait) {
  let timeout;

  return function(...args) {
    clearTimeout(timeout);
    timeout = setTimeout(() => func.apply(this, args), wait);
  };
}

// Usage
const search = debounce((query) => {
  console.log('Searching:', query);
}, 300);

input.addEventListener('input', (e) => search(e.target.value));
```"""
        }
    ],
    "python_matplotlib_viz.txt": [
        {
            "prompt": "Create an animated matplotlib visualization showing data trends over time with smooth transitions.",
            "answer": """<thinking>
For animated visualizations:
1. Use matplotlib.animation.FuncAnimation
2. Create update function that modifies plot data
3. Use intervals for smooth transitions
4. Save or display the animation
</thinking>

```python
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

fig, ax = plt.subplots(figsize=(10, 6))
x_data, y_data = [], []
line, = ax.plot([], [], lw=2, color='blue')

ax.set_xlim(0, 100)
ax.set_ylim(0, 100)

def update(frame):
    x_data.append(frame)
    y_data.append(np.sin(frame / 10) * 50 + 50)

    line.set_data(x_data, y_data)
    return line,

anim = animation.FuncAnimation(
    fig, update, frames=100,
    interval=50, blit=True, repeat=True
)

plt.show()
```"""
        }
    ],
    "complex_reasoning.txt": [
        {
            "prompt": "Explain the difference between OAuth 2.0 and JWT and when to use each.",
            "answer": """<thinking>
Key differences:
- OAuth 2.0: Authorization protocol, delegates authentication
- JWT: Token format for stateless authentication
- OAuth can use JWT as the token
- Different use cases and security implications
</thinking>

**OAuth 2.0** is an authorization protocol that allows users to grant third-party applications access to resources without sharing passwords.

**JWT (JSON Web Tokens)** is a compact, URL-safe means of representing claims to be transferred between parties.

**Key Differences:**
1. Purpose: OAuth is for authorization flow; JWT is a token format
2. Stateless: JWT is stateless; OAuth can maintain state
3. Delegation: OAuth delegates authentication; JWT doesn't
4. Revocation: JWT tokens can't be revoked mid-stream; OAuth can

**When to Use:**
- **OAuth 2.0**: Social login, third-party integrations, delegated access
- **JWT**: API authentication, microservices, stateless systems
- **Together**: OAuth can issue JWTs as access tokens

Security: JWT tokens should be stored securely and validated server-side."""
        }
    ],
    "kimi_coding.txt": [
        {
            "prompt": "Write a function to reverse a linked list recursively.",
            "answer": """<thinking>
Recursive reversal needs:
1. Base case: null or single node
2. Recursive call on rest of list
3. Reversal of pointers
4. Return new head
</thinking>

```python
class Node:
    def __init__(self, data):
        self.data = data
        self.next = None

def reverse_recursive(head):
    # Base case
    if not head or not head.next:
        return head

    # Recursive call
    new_head = reverse_recursive(head.next)

    # Reverse pointers
    head.next.next = head
    head.next = None

    return new_head
```"""
        }
    ]
}

def generate_synthetic_data():
    """Generate synthetic training data from prompts and answers."""
    Path("data/raw").mkdir(parents=True, exist_ok=True)
    output_file = Path("data/raw/synthetic_training.jsonl")

    total_entries = 0

    with open(output_file, "w") as f:
        # Generate from our synthetic answers
        for category_file, answers in SYNTHETIC_ANSWERS.items():
            for answer_dict in answers:
                entry = {
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": answer_dict["prompt"]},
                        {"role": "assistant", "content": answer_dict["answer"]}
                    ]
                }
                f.write(json.dumps(entry) + "\n")
                total_entries += 1

    print(f"✅ Generated {total_entries} synthetic examples → {output_file}")
    return total_entries

if __name__ == "__main__":
    generate_synthetic_data()
