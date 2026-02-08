---
title: "Flash Attention Explained"
layout: post
---

FlashAttention is a fast and memory-efficient algorithm for computing attention in transformer models.  
Normally, the attention mechanism requires forming an \(N \times N\) matrix of attention scores, which is expensive when the sequence length \(N\) is large.

FlashAttention avoids materializing the full attention matrix by computing attention in **streaming blocks**, allowing it to achieve:

- **O(N²)** compute but **O(N)** memory
- Much lower GPU memory usage
- Faster training and inference

## Why Regular Attention Is Slow

In a transformer, each token attends to all other tokens.  
This requires computing:

\[
\text{Attention}(Q,K,V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d}}\right)V
\]

The matrix \(QK^\top\) is \(N \times N\), and storing it is expensive.

For N = 4096 tokens, that's 16 million values — too large for GPU SRAM.

## Key Idea Behind FlashAttention

GPUs have a memory hierarchy:

- **High bandwidth but tiny SRAM (shared memory)**
- **Large but slow HBM (global memory)**

FlashAttention is an **IO-aware algorithm**:  
it rearranges computation so that all intermediate values fit into fast SRAM.

### How it works:

1. Split Q, K, V into **blocks**
2. Load small blocks of K and V into SRAM
3. Compute partial attention scores
4. Maintain a running softmax (numerically stable)
5. Never write the full attention matrix to memory

This avoids all unnecessary memory reads/writes.

## Pseudocode (simplified)

```python
for block_q in Q_blocks:
    m = -inf
    l = 0
    out = 0

    for block_k, block_v in KV_blocks:
        scores = block_q @ block_k.T
        m_new = max(m, max(scores))
        l = exp(m - m_new) * l + sum(exp(scores - m_new))
        out = exp(m - m_new) * out + exp(scores - m_new) @ block_v
        m = m_new

    output_block = out / l
