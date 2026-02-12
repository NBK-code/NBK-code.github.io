---
title: "Flash Attention"
layout: post
---

## Introduction

The Transformer architecture changed the way sequence data is modeled by replacing recurrence with
self-attention, where every token can directly interact with every other token. This design captures long-range
relationships effectively but comes with a high computational and memory cost that grows rapidly with
sequence length. Earlier methods such as Sparse Attention, Linformer, Performer, and Longformer tried
to reduce this cost by using approximations like sparsity or low-rank projections. While these approaches
improved efficiency, they often sacrificed exactness or generality.

FlashAttention addresses this challenge without compromising accuracy. It is an IO-aware algorithm
that accelerates the computation of self-attention by focusing on efficient data movement rather than
mathematical shortcuts. By reorganizing how attention is computed and how data flows through GPU
memory, FlashAttention achieves the same results as standard attention but with significantly improved
speed and scalability on long sequences.

## Background
### The Self-Attention Mechanism
In the self-attention mechanism, the output for each token is computed as
O = softmax
QKT
√
d

V,
where Q, K, and V are the query, key, and value matrices, and d is the head dimension. This operation
allows each token to attend to all others in the sequence, capturing contextual dependencies effectively.
### Computational and Memory Costs
The computational and memory requirements of self-attention increase rapidly with sequence length N. The
matrix multiplication QKT
involves O(N2d) operations, and the resulting attention matrix of size N × N
must be stored temporarily during computation, requiring O(N2
) space. Although this quadratic scaling is
inherent to the self-attention mechanism, in practice the major bottleneck arises not from the arithmetic itself
but from the repeated reading and writing of large intermediate matrices between different levels of memory.
### Modern GPU Architecture
GPUs employ a hierarchical memory structure with components that vary widely in size and speed. At the
top level, high-bandwidth memory (HBM) provides tens of gigabytes of storage but is relatively slow to access.
Closer to the processing cores are much smaller but significantly faster on-chip memories such as shared
memory (SRAM) and registers. For example, an NVIDIA A100 GPU offers 40–80 GB of HBM with around
1.5–2 TB/s bandwidth, compared to only about 192 KB of on-chip SRAM per streaming multiprocessor with
nearly an order of magnitude higher bandwidth. As GPU compute speed has grown faster than memory
bandwidth, performance for many workloads is now limited by how quickly data can be moved between these
memory levels rather than by arithmetic capability itself.
