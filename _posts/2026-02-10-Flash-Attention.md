---
title: "Flash Attention"
layout: post
---

#Introduction

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
