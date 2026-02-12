---
title: "Flash Attention"
mathjax: true
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
\begin{equation}
O = \text{softmax}\left(\frac{QK^{T}}{\sqrt{d}}\right)V,
\end{equation}
where $$Q, K$$, and $$V$$ are the query, key, and value matrices, and $$d$$ is the head dimension. This operation
allows each token to attend to all others in the sequence, capturing contextual dependencies effectively.

### Computational and Memory Costs
The computational and memory requirements of self-attention increase rapidly with sequence length $$N$$. The
matrix multiplication $$QK^T$$ involves $$O(N^2d)$$ operations, and the resulting attention matrix of size $$N \times N$$
must be stored temporarily during computation, requiring $$O(N^2)$$ space. Although this quadratic scaling is
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

GPU computations are executed through kernels, each representing a parallel operation launched across
thousands of lightweight threads. A kernel typically loads data from HBM into fast on-chip memory, performs
the required arithmetic, and writes the results back to HBM. The performance of a kernel depends on its
arithmetic intensity—the ratio of computation to data movement. If computation dominates, the operation
is said to be compute-bound; examples include large matrix multiplications or deep convolutions with many
channels. Conversely, when data transfer dominates, the operation is memory-bound; this is the case for
many elementwise functions, reductions (such as softmax or normalization), and attention mechanisms.

### The Need for IO-Aware Computation
In practice, many operations within the self-attention computation are bottlenecked by memory access
rather than arithmetic cost, as data must be frequently read from and written to different levels of GPU
memory. FlashAttention addresses this inefficiency by reorganizing the computation to minimize memory
traffic while preserving the exact mathematical result of standard self-attention. The algorithm introduces
the principle of IO-aware computation, which explicitly accounts for the cost of reading and writing data
between different levels of GPU memory. On modern hardware, computational throughput has grown much
faster than memory bandwidth, making memory access the dominant factor in the runtime of attention
operations. FlashAttention optimizes this by reorganizing the attention computation into small blocks that
fit in fast on-chip memory, avoiding the need to store or repeatedly access the large intermediate attention
matrix. It performs the softmax operation incrementally over these blocks (a technique known as tiling)
and recomputes the necessary quantities during the backward pass instead of reading them from memory.
Despite performing a comparable number of arithmetic operations, this IO-efficient design drastically reduces
memory traffic and leads to significant speedups over standard attention while maintaining exact results.

## Flash Attention Algorithm
### Forward Pass
There are three ingredients that go into constructing the forward pass of flash attention:

 - Numerically stable softmax
 - Online softmax
 - Block computation of matrix multiplication

#### Numerically Stable Softmax
The softmax operation converts raw attention scores into normalized probabilities, ensuring that the weights
assigned to all key positions for each query sum to one. However, directly computing
\begin{equation}
\text{softmax}(x_i) = \frac{e^{x_i}}{\sum_j e^{x_j}}
\end{equation}
can lead to numerical overflow or underflow when the values of $$x_i$$ are large in magnitude.
To improve stability, the softmax is implemented in its numerically stable form by subtracting the maximum value within each row:
\begin{equation}
\text{softmax}(x_i) = \frac{e^{x_i - \max_j(x_j)}}{\sum_k e^{x_k - \max_j(x_j)}}.
\end{equation}
Notice that the new $$\text{softmax}(x_i)$$ still computes the same value as $$e^{-\max_j(x_j)}$$ in the numerator and the denominator cancel each other out. But now $$x_i - \max_j(x_j)$$ is always less than or equal to zero and hence $$e^{-\max_j(x_j)}$$ do not explode. This formulation ensures that all exponential arguments are non-positive, preventing overflow while preserving the exact mathematical result.

#### Online Softmax
In the standard implementation, the softmax for each query vector requires access to all its attention scores
before normalization. This means that the entire set of scores must be computed and stored in memory,
which is not feasible when working with long sequences.
To avoid this, FlashAttention computes the softmax in an online manner, processing one segment
of the scores at a time while maintaining running statistics that allow exact normalization.

For a single query, let the attention scores be represented as a sequence of values
$$s_1, s_2, \dots, s_N$$. Instead of computing the maximum and summing over all the scores at once,
the algorithm updates them incrementally.
After processing the first $$i$$ scores, it keeps track of
the running maximum $$m_i$$ and normalization factor $$l_i$$:
\begin{equation}
m_i = \max(m_{i-1}, s_i), \qquad
l_i = e^{m_{i-1}-m_i}l_{i-1} + e^{s_i - m_i}.
\end{equation}

The first term, $$e^{m_{i-1}-m_i}l_{i-1}$$, rescales the previously accumulated normalization factor $$l_{i-1}$$
to account for any change in the running maximum from $$m_{i-1}$$ to $$m_i$$,
ensuring that the exponents remain numerically stable even when new scores exceed the previous maximum.
The second term, $$e^{s_i - m_i}$$, adds the contribution from the newly encountered score.
Together, these updates allow the algorithm to maintain the exact normalization constant as if all scores
had been processed simultaneously, but using only constant memory.

Once all $$N$$ scores have been processed, the final maximum $$m_N$$ and normalization factor $$l_N$$
are used to compute the softmax output for each element:
\begin{equation}
p_i = \frac{e^{s_i - m_N}}{l_N}.
\end{equation}
This yields the exact same result as the standard softmax,
while requiring only a single pass through the data and constant memory overhead.

### Block Computation
Large matrix multiplications are often too big to fit entirely in fast on-chip memory, requiring data to be repeatedly
loaded from and written back to slower main memory.
To address this, modern high-performance algorithms use a technique known as tiling or block computation.
The main idea is to divide the matrices into smaller submatrices (or tiles) that fit within fast memory, allowing
computation to proceed on one tile at a time while reusing loaded data efficiently.

Consider the matrix multiplication $$C = AB$$, where
$$A \in \mathbb{R}^{M \times K}$$, $$B \in \mathbb{R}^{K \times N}$$, and
$$C \in \mathbb{R}^{M \times N}$$.
Instead of computing the entire result at once, we partition the matrices into blocks of size
$$B_M \times B_K$$ and $$B_K \times B_N$$, such that

$$
A =
\begin{bmatrix}
A_{11} & A_{12} & \cdots & A_{1p} \\
A_{21} & A_{22} & \cdots & A_{2p} \\
\vdots & \vdots & \ddots & \vdots \\
A_{m1} & A_{m2} & \cdots & A_{mp}
\end{bmatrix},
\qquad
B =
\begin{bmatrix}
B_{11} & B_{12} & \cdots & B_{1n} \\
B_{21} & B_{22} & \cdots & B_{2n} \\
\vdots & \vdots & \ddots & \vdots \\
B_{p1} & B_{p2} & \cdots & B_{pn}
\end{bmatrix}.
$$

Each block $$A_{ij}$$ or $$B_{ij}$$ represents a small submatrix, for example:

$$
A_{11} =
\begin{bmatrix}
a_{11} & a_{12} & \cdots & a_{1B_K} \\
a_{21} & a_{22} & \cdots & a_{2B_K} \\
\vdots & \vdots & \ddots & \vdots \\
a_{B_M1} & a_{B_M2} & \cdots & a_{B_MB_K}
\end{bmatrix}
$$

Each output block $$C_{ij}$$ is computed as a sum of products between corresponding tiles:
\begin{equation}
C_{ij} = \sum_{k=1}^{p} A_{ik} B_{kj}
\end{equation}
This means that one pair of blocks $$A_{ik}$$ and $$B_{kj}$$ is loaded into fast memory,
multiplied to produce a partial result, and then accumulated into $$C_{ij}$$.
By performing the computation block by block, the algorithm minimizes memory movement while maximizing data reuse. This blockwise strategy forms the basis of most efficient matrix multiplication kernels,
as it balances the limited capacity of fast memory with the need to minimize data transfer.

### Putting It All Together

We now combine the ideas of numerically stable softmax, online softmax, and block computation
to describe the complete FlashAttention forward pass.

**High-Level Idea**
The FlashAttention forward pass can be summarized as follows:
 - **Tiling**: Split the query ($$Q$$), key ($$K$), and value ($V$) matrices into smaller tiles that fit in fast on-chip memory. Each query tile $$Q_i \in \mathbb{R}^{B_Q \times d}$$ is processed sequentially against all key/value tiles $$K_j, V_j \in \mathbb{R}^{B_K \times d}$$.
 - **Streaming computation**: Keep the query tile $$Q_i$$ resident in fast memory while streaming the key/value tiles one at a time from slower global memory.
 - **Online updates**: For each pair of tiles $$(Q_i, K_j, V_j)$$, compute partial attention scores and update running per-row quantities:
        \begin{itemize}
            \item \(m_{ij}\): running maximum (for numerical stability)
            \item \(l_{ij}\): normalization factor (for softmax scaling)
            \item \(O_i\): accumulated output
        \end{itemize}
 -  **Numerical stability**: Apply rescaling using the factor $$\exp(m_{ij-1} - m_{ij})$$ to keep all quantities consistent under changing maxima.
 -  **Final normalization**: After iterating through all key/value tiles, normalize $$O_i$$ by the final $$l_{iN}$$ to obtain the exact softmax output for that query tile.
     
**Algorithm**

For each pair of tiles \((Q_i, K_j, V_j)\):
\begin{equation}
S_{ij} = Q_i K_j^{T},
\end{equation}
\begin{equation}
m_{ij} = \max(\text{rowmax}(S_{ij}), m_{ij-1}),
\end{equation}
\begin{equation}
P_{ij} = \exp(S_{ij} - m_{ij}),
\end{equation}
\begin{equation}
l_{ij} = \text{rowsum}(P_{ij}) + l_{ij-1} \exp(m_{ij-1} - m_{ij}),
\end{equation}
\begin{equation}
O_i = \text{diag}(\exp(m_{ij-1} - m_{ij})) O_i + P_{ij} V_j.
\end{equation}

After all key/value tiles are processed, the final normalization is applied:
\begin{equation}
O_i = (\text{diag}(l_{iN}))^{-1} O_i,
\end{equation}
where $$N$$ is the total number of key/value tiles.

Here, $$m_{ij}$$ represents the updated running maximum for each query row,
and $$l_{ij}$$ is the corresponding normalization factor accumulated so far.
The matrix exponential and maximum operations are applied elementwise across rows of the tile.
The rescaling factor $$\exp(m_{ij-1} - m_{ij})$$ ensures that the previously accumulated quantities
are adjusted to the new numerical maximum, maintaining numerical stability across iterations.

Each query tile $$Q_i$$ remains in on-chip memory throughout computation,
while key/value tiles $$(K_j, V_j)$$ are streamed from global memory.
At every step, only small per-row quantities ($$m_i$$, $$l_i$$, and $$O_i$$) are updated,
avoiding the need to materialize the large $$N \times N$$ attention matrix.
This fusion of block computation and online softmax enables FlashAttention
to compute exact attention efficiently while drastically reducing memory traffic.


