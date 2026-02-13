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
 - **Tiling**: Split the query $$Q$$, key $$K$$, and value $$V$$ matrices into smaller tiles that fit in fast on-chip memory. Each query tile $$Q_i \in \mathbb{R}^{B_Q \times d}$$ is processed sequentially against all key/value tiles $$K_j, V_j \in \mathbb{R}^{B_K \times d}$$.
 - **Streaming computation**: Keep the query tile $$Q_i$$ resident in fast memory while streaming the key/value tiles one at a time from slower global memory.
 - **Online updates**: For each pair of tiles $$(Q_i, K_j, V_j)$$, compute partial attention scores and update running per-row quantities: $$m_{ij}$$: running maximum (for numerical stability), $$l_{ij}$$: normalization factor (for softmax scaling), $$O_i$$: accumulated output
 -  **Numerical stability**: Apply rescaling using the factor $$\exp(m_{ij-1} - m_{ij})$$ to keep all quantities consistent under changing maxima.
 -  **Final normalization**: After iterating through all key/value tiles, normalize $$O_i$$ by the final $$l_{iN}$$ to obtain the exact softmax output for that query tile.
     
**Algorithm**

For each pair of tiles $$(Q_i, K_j, V_j)$$:
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

## Backward Pass

We now describe the backward pass of flash attention, which computes gradients with respect to the query, key, and value matrices without storing the full $$N\times N$$ attention matrix. Like the forward pass, it uses the principles of tiling, online recomputation, and numerical stability.

### Gradient Calculations
Here I will provide a detailed derivation of the gradients. The derivation given in the flash attention paper is a little confusing, and hence I will provide my own derivation.

We have the following equations:
\begin{equation}
S = \frac{QK^T}{\sqrt{d}},\qquad P = \text{softmax}(S), \qquad O = PV.
\end{equation}
We can write these equations explicitly with all the indices as follows:
\begin{equation}
S_{ij} = \sum_k\frac{Q_{ik}K_{jk}}{\sqrt{d}},
\end{equation}
\begin{equation}
P_{ij}=\text{softmax}(S_{ij})=\frac{\exp{(\sum_kQ_{ik}K_{jk}/\sqrt{d})}}{\sum_l \exp{(\sum_mQ_{im}K_{lm}/\sqrt{d})}},
\end{equation}
\begin{equation}
O_{ij} = \sum_kP_{ik}V_{kj}.
\end{equation}
Our problem is to find the gradients
\begin{equation}
dQ_{ij} = \frac{\partial \mathcal{L}}{\partial Q_{ij}},\qquad dK_{ij} = \frac{\partial \mathcal{L}}{\partial K_{ij}},\qquad dV_{ij} = \frac{\partial \mathcal{L}}{\partial V_{ij}},
\end{equation}
given the gradient with respect to the output $$ dO_{ij} = \partial \mathcal{L}/\partial O_{ij}$$.

Now, let us get the gradients one by one. We will start with the easiest.

$$
\begin{split}
        dV_{ij} &= \frac{\partial \mathcal{L}}{\partial V_{ij}}\\
        &= \sum_{mn}\frac{\partial \mathcal{L}}{\partial O_{mn}}\frac{\partial O_{mn}}{\partial V_{ij}}\\
        &= \sum_{mn}\frac{\partial \mathcal{L}}{\partial O_{mn}}\frac{\partial (\sum_{l}P_{ml}V_{ln})}{\partial V_{ij}}\\
        &= \sum_{mnl}\frac{\partial \mathcal{L}}{\partial O_{mn}}P_{ml}\delta_{il}\delta_{nj}\\
        &= \sum_{m}\frac{\partial \mathcal{L}}{\partial O_{mj}}P_{mi}\\
        &=\sum_{m}(P^T)_{im}\frac{\partial \mathcal{L}}{\partial O_{mj}}.
\end{split}
$$

This can be written as

$$dV_{ij} = \sum_mP^T_{im}dO_{mj}.
$$
Now let us now on to $$dQ$$ and $$dK$$. 
In the above derivation, we used Kronecker delta function, which is defined as
$$
\delta_{ij} =
\begin{cases}
1, & \text{if } i = j,\\[4pt]
0, & \text{if } i \neq j.
\end{cases}
$$
Using chain rule, we have
$$
    \begin{split}
        dQ_{ij}&=\sum_{mnklrt}\frac{\partial \mathcal{L}}{\partial O_{mn}}\frac{\partial O_{mn}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}\frac{\partial S_{rt}}{\partial Q_{ij}}\\
        &=\sum_{klrt}\frac{\partial \mathcal{L}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}\frac{\partial S_{rt}}{\partial Q_{ij}}
    \end{split}
$$
Similarly
$$
    \begin{split}
        dK_{ij}&=\sum_{mnklrt}\frac{\partial \mathcal{L}}{\partial O_{mn}}\frac{\partial O_{mn}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}\frac{\partial S_{rt}}{\partial K_{ij}}\\
        &=\sum_{klrt}\frac{\partial \mathcal{L}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}\frac{\partial S_{rt}}{\partial K_{ij}}
    \end{split}
$$
So for both $$dQ$$ and $$dK$$ we need to find $$\partial P_{kl}/\partial S_{rt}$$. We have
$$
P_{kl}=\frac{e^{S_{kl}}}{\sum_me^{S_{km}}}
$$
Therefore
$$
    \begin{split}
        \frac{\partial P_{kl}}{\partial S_{rt}}&=\frac{\partial}{\partial S_{rt}}\left(\frac{e^{S_{kl}}}{\sum_me^{S_{km}}}\right)\\
        &=\frac{e^{S_{kl}}\delta_{kr}\delta_{lt}(\sum_me^{S_{km}})-e^{S_{kl}}(\sum_me^{S_{km}}\delta_{kr}\delta_{mt})}{(\sum_me^{S_{km}})^2}\\
        &=\frac{e^{S_{kl}}\delta_{kr}\delta_{lt}(\sum_me^{S_{km}})-e^{S_{kl}}e^{S_{kt}}\delta_{kr}}{(\sum_me^{S_{km}})^2}\\
        &=\frac{e^{S_{kl}}}{\sum_me^{S_{km}}}\delta_{kr}\left(\delta_{lt}-\frac{e^{S_{kt}}}{\sum_me^{S_{km}}}\right)\\
        &=P_{kl}\delta_{kr}(\delta_{lt} - P_{kt}).
    \end{split}
$$
Using the above result,
\begin{equation*}
    \begin{split}
        \sum_{kl}\frac{\partial \mathcal{L}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}&=\sum_{kl}\frac{\partial \mathcal{L}}{\partial P_{kl}}P_{kl}\delta_{kr}(\delta_{lt} - P_{kt})\\
        &=P_{rt}\left(\frac{\partial \mathcal{L}}{\partial P_{rt}}-\sum_{l}P_{rl}\frac{\partial \mathcal{L}}{\partial P_{rl}}\right)
    \end{split}
\end{equation*}
Now, the second term
\begin{equation*}
    \begin{split}
        \sum_{l}P_{rl}\frac{\partial \mathcal{L}}{\partial P_{rl}}&=\sum_{mnl}P_{rl}\frac{\partial \mathcal{L}}{\partial O_{mn}}\frac{\partial O_{mn}}{\partial P_{rl}}\\
        &=\sum_{mnlt}P_{rl}\frac{\partial \mathcal{L}}{\partial O_{mn}}\delta_{mr}\delta_{tl}V_{tn}\\
        &=\sum_{nl}P_{rl}\frac{\partial \mathcal{L}}{\partial O_{rn}}V_{ln}\\
        &=\sum_{n}O_{rn}\frac{\partial \mathcal{L}}{\partial O_{rn}}
    \end{split}
\end{equation*}
The above result allows us to trade the summation over the sequence length ($l$) for the summation over the head dimension ($d$). Therefore,
\begin{equation*}
    \sum_{kl}\frac{\partial \mathcal{L}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}=P_{rt}\frac{\partial \mathcal{L}}{\partial P_{rt}}-P_{rt}\sum_{n}O_{rn}\frac{\partial \mathcal{L}}{\partial O_{rn}}.
\end{equation*}
Notice that
\[dS_{rt}=\frac{\partial \mathcal{L}}{\partial S_{rt}} = \sum_{kl}\frac{\partial \mathcal{L}}{\partial P_{kl}}\frac{\partial P_{kl}}{\partial S_{rt}}\]
So we have,
\[dS_{rt} = P_{rt}\left(dP_{rt}-\sum_nO_{rn}dO_{rn}\right).\]
We will define $D_r=\sum_nO_{rn}dO_{rn}$. Therefore, we finally have
\[dS_{rt}=P_{rt}(dP_{rt}-D_r).\]
We still have to find $dP_{rt}$ to make use of the above formula in computations. We have
\begin{equation*}
    \begin{split}
        dP_{rt} = \frac{\partial\mathcal{L}}{\partial P_{rt}} &= \sum_{mn}\frac{\partial\mathcal{L}}{\partial O_{mn}}\frac{\partial O_{mn}}{\partial P_{rt}}\\
        &=\sum_{mnl}dO_{mn}\delta_{mr}\delta_{lt}V_{ln}\\
        &=\sum_{n}dO_{rn}V_{tn}\\
        &=\sum_{n}dO_{rn}V^T_{nt}
    \end{split}
\end{equation*}
Now we can use $dS_{rt}$ to find the gradients with respect to $Q$ and $K$,
\begin{equation*}
    \begin{split}
dQ_{ij}&=\sum_{rt}\frac{\partial\mathcal{L}}{\partial S_{rt}} \frac{\partial S_{rt}}{\partial Q_{ij}}\\
&=\frac{1}{\sqrt{d}}\sum_{rtm}dS_{rt}\delta_{ri}\delta_{mj}K_{tm}\\
&=\frac{1}{\sqrt{d}}\sum_tdS_{it}K_{tj}.
    \end{split}
\end{equation*}
Similarly,
\begin{equation*}
    \begin{split}
dK_{ij}&=\sum_{rt}\frac{\partial\mathcal{L}}{\partial S_{rt}} \frac{\partial S_{rt}}{\partial K_{ij}}\\
&=\frac{1}{\sqrt{d}}\sum_{rtm}dS_{rt}Q_{rm}\delta_{ti}\delta_{mj}\\
&=\frac{1}{\sqrt{d}}\sum_rdS_{ri}Q_{rj}\\
&=\frac{1}{\sqrt{d}}\sum_rdS^T_{ir}Q_{rj}.
    \end{split}
\end{equation*}
The final set of equations are
\begin{equation*}
    \begin{split}
        dV_{ij} &= \sum_kP^T_{ik}dO_{kj},\\
        dP_{ij}&=\sum_{k}dO_{ik}V^T_{kj}\\
        D_i&=\sum_jO_{ij}dO_{ij},\\
        dS_{ij}&=P_{ij}(dP_{ij}-D_i),\\
        dQ_{ij} &= \frac{1}{\sqrt{d}}\sum_kdS_{ik}K_{kj},\\
        dK_{ij} &= \frac{1}{\sqrt{d}}\sum_kdS^T_{ik}Q_{kj}.
    \end{split}
\end{equation*}
We can use all the above formulae to get the gradients.

\subsubsection{The Log-Sum-Exp Trick and Forward-Pass Statistics}

A standard implementation of attention would store the full probability matrix
\(P \in \mathbb{R}^{N \times N}\) during the forward pass so that it can be
reused in the backward pass.  
However, this is infeasible for long sequences: the matrix \(P\) requires
\(O(N^{2})\) memory, which quickly exceeds GPU capacity even for moderate
values of \(N\).  FlashAttention avoids this cost entirely by \emph{never
storing \(P\)}.  Instead, the backward pass simply recomputes the relevant
blocks of \(P\) on the fly.

To make this recomputation possible, the forward pass stores only a single
scalar per query row: the \emph{log-sum-exp} value
\[
M_i \;=\; m_i + \log(\ell_i)
\;=\; \log\!\left(\sum_{j} e^{S_{ij}}\right),
\]
where \(m_i = \max_j S_{ij}\) is the running maximum accumulated across key
tiles, and
\(\ell_i = \sum_j e^{S_{ij}-m_i}\) is the corresponding normalization factor
computed in a numerically stable way.  
These two quantities are maintained incrementally during the tiled forward
computation, and combined into \(M_i\) at the end of the forward pass.

During the backward pass, when a block of scores \(S_{ij}\) is recomputed,
the corresponding block of softmax probabilities is recovered using the
log-sum-exp identity:
\[
P_{ij}
\;=\;
\exp\!\bigl(S_{ij} - M_i\bigr).
\]
This works because
\[
\exp(S_{ij} - M_i)
=\frac{e^{S_{ij}-m_i}}{\sum_k e^{S_{ik}-m_i}}
=\operatorname{softmax}(S_{ij}).
\]
Thus, storing \(M_i\) is sufficient to compute the correct softmax of\(S\) exactly,
without ever computing maximum and normalization factor again.

\subsubsection{Blockwise Form of the Backward Pass}

To connect the index-level gradients with the FlashAttention implementation, we now express all
quantities in terms of \emph{blocks} (tiles).  
Let the sequence dimension be partitioned into query tiles \(Q_i \in \mathbb{R}^{B_Q \times d}\) and 
key/value tiles \(K_j, V_j \in \mathbb{R}^{B_K \times d}\).
For each pair of tiles \((i,j)\), define the block matrices
\[
S_{ij} \in \mathbb{R}^{B_Q \times B_K},\qquad
P_{ij} \in \mathbb{R}^{B_Q \times B_K},\qquad
dP_{ij}, dS_{ij} \in \mathbb{R}^{B_Q \times B_K}.
\]

\paragraph{Blockwise equations.}
The global elementwise formulas translate blockwise into
\[
S_{ij} = \frac{Q_i K_j^{T}}{\sqrt{d}},
\qquad
P_{ij} = \exp\!\left(S_{ij} - M_i[:,{\!}\text{ None}]\right),
\]
\[
dP_{ij} = dO_i V_j^{T},
\qquad
D_i = \mathrm{rowsum}(O_i \odot dO_i),
\]
\[
dS_{ij} = P_{ij} \odot \left(dP_{ij} - D_i[:,{\!}\text{ None}]\right),
\]
\[
dV_j \;\mathrel{+}= P_{ij}^{T} dO_i,\qquad
dK_j \;\mathrel{+}= \frac{1}{\sqrt{d}}\, dS_{ij}^{T} Q_i,\qquad
dQ_i \;\mathrel{+}= \frac{1}{\sqrt{d}}\, dS_{ij} K_j.
\]

Here, \(M_i\) is the log-sum-exp normalization vector stored from the forward pass (None in $M_i[:, \text{ None}]$ is to broadcast the values of $M_i$ across the columns), 
and \(D_i\) ($\odot$ is elementwise multiplication) is a per-row scalar defined by
\[
D_i = \sum_{\alpha} O_{i\alpha} dO_{i\alpha}.
\]
The notation \(\mathrm{rowsum}(\cdot)\) sums across columns.

\paragraph{Blockwise backward algorithm.}
Using these quantities, the backward pass proceeds as follows.

\begin{itemize}
    \item \textbf{Preprocess:}  
    For each query tile \(i\), load \(O_i\) and \(dO_i\) and compute  
    \[
        D_i = \mathrm{rowsum}(O_i \odot dO_i).
    \]
    This is a small per-row vector reused throughout the backward pass.

    \item \textbf{Gradients for \(K\) and \(V\):}  
    For each key/value tile \(j\):
    \begin{enumerate}
        \item Load \(K_j, V_j\).
        \item For each query tile \(i\):
        \[
        S_{ij} = \frac{Q_i K_j^{T}}{\sqrt{d}},\qquad
        P_{ij} = \exp\!\left(S_{ij} - M_i[:,\text{ None}]\right),
        \]
        \[
        dV_j \mathrel{+}= P_{ij}^{T} dO_i,
        \qquad
        dP_{ij} = dO_i V_j^{T},
        \]
        \[
        dS_{ij} = P_{ij} \odot \left(dP_{ij} - D_i[:,\text{ None}]\right),
        \qquad
        dK_j \mathrel{+}= \frac{1}{\sqrt{d}}\, dS_{ij}^{T} Q_i.
        \]
    \end{enumerate}

    \item \textbf{Gradients for \(Q\):}  
    For each query tile \(i\):
    \begin{enumerate}
        \item Initialize \(dQ_i = 0\).
        \item For each key/value tile \(j\):
        \[
        S_{ij} = \frac{Q_i K_j^{T}}{\sqrt{d}},\qquad
        P_{ij} = \exp\!\left(S_{ij} - M_i[:,\text{ None}]\right),
        \]
        \[
        dP_{ij} = dO_i V_j^{T},\qquad
        dS_{ij} = P_{ij} \odot \left(dP_{ij} - D_i[:,\text{ None}]\right),
        \]
        \[
        dQ_i \mathrel{+}= \frac{1}{\sqrt{d}}\, dS_{ij} K_j.
        \]
    \end{enumerate}
\end{itemize}

\noindent
This blockwise formulation reproduces the exact gradients of standard attention while avoiding
materialization of the full \(N\times N\) matrices \(P\) and \(dP\), enabling the memory-efficient backward pass used in FlashAttention.


