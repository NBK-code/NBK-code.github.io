---
title: "LLM Post Training"
mathjax: true
layout: post
---

# 1. Introduction

The remarkable capabilities of modern large language models (LLMs) are the result of a multi-stage training pipeline. After large-scale pretraining on internet-scale text corpora, models are typically refined through a post-training stage designed to align their behavior with human expectations and task-specific objectives. Early post-training approaches relied primarily on supervised fine-tuning (SFT), where models were trained to imitate high-quality demonstrations produced by human annotators. While SFT significantly improves instruction following and conversational abilities, much of the recent progress in reasoning has been driven by reinforcement learning (RL) based methods. These methods optimize the model's policy using reward signals derived from the quality of its generated responses. Alongside reinforcement learning, preference optimization has emerged as another important paradigm, where models are trained directly from human preference data to better align their outputs with human choices, without explicitly relying on reinforcement learning.

Although supervised fine-tuning provides a strong initialization for language models, it fundamentally optimizes the probability of reproducing demonstration trajectories rather than the quality of the final outcomes. In many reasoning tasks, there may exist multiple valid chains of thought leading to the same correct answer, yet SFT only reinforces the specific trajectories present in the training data. Furthermore, during inference, language models generate responses autoregressively by conditioning on their own previously generated tokens, creating a mismatch between training and deployment known as exposure bias. Consequently, imitation alone may be insufficient for developing robust reasoning capabilities. Reinforcement learning addresses these limitations by optimizing models directly with respect to the quality of their generated outputs using reward or feedback signals, allowing models to discover behaviors that extend beyond the demonstrations in the training data. Preference optimization approaches the same problem from a different perspective by learning directly from pairwise human preferences, avoiding the need for explicit reward modeling while still optimizing the policy to better align model behavior with human judgments.

This article presents a mathematical overview of the major post-training methods used to train modern large language models after pretraining. We begin with supervised fine-tuning (SFT), the foundation of nearly all contemporary post-training pipelines, before introducing the mathematical foundations of reinforcement learning through policy gradient methods and their evolution into Proximal Policy Optimization (PPO), Group Relative Policy Optimization (GRPO), and On-Policy Distillation (OPD). Finally, we examine Direct Preference Optimization (DPO), one of the most influential preference optimization methods. Together, these methods trace the evolution of modern LLM post-training and provide the mathematical foundations needed to understand today's state-of-the-art reasoning models.

# 2. Supervised Fine-Tuning

Supervised fine-tuning (SFT) serves as the foundation of nearly every modern LLM post-training pipeline. After pretraining, a language model already possesses broad linguistic knowledge and general reasoning capabilities, but its responses may not consistently follow user instructions or align with human expectations. SFT addresses this problem by training the model on high-quality prompt-response pairs collected from human annotators or curated datasets. The objective is straightforward: given an input prompt, the model should generate the corresponding reference response. Despite the emergence of more sophisticated post-training algorithms, SFT remains an indispensable first stage because it provides the initialization upon which nearly all subsequent optimization methods are built.

Let (x) denote an input prompt and let

[
y = (y_1, y_2, \ldots, y_T)
]

denote the corresponding response consisting of (T) tokens. Modern language models generate responses autoregressively, predicting one token at a time conditioned on the prompt and all previously generated tokens. Throughout this article, we represent the language model by the policy

[
\pi_\theta(y_t \mid x, y_{<t}),
]

where (y_{<t} = (y_1, \ldots, y_{t-1})) denotes the prefix generated before the (t)-th token, and (\theta) represents the model parameters. Under the autoregressive assumption, the probability of generating the complete response can be factorized as

[
\pi_\theta(y \mid x)
====================

\prod_{t=1}^{T}
\pi_\theta(y_t \mid x, y_{<t}).
]

Suppose we are given a supervised dataset

[
\mathcal{D}
===========

{(x_i, y_i)}_{i=1}^{N},
]

where each prompt (x_i) is paired with a reference response (y_i). Supervised fine-tuning learns the model parameters by maximizing the likelihood of these demonstrations. The corresponding optimization problem is

[
\max_{\theta}
\sum_{(x,y)\in\mathcal{D}}
\log
\pi_\theta(y \mid x).
]

Using the autoregressive factorization, the log-likelihood of each response can be written as

[
\log
\pi_\theta(y \mid x)
====================

\sum_{t=1}^{T}
\log
\pi_\theta(y_t \mid x, y_{<t}),
]

which leads to the familiar negative log-likelihood objective

[
\mathcal{L}_{\mathrm{SFT}}
==========================

*

\sum_{(x,y)\in\mathcal{D}}
\sum_{t=1}^{T}
\log
\pi_\theta(y_t \mid x, y_{<t}).
]

Minimizing this objective is equivalent to minimizing the token-level cross-entropy loss between the model's predicted distribution and the reference response. During training, the ground-truth prefix (y_{<t}) is always provided as input when predicting the next token, a procedure commonly known as *teacher forcing*. This provides a dense supervision signal, since every token in every demonstration contributes directly to the optimization objective.

The success of supervised fine-tuning stems from its simplicity. It is computationally efficient, stable to optimize, and requires only high-quality demonstration data. More importantly, it produces a strong policy that serves as the starting point for nearly all modern post-training algorithms. Reinforcement learning methods such as PPO, GRPO, and OPD, as well as preference optimization methods such as DPO, all begin from an SFT-trained model rather than a randomly initialized language model.

Despite these advantages, supervised fine-tuning has several important limitations. First, it optimizes the likelihood of reproducing demonstrations rather than the quality of the generated responses. In other words, the objective encourages the model to imitate a particular demonstration instead of directly optimizing the desired outcome. Second, many reasoning tasks admit multiple valid solution paths, yet SFT reinforces only the demonstrations available in the training data. Finally, because training conditions on the ground-truth prefix while inference conditions on the model's own predictions, a mismatch known as *exposure bias* arises, allowing errors to accumulate during generation. These limitations motivate the development of post-training methods that optimize language models using richer feedback signals, beginning with reinforcement learning methods based on policy optimization.

