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

Supervised Fine-Tuning (SFT) forms the foundation of almost every modern LLM post-training pipeline. After pretraining on massive text corpora, a language model already possesses a broad understanding of language and factual knowledge. However, it may not reliably follow user instructions or produce responses that align with human expectations. SFT addresses this problem by training the model on a collection of high-quality prompt-response pairs created by human annotators. Given a prompt, the objective of the model is simply to reproduce the corresponding demonstration.

Let $x$ denote the input prompt and let $$y = (y_1,y_2,\ldots,y_T)$$ denote the corresponding response consisting of $T$ tokens. Modern language models generate responses autoregressively, producing one token at a time conditioned on the prompt and all previously generated tokens. Throughout this article, we denote the language model by the policy $$\pi_\theta(y_t|x,y_{<t}),$$ where $y_{<t}=(y_1,\ldots,y_{t-1})$ denotes the prefix generated before the $$t$$-th token and $$\theta$$ denotes the model parameters. Under the autoregressive assumption, the probability of generating the complete response can therefore be written as

\begin{equation}
\pi_\theta(y|x)=\prod_{t=1}^{T}\pi_\theta(y_t|x,y_{<t}).
\end{equation}

Suppose we are given a supervised dataset

\begin{equation}
\mathcal{D}={(x_i,y_i)}_{i=1}^{N},
\end{equation}

where each prompt $$x_i$$ is paired with a reference response $$y_i$$. The goal of supervised fine-tuning is to find the model parameters $$\theta$$ that maximize the likelihood of the demonstrations contained in this dataset. Mathematically, this can be written as

\begin{equation}
\max_\theta
\sum_{(x,y)\in\mathcal{D}}
\log\pi_\theta(y|x).
\end{equation}

Using the autoregressive decomposition given by equation (3), we obtain

\begin{equation}
\log\pi_\theta(y|x)
===================

\sum_{t=1}^{T}
\log\pi_\theta(y_t|x,y_{<t}),
\end{equation}

which leads to the familiar supervised fine-tuning objective

\begin{equation}
\mathcal{L}_{\mathrm{SFT}}
==========================

*

\sum_{(x,y)\in\mathcal{D}}
\sum_{t=1}^{T}
\log
\pi_\theta(y_t|x,y_{<t}).
\end{equation}

Equation (6) is simply the negative log-likelihood or, equivalently, the token-level cross-entropy loss used to train modern language models. During training, the correct prefix $$y_{<t}$$ is always provided to the model when predicting the next token. This procedure, known as *teacher forcing*, provides a dense supervision signal because every token in every demonstration contributes directly to the optimization objective. As a result, SFT is computationally efficient, stable to optimize, and produces an excellent initialization for subsequent post-training algorithms.

Despite its simplicity and effectiveness, supervised fine-tuning has several important limitations. The most fundamental limitation is that the objective in equation (6) encourages the model to imitate demonstrations rather than optimize the quality of its responses. In many reasoning tasks, there may exist multiple valid reasoning paths leading to the same correct answer, yet SFT only reinforces the particular demonstrations present in the training data. Furthermore, during training the model always conditions on the ground-truth prefix, whereas during inference it must condition on its own previously generated tokens. This mismatch, commonly referred to as *exposure bias*, can cause errors to accumulate during generation. These limitations motivate the need for post-training methods that optimize models using richer feedback signals, beginning with reinforcement learning.
