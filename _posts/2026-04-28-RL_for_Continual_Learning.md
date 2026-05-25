# RL for Continual Learning

## This page is under construction.

## Why RL over SFT?

Given a some text, say a solution to a hard math problem, there are three ways we could use this text to train a model: 

1) make the text part of the pretraining data corpus,
2) make the text part of post training semi-supervised finetuning or
3) make the text part of RL fine tuning.

Where would you put the text to gain maximum performance as measured by a) ability to solve similar problems and b) ability to solve harder but similar problems?

The paper "SFT Memorizes, RL Generalizes: A Comparative Study of Foundation Model Post-training" showed that RL tends to generalize while SFT tends to memorize.

## Why RL performs better than SFT?

## RL paradigms
### Reinforcement Learning with Verifiable Rewards (PLVR)
### On-Policy Distillation (OPD)

1. RL exhibits less forgetting than SFT.
2. RL = Reverse KL, SFT = Forward KL
3. Reverse KL = Mode seeking, Forward KL = Mode covering
4. RL = On-policy, SFT = Off-policy
5. Forgetting correlates well with KL divergence between the fine-tuned model and the base model on the new task.
6. On-policy nature of RL is responsible for less forgetting when compared to SFT. This produces minimal shift in the KL divergence.
7. SDPO achieves continual learning by using demonstrations inside the LLM context window to get on-policy response and optimizing on it.


<p align="center">
  <img src="/assets/RL_continual_learning_1.png" width="800">
</p>

<p align="center">
  <img src="/assets/RL_continual_learning_2.png" width="800">
</p>

<p align="center">
  <img src="/assets/RL_continual_learning_3.png" width="800">
</p>

<p align="center">
  <img src="/assets/RL_continual_learning_4.png" width="800">
</p>

<p align="center">
  <img src="/assets/RL_continual_learning_5.png" width="800">
</p>
