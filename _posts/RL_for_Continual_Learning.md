# RL for Continual Learning

## This page is under construction.

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
