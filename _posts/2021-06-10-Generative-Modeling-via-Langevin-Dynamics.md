---
title: "Generative Modeling via Langevin Dynamics"
mathjax: true
layout: post
---

Recently I came across a very interesting method to generative modeling that has its foundations in Physics. Naturally I had to dig in. Generative modeling involves learning a probability distribution from the given dataset and using the learned distribution to produce new samples. Over the last few years, the state of the art results in this type of modeling were all produced by Generative Adversarial Networks (GANs). Recently, a new class of generative models based on learning the gradient of the distribution have gained attention. In this blog post, I will try to explain the Physics principles behind such models. 

To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

[Euler's formula](https://en.wikipedia.org/wiki/Euler%27s_formula) relates the  complex exponential function to the trigonometric functions.

$$ e^{i\theta}=\cos(\theta)+i\sin(\theta) $$

The [Euler-Lagrange](https://en.wikipedia.org/wiki/Lagrangian_mechanics) differential equation is the fundamental equation of calculus of variations.

$$ \frac{\mathrm{d}}{\mathrm{d}t} \left ( \frac{\partial L}{\partial \dot{q}} \right ) = \frac{\partial L}{\partial q} $$

The [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation) describes how the quantum state of a quantum system changes with time.

$$ i\hbar\frac{\partial}{\partial t} \Psi(\mathbf{r},t) = \left [ \frac{-\hbar^2}{2\mu}\nabla^2 + V(\mathbf{r},t)\right ] \Psi(\mathbf{r},t) $$
