---
title: "Generative Modeling via Langevin Dynamics"
mathjax: true
layout: post
---

Recently I came across a very interesting method to generative modeling that has its foundations in Physics. Naturally I had to dig in. Generative modeling involves learning a probability distribution from the given dataset and using the learned distribution to produce new samples. Over the last few years, the state of the art results in this type of modeling were all produced by Generative Adversarial Networks (GANs). Recently, a new class of generative models called score-based models have gained attention. As it happens, the main tools of these models come from what is called the Brownian motion - the chaotic motion of a particle (like a pollen) floating on a liquid (like water). In this blog post, I will try to explain the physics principles behind score based models. First, I will go over the physics and then come back to its connection to score-based models.

When a particle like a pollen floats on water, it experiences a constant barrage of collisions from the molecules of the water. Occasionally, these collisions are energetic enough to kick the particle in a particular direction. The particle as a result, exhibits a random zig zag motion. This is called the Brownian motion, named after Raboert Brown who first discovered the phenomenon in 1827, while looking through a microscope a pollen immersed in water.


It can be established that the dynamical equation describing the chaotic motion of the particle to be

$$ \frac{\mathrm{d}x}{\mathrm{d}t} = - a \nabla U + b\eta(t) $$

where the first term describes the friction force on the particle and the second term describes the random fluctuations in the density of the liquid. In fact $$\eta(t)$$ is a white noise term with properties $$\langle \eta (t) \rangle = 0$$ and $$\langle \eta (t_1) \eta (t_2) \rangle = \delta(t_1-t_2)$$. One important aspect of this equation is that it only captures a coarse grained state of the particle as opposed to a fine grained state that precisely takes into account each and every collision with the molecules. In fact, we do not desire such a description as we will have to then specify the exact position and momentum of each and every molecule in order to find the motion of the particle. In the above equation, all that random fluctuations is abstracted away in the white noise term $$\eta(t)$$. To add new posts, simply add a file in the `_posts` directory that follows the convention `YYYY-MM-DD-name-of-post.ext` and includes the necessary front matter. Take a look at the source for this post to get an idea about how it works.

[Euler's formula](https://en.wikipedia.org/wiki/Euler%27s_formula) relates the  complex exponential function to the trigonometric functions.

$$ e^{i\theta}=\cos(\theta)+i\sin(\theta) $$

The [Euler-Lagrange](https://en.wikipedia.org/wiki/Lagrangian_mechanics) differential equation is the fundamental equation of calculus of variations.

$$ \frac{\mathrm{d}}{\mathrm{d}t} \left ( \frac{\partial L}{\partial \dot{q}} \right ) = \frac{\partial L}{\partial q} $$

The [Schrödinger equation](https://en.wikipedia.org/wiki/Schr%C3%B6dinger_equation) describes how the quantum state of a quantum system changes with time.

$$ i\hbar\frac{\partial}{\partial t} \Psi(\mathbf{r},t) = \left [ \frac{-\hbar^2}{2\mu}\nabla^2 + V(\mathbf{r},t)\right ] \Psi(\mathbf{r},t) $$
