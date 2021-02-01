# GAN Training for Discrete Generators

This project was made for the University of Cambridge's Part III "Probabilistic Machine Learning" course.

The repository contains an implementation of a GAN network generating English first names. The training method is based on a rewriting of the usual optimisation problem using importance sampling. The derived generator training equation requires the computation (and differentiation) of the generation likelihood of a given sequence.

For more information, read the ```docs/``` folder.
