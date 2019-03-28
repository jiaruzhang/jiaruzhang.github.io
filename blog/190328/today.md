---
title: Some thoughts about feature fusion network
create: 2019.3.28
modified: 2019.3.28
tags: Notes
---
[TOC]

Feature fusion neural network I defined before is a network with multiple input features and single output features.
Here I use a two feature fusion network to explain it: $\mathbf{x}, \mathbf{z} \rightarrow^{NN} \mathbf{y}$.

I list some of questions and my thoughts here:

1. Why $\mathbf{x}$ and $\mathbf{y}$ can be called two features here? Maybe they are taken from different
domains and can be distinguished by humans easily, but is there any quantified criteria to measure
whether they are two different features or just two parts of one features? On the other hand, if we mix them
as one feature $[x;z]$, can they be separated as two features again?

    I guess different feature vectors come from different high-dimensional probability distributions.
    Moreover, the inner correlation coefficient is larger than outer correlation coefficient.

2. Why tensor-based feature fusion is helpful? If the neural network to make final decision is powerful enough
(can always find the global optimal solution), the feature fusion method will be useless. But how to combine the
feature fusion process and the neural network structure?

