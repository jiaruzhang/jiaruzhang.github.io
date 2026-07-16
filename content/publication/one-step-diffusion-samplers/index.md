---
title: 'One-Step Diffusion Samplers via Self-Distillation and Deterministic Flow'

authors:
  - Pascal Jutras-Dubé
  - admin
  - Ziran Wang
  - Ruqi Zhang

date: '2026-05-02T00:00:00Z'
doi: ''

publishDate: '2026-05-02T00:00:00Z'

# 1 = Conference paper
publication_types: ['1']

publication: In *Proceedings of the 29th International Conference on Artificial Intelligence and Statistics*
publication_short: In *AISTATS*

abstract: >-
  Sampling from unnormalized target distributions is a fundamental yet challenging task in machine learning and statistics. Existing sampling algorithms typically require many iterative steps to produce high-quality samples, leading to high computational costs. We introduce one-step diffusion samplers which learn a step-conditioned ODE so that one large step reproduces the trajectory of many small ones via a state-space consistency loss. We further show that standard ELBO estimates in diffusion samplers degrade in the few-step regime because common discrete integrators yield mismatched forward/backward transition kernels. Motivated by this analysis, we derive a deterministic-flow (DF) importance weight for ELBO estimation without a backward kernel. To calibrate DF, we introduce a volume-consistency regularization that aligns the accumulated volume change along the flow across step resolutions. Our proposed sampler therefore achieves both fast sampling and stable evidence estimate in only one or a few steps. Across challenging synthetic and Bayesian benchmarks, it achieves competitive sample quality with orders-of-magnitude fewer network evaluations while maintaining robust ELBO estimates.

summary: >-
  Self-Distilled One-Step Diffusion Samplers achieve fast one/few-step sampling and stable evidence estimation through state-space and volume consistency with a deterministic-flow importance weight.

tags: []
featured: false

links:
  - name: AISTATS
    url: 'https://virtual.aistats.org/virtual/2026/poster/13446'
  - name: OpenReview
    url: 'https://openreview.net/forum?id=lex9dSXFNn'
  - name: arXiv
    url: 'https://arxiv.org/abs/2512.05251'

url_pdf: 'https://openreview.net/pdf?id=lex9dSXFNn'
url_code: 'https://github.com/PascalJD/one-step-diffusion-samplers'
url_slides: ''
url_video: ''

image:
  caption: ''
  focal_point: ''
  preview_only: false

projects:
---
