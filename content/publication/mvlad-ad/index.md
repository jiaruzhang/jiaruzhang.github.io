---
title: 'Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion'

authors:
  - admin
  - Manav Gagvani
  - Can Cui
  - Juntong Peng
  - Ruqi Zhang
  - Ziran Wang

author_notes:
  - 'Equal contribution'
  - 'Equal contribution'
  - 'Equal contribution'
  - ''
  - ''
  - ''

date: '2026-09-27T00:00:00Z'
doi: ''

# Keep the page visible before the conference date.
publishDate: '2026-07-16T00:00:00Z'

# 1 = Conference paper
publication_types: ['1']

publication: In *Proceedings of the 2026 IEEE/RSJ International Conference on Intelligent Robots and Systems*
publication_short: In *IROS 2026*

abstract: >-
  Large Language Models (LLMs) and Vision-Language Models (VLMs) have emerged as promising candidates for end-to-end autonomous driving. However, these models typically face challenges in inference latency, action precision, and explainability. Existing autoregressive approaches struggle with slow token-by-token generation, while prior diffusion-based planners often rely on verbose, general-purpose language tokens that lack explicit geometric structure. In this work, we propose Masked Vision-Language-Action Diffusion for Autonomous Driving (MVLAD-AD), a novel framework designed to bridge the gap between efficient planning and semantic explainability via a masked vision-language-action diffusion model. Unlike methods that force actions into the language space, we introduce a discrete action tokenization strategy that constructs a compact codebook of kinematically feasible waypoints from real-world driving distributions. Moreover, we propose geometry-aware embedding learning to ensure that embeddings in the latent space approximate physical geometric metrics. Finally, an action-priority decoding strategy is introduced to prioritize trajectory generation. Extensive experiments on nuScenes and derived benchmarks demonstrate that MVLAD-AD achieves superior efficiency and outperforms state-of-the-art autoregressive and diffusion baselines in planning precision, while providing high-fidelity and explainable reasoning.

summary: >-
  MVLAD-AD combines masked vision-language-action diffusion with discrete action tokenization, geometry-aware embeddings, and action-priority decoding for efficient, precise, and explainable end-to-end autonomous driving.

tags: []
featured: false

url_pdf: 'mvlad-ad.pdf'
url_code: 'https://github.com/lan-qing/MVLAD-AD'
url_slides: ''
url_video: ''

image:
  caption: ''
  focal_point: ''
  preview_only: false

projects:
---
