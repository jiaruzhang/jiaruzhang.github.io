---
title: 'AMPERE: A Generic Energy Estimation Approach for On-Device Training'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Zesong Wang
  - Hao Wang
  - Tao Song
  - Huai-an Su
  - Rui Chen
  - Yang Hua
  - Xiangwei Zhou
  - Ruhui Ma
  - Miao Pan
  - Haibing Guan

# Author notes (optional)
author_notes:

date: '2025-08-26T00:00:00Z'
doi: '10.1145/3764944.3764951'

# Schedule page publish date (NOT publication's date).
publishDate: '2025-08-26T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['2']

# Publication name and optional abbreviated publication name.
publication: In *ACM SIGMETRICS Performance Evaluation Review*, 53(2), 27–32
publication_short: In *ACM SIGMETRICS Performance Evaluation Review*

abstract: Battery-powered mobile devices (e.g., smartphones, AR/VR glasses, and various IoT devices) are increasingly being used for AI training due to their growing computational power and easy access to valuable, diverse, and real-time data. On-device training is highly energy-intensive, making accurate energy consumption estimation crucial for effective job scheduling and sustainable AI. However, the heterogeneity of devices and the complexity of models challenge the accuracy and generalizability of existing methods. This paper proposes AMPERE, a generic approach for energy consumption estimation in deep neural network (DNN) training. First, we examine the layer-wise energy additivity property of DNNs and strategically partition the entire model into layers for fine-grained energy consumption profiling. Then, we fit Gaussian Process (GP) models to learn from layer-wise energy consumption measurements and estimate a DNN's overall energy consumption based on its layer-wise energy additivity property. We conduct extensive experiments with various types of models across different real-world platforms. The results demonstrate that AMPERE has effectively reduced the Mean Absolute Percentage Error (MAPE) by up to 30%. Moreover, AMPERE is applied in guiding energy-aware pruning, successfully reducing energy consumption by 50%, thereby further demonstrating its generality and potential.
## Summary. An optional shortened abstract.
summary: AMPERE provides a generic, layer-wise Gaussian Process approach for estimating the energy consumption of on-device DNN training across heterogeneous models and devices.
tags: []

# Display this page in the Featured widget?
featured: false

# Custom links
links:
  - name: ACM Digital Library
    url: 'https://dl.acm.org/doi/10.1145/3764944.3764951'
  - name: arXiv (THOR preprint)
    url: 'https://arxiv.org/abs/2501.16397'

url_pdf: 'https://dl.acm.org/doi/pdf/10.1145/3764944.3764951'
url_code: ''
#url_dataset: 'https://github.com/wowchemy/wowchemy-hugo-themes'
#url_poster: ''
#url_project: ''
url_slides: ''
#url_source: 'https://github.com/wowchemy/wowchemy-hugo-themes'
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: ''
  focal_point: ''
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects:
#  - example

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
#slides: example
---
[//]: # (Supplementary notes can be added here, including [code, math, and images]&#40;https://wowchemy.com/docs/writing-markdown-latex/&#41;.)
