---
title: 'Improving Bayesian Neural Networks by Adversarial Sampling'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Yang Hua
  - Tao Song
  - Hao Wang
  - Zhengui Xue
  - Ruhui Ma
  - Haibing Guan

# Author notes (optional)
author_notes:
#  - 'Equal contribution'
#  - 'Equal contribution'

date: '2022-06-28T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2022-06-28T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *Proceedings of the AAAI Conference on Artificial Intelligence*
publication_short: In *AAAI*

abstract: Bayesian neural networks (BNNs) have drawn extensive interest due to the unique probabilistic representation framework. However, Bayesian neural networks have limited publicized deployments because of the relatively poor model performance in real-world applications. In this paper, we argue that the randomness of sampling in Bayesian neural networks causes errors in the updating of model parameters during training and some sampled models with poor performance in testing. To solve this, we propose to train Bayesian neural networks with Adversarial Distribution as a theoretical solution. To avoid the difficulty of calculating Adversarial Distribution analytically, we further present the Adversarial Sampling method as an approximation in practice. We conduct extensive experiments with multiple network structures on different datasets, eg, CIFAR-10 and CIFAR-100. Experimental results validate the correctness of the theoretical analysis and the effectiveness of the Adversarial Sampling on improving model performance. Additionally, models trained with Adversarial Sampling still keep their ability to model uncertainties and perform better when predictions are retained according to the uncertainties, which further verifies the generality of the Adversarial Sampling approach.
  
# Summary. An optional shortened abstract.
summary:  In this paper, we argue that the randomness of sampling in Bayesian neural networks causes errors in the updating of model parameters during training and some sampled models with poor performance in testing. We propose to train Bayesian neural networks with Adversarial Distribution as a theoretical solution. We further present the Adversarial Sampling method as an approximation in practice.

tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: ''
url_code: 'https://github.com/AISIGSJTU/AS'
#url_dataset: 'https://github.com/wowchemy/wowchemy-hugo-themes'
#url_poster: ''
#url_project: ''
url_slides: 'slides/AS.pdf'
#url_source: 'https://github.com/wowchemy/wowchemy-hugo-themes'
url_video: 'https://aaai-2022.virtualchair.net/poster_aaai385'

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

{{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

[//]: # (Supplementary notes can be added here, including [code, math, and images]&#40;https://wowchemy.com/docs/writing-markdown-latex/&#41;.)
