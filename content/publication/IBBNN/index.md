---
title: 'Information Bound and its Applications in Bayesian Neural Networks'

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

date: '2023-10-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2023-09-23T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *Proceedings of the 26th European Conference on Artificial Intelligence*
publication_short: In *ECAI*

abstract: "Bayesian neural networks (BNNs) have drawn extensive interest because of their distinctive probabilistic representation framework. However, despite its recent success, little work focuses on the information-theoretic understanding of Bayesian neural networks. In this paper, we propose Information Bound as a metric of the amount of information in Bayesian neural networks. Different from mutual information on deterministic neural networks where modification of network structure or specific input data is usually necessary, Information Bound can be easily estimated on current Bayesian neural networks without any modification of network structures or training processes. By observing the trend of Information Bound during training, we demonstrate the existence of the ``critical period'' in Bayesian neural networks. Besides, we show that the Information Bound can be used to judge the confidence of the model prediction and to detect out-of-distribution datasets. Based on these observations of model interpretation, we propose Information Bound regularization and Information Bound variance regularization methods. The Information Bound regularization encourages models to learn the minimum necessary information and improves the model generality and robustness. The Information Bound variance regularization encourages models to learn more about complex samples with low Information Bound. Extensive experiments on KMNIST, Fashion-MNIST, CIFAR-10, and CIFAR-100 verify the effectiveness of the proposed regularization methods."
  
# Summary. An optional shortened abstract.
summary:  "In this paper, we propose Information Bound as a metric of the amount of information in Bayesian neural networks. Different from mutual information on deterministic neural networks where modification of network structure or specific input data is usually necessary, Information Bound can be easily estimated on current Bayesian neural networks without any modification of network structures or training processes."
tags: []

# Display this page in the Featured widget?
featured: true

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'IBBNN.pdf'
url_code: 'https://github.com/AISIGSJTU/IBBNN'
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

{{% callout note %}}
Click the _Cite_ button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the _Slides_ button to check out the example.
{{% /callout %}}

[//]: # (Supplementary notes can be added here, including [code, math, and images]&#40;https://wowchemy.com/docs/writing-markdown-latex/&#41;.)
