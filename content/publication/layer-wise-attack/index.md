---
title: "Leveraging Model Guidance to Extract Training Data from Personalized Diffusion Models"

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Xiaoyu Wu
  - admin
  - Steven Wu

# Author notes (optional)
author_notes:

date: '2025-05-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate:  '2025-05-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *Proceedings of the 42th International Conference on Machine Learning*
publication_short: In *ICML*

abstract: Diffusion Models (DMs) have evolved into advanced image generation tools, especially for few-shot fine-tuning where a pretrained DM is fine-tuned on a small set of images to capture specific styles or objects. Many people upload these personalized checkpoints online, fostering communities such as Civitai and HuggingFace. However, model owners may overlook the potential risks of data leakage by releasing their fine-tuned checkpoints. Moreover, concerns regarding copyright violations arise when unauthorized data is used during fine-tuning.  In this paper, we ask *"Can training data be extracted from these fine-tuned DMs shared online?”* A successful extraction would present not only data leakage threats but also offer tangible evidence of copyright infringement. To answer this, we propose FineXtract, a framework for extracting fine-tuning data.  Our method approximates fine-tuning as a gradual shift in the model's learned distribution---from the original pretrained DM toward the fine-tuning data. By extrapolating the models before and after fine-tuning, we guide the generation toward high-probability regions within the fine-tuned data distribution. We then apply a clustering algorithm to extract the most probable images from those generated using this extrapolated guidance. Experiments on DMs fine-tuned with datasets such as WikiArt, DreamBooth, and real-world checkpoints posted online validate the effectiveness of our method, extracting approximately 20\% of fine-tuning data in most cases, significantly surpassing baseline performance.

## Summary. An optional shortened abstract.
#summary:  we propose a novel DNN-based SCL approach, PAIRE, which incorporates a unique pairwise encoder module with a unidirectional attention layer. By taking both node features and pairwise features as layer input, it can model the internal and external relationships of variable pairs. In addition, we use a skeleton matrix along with a v-tensor, a third-order tensor representing v-structures, as our output, so as to represent the Markov Equivalence Class (MEC), which resolves identifiability inconsistency.
tags: []

# Display this page in the Featured widget?
featured: false

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: 'finextract'
url_code: 'https://github.com/Nicholas0228/FineXtract'
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
