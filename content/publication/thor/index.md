---
title: 'THOR: Generic Energy Estimation for On-device Training'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Zesong Wang
  - admin
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
  - 'Equal contribution'
  - 'Equal contribution'

date: '2023-10-27T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2023-10-27T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['3']

# Publication name and optional abbreviated publication name.
publication: In Submission
publication_short: In Submission

#abstract: Supervised Causal Learning (SCL) aims to obtain causal relations from observational data, leveraging the model learned from prior datasets with ground truth causal relations. Deep Neural Network (DNN) based SCL, which learns DNNs as causal models, has gained significant attention with its numerous advantages. A recently proposed transformer-based architecture employs sample-wise and node-wise attention mechanisms to capture representations of individual variables. In the inference stage, the trained model takes the test data as input and outputs a Directed Acyclic Graph (DAG) represented as a weighted adjacency matrix. However, this paper identifies two limitations of these approaches. First, using the adjacency matrix as a learning target yield inconsistent results w.r.t. structure identifiability. Second, current network architecture does not adequately encode the essential characteristics for learning causal structures. To address these issues, we propose a novel DNN-based SCL approach, PAIRE, which incorporates a unique pairwise encoder module with a unidirectional attention layer. By taking both node features and pairwise features as layer input, it can model the internal and external relationships of variable pairs. In addition, we use a skeleton matrix along with a v-tensor, a third-order tensor representing v-structures, as our output, so as to represent the Markov Equivalence Class (MEC), which resolves identifiability inconsistency. Empirical evidence indicates PAIRE significantly outperforms other DNN-based SCL approaches.
## Summary. An optional shortened abstract.
#summary:  we propose a novel DNN-based SCL approach, PAIRE, which incorporates a unique pairwise encoder module with a unidirectional attention layer. By taking both node features and pairwise features as layer input, it can model the internal and external relationships of variable pairs. In addition, we use a skeleton matrix along with a v-tensor, a third-order tensor representing v-structures, as our output, so as to represent the Markov Equivalence Class (MEC), which resolves identifiability inconsistency.
tags: []

# Display this page in the Featured widget?
featured: false

# Custom links (uncomment lines below)
# links:
# - name: Custom Link
#   url: http://example.org

url_pdf: ''
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
