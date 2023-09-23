---
title: "Hierarchical Satellite System Graph for Approximate Nearest Neighbor Search on Big Data"
authors:
- admin
- Ruhui Ma
- Tao Song
- Yang Hua
- Zhengui Xue
- Chenyang Guan
- Haibing Guan

author_notes:
#- "Equal contribution"
#- "Equal contribution"
date: "2022-02-03T00:00:00Z"
doi: ""

# Schedule page publish date (NOT publication's date).
publishDate: "2022-02-03T00:00:00Z"

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ["2"]

# Publication name and optional abbreviated publication name.
publication: "ACM/IMS Transactions on Data Science (TDS)"
publication_short: "ACM TDS"

abstract: Approximate nearest neighbor search is a classical problem in data science, which is widely applied in many fields. With the rapid growth of data in the real world, it becomes more and more important to speed up the nearest neighbor search process. Satellite System Graph (SSG) is one of the state-of-the-art methods to solve the problem. However, with the further increase of the data scale of problems, SSG still needs a considerable amount of time to finish the search due to the limitation of step length and start point locations. To solve the problem, we propose Hierarchical Satellite System Graph (HSSG) and present its index algorithm and search algorithm. The index process can be distributed deployed due to the good parallelism of our designed hierarchical structure. The theoretical analysis reveals that HSSG decreases the search steps and reduces the computational cost and reduces the search time by searching on the hierarchical structure with a similar indexing time compared with SSG, hence reaches a better search efficiency. The experiments on multiple datasets present that HSSG reduces the distance computations, accelerates the search process, and increases the search precision in the real tasks, especially under the tasks with large scale and crowded distributions, which presents a good application prospect of HSSG.
# Summary. An optional shortened abstract.
summary: We propose Hierarchical Satellite System Graph (HSSG) and present its index algorithm and search algorithm. The index process can be distributed deployed due to the good parallelism of our designed hierarchical structure. The theoretical analysis reveals that HSSG decreases the search steps and reduces the computational cost and reduces the search time by searching on the hierarchical structure.
tags:
- Source Themes
featured: false

# links:
# - name: ""
#   url: ""
url_pdf: https://dl.acm.org/doi/pdf/10.1145/3488377
url_code: 'https://github.com/lan-qing/HSSG'
url_dataset: ''
url_poster: ''
url_project: ''
url_slides: ''
url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder. 
image:
  caption: ''
  focal_point: ""
  preview_only: false

# Associated Projects (optional).
#   Associate this publication with one or more of your projects.
#   Simply enter your project's folder or file name without extension.
#   E.g. `internal-project` references `content/project/internal-project/index.md`.
#   Otherwise, set `projects: []`.
projects: []

# Slides (optional).
#   Associate this publication with Markdown slides.
#   Simply enter your slide deck's filename without extension.
#   E.g. `slides: "example"` references `content/slides/example/index.md`.
#   Otherwise, set `slides: ""`.
#slides: example
---

{{% callout note %}}
Click the *Cite* button above to demo the feature to enable visitors to import publication metadata into their reference management software.
{{% /callout %}}

{{% callout note %}}
Create your slides in Markdown - click the *Slides* button to check out the example.
{{% /callout %}}

[//]: # (Supplementary notes can be added here, including [code, math, and images]&#40;https://wowchemy.com/docs/writing-markdown-latex/&#41;.)
