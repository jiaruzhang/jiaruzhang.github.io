---
title: 'Analytical Correction for Subsampling Bias in Drifting Models'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - admin
  - Zeyun Deng
  - Juanwu Lu
  - Ziran Wang
  - Ruqi Zhang

date: '2026-12-09T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2026-04-29T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *Advances in Neural Information Processing Systems*
publication_short: In *NeurIPS*

abstract: Drifting models are capable one-step generative models trained to follow a drifting field. The field combines attractive and repulsive softmax-weighted centroids over the data and current-generator distributions. In practice, only a minibatch of *n* samples from each distribution is available, and each centroid is approximated by an empirical estimate. In this paper, we begin by showing that the minibatch centroid is in general a biased estimator of the target centroid, with a pointwise O(1/*n*) bias arising from softmax self-normalization. Correcting this bias requires the expectation over the full distribution, which is intractable. We instead approximate the leading bias term from in-batch statistics and propose Analytical Bias Correction (ABC), a closed-form plug-in adjustment. We prove that ABC reduces the bias from O(1/*n*) to O(1/*n*²), introduces no first-order increase in total variance, and preserves convex-hull containment of the corrected centroid. In practice, ABC requires only two additional lines of code and has negligible wall-time overhead under compiled execution. Toy experiments confirm the theoretical O(1/*n*) and O(1/*n*²) scaling. On CIFAR-10, ABC reduces FID and trains faster, with the largest gains at small *n*, where the bias is most significant.

# Summary. An optional shortened abstract.
summary: We show that the minibatch centroid in drifting models carries an O(1/*n*) bias from softmax self-normalization, and propose Analytical Bias Correction (ABC), a two-line closed-form fix that reduces the bias to O(1/*n*²).

tags: []

# Display this page in the Featured widget?
featured: false

# Custom links
links:
  - name: arXiv
    url: 'https://arxiv.org/abs/2604.27239'
    icon_pack: ai
    icon: arxiv

# url_pdf: 'https://arxiv.org/pdf/2604.27239'
url_code: 'https://github.com/lan-qing/DriftingABC'
#url_dataset: ''
#url_poster: ''
#url_project: ''
url_slides: ''
#url_source: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: ''
  focal_point: ''
  preview_only: false

projects: []
---
