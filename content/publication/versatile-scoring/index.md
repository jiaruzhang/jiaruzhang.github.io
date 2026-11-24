---
title: 'Designing Versatile Samples for Learned Trajectory Scoring'

# Authors
# If you created a profile for a user (e.g. the default `admin` user), write the username (folder name) here
# and it will be replaced with their full name and linked to their profile.
authors:
  - Yaguang Li
  - admin
  - Chuheng Wei
  - Can Cui
  - Ziran Wang

date: '2027-03-01T00:00:00Z'
doi: ''

# Schedule page publish date (NOT publication's date).
publishDate: '2026-09-01T00:00:00Z'

# Publication type.
# Legend: 0 = Uncategorized; 1 = Conference paper; 2 = Journal article;
# 3 = Preprint / Working Paper; 4 = Report; 5 = Book; 6 = Book section;
# 7 = Thesis; 8 = Patent
publication_types: ['1']

# Publication name and optional abbreviated publication name.
publication: In *IEEE/CVF Winter Conference on Applications of Computer Vision*
publication_short: In *WACV*

abstract: Many current end-to-end driving policies emit a pool of candidate trajectories and select one, which makes selection a separable component; a scorer can be retrained while the planner, its backbone, and its trajectory generator all stay frozen. However, many strong planners concentrate their proposals around safe mode, providing limited supervision near decision boundaries. In this work, we design a training dataset that provides more informative supervision for the scorer. In particular, we construct two generators that perturb the logged human trajectory along the two axes a vehicle can be displaced, laterally toward the drivable boundary and longitudinally toward a leading vehicle. The designed dataset produces more informative positive and negative samples than the base planner's proposal pool. We attach a transformer-based scorer to two frozen generative planners, DiffusionDrive and MeanFuser, and train it on the NAVSIM navtrain dataset. The results of the experiments show that we achieve 90.1 EPDMS on DiffusionDrive and 90.4 EPDMS on MeanFuser when using ResNet-34, with 0.4 and 0.3 EPDMS respectively, from the designed training dataset.

# Summary. An optional shortened abstract.
summary: We perturb logged human trajectories laterally and longitudinally to build training samples near decision boundaries, giving a learned trajectory scorer more informative supervision than the planner's own proposal pool.

tags: []

# Display this page in the Featured widget?
featured: false

# Custom links
links:
  - name: arXiv
    url: 'https://arxiv.org/abs/2609.01799'
    icon_pack: ai
    icon: arxiv

url_code: 'https://github.com/PurdueDigitalTwin/Versatile-Scoring'
url_slides: ''
url_video: ''

# Featured image
# To use, add an image named `featured.jpg/png` to your page's folder.
image:
  caption: ''
  focal_point: ''
  preview_only: false

projects: []
---
