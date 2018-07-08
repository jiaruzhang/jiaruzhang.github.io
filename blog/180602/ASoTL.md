---
title: A note of A Survey on Transfer Learning
create: 2018.6.2
modified: 2018.7.8
tags: Notes
---
[TOC]
# A Survey on Transfer Learning

## Introduction

#### Applied range
*knowledge transfer*  or *transfer learning* would be desirable when it is expensive or impossible to recollect the needed training data and rebuild the models.

#### Examples

* Web-document classification
 
    To classify a given Web document into several predefined categories. 

    For example, the labeled examples may be the university webpages. But the newly created website may have different data features or data distributions. 

* Data which outdate easily 

    Example: indoor WiFi localization problems. We wish to adapt the localization model trained in one time period (the source domain) for a new time period (the target domain), or to adapt the localization model trained on a mobile device (the source domain) for a new mobile device (the target domain)

* The problem of sentiment classification

    To adapt a classification model that is trained on some products to help learn classification models for some other products

## Overview

### Brief History

#### History
* Traditional machine learning algorithms: make predictions on the future data using statistical models that are trained on previously collected training data

* Semisupervised classification: much unlabeled data, little labeled data. Assume they are the same.

#### Transfer Learning
* Motivation: People can intelligently apply knowledge learned previously to solve new problems

* Fundamental motivation: A NIPS-95 workshop on “Learning to Learn”

* Different names: learning to learn, life-long learning, knowledge transfer, inductive transfer, multitask learning, knowledge consolidation, context-sensitive learning, knowledge-based inductive bias, metalearning, and incremental/cumulative learning

* New definition: the ability of a system to recognize and apply knowledge and skills learned in previous tasks to novel tasks.
    * Difference between transfer learning and mutitask learning: transfer learning cares most about the target task, while multitask learning learning all of the source and target tasks simultaneously

* Top venues: 
    * data mining: ACM KDD, IEEE ICDM, and PKDD, for example
    * machine learning: ICML, NIPS, ECML, AAAI, and IJCAI, for example
    * applications of machine learning and data mining: ACM SIGIR, WWW, and ACL, for example


### Notations and Definitions

#### Definitions of "Domain" and "Task"
A *domain* $D$ consists of two components: a feature space $\mathcal{X}$ and a marginal probability distribution $P(X)$, where $X = \{x_1, \dots, x_n\} \in \mathcal{X}$.

A *task* consists of two components: a label space $\mathcal{Y}$ and an objective predictive function $f(\cdot)$.

This survey only consider one source domain $\mathcal{D}_S$, one target domain $\mathcal{D}_T$ and usually $0 \le N_T \ll N_S$.

#### Unified definition
**Transfer Learning**: Given a source domain $\mathcal{D}_S$ and learning task $\mathcal{T}_S$, a target domain $\mathcal{D}_T$ and learning task $\mathcal{T}_T$, transfer learning aims to help improve the learning of the target predictive function $f_T(\cdot)$ in $\mathcal{D}_T$ using the knowledge in $\mathcal{D}_S$ and $\mathcal{T}_S$,where $\mathcal{D}_S \neq \mathcal{D}_T$, or $\mathcal{T}_S \neq \mathcal{T}_T$.

### A Categorization of Transfer Learning Techniques

Three main research issues:

1. what to transfer
2. how to transfer
3. when to transfer

Three subsettings:

1. inductive transfer learning
2. transductive transfer learning
3. unsupervised transfer learning

##### inductive transfer learning
The target task is different from the source task, no matter when the source and target domains are the same or not.

Two cases:
1. A lot of labeled data in the source domain are available.
2. No labeled data in the source domain are available.

##### transductive transfer learning
The source and target tasks are the same, while the source and target domains are different.

Two cases:
1. $\mathcal{X}_S \neq \mathcal{X}_T$.
2. $\mathcal{X}_S = \mathcal{X}_T$ but $P(X_S) \neq P(X_T)$.

##### unsupervised transfer learning
no labeled data available in both source and target domains in training.

## Inductive transfer learning

**Inductive transfer learning**: Given a source domain $\mathcal{D}_S$ and a learning task $\mathcal{T}_S$, a target domain $\mathcal{D}_T$ and a learning task $\mathcal{T}_T$, inductive transfer learning aims to help improve the learning of the target predictive function $f_T(\cdot)$ in $\mathcal{D}_T$ using the knowledge in $\mathcal{D}_S$ and $\mathcal{T}_S$ where $\mathcal{T}_S \neq \mathcal{T}_T$.

### Transferring Knowledge of Instances

**a boosting algorithm**: *TrAdaBoost*

### Transferring Knowledge of Feature Representations

Aims at finding “good” feature representations to minimize domain divergence and classi- fication or regression model error. 

Similar to *common feature learning* in the field of multitask learning

#### Supervised Feature Construction

In the *inductive transfer learning* setting, the common features can be learned by solving an optimization problem, given as follows:

$$ argmin_{A, U} \sum_{t \in \{T, S\}} \sum_{i=1}^{n_t} L(y_{t_i}, \langle a_t, U^T x_{t_i}\rangle) + \gamma ||A||^2_{2, 1}$$ 

$$s.t. U \in \mathbf{O}^d .$$

#### Unsupervised Feature Construction

Sparse coding is an unsupervised feature construction method for learning *higher level* features for transfer learning. It consists of two steps.

In the first step, higher level basis vectors $b = \{b_1, b_2, \dots, b_s\}$ are learned from

$$ min_{a, b}\sum_i ||x_{S_i} - \sum_j a_{S_i}^j b_j||^2_2 + \beta ||a_{S_i}||_1$$

$$s.t. ||b_j||_2 \le 1 , \forall j \in 1, \dots, s. $$

In the second step, *higher level* features on the target-domain data will be learned based on the basis vectors $b$:

$$a_{T_i}^* = argmin_{a_{T_i}} ||x_{T_i} - \sum_j a_{T_i}^j b_j||^2_2 + \beta||a_{T_i}||_1$$

### Transferring Knowledge of Parameters

Most approaches are designed to work under multitask learning. But they can be easily modified for transfer learning. Intuitively, we may assign a larger weight to the loss function of the target domain to achieve better performance in the target domain.

**Algorithms**: 

- MT-IVM (Lawrence and Platt)
- SVMs for multitask learning (Evgeniou and Pontil)
- A locally weighted ensemble learning framework (Gao et al)

### Transferring Relational Knowledge
It deals with transfer learning problems in relational domains. It does not assume that the data drawn from each domain be independent and identically distributed (i.i.d.).

*Statistical relational learning techniques* are proposed to solve these problems.

**Algorithms**: 

- TAMAR (Mihalkova et al)

## Transductive transfer learning

### Transferring the Knowledge of Instances
We want to minimize
$$ \theta^* = argmin_{\theta \in \Theta} \sum_{(x, y)\in D_S} \frac{P(D_T)}{P(D_S)}P(D_S)l(x,y,\theta) \\
\approx argmin_{\theta \in \Theta} \sum_{i=1}^{n_S}\frac{P_T(x_{T_i}, y_{T_i})}{P_S(x_{S_i}, y_{S_i})} l(x_{S_i}, y_{S_i}, \theta).$$
and
$$\frac{P_T(x_{T_i}, y_{T_i})}{P_S(x_{S_i}, y_{S_i})}= \frac{P(x_{S_i})}{P(x_{T_i})}$$

Algorithms to estimate $\frac{P(x_{S_i})}{P(x_{T_i})}$:

- kernel-mean matching(KMM）
- Kullback-Leibler Importance Estimation Procedure(KLIEP)

### Transferring Knowledge of Feature Representations
Structural correspondence learning (SCL) algorithm：
 1. Define a set of *pivot* features from both domains
 2. Remove these *pivot* features from the data and treats each *pivot* feature as a new label vector. Solve $m$ classification problem:
 $$f_1(x)= sgn(w_l^T \cdot x), l=1, \dots, m.$$
 3. Use SVD on matrix $W$: $W = UDV^T$, and $\theta = U^T_{[1:h,:]}$(h is the number of the shared features)is the matrix (linear mapping) whose rows are the top left singular vectors of W.
 4. Use the augmented feature vector to build models.

Disadvantages: How to select the pivot features is difficult and domain dependent.

MI-SCL: Use Mutual Information (MI) to choose the pivot features instead of using more heuristic criteria.

Other algorithms:

- coclustering-based algorithm: propagate the label information across different domains.
- bridged refinement
- spectral classification framework for cross-domain transfer learning problem
-  a cross-domain text classification algorithm that extended the traditional prob- abilistic latent semantic analysis (PLSA) algorithm

Transfer learning via dimensionality reduction:
Maximum Mean Discrepancy Embedding(MMDE) can learn a low-dimensional space to reduce the difference of distributions between different domains. Transfer Component Analysis (TCA) overcomes the drawback of computational burden.

## Unsupervised Transfer Learning
Little research works on this setting. *Self-taught clustering* (STC) and transferred discriminative analysis (TDA) algorithms are proposed to transfer clustering and transfer dimensionality reduction problems, respectively.

### Transferring Knowledge of Feature Representations

*Self-taught clustering* is an instance of *unsupervised transfer learning*, which aims at clustering a small collection of unlabeled data in the target domain with the help of a large amount of unlabeled data in the source domain.

 STC: tries to learn a common feature space across domains.
 
 *TDA* algorithm: solves the transfer dimensionality reduction problem. First applies clustering methods to generate pseudoclass labels for the target unlabeled data, then applies dimensionality reduction methods to the target data and labeled source data to reduce the dimensions. Run iteratively.
 
## Transfer Bounds and Negative Transfer
Conditional Kolmogorov complexity is used to measure relatedness between tasks and transfer the “right” amount of information. 

A novel graph-based method is used for knowledge transfer.

How to avoid negative transfer is a very important issue. If two tasks are too dissimilar, then brute-force transfer may hurt the performance.

## Applications of Transfer Learning
Datasets:
- text mining data sets
- Email spam-filtering data set
- WiFi localization over time periods data set
- Sentiment classification data set

Toolboxes:
a MATLAB toolkit for transfer learning. http://multitask.cs.berkeley.edu/

Other applications:
In sequential machine learning

## Conclusion
This survey reviewed several current trends of transfer learning. Three different settings of Transfer Learning: inductive transfer learning, transductive transfer learning, and unsupervised transfer learning. 

Approaches can be classified into four contexts based on “what to transfer” in learning.

Future research issues:
- how to avoid negative transfer
- how to make sure that no negative transfer happens
- when an entire domain cannot be used for transfer learning and whether we can still transfer part of the domain for useful learning in the target domain.

Most transfer learning algorithms assumed that the feature spaces between the source and target domains are the same. However, we may wish to transfer knowledge across domains or tasks that have different feature spaces, and transfer from multiple such source domains. This type is *heterogeneous transfer learning*.

Use transfer learning to solve other challenging applications.
