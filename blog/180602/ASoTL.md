---
title: A note of <i>A survey on Transfer Learning</i>
create: 2018.6.1
modified: 2016.6.2
tags: Note
---
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