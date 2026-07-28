---
title: "Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning"
subtitle: "Full text — web transcription"

summary: "Web transcription of the full text of Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning (AISTATS 2025), with all equations, tables and figures."

date: '2025-01-18T00:00:00Z'
publishDate: '2025-01-18T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/paire/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/paire/en/'

authors: []
tags: []
categories: []
featured: false

share: false
profile: false
commentable: false
editable: false
show_breadcrumb: false

_build:
  list: never

image:
  caption: ''
  focal_point: ''
  preview_only: false
---

{{% callout note %}}
This page is a web transcription of the full text of **“Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning”** (AISTATS 2025), reproduced here for easier reading, searching and linking.
It is provided **for reference only**; please read and cite the [original PDF](https://arxiv.org/pdf/2502.10883), which is authoritative. A [Chinese translation]({{< relref "/publication/paire-cn" >}}) is also available.
{{% /callout %}}

{{< toc >}}

Jiaru Zhang, Rui Ding, Qiang Fu, Huang Bojun, Zizhen Deng, Yang Hua, Haibing Guan, Shi Han, and Dongmei Zhang

> Jiaru Zhang (jiaruzhang@sjtu.edu.cn), Shanghai Jiao Tong University. The work was done during his internship at Microsoft Research Asia.
> Rui Ding (juding@microsoft.com), Microsoft. Corresponding author.
> Qiang Fu (qifu@microsoft.com), Microsoft.
> Huang Bojun (bojhuang@gmail.com), Sony Research.
> Zizhen Deng (dengzizhen557@outlook.com), Peking University.
> Yang Hua (y.hua@qub.ac.uk), Queen’s University Belfast.
> Haibing Guan (hbguan@sjtu.edu.cn), Shanghai Jiao Tong University.
> Shi Han (shihan@microsoft.com), Microsoft.
> Dongmei Zhang (dongmeiz@microsoft.com), Microsoft.

## Abstract

Causal discovery is a structured prediction task that aims to predict causal relations among variables based on their data samples. Supervised Causal Learning (SCL) is an emerging paradigm in this field. Existing Deep Neural Network (DNN)-based methods commonly adopt the “Node-Edge approach”, in which the model first computes an embedding vector for each variable-node, then uses these variable-wise representations to concurrently and independently predict for each directed causal-edge. In this paper, we first show that this architecture has some systematic bias that cannot be mitigated regardless of model size and data size. We then propose SiCL, a DNN-based SCL method that predicts a skeleton matrix together with a v-tensor (a third-order tensor representing the v-structures). According to the Markov Equivalence Class (MEC) theory, both the skeleton and the v-structures are *identifiable* causal structures under the canonical MEC setting, so predictions about skeleton and v-structures do not suffer from the identifiability limit in causal discovery, thus SiCL can avoid the systematic bias in Node-Edge architecture, and enable consistent estimators for causal discovery. Moreover, SiCL is also equipped with a specially designed pairwise encoder module with a unidirectional attention layer to model both internal and external relationships of pairs of nodes. Experimental results on both synthetic and real-world benchmarks show that SiCL significantly outperforms other DNN-based SCL approaches.

## 1. Introduction

Causal discovery seeks to infer causal structures from an observational data sample.
Supervised Causal Learning (SCL) (<a href="#ref-6">Dai et al., 2023</a>; <a href="#ref-13">Ke et al., 2023</a>; <a href="#ref-22">Ma et al., 2022</a>) is an emerging paradigm in this field. The basic idea is to consider causal discovery as a *structured prediction* task, and to train a prediction model using supervised learning techniques.
At training time, a training dataset comprising a variety of causal mechanisms and their associated data samples is generated.
The prediction model is then trained to take such a data sample as input, and to output predictions about the causal mechanism behind the data sample.
Compared to traditional rule-based or unsupervised methods (<a href="#ref-9">Glymour et al., 2019</a>), the SCL method has demonstrated strong empirical performance (<a href="#ref-6">Dai et al., 2023</a>; <a href="#ref-22">Ma et al., 2022</a>), as well as robustness against sample size and distribution shift (<a href="#ref-13">Ke et al., 2023</a>; <a href="#ref-21">Lorch et al., 2022</a>).

Deep Neural Network (DNN)-based SCL employs DNN as the prediction model. It allows end-to-end training, removing the need for manual feature engineering. Additionally, it can handle both continuous and discrete data types effectively, and can learn latent representations.
A specific DNN architecture, first introduced by <a href="#ref-21">Lorch et al. (2022)</a>, is particularly popular in recent DNN-based SCL works. The model first transforms the given data sample into a set of node-wise feature vectors, each representing an individual variable (corresponding to a node in the associated causal graph).
Based on these node-wise features, the model then outputs a weighted adjacency matrix A, where {{< math >}}$A_{ij}\in[0,1]${{< /math >}} is an estimated probability for the directed edge {{< math >}}$i \rightarrow j${{< /math >}} (meaning that i is a direct cause of j).
Finally, the adjacency matrix of an inferred causal graph G is obtained as a Bernoulli sample of A, where each entry {{< math >}}$G_{ij} \in \{0,1\}${{< /math >}} is sampled *independently*, following probability {{< math >}}$A_{ij}${{< /math >}}.
For convenience, we call such a model architecture as the “Node-Edge” architecture, as the representation is learned for individual nodes and the probability is estimated and sampled for individual directed edges.

Despite its popularity and encouraging results (<a href="#ref-21">Lorch et al., 2022</a>; <a href="#ref-45">Zhu et al., 2020</a>; <a href="#ref-2">Charpentier et al., 2022</a>; <a href="#ref-35">Varambally et al., 2024</a>), we identify two limitations for the Node-Edge approach:

First, the Node-Edge architecture imposes a fundamental bias in the inferred causal relations. Specifically, given an observational data sample D, the existence of a directed causal edge {{< math >}}$i \rightarrow j${{< /math >}} may *necessarily* depend on the existence of other edges. But the existing Node-Edge models predict each edge separately and independently, so the probability prediction {{< math >}}$A_{ij}${{< /math >}} made by such models is only conditioned on the input sample D, not on the sampling result of other entries of A, thereby failing to capture the crucial inter-edge dependency in its probability estimation.

As a simple example, a Node-Edge model maintaining the possibility of both {{< math >}}$G_1: X\rightarrow T \rightarrow Y${{< /math >}} and {{< math >}}$G_2: X\leftarrow T \leftarrow Y${{< /math >}} would necessarily have a non-zero probability to output the edges {{< math >}}$X\rightarrow T${{< /math >}} and {{< math >}}$T \leftarrow Y${{< /math >}}, thus cannot rule out the possibility of {{< math >}}$G_3: X\rightarrow T \leftarrow Y${{< /math >}}, even though {{< math >}}$G_3${{< /math >}} is impossible to be the groundtruth causal graph behind a data sample D compatible with {{< math >}}$G_1${{< /math >}} and {{< math >}}$G_2${{< /math >}} (<a href="#ref-36">Verma and Pearl, 1990</a>).
Crucially, there is no way to tell {{< math >}}$G_1${{< /math >}} from {{< math >}}$G_2${{< /math >}} based on observational data in general cases (<a href="#ref-1">Andersson et al., 1997</a>; <a href="#ref-24">Meek, 1995b</a>).
It means that for any Node-Edge model to be sound, it has to maintain the possibility of both {{< math >}}$G_1${{< /math >}} and {{< math >}}$G_2${{< /math >}} (when observing a data sample compatible with any of them), leading to an inevitable error probability to output the impossible graph {{< math >}}$G_3${{< /math >}} on the other hand.

Second, the Node-Edge architecture does not explicitly represent the features about node pairs, which we argue are essential for observational causal discovery.
For example, a causal edge {{< math >}}$X\rightarrow Y${{< /math >}} can exist only if the node pair {{< math >}}$\langle X, Y\rangle${{< /math >}} demonstrates *persistent dependency* (<a href="#ref-22">Ma et al., 2022</a>; <a href="#ref-33">Spirtes et al., 2000</a>), meaning that X and Y remain statistically dependent regardless of conditioning on any subset of other variables. As another example, for causal DAGs, a sufficient condition to determine the causal direction between a persistently dependent node pair {{< math >}}$\langle X, Y\rangle${{< /math >}} is that X and Y exhibits *orientation asymmetry*, meaning that there exists a third variable Z such that X is persistently dependent to Z but Y can become independent to Z conditioned on a variable-set {{< math >}}$\mathbf{S}\not\ni X${{< /math >}} (or vice versa). A feature like persistent dependency or orientation asymmetry is, in its nature, a collective property of a node pair, but not of any individual node alone.

To address these limitations, in this paper, we propose a novel DNN-based SCL approach, called Supervised Identifiable Causal Learning (SiCL).
The neural network in SiCL does not seek to predict the probabilities of directed edges, but tries to predict a skeleton matrix together with v-tensor, a third-order tensor representing the v-structures.
According to the Markov Equivalence Class (MEC) theory of causal discovery, skeleton and v-structures are *identifiable* causal structures under the canonical MEC setting (while the directed edges are not), so predictions about skeleton and v-structures do not suffer from the (non-)identifiability limit.
By leveraging this insight, our theory-inspired DNN architecture completely avoids the systematic bias in edge-prediction models as previously discussed, and enables *consistent* neural-estimators (Note: Recall that a statistical estimator is *consistent* if it converges to the groundtruth given infinite data.) for causal discovery.
Moreover, SiCL is also equipped with a specially designed pairwise encoder module with a unidirectional attention layer.
With both node features and node-pair features as the layer input, it can model both internal and external relationships of pairs of nodes.
Experimental results on both synthetic and real-word benchmarks show that SiCL can effectively address the two above-mentioned limitations, and the resulted SiCL solution significantly outperforms other DNN-based SCL approaches with more than 50% performance improvement in terms of SHD (Structural Hamming Distance) on the real-world Sachs data. The codes are publicly available at <https://github.com/microsoft/reliableAI/tree/main/causal-kit/SiCL>.

## 2. Background and Related Work

A Causal Graphical Model is defined by a joint probability distribution {{< math >}}$P${{< /math >}} over multiple random variables and a DAG {{< math >}}$G${{< /math >}}. Each node {{< math >}}$X_i${{< /math >}} in {{< math >}}$G${{< /math >}} represents a variable in {{< math >}}$P${{< /math >}}, and a directed edge {{< math >}}$X_i \rightarrow X_j${{< /math >}} represents a direct cause-effect relation from {{< math >}}$X_i${{< /math >}} to {{< math >}}$X_j${{< /math >}}. A causal discovery task generally asks to infer about {{< math >}}$G${{< /math >}} from an i.i.d. sample of {{< math >}}$P${{< /math >}}.

However, there is a well-known identifiability limit for causal discovery. In general, the causal DAG is only identifiable up to an equivalence class. Studies of this identifiability limit under a canonical assumption setting have led to the well-established MEC theory (<a href="#ref-7">Frydenberg, 1990</a>; <a href="#ref-36">Verma and Pearl, 1990</a>). We call a causal feature, *MEC-identifiable*, if the value of this feature is invariant among the equivalence class under the canonical MEC assumption setting. It is known that such MEC-identifiable features include the skeleton and the set of v-structures, which we briefly present in the following.

A *skeleton* {{< math >}}$E${{< /math >}} defined over the data distribution {{< math >}}$P${{< /math >}} is an undirected graph where an edge exists between {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} if and only if {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} are always dependent in {{< math >}}$P${{< /math >}}, i.e., {{< math >}}$\forall Z \subseteq\left\{X_1, X_2, \cdots, X_d\right\} \backslash \left\{X_i, X_j \right\}${{< /math >}}, we have {{< math >}}$X_i \nperp X_j | Z${{< /math >}}. Under mild assumptions (such as that {{< math >}}$P${{< /math >}} is Markovian and faithful to the DAG {{< math >}}$G${{< /math >}}; see details in Appendix Sec. A1.1), the skeleton is the same as the corresponding undirected graph of the DAG {{< math >}}$G${{< /math >}} (<a href="#ref-33">Spirtes et al., 2000</a>). A triple of variables {{< math >}}$\langle X, T, Y \rangle${{< /math >}} is an *Unshielded Triple (UT)* if {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}} are both adjacent to {{< math >}}$T${{< /math >}} but not adjacent to each other in (the skeleton of) {{< math >}}$G${{< /math >}}. It becomes a *v-structure* denoted as {{< math >}}$X \rightarrow T \leftarrow Y${{< /math >}} if the directions of the edges are from {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}} to {{< math >}}$T${{< /math >}}.

Two graphs are Markov equivalent if and only if they have the same skeleton and v-structures. The *Markov equivalence class (MEC)* can be represented by a *Completed Partially Directed Acyclic Graph (CPDAG)* consisting of both directed and undirected edges. We use {{< math >}}$CPDAG(G)${{< /math >}} to denote the CPDAG derived from {{< math >}}$G${{< /math >}}. According to the theorem of Markov completeness (<a href="#ref-24">Meek, 1995b</a>), we can only identify a causal graph up to its MEC, i.e., the CPDAG, unless additional assumptions are made (see the remark below). This means that each (un)directed edge in {{< math >}}$CPDAG(G)${{< /math >}} indicates a (non)identifiable causal relation.

**Remark:** The MEC-based identifiability theory is applicable in the general-case setting, when we take into account all possible distributions {{< math >}}$P${{< /math >}}. It is known this identifiability limit could be broken (i.e., an undirected edge in the CPDAG could be oriented) *if* we assume that the data follows some special class of distributions, e.g., linear non-Gaussians, additive noise models, post-nonlinear or location-scale models (<a href="#ref-27">Peters et al., 2014</a>; <a href="#ref-31">Shimizu et al., 2011</a>; <a href="#ref-41">Zhang and Hyvärinen, 2009</a>; <a href="#ref-11">Immer et al., 2023</a>). These assumptions are sometimes hard to verify in practice, so this paper considers the general-case setting. More discussions on the identifiability and causal assumptions are provided in Appendix Sec. A2.

### 2.1. Related Work

In traditional methods of causal discovery, constraint-based methods are mostly related to our work. They aim to identify the DAG that is consistent with inter-variable conditional independence constraints. These methods first identify the skeleton and then conduct orientation based on v-structure identification (<a href="#ref-39">Yu et al., 2016</a>). The output is a CPDAG which represents the MEC. Notable algorithms in this category include PC (<a href="#ref-33">Spirtes et al., 2000</a>), along with variations such as Conservative-PC (<a href="#ref-29">Ramsey et al., 2012</a>), PC-stable (<a href="#ref-4">Colombo et al., 2014</a>), and Parallel-PC (<a href="#ref-17">Le et al., 2016</a>). Compared to constraint-based methods, both our approach and theirs are founded upon the principles of MEC theory for estimating skeleton and v-structures. However, whereas traditional methods rely on symbolic reasoning based on explicit constraints, we employ DNNs to capture the essential causal information intricately linked with these constraints.

Score-based methods aim to find an optimal DAG according to a predefined score function, subject to combinatorial constraints. These methods employ specific optimization procedures such as forward-backward search GES (<a href="#ref-3">Chickering, 2002</a>), hill-climbing (<a href="#ref-14">Koller and Friedman, 2009</a>), and integer programming (<a href="#ref-5">Cussens, 2011</a>). Continuous optimization methods transform the discrete search procedure into a continuous equality constraint. NOTEARS (<a href="#ref-43">Zheng et al., 2018</a>) formulates the acyclic constraint as a continuous equality constraint and is further extended by DAG-GNN (<a href="#ref-40">Yu et al., 2019</a>), DECI (<a href="#ref-8">Geffner et al., 2022</a>) to support non-linear causal relations. DECI (<a href="#ref-8">Geffner et al., 2022</a>) is a flow-based model which can perform both causal discovery and inference on non-linear additive noise data. Recently, ENCO (<a href="#ref-18">Lippe et al., 2022</a>) is proposed as a continuous optimization method where the edge orientation is modeled as a separate parameter to maintain the acyclicity. It is guaranteed to converge to the correct graph if interventions on all variables are available. RL-BIC (<a href="#ref-45">Zhu et al., 2020</a>) utilizes Reinforcement Learning to search for the optimal DAG. These methods can be viewed as unsupervised since they do not access additional datasets associated with ground truth causal relations. We refer to <a href="#ref-9">Glymour et al. (2019)</a>; <a href="#ref-37">Vowels et al. (2022)</a> for a thorough exploration of this literature.

SCL begins from orienting edges in the bivariate cases under the functional causal model formalism. Methods such as RCC (<a href="#ref-19">Lopez-Paz et al., 2015</a>) and NCC (<a href="#ref-20">Lopez-Paz et al., 2017</a>) have outperformed unsupervised approaches like ANM (<a href="#ref-10">Hoyer et al., 2008</a>) or IGCI (<a href="#ref-12">Janzing et al., 2012</a>). For multivariate cases, ML4S (<a href="#ref-22">Ma et al., 2022</a>) proposes a supervised approach specifically for skeleton learning. Complementary to ML4S, ML4C (<a href="#ref-6">Dai et al., 2023</a>) takes both data and skeleton as input and classifies unshielded triples as either v-structures or non-v-structures. <a href="#ref-28">Petersen et al. (2023)</a> proposes a SLdisco method, utilizing SCL approach to address some limitations of PC and GES.

DNN-based SCL has emerged as a prominent approach for enabling end-to-end causal learning. Two notable works in this line, namely AVICI (<a href="#ref-21">Lorch et al., 2022</a>) and CSIvA (<a href="#ref-13">Ke et al., 2023</a>), introduced an alternating attention mechanism to enable permutation invariance across samples and variables. Both methods learn individual representation for each node, which is then used to predict directed edges. Among them, AVICI considers the task of predicting DAG from observational data and adopts exactly the Node-Edge architecture, hence suffers from the issues as discussed in Sec. 1. On the other hand, CSIvA requires additional interventional data as input to identify the full DAG, and applies an autoregressive DNN architecture where edges are predicted sequentially by multiple inference runs. Therefore, this autoregressive approach incurs very high inference cost due to the quadratic number of model runs required (w.r.t. the number of variables in question), as we experimentally verify in Appendix Sec. A8.4. In contrast, the method proposed in this paper only requires a single run of the DNN model. Besides that, our method also differs from both AVICI and CSIvA in terms of the usage of pairwise embedding vectors.

## 3. Limitations of the Node-Edge Architecture

The Node-Edge architecture is common and has been adopted to generate the output DAG {{< math >}}$G${{< /math >}} in the literature (<a href="#ref-21">Lorch et al., 2022</a>; <a href="#ref-45">Zhu et al., 2020</a>; <a href="#ref-2">Charpentier et al., 2022</a>; <a href="#ref-35">Varambally et al., 2024</a>).

In this architecture, each entry {{< math >}}$G_{ij}${{< /math >}} in the DAG is independently sampled from {{< math >}}$A_{ij}${{< /math >}}, an entry in the adjacency matrix {{< math >}}$A${{< /math >}}. This entry {{< math >}}$A_{ij}${{< /math >}} represents the probability that {{< math >}}$i${{< /math >}} directly causes {{< math >}}$j${{< /math >}}.
We introduce a simple yet effective example setting with only three variables {{< math >}}$X${{< /math >}}, {{< math >}}$Y${{< /math >}}, and {{< math >}}$T${{< /math >}} to reveal its limitation.

Considering a simulator that generates DAGs with equal probability from two causal models: In model 1, the causal graph is {{< math >}}$G_1: X \rightarrow T \rightarrow Y${{< /math >}}, and the variables follow {{< math >}}$X \sim \mathcal{N} (0, 1)${{< /math >}}, {{< math >}}$T = X + \mathcal{N}(0, 1)${{< /math >}}, {{< math >}}$Y = T + \mathcal{N}(0, 1)${{< /math >}}.
In model 2, the causal graph is {{< math >}}$G_2: X \leftarrow T \leftarrow Y${{< /math >}}, and the variables follow {{< math >}}$ Y = \mathcal{N}(0, 3)${{< /math >}}, {{< math >}}$T = \frac{2}{3}Y + \mathcal{N}(0, \frac{2}{3})${{< /math >}}, {{< math >}}$X = 0.5T + \mathcal{N}(0, 0.5)${{< /math >}}.
In this case, data samples coming from both causal models follow the same joint distribution, which makes {{< math >}}$G_1${{< /math >}} and {{< math >}}$G_2${{< /math >}} inherently indistinguishable (from observational data sample).

More importantly, when the fully-directed causal DAGs are used as the learning target (as the Node-Edge approach does), an optimally trained neural network will predict 0.5 probabilities on the directions of the two edges {{< math >}}$X - T${{< /math >}} and {{< math >}}$T - Y${{< /math >}}.
As a result, with 0.25 probability the graph sampling outcome would be {{< math >}}$X \rightarrow T \leftarrow Y${{< /math >}} (see Fig. A4 in the Appendix).

This error probability is rooted from the fact that the Bernoulli sampling of the edge {{< math >}}$X \rightarrow T${{< /math >}} is not conditioned on the sampling result of the edge {{< math >}}$T \leftarrow Y${{< /math >}}. Consequently, it is a bias that cannot be avoided even if the DNN has perfectly modeled the *marginal probability* of each edge (marginalized over other edges) given input data.

We further find that 0.25 is not the worst-case error rate yet.
Formally, for a distribution {{< math >}}$Q${{< /math >}} over a set of graphs, we define the graph distribution where the edges are independently sampled from the marginal distribution as {{< math >}}$M(Q)${{< /math >}}, i.e., for any causal edges {{< math >}}$e_1${{< /math >}} and {{< math >}}$e_2${{< /math >}}, {{< math >}}$P_{G\sim Q}(e_1 \in G) = P_{G\sim M(Q)}(e_1 \in G) = P_{G\sim M(Q)}(e_1 \in G | e_2 \in G)${{< /math >}}.
In general, a Node-Edge model optimally trained on data samples {{< math >}}$D${{< /math >}} coming from the distribution {{< math >}}$Q${{< /math >}} will essentially learn to predict {{< math >}}$M(Q)${{< /math >}} (when given the same data samples {{< math >}}$D${{< /math >}} at test time).
The following proposition shows that for causal graphs with star-shaped skeleton, with a chance of 26.42% the graph sampled from the marginal distribution {{< math >}}$M(Q)${{< /math >}} would be incorrect.

**Proposition 3.1.** Let {{< math >}}$\mathcal{G}_n${{< /math >}} be the set of graphs with {{< math >}}$n+1${{< /math >}} nodes where there is a central node {{< math >}}$y${{< /math >}} such that (1) every other node is connected to {{< math >}}$y${{< /math >}}, (2) there is no edge between the other nodes, and (3) there is at most one edge pointing to {{< math >}}$y${{< /math >}}.
We have

{{< math >}}
$$
\sup_n \max_{Q} P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 
1 - \frac{2}{e} \approx 0.2642. \tag{1}
$$
{{< /math >}}

The proof is provided in Appendix Sec. A6.

It indicates that an edge-predicting neural network could suffer from an inevitable error rate of 0.2642 even if it is perfectly trained.

In contrast, models that predict skeleton and v-structures would have a theoretical asymptotic guarantee of the consistency under canonical assumption.
The details, proof and relevant discussions are provided in Appendix Sec. A1.

## 4. The SiCL Method

In light of the limitations as discussed, we propose a new DNN-based SCL method in this section, named **SiCL** (**S**upervised **i**dentifiable **C**ausal **L**earning).

### 4.1. Overall Workflow

{{< figgrid caption="**Figure 1**: The inference workflow of SiCL." >}}
paire/new_workflow.png | 100
{{< /figgrid >}}


Following the standard DNN-based SCL paradigm, the core inference process is implemented as a DNN. The DNN takes a data sample encoded by a matrix {{< math >}}$D\in\mathbb{R}^{n \times d}${{< /math >}} as input, with {{< math >}}$d${{< /math >}} being the number of observable variables and {{< math >}}$n${{< /math >}} being the number of observations. In contrast to previous Node-Edge approaches, the SiCL method does not use the DNN to directly predict the causal graph, but instead seeks to predict the skeleton and the v-structures of the causal graph, which amount to the *MEC-identifiable* causal structures as mentioned previously.

Specifically, our DNN outputs two objects: (1) a skeleton prediction matrix {{< math >}}$S \in [0,1]^{d \times d}${{< /math >}} where {{< math >}}$S_{ij}${{< /math >}} models the conditional probability of the existence of the *undirected* edge {{< math >}}$X_i - X_j${{< /math >}}, conditioned on the input data sample {{< math >}}$D${{< /math >}}, and (2) a v-structure prediction tensor {{< math >}}$V \in [0,1]^{d\times d\times d}${{< /math >}} where {{< math >}}$V_{ijk}${{< /math >}} models the conditional probability of the existence of the v-structure component {{< math >}}$X_j \rightarrow X_i \leftarrow X_k${{< /math >}}, again conditioned on {{< math >}}$D${{< /math >}}. In our implementation, {{< math >}}$S${{< /math >}} and {{< math >}}$V${{< /math >}} are generated by two separate sub-networks, called Skeleton Predictor Network (SPN) and V-structure Predictor Network (VPN), respectively.

To further address the limitation of only having node-wise features for Node-Edge models, we propose to equip SPN and VPN with Pairwise Encoder modules to explicitly capture node-pair-wise features.

Based on the skeleton prediction matrix {{< math >}}$S${{< /math >}} and v-structure prediction tensor {{< math >}}$V${{< /math >}}, we infer the skeleton and v-structures of the causal graph; from the two we can determine a unique CPDAG. The CPDAG encodes a Markov equivalence class, from which we can pick up a graph instance as the prediction of the causal DAG (if needed).

Figure 1 provides a diagram of the overall inference workflow of SiCL, and a pseudo-code of the workflow is given by Algorithm A1 in Appendix.

Parameters of the SPN and VPN are trained following the standard supervised learning procedure. In our implementation, we only use synthetic training data, which is relatively easy to obtain, and yet could often lead to strong performance on real-world workloads (<a href="#ref-13">Ke et al., 2023</a>).

In the following, we elaborate the DNN architecture, the learning targets, as well as the post-processing procedure used in the SiCL method.

### 4.2. Feature Extraction

**Input Processing and Node Feature Encoder.** Given input data sample, which is a matrix {{< math >}}$D\in\mathbb{R}^{n \times d}${{< /math >}}, the input processing module contains a linear layer for continuous input data or an embedding layer for discrete input data, yielding the raw node features {{< math >}}$\mathcal{F}^{raw}_{il}${{< /math >}} for each node {{< math >}}$i${{< /math >}} in each observation {{< math >}}$l${{< /math >}}.

After that, we employ a node feature encoder to further process the raw node features into the final node features {{< math >}}$\mathcal{F}_{il}${{< /math >}}. Similar to previous papers (<a href="#ref-13">Ke et al., 2023</a>; <a href="#ref-21">Lorch et al., 2022</a>), the node feature encoder is a transformer-like network comprising attention layers over the observation dimension and the node dimension alternately, which naturally maintains permutation equivalence across both variable and data dimension because of the intrinsic symmetry of attention operations. More details about the node feature encoder are presented in Appendix Sec. A4 due to page limit.

**Pairwise Encoder.** Given node features {{< math >}}$\mathcal{F} \in \mathbb{R}^{d\times n \times h}${{< /math >}} for all the {{< math >}}$d${{< /math >}} nodes, the goal of pairwise encoder is to encode their pairwise relationships by {{< math >}}$d^2${{< /math >}} pairwise features, represented as a tensor {{< math >}}$\mathcal{P} \in \mathbb{R}^{d\times d \times n \times h}${{< /math >}}, where {{< math >}}$\mathcal{P}_{ijl} \in \mathbb{R}^{h}${{< /math >}} is a pairwise feature corresponding to the node pair {{< math >}}$(i,j)${{< /math >}} in observation {{< math >}}$l${{< /math >}}.

As argued in Sec. 1, both “internal” information (i.e., the pairwise relationship) and “external information (e.g., the context of the conditional separation set) of node pairs are needed to capture persistent dependency and orientation asymmetry. Our pairwise encoder module is designed to model the internal relationship via node feature concatenation and the non-linear mapping by MLP. On the other hand, we employ attention operations within the pairwise encoder to capture the contextual relationships (including persistent dependency and orientation asymmetry).

More specifically, the pairwise encoder module consists of the following parts (see Appendix Fig. A3 for diagrammatic illustration):

1. *Pairwise Feature Initialization.* The initial step is to concatenate the node features from the previous node feature encoder module for every pair of nodes. Subsequently, we employ a three-layer MLP to convert each concatenated vector {{< math >}}$\mathcal{P}_{ijl} \in \mathbb{R}^{2h}${{< /math >}} to an {{< math >}}$h${{< /math >}}-dimensional raw pairwise feature, i.e., {{< math >}}$\mathcal{P}_{ijl}^1 =  \mathrm{MLP}([\mathcal{F}_{il}; \mathcal{F}_{jl}])${{< /math >}}. It is designed to capture the intricate relations that exist inside the pairs of nodes.
2. *Unidirectional Multi-Head Attention.* In order to model the external information, we employ an attention mechanism where the query is composed of the aforementioned {{< math >}}$d^2${{< /math >}} {{< math >}}$h${{< /math >}}-dimensional raw pairwise features, while the keys and values consist of {{< math >}}$h${{< /math >}}-dimensional features of {{< math >}}$d${{< /math >}} individual nodes, i.e., {{< math >}}$\mathcal{P}^2 = \mathrm{MultiHeadAttention}(\mathcal{P}^1, \mathcal{F}, \mathcal{F})${{< /math >}}. Note that, this attention operation is unidirectional, which means we only calculate cross attention from raw pairwise features {{< math >}}$\mathcal{P}^1${{< /math >}} to node features {{< math >}}$F${{< /math >}}. This design is meant to capture both pair-wise and node-wise information (as both are critical to model the causality, as discussed in Sec. 1) while at the same time to maintain a reasonable computational cost.
3. *Final Processing.* Following the widely-adopted transformer architecture, we incorporate a residual structure and a dropout layer after the previous part, i.e., {{< math >}}$\mathcal{P}^3 = \mathrm{Norm}(\mathcal{P}^1 + \mathcal{P}^2)${{< /math >}}. Finally, we introduce a three-layer MLP to further capture intricate patterns and non-linear relationships between the input embeddings, as well as to more effectively process the information from the attention mechanism: {{< math >}}$\mathcal{P} = \mathrm{Norm}(\mathrm{MLP}(\mathcal{P}^3) + \mathcal{P}^3)${{< /math >}}.

    It yields the final pairwise feature tensor {{< math >}}$\mathcal{P} \in \mathbb{R}^{d \times d \times n \times h}${{< /math >}}.

### 4.3. Learning Targets

As mentioned above, our learning target is a combination of the skeleton and the set of v-structures, which together represent an MEC. Two separate neural (sub-)networks are trained for these two targets.

**Skeleton Prediction.** As the persistent dependency between pairs of nodes determines the existence of edges in the skeleton, the pairwise features correspond to edges in the skeleton naturally. Therefore, for the skeleton learning task, we initially employ a max-pooling layer over the observation dimension to obtain a single vector {{< math >}}$\mathcal{S}_{ij} \in \mathbb{R}^{h}${{< /math >}} for each pair of nodes, i.e., {{< math >}}$\mathcal{S}_{ij} = \max_{k} \mathcal{P}_{ijk}${{< /math >}}. Then, a linear layer and a sigmoid function are applied to map the pairwise features to the final prediction of edges, i.e., {{< math >}}$S_{ij} = \mathrm{Sigmoid}(\mathrm{Linear}(\mathcal{S}_{ij}))${{< /math >}}. Our learning label, the undirected graph representing the skeleton, can be easily calculated by summing the adjacency of the DAG {{< math >}}$G${{< /math >}} and its transpose {{< math >}}$G^T${{< /math >}}.

Therefore, our learning target for the skeleton prediction task can be formulated as {{< math >}}$\min \mathcal{L}(S, G + G^T)${{< /math >}}, where {{< math >}}$\mathcal{L}${{< /math >}} is the popularly used binary cross-entropy loss function.

**V-structure Prediction.** A UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} is a v-structure when {{< math >}}$\exists \mathbf{S}${{< /math >}}, such that {{< math >}}$X_k \notin \mathbf{S}${{< /math >}} and {{< math >}}$X_i \perp X_j | \mathbf{S}${{< /math >}}. Motivated by this, we concatenate the corresponding pairwise features of the pair {{< math >}}$\langle X_i, X_j \rangle${{< /math >}} with the node features of {{< math >}}$X_k${{< /math >}} as the feature for each UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} after a max-pooling along the observation dimension, i.e., {{< math >}}$\mathcal{U}_{kij} = [\max_l \mathcal{P}_{ijl};\max_l \mathcal{F}_{kl}]${{< /math >}}. After that, we use a three-layer MLP with a sigmoid function to predict the existence of v-structures among all UTs, i.e., {{< math >}}$\mathcal{U}_{kij} = \mathrm{Sigmoid}(\mathrm{MLP}( \mathcal{U}_{kij}))${{< /math >}}. Given a data sample of {{< math >}}$d${{< /math >}} nodes, it outputs a third-order tensor of shape {{< math >}}$\mathbb{R}^{d \times d \times d}${{< /math >}}, namely v-tensor, corresponding to the predictions of the existence of v-structures. The v-tensor label can be obtained by {{< math >}}$\mathcal{V}_{kij} = G_{ik} G_{jk} (1 - G_{ij})(1 - G_{ji})${{< /math >}}, where {{< math >}}$\mathcal{V}_{kij}${{< /math >}} indicates the existence of v-structure {{< math >}}$X_i \rightarrow X_k \leftarrow X_j${{< /math >}}. Therefore, the learning target for the v-structure prediction task can be formulated as {{< math >}}$\min \mathcal{L}_{UT}(\mathcal{U}, \mathcal{V})${{< /math >}}, where {{< math >}}$\mathcal{L}_{UT}${{< /math >}} is the binary cross-entropy loss masked by UTs, i.e., we only calculate such loss on the valid UTs. In our current implementation, the parameters of the feature encoders are fine-tuned from the skeleton prediction task, as the UTs to be classified are obtained from the predicted skeleton and the skeleton prediction can be seen as a general pre-trained task.

Note that neural networks with our learning targets have a theoretical guarantee for correctness in asymptotic sense, as mentioned around the end of Sec. 3.

### 4.4. Post-Processing

Although our method theoretically guarantees asymptotic correctness, conflicts in predicted v-structures might occasionally occur in practice. Therefore, in the post-processing stage, we apply a straightforward heuristic to resolve the potential conflicts and cycles among predicted v-structures following previous work (<a href="#ref-6">Dai et al., 2023</a>). After that, we use an improved version of Meek rules (<a href="#ref-23">Meek, 1995a</a>; <a href="#ref-34">Tsagris, 2019</a>) to obtain other MEC-identifiable edges without introducing extra cycles. Combining the skeleton from the skeleton predictor model with all MEC-identifiable edge directions, we get the CPDAG predictions.

We provide a more detailed description of the post-processing process in Appendix Sec. A3. It is worth noting that our current design of post-processing is a very conservative one, and this module is also non-essential in our whole framework; see Appendix Sec. A3 for more discussions and evidences.

## 5. Experiments

**Table 1**: **General comparison of SiCL and other methods**. The average performance results in three runs are reported for SiCL. GES takes more than 24 hours per graph on WS-L-G. SLdicso is unsuitable on non-linear-Gaussian data. Full results on all metrics are provided in Appendix Tab. A5.

| Method | WS-L-G s-F1 ↑ | WS-L-G o-F1 ↑ | SBM-L-G s-F1 ↑ | SBM-L-G o-F1 ↑ | WS-RFF-G s-F1 ↑ | WS-RFF-G o-F1 ↑ | SBM-RFF-G s-F1 ↑ | SBM-RFF-G o-F1 ↑ | ER-CPT-MC s-F1 ↑ | ER-CPT-MC o-F1 ↑ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| PC | 30.4 | 16.0 | 58.8 | 35.9 | 36.1 | 16.1 | 57.5 | 34.2 | 82.2 | 40.6 |
| GES | * | * | 70.8 | 55.0 | 41.7 | 23.6 | 56.5 | 38.0 | 82.1 | 42.4 |
| NOTEARS | 33.3 | 31.5 | 80.1 | 77.8 | 37.7 | 33.4 | 55.6 | 48.5 | 16.7 | 0.6 |
| DAG-GNN | 35.5 | 32.7 | 66.2 | 62.5 | 33.2 | 28.9 | 47.1 | 40.6 | 24.8 | 3.7 |
| GRAN-DAG | 16.6 | 11.7 | 22.6 | 14.4 | 4.7 | 1.1 | 17.4 | 3.8 | 40.8 | 7.3 |
| GOLEM | 30.0 | 19.3 | 68.5 | 65.2 | 27.6 | 17.7 | 41.1 | 24.8 | 37.6 | 9.3 |
| SLdisco | 0.1 | 0.1 | 1.9 | 1.2 | * | * | * | * | * | * |
| AVICI | 39.9 | 35.8 | 84.3 | 81.6 | 47.7 | 45.2 | 76.6 | 72.7 | 76.9 | 57.6 |
| SiCL | **44.7** | **38.5** | **85.8** | **82.7** | **51.8** | **46.3** | **82.1** | **78.0** | **84.2** | **59.9** |

In this section, we report the performance of SiCL on both synthetic and real-world benchmarks, followed by an ablation study.
More results and discussions about time cost, generality, and acyclicity are deferred to Appendix Sec. A8, due to page limit.

### 5.1. Experiment Design

**Metrics.** We profile a causal discovery method's performance using the following two tasks:

*Skeleton Prediction*:
Given a data sample {{< math >}}$D${{< /math >}} of {{< math >}}$d${{< /math >}} variables, for each variable pair, we want to infer if there exists direct causation between them. The standard metric **s-F1** (short for **skeleton-F1**) is used, which considers skeleton prediction as a binary classification task over the {{< math >}}$d(d-1)/2${{< /math >}} variable pairs.
For completion, we also report classification accuracy results.
For methods with probabilistic outputs, AUC and AUPRC scores are also measured.

*CPDAG Prediction*:
Given a data sample {{< math >}}$D${{< /math >}} of {{< math >}}$d${{< /math >}} variables, for each pair of variables, we aim to determine if there is direct causality between them. If so, we then assess whether the causal direction is MEC-identifiable, and if it is, we attempt to infer the specific causal direction.
As this task involves both directed and undirected edge prediction, we use **SHD** (Structural Hamming Distance) to measure the difference between the true CPDAG and the inferred CPDAG. Besides that, we also measure the **o-F1** (short for **orientation-F1**) of the directed sub-graph of the inferred CPDAG (compared against the directed sub-graph of the true CPDAG), which focuses on capturing the inference method's orientation capability in identifying *MEC-identifiable* causal edges.

Finally, we calculate the **v-F1** score, where the F1 score is based on the set of v-structures in the true CPDAG as the positive instances (from all ordered triples), and the v-structures in the inferred CPDAG as the positive predictions.

**Testing Data.**
A testing instance consists of a groundtruth causal graph {{< math >}}$G${{< /math >}}, the structural equations {{< math >}}$f_i${{< /math >}} and noise variables {{< math >}}$\epsilon_i${{< /math >}} for each variable {{< math >}}$X_i${{< /math >}}, and an i.i.d. sample {{< math >}}$D${{< /math >}}. We use two categories of testing instances in our experiments:

*Analytical Instances*:
where {{< math >}}$G${{< /math >}} is sampled from a DAG distribution {{< math >}}$\mathcal{G}${{< /math >}}, and {{< math >}}$\{f_i,\epsilon_i\}${{< /math >}} sampled from a structural-equation distribution {{< math >}}$\mathcal{F}${{< /math >}} and a noise meta-distribution {{< math >}}$\mathcal{N}${{< /math >}}.

We consider three random graph distributions for {{< math >}}$\mathcal{G}${{< /math >}}: Watts-Strogatz (WS), Stochastic Block Model (SBM), Erdos-Rényi (ER); and three {{< math >}}$\mathcal{F}${{< /math >}}'s: random linear (L), Random Fourier Features (RFF), and conditional probability table (CPT). {{< math >}}$\mathcal{N}${{< /math >}} is a uniform distribution over Gaussian's for continuous data, and a Dirichlet distribution over Multinomial Categorical distributions for discrete data.

We examine five combinations of testing instances: **WS-L-G**, **SBM-L-G**, **WS-RFF-G**, **SBM-RFF-G**, and **ER-CPT-MC**.

*Real-world Instance*:
The classic dataset **Sachs** is used to evaluate performance in real-world scenarios.
It consists of a data sample recording the concentration levels of 11 phosphorylated proteins in 853 human immune system cells, and of a causal graph over these 11 variables identified by <a href="#ref-30">Sachs et al. (2005)</a> based on expert consensus and biology literature.

**Algorithms.**
As baselines, we compare with a series of representative unsupervised methods, including **PC** (using the recent Parallel-PC variation by <a href="#ref-17">Le et al. (2016)</a>),
**GES** (<a href="#ref-3">Chickering, 2002</a>),
**NOTEARS** (<a href="#ref-43">Zheng et al., 2018</a>),
**GOLEM** (<a href="#ref-25">Ng et al., 2020</a>),
**DAG-GNN** (<a href="#ref-40">Yu et al., 2019</a>),
**GRANDAG** (<a href="#ref-15">Lachapelle et al., 2020</a>), **SLdisco** (<a href="#ref-28">Petersen et al., 2023</a>) as well as **AVICI** (<a href="#ref-21">Lorch et al., 2022</a>), a DNN-based SCL method regarded as current state-of-the-art method.

For our method, besides the full **SiCL** implementation as described by Sec. 4,
we also implement
**SiCL-Node-Edge**, which predicts the causal graph using the node features and can be regarded as equivalent to AVICI, and
**SiCL-no-PF**, which skips pairwise feature extraction and predicts the skeleton and v-tensor using node-wise features (see Appendix Fig. A5).

Notably, SiCL contains 2.8M parameters, while SiCL-Node-Edge and SiCL-no-PF contain 3.2M parameters, because SiCL contains fewer layers on the node feature encoder to eliminate potential bias from size difference.

For DNN-based SCL methods, the DNNs are trained with synthetic data where the causal graphs follow the Erdos-Rényi (ER) and Scale-Free (SF) models and the structural equations and noise variables follow the same distribution type as the corresponding testing data.
Therefore, the disparities between the causal graph distribution at training and testing time help to examine the generality of SiCL in **OOD** settings to some extent.

More details of the experimental setting are presented in Appendix Sec. A7.

### 5.2. Results on Synthetic Dataset

We conduct a comprehensive comparison of SiCL with various baselines in both skeleton prediction and CPDAG prediction tasks.
The main results of metrics skeleton-F1 and orientation-F1 are presented in Tab. 1, and results on full metrics are provided in Appendix Tab. A5.

On continuous data, DNN-based SCL methods (i.e., AVICI and SiCL) demonstrate consistent and obvious advantages over traditional approaches.
SiCL consistently outperforms the other methods on both skeleton prediction task and CPDAG prediction task.
On the other hand, some unsupervised methods achieve comparable performance among DNN-based SCL methods on the discrete data ER-CPT-MC.
Nonetheless, our proposed SiCL emerges as the top performer, further substantiating its superiority in addressing the causal learning problem.

### 5.3. Results on Real-world Dataset

**Table 2**: Comparison on Sachs dataset.

| Method | Skeleton Prediction s-F1 ↑ | Skeleton Prediction s-Acc. ↑ | CPDAG Prediction SHD ↓ | CPDAG Prediction #v-struc. ↓ |
| :---: | :---: | :---: | :---: | :---: |
| PC | 68.6 | 80.0 | 19 | 12 |
| GES | 70.6 | 81.8 | 19 | 8 |
| DAG-GNN | 21.1 | 72.7 | 15 | **0** |
| NOTEARS | 11.1 | 70.9 | 16 | **0** |
| GRAN-DAG | 45.5 | 78.2 | 12 | **0** |
| GOLEM | 36.4 | 74.5 | 14 | **0** |
| AVICI | 66.7 | 83.5 | 18 | 14 |
| SiCL | **71.4** | **86.8** | **6** | **0** |

To assess the practical applicability of SiCL, we conduct a comparison using the real-world dataset Sachs.
The discretized Sachs data obtained from the bnlearn library (Note: <https://www.bnlearn.com/>) is used.
The DNN-based SCL methods are trained on random synthetic graphs, making this also an **OOD** prediction task.
The results are provided in Tab. 2.

For the skeleton prediction task, SiCL performs the best, albeit with a modest gap (generally 1∼3 scores higher than the runners-up). For the CPDAG prediction task, SiCL performs significantly better than all other methods (reducing SHD from 12 to 6, against the second best). Interestingly, the true causal DAG of the Sachs benchmark actually contains no v-structure, so any predicted v-structure is an error. We see that methods competitive with SiCL in skeleton prediction (AVICI, PC, GES) mistakenly predicted a large number of v-structures on the Sachs data, while SiCL correctly predict zero v-structure.

**Table 3**: Ablation study of SiCL components. Full metrics are available in Appendix Tab. A6.

| Method | WS-L-G s-F1 ↑ | WS-L-G o-F1 ↑ | SBM-L-G s-F1 ↑ | SBM-L-G o-F1 ↑ |
| :---: | :---: | :---: | :---: | :---: |
| SiCL-Node-Edge | 39.9 | 35.8 | 84.3 | 81.6 |
| SiCL-no-PF | 42.4 | 37.9 | 85.5 | 82.2 |
| SiCL | **44.7** | **38.5** | **85.8** | **82.7** |

### 5.4. Ablation Study

{{< figgrid caption="**Figure 2**: Comparison of SiCL-Node-Edge and SiCL-no-PF in o-F1 trend as observation samples increase on a constructed dataset." >}}
paire/Cmp_on_of1_font.png | 100
{{< /figgrid >}}


**Effectiveness of Learning Identifiable Structures.** As discussed in Section 4.3, SiCL focuses on learning MEC-identifiable causal structures rather than directly learning the adjacency matrix. To verify the effectiveness of this idea, we compare SiCL-Node-Edge with SiCL-no-PF. These two models share a similar node feature encoder architecture but have different learning targets: SiCL-Node-Edge predicts the adjacency matrix, while the SiCL-no-PF predicts the skeleton and v-tensor. The results are shown in Table 3. Consistently, SiCL-no-PF demonstrates higher performance on both skeleton and CPDAG prediction tasks. This observation echoes our theoretical conclusion regarding the necessity and benefits of learning identifiable causal structures to improve overall performance.

To further underscore the significance of learning identifiable causal structures (especially in the asymptomatic sense), we conduct a comparative analysis using a specially constructed dataset, which contains six nodes forming an independent v-structure and a UT. Figure 2 illustrates that the orientation F1 scores of CPDAG predictions from SiCL-Node-Edge suffer from an unavoidable error and do not improve with the addition of more observational samples. In contrast, predictions from SiCL-no-PF reach perfect accuracy, confirming the value of learning identifiable causal structures.

**Effectiveness of Pairwise Representation.** To assess the effectiveness of pairwise representation, we compare the full version of SiCL with a variant lacking pairwise features (SiCL-no-PF). As shown in Table 3, the full-version SiCL consistently outperforms SiCL-no-PF in both skeleton prediction and CPDAG prediction tasks. Notably, we have intentionally set the model size of the full-version SiCL (2.8M parameters) to be smaller than that of SiCL-no-PF (3.2M parameters) so as to avoid any potential advantage from increased model complexity brought by the pairwise feature encoder module. The observed performance gains in this case underscore the critical role of pairwise features in identifying causal structures. Additionally, we conduct further comparisons across more diverse settings, with results detailed in Appendix Sec. A7. These results demonstrate even more pronounced improvements in favor of SiCL, reinforcing the importance of pairwise representations in causal discovery.

## 6. Conclusion

We proposed SiCL, a novel DNN-based SCL approach designed to predict the corresponding skeleton and a set of v-structures. We showed that such design do not suffer from the (non-)identifiability limit that exists in current architectures. Moreover, SiCL is equipped with a pairwise encoder module to explicitly model relationships between node-pairs. Experimental results validated the effectiveness of these ideas.

This paper also introduces a few interesting open problems. The proposed DNN model works in the canonical setting under the classic MEC theory, in which the skeleton and v-structures are the identifiable structure. It can be an interesting future-work direction to explore how to learn other identifiable causal structure in other assumption settings following the same principle. Due to the inherent complexity of DNNs, the explanation of the decision mechanism of our model remains an open question. Therefore, future work could consider to explore how decisions are made within the networks and provide some insights for traditional methods. Moreover, the proposed pairwise encoder modules needs {{< math >}}$O(d^3)${{< /math >}} computational complexity, which may restrict its current application to scenarios with huge number of nodes. Future work could focus on simplifying these operations or exploring features with less complexity (e,g., low rank features) to reduce the overall computational cost.

## Checklist

1. For all models and algorithms presented, check if you include:
   1. A clear description of the mathematical setting, assumptions, algorithm, and/or model. [Yes]
   2. An analysis of the properties and complexity (time, space, sample size) of any algorithm. [Yes]
   3. (Optional) Anonymized source code, with specification of all dependencies, including external libraries. [Yes]

2. For any theoretical claim, check if you include:
   1. Statements of the full set of assumptions of all theoretical results. [Yes]
   2. Complete proofs of all theoretical results. [Yes]
   3. Clear explanations of any assumptions. [Yes]

3. For all figures and tables that present empirical results, check if you include:
   1. The code, data, and instructions needed to reproduce the main experimental results (either in the supplemental material or as a URL). [Yes]
   2. All the training details (e.g., data splits, hyperparameters, how they were chosen). [Yes]
   3. A clear definition of the specific measure or statistics and error bars (e.g., with respect to the random seed after running experiments multiple times). [Yes]
   4. A description of the computing infrastructure used. (e.g., type of GPUs, internal cluster, or cloud provider). [Yes]

4. If you are using existing assets (e.g., code, data, models) or curating/releasing new assets, check if you include:
   1. Citations of the creator If your work uses existing assets. [Yes]
   2. The license information of the assets, if applicable. [Not Applicable]
   3. New assets either in the supplemental material or as a URL, if applicable. [Not Applicable]
   4. Information about consent from data providers/curators. [Yes]
   5. Discussion of sensible content if applicable, e.g., personally identifiable information or offensive content. [Not Applicable]

5. If you used crowdsourcing or conducted research with human subjects, check if you include:
   1. The full text of instructions given to participants and screenshots. [Not Applicable]
   2. Descriptions of potential participant risks, with links to Institutional Review Board (IRB) approvals if applicable. [Not Applicable]
   3. The estimated hourly wage paid to participants and the total amount spent on participant compensation. [Not Applicable]

**Algorithm A1**: SiCL Workflow for Predicting Causal Structures

```
Procedure INFERENCE(data, target)
    Calculate node features with node encoder
    Calculate pairwise features with pairwise encoder following Sec. 4.2
    if target is skeleton then
        Calculate skeleton with Sec. 4.3
    else
        Calculate v-structures with Sec. 4.3
    end if
End Procedure

Procedure TRAINING_PHASE()
    skeleton_predictor <- init_skeleton_predictor()
    Sample graphs and corresponding data
    Training the skeleton predictor with INFERENCE(data, skeleton)
    Training the v-structure predictor with INFERENCE(data, v-structure), with feature encoders fine-tuned from skeleton predictor
End Procedure

Procedure TESTING_PHASE(test_data)
    Calculate predicted skeleton with the trained skeleton predictor
    Calculate predicted v-structures with the trained v-structure predictor
    Combine predicted skeleton and v-structures to obtain predicted CPDAG
End Procedure
```

{{< figgrid caption="**Figure A3**: Illustration of the pairwise encoder module. In Part ①, it initializes raw pairwise features. In Part ②, a unidirectional attention is applied to utilized information from node features and pairwise features. In Part ③, an MLP and residual connection is used to yield final pairwise features." >}}
paire/pairwiseencoder.png | 100
{{< /figgrid >}}


## A1. Theoretical Guarantee

In this section, we delve into the theoretical analysis concerning the asymptotic correctness of our proposed model with respect to the sample size. Sec. A1.1 lays out the essential definitions and assumptions pertinent to the problem under study. Following this, from Sec. A1.2 to A1.3, we rigorously demonstrate the asymptotic correctness of the neural network model. Finally, in Sec. A1.4, we engage in a detailed discussion about the practical advantages and superiority of neural network models.

### A1.1. Definitions and Assumptions

As outlined in Sec. 2, a Causal Graphical Model is defined by a joint probability distribution {{< math >}}$P${{< /math >}} over {{< math >}}$d${{< /math >}} random variables {{< math >}}$X_1, X_2, \cdots, X_{d}${{< /math >}}, and a DAG {{< math >}}$G${{< /math >}} with {{< math >}}$d${{< /math >}} vertices representing the {{< math >}}$d${{< /math >}} variables.
An observational dataset {{< math >}}$D${{< /math >}} consists of {{< math >}}$n${{< /math >}} records and {{< math >}}$d${{< /math >}} columns, which represents {{< math >}}$n${{< /math >}} instances drawn i.i.d. from {{< math >}}$P${{< /math >}}.
In this work, we assume causal sufficiency:

**Assumption A1.1 (Causal Sufficiency).** There are no latent common causes of any of the variables in the graph.

Moreover, we assume the data distribution {{< math >}}$P${{< /math >}} is Markovian to the DAG {{< math >}}$G${{< /math >}}:

**Assumption A1.2 (Markov Factorization Property).** Given a joint probability distribution {{< math >}}$P${{< /math >}} and a DAG {{< math >}}$G, P${{< /math >}} is said to satisfy Markov factorization property w.r.t. {{< math >}}$G${{< /math >}} if {{< math >}}$P:=${{< /math >}} {{< math >}}$P\left(X_1, X_2, \cdots, X_d\right)=\prod_{i=1}^d P\left(X_i \mid \mathrm{pa}_i^G\right)${{< /math >}}, where {{< math >}}$\mathrm{pa}_i^G${{< /math >}} is the parent set of {{< math >}}$X_i${{< /math >}} in {{< math >}}$G${{< /math >}}.

It is noteworthy that the Markov factorization property is equivalent to the Global Markov Property (GMP) (<a href="#ref-16">Lauritzen, 1996</a>), which is

**Definition A1.1 (Global Markov Property (GMP)).** {{< math >}}$P${{< /math >}} is said to satisfy GMP (or Markovian) w.r.t. a DAG {{< math >}}$G${{< /math >}} if {{< math >}}$X \perp_G Y|Z \Rightarrow X \perp Y| Z${{< /math >}}. Here {{< math >}}$\perp_G${{< /math >}} denotes d-separation, and {{< math >}}$\perp${{< /math >}} denotes statistical independence.

GMP indicates that any d-separation in graph {{< math >}}$G${{< /math >}} implies conditional independence in distribution {{< math >}}$P${{< /math >}}. We further assume that {{< math >}}$P${{< /math >}} is faithful to {{< math >}}$G${{< /math >}} by:

**Assumption A1.3 (Faithfulness).** Distribution {{< math >}}$P${{< /math >}} is faithful w.r.t. a DAG {{< math >}}$G${{< /math >}} if {{< math >}}$X \perp Y\left|Z \Rightarrow X \perp_G Y\right| Z${{< /math >}}.

**Definition A1.2 (Canonical Assumption).** We say our settings satisfy the canonical assumption if the Assumptions A1.1 - A1.3 are all satisfied.

We restate the definitions of skeletons, Unshielded Triples (UTs) and v-strucutres as follows.

**Definition A1.3 (Skeleton).** A skeleton {{< math >}}$E${{< /math >}} defined over the data distribution {{< math >}}$P${{< /math >}} is an undirected graph where an edge exists between {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} if and only if {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} are always dependent in {{< math >}}$P${{< /math >}}, i.e., {{< math >}}$\forall Z \subseteq\left\{X_1, X_2, \cdots, X_d\right\} \backslash \left\{X_i, X_j \right\}${{< /math >}}, we have {{< math >}}$X_i \nperp X_j | Z${{< /math >}}.

Under our assumptions, the skeleton is the same as the corresponding undirected graph of {{< math >}}$G${{< /math >}} (<a href="#ref-33">Spirtes et al., 2000</a>).

**Definition A1.4 (Unshielded Triples (UTs) and V-structures).** A triple of variables {{< math >}}$X, T, Y${{< /math >}} is an Unshielded Triple (UT) denoted as {{< math >}}$\langle X, T, Y \rangle${{< /math >}}, if {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}} are both adjacent to {{< math >}}$T${{< /math >}} but not adjacent to each other in the DAG {{< math >}}$G${{< /math >}} or the corresponding skeleton.
It becomes a v-structure denoted as {{< math >}}$X \rightarrow T \leftarrow Y${{< /math >}}, if the directions of the edges are from {{< math >}}$X${{< /math >}} and {{< math >}}$Y${{< /math >}} to {{< math >}}$T${{< /math >}} in {{< math >}}$G${{< /math >}}.

We introduce the definition of separation set as:

**Definition A1.5 (Separation Set).** For a node pair {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}}, a node set {{< math >}}$Z${{< /math >}} is a separation set if {{< math >}}$X_i \perp X_j | Z ${{< /math >}}. Under faithfulness assumption, a separation set {{< math >}}$Z${{< /math >}} is a subset of variables within the vicinity that d-separates {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}}.

Finally, we assume a neural network can be used as a universal approximator in our settings.

**Assumption A1.4 (Universal Approximation Capability).** A neural network model can be trained to approximate a function under our settings with arbitary accuracy.

### A1.2. Skeleton Learning

In this section, we prove the asymptotic correctness of neural networks on the skeleton prediction task by constructing a perfect model and then approximating it with neural networks.
For the sake of convenience and brevity in description, we define the skeleton predictor as follows.

**Definition A1.6 (Skeleton Predictor).** Given observational data {{< math >}}$D${{< /math >}}, a skeleton predictor is a predicate function with domain as observational data {{< math >}}$D${{< /math >}} and predicts the adjacency between each pair of the vertices.

Now we restate the Remark from <a href="#ref-22">Ma et al. (2022)</a> as the following proposition.
It proves the existence of a perfect skeleton predictor by viewing the skeleton prediction step of PC (<a href="#ref-33">Spirtes et al., 2000</a>) as a skeleton predictor, which is proved to be sound and complete.

**Proposition A1.1 (Existence of a Perfect Skeleton Predictor).** There exists a skeleton predictor that always yields the correct skeleton with sufficient samples in {{< math >}}$D${{< /math >}}.

**Proof.** We construct a skeleton predictor {{< math >}}$SP${{< /math >}} consisting of two parts by viewing PC (<a href="#ref-33">Spirtes et al., 2000</a>) as a skeleton predictor.
In the first part, it extracts a pairwise feature {{< math >}}$\boldsymbol{x}_{i j}${{< /math >}} for each pair of nodes {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}}:

{{< math >}}
$$
\boldsymbol{x}_{i j}=\min _{Z \subseteq V \backslash\left\{X_i, X_j\right\}}\left\{X_i \sim X_j \mid Z\right\}, \tag{2}
$$
{{< /math >}}

where {{< math >}}$\left\{X_i \sim X_j \mid Z\right\} \in [0, 1] ${{< /math >}} is a scalar value that measures the conditional dependency between {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} given a node subset {{< math >}}$Z${{< /math >}}.

Consequently, {{< math >}}$\boldsymbol{x}_{i j} &gt; 0${{< /math >}} indicates the persistent dependency between the two nodes.

In the second part, it predicts the adjacency based on {{< math >}}$\boldsymbol{x}_{i j}${{< /math >}}:

{{< math >}}
$$
\left(X_i,  X_j\right)= \begin{cases} 1 \text { (adjacent) } & \boldsymbol{x}_{i j} \neq 0 \\ 0 \text { (non-adjacent) } & \boldsymbol{x}_{i j} = 0\end{cases} \tag{3}
$$
{{< /math >}}

Now we prove that {{< math >}}$SP${{< /math >}} always yields the correct skeleton by proving the absence of false positive predictions and false negative predictions. Here, false positive prediction denotes {{< math >}}$SP${{< /math >}} predicts a non-adjacent node pair as adjacent and false negative predictions denote {{< math >}}$SP${{< /math >}} predicts an adjacent node pair as non-adjacent.

- **False Positive.** Suppose {{< math >}}$X_i, X_j${{< /math >}} are non-adjacent. Under the Markovian assumption, there exists a set of nodes {{< math >}}$Z${{< /math >}} such that {{< math >}}$\left\{X_i \sim X_j \mid Z\right\} = 0${{< /math >}} and hence {{< math >}}$\boldsymbol{x}_{ij} = 0${{< /math >}}. According to Eq. (3), {{< math >}}$SP${{< /math >}} will always predicts them as non-adjacent.
- **False Negative**. Suppose {{< math >}}$X_i, X_j${{< /math >}} are adjacent. Under the faithfulness assumption, for any {{< math >}}$Z \in V \backslash \left\{X_i, X_j\right\}, \left\{X_i \sim X_j \mid Z\right\} &gt; 0${{< /math >}}, which implies {{< math >}}$\boldsymbol{x}_{ij} &gt; 0${{< /math >}}. Therefore, {{< math >}}$SP${{< /math >}} always predicts them as adjacent.

Therefore, {{< math >}}$SP${{< /math >}} never yields any false positive predictions or false negative predictions under the Markovian assumption and faithfulness assumption, i.e., it always yields the correct skeleton.

With the existence of a perfect skeleton predictor, we prove the correctness of neural network models with sufficient samples under our assumptions.

**Theorem A1.1.** Under the canonical assumption and the assumption that neural network can be used as a universal approximator (Assumption A1.4),
there exists a neural network model that always predicts the correct skeleton with sufficient samples in {{< math >}}$D${{< /math >}}.

**Proof.** From Proposition A1.1, there exists a perfect skeleton predictor that predicts the correct skeleton.
Thus, according to the Assumption A1.4, a neural network model can be trained to approximate the perfect skeleton prediction hence predicts the correct skeleton.

### A1.3. Orientation Learning

Similarly to the overall thought process in Sec. A1.2, in this section we prove the asymptotic correctness of neural networks on the v-structure prediction task by constructing a perfect model and then approximating it with neural networks.

**Definition A1.7 (V-structure Predictor).** Given observational data {{< math >}}$D${{< /math >}} with sufficient samples from a {{< math >}}$BN${{< /math >}} with vertices {{< math >}}$V = \{X_1, \dots, X_p\}${{< /math >}}, a v-structure predictor is a predicate function with domain as observational data {{< math >}}$D${{< /math >}} and predicts existence of the v-structure for each unshielded triple.

The following proposition proves the existence of a perfect v-structure predictor by viewing the orientation step of PC (<a href="#ref-33">Spirtes et al., 2000</a>) as a v-structure predictor.

**Proposition A1.2 (Existence of a Perfect V-structure Predictor).** Under the Markov assumption and faithfulness assumption, there exists skeleton predictor that always yields the correct skeleton.

**Proof.** We construct a v-structure predictor {{< math >}}$VP${{< /math >}} consisting of two parts by viewing PC (<a href="#ref-33">Spirtes et al., 2000</a>) as a v-structure predictor.

In the first part, it extracts a boolean feature {{< math >}}$\boldsymbol{z}_{kij}${{< /math >}} for each UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}}:

{{< math >}}
$$
\boldsymbol{z}_{kij} = (X_k \in Z), \text{ where } Z \text{ is called as a sepset, i.e. } X_i\perp Y_j | Z.  \tag{4}
$$
{{< /math >}}

Note that the sepset {{< math >}}$Z${{< /math >}} always exists because the separation set of a UT always exists (See Lemma 4.1 in <a href="#ref-6">Dai et al. (2023)</a>).

In the second part, it predicts the v-structures based on {{< math >}}$\boldsymbol{z}_{ijk}${{< /math >}}:

{{< math >}}
$$
\langle X_i, X_k, X_j\rangle = \begin{cases} 0 \text { (not v-structure) } & \boldsymbol{z}_{k i j } = True \\ 1 \text { (v-structure) } & \boldsymbol{z}_{k i j } = False\end{cases} \tag{5}
$$
{{< /math >}}

Now we prove that {{< math >}}$VP${{< /math >}} always yields the correct predictions of v-structures.
According to Theorem 5.1 on p.410 of <a href="#ref-33">Spirtes et al. (2000)</a>, assuming faithfulness and sufficient samples, if a UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} is a v-structure, then {{< math >}}$X_k${{< /math >}} does not belong to any separation sets of {{< math >}}$(X_i, X_j)${{< /math >}}; if a UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} is not a v-structure, then {{< math >}}$X_k${{< /math >}} belongs to every separation sets of {{< math >}}$(X_i, X_j)${{< /math >}}. Therefore, we have {{< math >}}$\boldsymbol{z}_{kij} = False${{< /math >}} if and only if {{< math >}}$X_k${{< /math >}} is not in any separation set of {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}}, i.e., {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} is a v-structure.

With the existence of a perfect v-structure predictor, we prove the correctness of neural network models with sufficient samples under our assumptions.

**Theorem A1.2.** Under the canonical assumption and the assumption that neural network can be used as a universal approximator (Assumption A1.4), there exists a neural network model that always predicts the correct v-structures with sufficient samples in {{< math >}}$D${{< /math >}}.

**Proof.** From Proposition A1.1, there exists a perfect skeleton predictor that predicts the correct v-structures.
Thus, according to the Assumption A1.4, a neural network model can be trained to approximate the perfect v-structure predictions hence predicts the correct v-structures.

### A1.4. Discussion

In the sections above, we prove the asymptotic correctness of neural network models by constructing theoretically perfect predictors. These predictors both consist of two parts: feature extractors providing features {{< math >}}$\boldsymbol{x}_{ij}${{< /math >}} and {{< math >}}$\boldsymbol{z}_{ijk}${{< /math >}}, and final predictors of adjacency and v-structures. Even though they have a theoretical guarantee of the correctness with sufficient samples, it is noteworthy that they are hard to be applied practically. For example, to obtain {{< math >}}$\boldsymbol{x}_{ij}${{< /math >}} in Eq. (2), we need to calculate the conditional dependency between {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} given every node subset {{< math >}}$Z \subseteq V \backslash\left\{X_i, X_j\right\}${{< /math >}}. Leaving aside the fact that the number of {{< math >}}$Z${{< /math >}}s itself presents factorial complexity, the main issue is that when {{< math >}}$Z${{< /math >}} is relatively large, due to the curse of dimensionality, it becomes challenging to find sufficient samples to calculate the conditional dependency. This difficulty significantly hampers the ability to apply the constructed prefect predictors in practical scenarios.

Some existing methods can be interpreted as constructing more practical predictors. Majority-PC (MPC) (<a href="#ref-4">Colombo et al., 2014</a>) achieves better performance on finite samples by modifying Eq. (4) - (5) as:

{{< math >}}
$$
\boldsymbol{z}_{kij} = \frac{\left|\left\{ (X_k, Z) | \{ X_i \sim X_j | Z\} = 0 \wedge X_k \in Z \right\}\right|}{\left|\left\{ Z | \{ X_i \sim X_j | Z\} = 0 \right\}\right|}, \tag{6}
$$
{{< /math >}}

and

{{< math >}}
$$
\left\langle X_i, X_k, X_j\right\rangle= \begin{cases}0 \text { (not v-structure) } &amp; \boldsymbol{z}_{i j k} &gt; 0.5 \\ 1(\mathrm{v} \text {-structure) } &amp; \boldsymbol{z}_{i j k} \leq 0.5,\end{cases} \tag{7}
$$
{{< /math >}}

where {{< math >}}$\left\{X_i \sim X_j \mid Z\right\} \in [0, 1]${{< /math >}} is a scalar value that measures the conditional dependency between {{< math >}}$X_i${{< /math >}} and {{< math >}}$X_j${{< /math >}} given a node subset {{< math >}}$Z${{< /math >}}, and {{< math >}}$|\cdot|${{< /math >}} represents the cardinality of a set. Due to its more complex classification mechanism, it achieves better performance empirically. However, from the machine learning perspective, features from both the PC and MPC predictors are relatively simple. As supervised causal learning methods, ML4S (<a href="#ref-22">Ma et al., 2022</a>) and ML4C (<a href="#ref-6">Dai et al., 2023</a>) provide more systematic featurizations by manual feature engineering and utilization of powerful machine learning models for classification.

While these methods show enhanced practical efficacy, their manual feature engineering processes are complex. In our paper, we utilize neural networks as universal approximators for learning the prediction of identifiable causal structures. It not only simplifies the procedure but also potentially uncovers more nuanced and complex patterns within the data that manual methods might overlook. It is noteworthy that the benefits of supervised causal learning using neural networks are also discussed elsewhere, as mentioned in SLdisco (<a href="#ref-28">Petersen et al., 2023</a>) and CSIvA (<a href="#ref-13">Ke et al., 2023</a>).

**Algorithm A2**: Post-processing

```
Input: weighted skeleton matrix S, weighted V-tensor U, threshold for skeleton τ_s, threshold for v-structure τ_v
Output: predicted oriented edge set oriEdges, predicted skeleton skeleton

Step 1:
// Obtaining a predicted skeleton by thresholding.
skeleton = {(i, j) | max(S_ij, S_ji) > τ_s}
// Obtaining raw v-structures vstructs_raw by thresholding.
vstructs_raw = {(i, j, k) | (i, j) ∈ skeleton and (i, k) ∈ skeleton and (j, k) ∉ skeleton and max(U_ijk, U_ikj) > τ_v}

Step 2:
// V-structure conflict resolving: discard any v-structure if there exists another conflicted v-structure with a higher predicted score, following (Dai et al., 2023).
vstructs = {(i, j, k) ∈ vstructs_raw | ∀ (i', j', k') ∈ vstructs_raw, (i' ≠ k and i' ≠ j) or (k' ≠ i and j' ≠ i) or U_i'j'k' < U_ijk}

Step 3:
// Obtaining the predicted directed edge from vstructs.
oriEdges_raw = {(j, i) | ∃ k, (i, j, k) ∈ vstructs}
// Set a score for each edge with the highest v-structure's score containing this edge.
Set {p_ij} such that p_ij = max_v U_v for v ∈ oriEdges_raw and v ∋ (i, j).
// If there exist any cycles, remove the edge with the smallest score in each cycle.
oriEdges = {(i, j) ∈ oriEdges_raw | ∀ cycle C, (i, j) ∉ C or (∃ (i', j') ∈ C, p_ij > p_i'j')}

Step 4:
// Meek rules: Add edges to oriEdges for directed edges that (1) introducing the edges does not lead to cycles or new v-structures; (2) adding the opposite edges necessarily leads to cycles or new v-structures.
oriEdges = oriEdges ∪ {(i, j) ∈ skeleton | (i, j) complies with Meek rules}
```

## A2. More Discussions on Identifiability and Causal Assumptions

**Advocation of Learning Identifiable Structures under All Settings.**
In this paper, we have to work on a concrete setting for demonstration purpose with concrete identifiable causal structures in this paper.
Nonetheless, we want to emphasize that the very concept of identifiability, as well as its ramifications in SCL, is indeed a general issue that is less bound to the issue of “which causal structures are identifiable under which assumptions”.
The simple fact that in some situations the causal edge cannot be identified – no matter what feature can be identified in that case – this identifiability limit has a general effect on SCL.
Unless the causal graph/edge itself becomes fully invariant/identifiable (a special case that is important but certainly not universally true), the presence of the identifiability limit entails a fundamental bias for a popular SCL model architecture (i.e., Node-Edge) that cannot be mitigated by larger model or bigger data at all.
This “identifiability-limit-causes-learning-error” effect is the main thesis of this paper, and we advocate to design neural networks that focus on learning the identifiable features (no matter what those features are).
In other words, there is nothing stopping one from studying another setting where another feature is identifiable though, and in that case we would also advocate to learn that feature instead of v-structures.
For example, if we assume canonical MEC assumptions and non-existence of causal-fork and v-structures, the identifiable causal structure becomes a kind of chains.
In that case, one may want to design neural networks that predict about causal chains.

**Rationality of Canonical Assumptions.**
In this paper, we choose the canonical setting under the classic MEC theory, in which the skeleton and v-structures are the identifiable structure.
This setting includes the assumptions of the Markov and faithfulness conditions.
Unlike scenario-specific assumptions, such as those tied to a particular data-generating process, these assumptions are classic assumptions about causality that are often adopted as “postulates” about some general aspects of the world. For example,

- <a href="#ref-26">Pearl (2009)</a> argues that stability (faithfulness) stems from the natural improbability of strict equality constraints among parameters, which aligns with the autonomy of causal mechanisms.
- <a href="#ref-32">Spirtes et al. (2001)</a> support the Causal Faithfulness Condition (CFC) by noting that the exact cancellation of causal paths is highly improbable under natural conditions.
- <a href="#ref-38">Weinberger (2018)</a> reinforces this argument, proposing that coincidences leading to CFC violations are rare and lack explanatory power, further justifying its adoption within a general modeling framework.

These considerations underscore the rationality and generality of the assumptions, making them a natural choice for our analysis.

## A3. Details and Discussion about Post-processing

For comprehensive clarity, we provide a clear process about the post-processing algorithm in Alg. A2.

**Discussion.**
It is worth noting that our design in post-processing is as conservative as possible. In fact, we simply adhere to the conventions in deep learning (i.e., thresholding) to obtain the skeleton and the initial v-structure set. Subsequently, we follow the conventions in constraint-based causal discovery methods to derive the final oriented edges. Therefore, we have not dedicated extensive efforts towards the meticulous design, nor do we intend to emphasize this aspect of our workflow.

The conflicts and cycles are not unique to SiCL; they are, in fact, common issues encountered by all constraint-based algorithms like PC. Moreover, it's worth noting that they never appear if the networks perform perfect. Therefore, the conflict resolving of v-structures and the removal of cycles are designed as fallback mechanisms to ensure the soundness of our workflow, rather than being central elements of our approach. To illustrate it, we experimented with an opposite variant (Intuitively, this is a bad choice) that prioritizes discarding the v-structure with the higher predicted probability. The minimal differences in outcomes between this variant and its counterpart, as detailed in Tab. A4, support our viewpoint that the conflict resolution process is of limited significance within our workflow. On the other hand, experimental results presented in Tab. A11 underscore the infrequency of cycles in the predictions, reinforcing the non-essential nature of the cycle removal component.

**Table A4**: The o-F1 comparison between the used conflict resolving method with an opposite variant.

| Conflict Resolving Method | WS-L-G | SBM-L-G |
| --- | --- | --- |
| Original Conflict Resolving | **41.1** | **83.3** |
| Opposite Conflict Resolving | 40.7 | 83.2 |

## A4. Details about Node Feature Encoder

Motivated by previous approaches (<a href="#ref-21">Lorch et al., 2022</a>; <a href="#ref-13">Ke et al., 2023</a>), we employ a transformer-like architecture comprising attention layers over either the observation dimension or the node dimension alternately as the node feature encoder.
Concretely, for the raw node features {{< math >}}$\mathcal{F} \in \mathbb{R}^{d \times n \times h}${{< /math >}} corresponding to {{< math >}}$d${{< /math >}} nodes and {{< math >}}$n${{< /math >}} observations, our goal is to capture the correlations between both different nodes and different observations.
Therefore, we utilize two transformer encoder layers over the observation dimension and the node dimension alternatively:

{{< math >}}
$$
\begin{aligned}
    \mathcal{F} & \leftarrow TransformerEncoderLayer(\mathcal{F}, \mathcal{F}, \mathcal{F}) \\
    \mathcal{F} & \leftarrow \mathcal{F}.transpose(0, 1) \\
        \mathcal{F} & \leftarrow TransformerEncoderLayer(\mathcal{F}, \mathcal{F}, \mathcal{F})\\
    \mathcal{F} & \leftarrow \mathcal{F}.transpose(0, 1). \\
\end{aligned}
\tag{8}
$$
{{< /math >}}

The above operation is repeated multiple times for sufficiently feature encoding.
It yields the final node feature tensor {{< math >}}$\mathcal{F} \in \mathcal{R}^{d \times \times h}${{< /math >}}.

## A5. Illustration of the Case Study in Sec. 3

Fig. A4 presents an illustration for the case study of the Node-Edge approach in Sec. 3.
It clearly shows that observational data with the two different parametrized forms follow the same joint distribution:

{{< math >}}
$$
P(\left[X, Y, T\right]) =\mathcal{N}\left([0,0,0],\left[\begin{array}{lll}1 & 1 & 1 \\ 1 & 3 & 2 \\ 1 & 2 & 2\end{array}\right]\right). \tag{9}
$$
{{< /math >}}

Therefore, the observational datasets coming from the two DAGs are inherently indistinguishable.

{{< figgrid caption="**Figure A4**: The problem setting to emphasize the limitations of the Node-Edge approach. *Best viewed in color.*" >}}
paire/ps.png | 90
{{< /figgrid >}}


## A6. Proof and Discussion for Proposition 3.1

We first restate the Proposition 3.1 with more details and provide the proof.

**Proposition A6.1.** Let {{< math >}}$\mathcal{G}_n${{< /math >}} be the set of graphs with {{< math >}}$n+1${{< /math >}} nodes where there is a central node {{< math >}}$y${{< /math >}} such that (1) every other node is connected to {{< math >}}$y${{< /math >}}, (2) there is no edge between the other nodes, (3) there is at most one edge pointing to {{< math >}}$y${{< /math >}}.
For any distribution {{< math >}}$Q${{< /math >}} over {{< math >}}$\mathcal{G}_n${{< /math >}}, let {{< math >}}$M(Q)${{< /math >}} be another distribution over {{< math >}}$\mathcal{G}_n${{< /math >}} such that for any causal edges {{< math >}}$e, e'${{< /math >}}, {{< math >}}$P_{G\sim Q}(e \in G) = P_{G\sim M(Q)}(e \in G) = P_{G\sim M(Q)}(e \in G | e' \in G)${{< /math >}}. We have

{{< math >}}
$$
\max_{Q} P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 
1 - \frac{2n-1}{n-1}(1 - \frac{1}{n})^n. \tag{10}
$$
{{< /math >}}

As a corollary, we have

{{< math >}}
$$
\sup_n \max_{Q} P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 
1 - \frac{2}{e} \approx 0.2642, \tag{11}
$$
{{< /math >}}

**Proof.** Denote other nodes except for the central node as {{< math >}}$x_i${{< /math >}} where {{< math >}}$i \in \left\{1, 2, \dots, n\right\}${{< /math >}}.
In our setting, the set {{< math >}}$\mathcal{G}_n${{< /math >}} contains {{< math >}}$n + 1${{< /math >}} DAGs
with the same skeleton and no v-structure: {{< math >}}$G_0: y \rightarrow x_i${{< /math >}} for all {{< math >}}$x_i${{< /math >}}, and {{< math >}}$G_i: y \rightarrow x_j${{< /math >}} for all {{< math >}}$x_j \neq x_i${{< /math >}} together with {{< math >}}$x_i \rightarrow y${{< /math >}}.
Denote the sampling probability of DAG {{< math >}}$G_i${{< /math >}} from {{< math >}}$\mathcal{G}_n${{< /math >}} as {{< math >}}$P_i${{< /math >}}.
Therefore, the marginal probability of the edge {{< math >}}$y \rightarrow x_i${{< /math >}} is {{< math >}}$1 - P_i${{< /math >}}.

If {{< math >}}$\exists i${{< /math >}}, {{< math >}}$P_i = 1${{< /math >}}, it means that {{< math >}}$\mathcal{G}_n${{< /math >}} only contains the DAG {{< math >}}$G_i${{< /math >}}. Therefore, {{< math >}}$M(Q)${{< /math >}} is equivalent to {{< math >}}$Q${{< /math >}} and we have {{< math >}}$P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 0${{< /math >}}.

If {{< math >}}$\forall i${{< /math >}}, {{< math >}}$P_i &lt; 1${{< /math >}}, denoting {{< math >}}$Q_i = 1 - P_i${{< /math >}} and {{< math >}}$P(v)${{< /math >}} as the probability of {{< math >}}$G${{< /math >}} containing no v-structures. In other words, {{< math >}}$P(v) = P_{G \sim M(Q)}(G \in \mathcal{G}_n)${{< /math >}}. We have

{{< math >}}
$$
\begin{aligned}
P(v) = \prod_{i=1}^n Q_i + \sum_{j=1}^n \frac{\prod_i^n Q_i}{Q_j} (1 - Q_j)
&= ( \prod_{i=1}^n Q_i) \cdot (1 + \sum_{j=1}^{n} \frac{1-Q_j}{Q_j}).
\end{aligned}
\tag{12}
$$
{{< /math >}}

As {{< math >}}$P(v)${{< /math >}} is a probability, we have {{< math >}}$P(v) &gt; 0${{< /math >}}. Denoting function

{{< math >}}
$$
f(Q_1, Q_2, \dots, Q_n) = \log P(v) = \sum_{i=1}^n \log Q_i + \log (1 + \sum_{j=1}^n \frac{1-Q_j}{Q_j}), \tag{13}
$$
{{< /math >}}

we would like to find its minimum s.t. {{< math >}}$\sum_i Q_i \geq n - 1${{< /math >}} and {{< math >}}$Q_i \in (0, 1]${{< /math >}}.

Define its Lagrange function

{{< math >}}
$$
L(Q_1, Q_2, \dots, Q_n, \lambda) = f + \lambda (n-1-\sum_i Q_i). \tag{14}
$$
{{< /math >}}

We have

{{< math >}}
$$
\frac{\partial L}{\partial \lambda} = n - 1 - \sum_i Q_i, \tag{15}
$$
{{< /math >}}

and

{{< math >}}
$$
\frac{\partial L}{\partial Q_i} = \frac{1}{Q_i}(1 - \frac{1}{Q_i(1-n + \sum_{k=1}^{n}\frac{1}{Q_k})}) - \lambda. \tag{16}
$$
{{< /math >}}

Now we are going to find the extremums for {{< math >}}$L(Q_1, Q_2, \dots, Q_n, \lambda)${{< /math >}}.

**(1)** If {{< math >}}$\lambda = 0${{< /math >}}, we have {{< math >}}$\forall i${{< /math >}}, {{< math >}}$\frac{\partial f}{\partial Q_i} = 0${{< /math >}}, then

{{< math >}}
$$
\forall i, Q_i = \frac{1}{(1 - n + \sum_{k=1}^{n} \frac{1}{Q_k})}. \tag{17}
$$
{{< /math >}}

It indicates that {{< math >}}$\forall i, Q_i = 1${{< /math >}}, hence {{< math >}}$f = 0${{< /math >}} and {{< math >}}$P(v) = 1${{< /math >}}.

**(2)** If {{< math >}}$\lambda \neq 0${{< /math >}}, {{< math >}}$\exists i${{< /math >}}, we have {{< math >}}$\forall i${{< /math >}}, {{< math >}}$\frac{\partial f}{\partial Q_i} = \lambda${{< /math >}} and {{< math >}}$\sum_{i=1}^n = n - 1${{< /math >}}. In other words, we have

{{< math >}}
$$
\forall i, j, \frac{\partial f}{\partial Q_i} = \frac{\partial f}{\partial Q_j} = \lambda. \tag{18}
$$
{{< /math >}}

Define function

{{< math >}}
$$
h(Q_i) = \frac{\partial f}{\partial Q_i} = \frac{1}{Q_i}(1 - \frac{1}{Q_i(1-n + \sum_{k=1}^{n}\frac{1}{Q_k})}). \tag{19}
$$
{{< /math >}}

we can rewrite the function as

{{< math >}}
$$
h(Q_i) = \frac{1}{Q_i}(1 - \frac{1}{1 + AQ_i}), \tag{20}
$$
{{< /math >}}

where {{< math >}}$A = 1 - n + \sum_{k\neq i} \frac{1}{Q_k} \geq 1 - n + \frac{(n-1)^2}{n-1-Q_i} &gt; 0${{< /math >}}.
Therefore, {{< math >}}$h(x)${{< /math >}} is a monotonic function in its domain.

It indicates that {{< math >}}$\forall i, j${{< /math >}}, {{< math >}}$Q_i = Q_j = \frac{n-1}{n}${{< /math >}}, where {{< math >}}$P(v) = \frac{2n-1}{n-1}(1 - \frac{1}{n})^n${{< /math >}}.

Now we are going to list the boundary points for {{< math >}}$f${{< /math >}}.

**(1)** {{< math >}}$\forall i${{< /math >}}, {{< math >}}$Q_i = 1${{< /math >}}, it becomes the first extremum point.

**(2)** {{< math >}}$\exists i${{< /math >}}, {{< math >}}$Q_i${{< /math >}} is approaching to 0. Due to the constraint of {{< math >}}$\sum Q_i \geq n - 1${{< /math >}}, other {{< math >}}$Q${{< /math >}}s are approaching to 1. We have {{< math >}}$\lim_{Q_i \rightarrow 0} f = 0${{< /math >}} and {{< math >}}$P(v) = 1${{< /math >}}.

In conclusion, the maximum point of function {{< math >}}$f${{< /math >}} is {{< math >}}$\forall i${{< /math >}}, {{< math >}}$Q_i = \frac{n - 1}{n}${{< /math >}}, where

{{< math >}}
$$
P(v) = \frac{2n-1}{n-1}(1 - \frac{1}{n})^n, \tag{21}
$$
{{< /math >}}

and

{{< math >}}
$$
P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 1- P(v) = 1 - \frac{2n-1}{n-1}(1 - \frac{1}{n})^n. \tag{22}
$$
{{< /math >}}

**Discussion.** It is worth noting that {{< math >}}$\mathcal{G}_n${{< /math >}} is exactly the MEC of any graph in {{< math >}}$\mathcal{G}_n${{< /math >}}.
Hence, {{< math >}}$P_{G \sim M(Q)}(G \notin \mathcal{G}_n)${{< /math >}} represents the probability that the graph sampled from {{< math >}}$M(Q)${{< /math >}} is incorrect.
It indicates that a Node-Edge model could suffer from an inevitable error rate of 0.2642 though has been perfectly trained to predict {{< math >}}$M(Q)${{< /math >}}.

## A7. Experimental Settings

**Baselines.** To demonstrate the effectiveness and superiority of the proposed framework, several representative baselines from multiple categories are selected for comparison. The PC algorithm is a classic constraint-based causal discovery algorithm based on conditional independence tests, and the version with parallelized optimization is selected (<a href="#ref-17">Le et al., 2016</a>). GES, a classic score-based greedy equivalence search algorithm, is also included (<a href="#ref-3">Chickering, 2002</a>). For continuous optimization methods, we compare with NOTEARS (<a href="#ref-43">Zheng et al., 2018</a>), a representative gradient-based optimization method, and GOLEM (<a href="#ref-25">Ng et al., 2020</a>), regarded as a more efficient variant of NOTEARS. For neural network based optimization algorithms, we compare with DAG-GNN (<a href="#ref-40">Yu et al., 2019</a>), an optimization algorithm based on graph neural networks, and GRAN-DAG, a gradient-based algorithm using neural network modeling (<a href="#ref-15">Lachapelle et al., 2020</a>). For DNN-based SCL methods, we compare with AVICI, which is the most related work to ours and regarded as the current state-of-the-art method (<a href="#ref-21">Lorch et al., 2022</a>).

**Implementation Details.** The implementation from gCastle (<a href="#ref-42">Zhang et al., 2021</a>) is utilized for baselines except the SCL methods (i.e., SLdisco and AVICI). For PC algorithm, we employ the Fisher-Z transformation with a significance threshold of 0.05 for conditional independence tests, which is a prevalent choice in statistical analyses and current PC implementations (<a href="#ref-42">Zhang et al., 2021</a>; <a href="#ref-44">Zheng et al., 2024</a>). Our criterion for graph selection in GES experiments is the Gaussian Bayesian Information Criterion (BIC), specifically the {{< math >}}$l_\infty${{< /math >}}-penalized Gaussian likelihood score. It is used in the original paper (<a href="#ref-3">Chickering, 2002</a>), and remains a favored variant in the literature. For NOTEARS, adhering to the official implementation's settings, we configure NOTEARS with a maximum of 100 dual ascent steps, and an edge dropping threshold of 0.3. For hyperparameters lacking specific default settings, such as the L1 penalty and loss function type, we default to settings used by gCastle <a href="#ref-42">Zhang et al. (2021)</a>, employing an L1 penalty of 0.1 and an L2 loss function. For DAG-GNN, we utilize hyperparameter settings directly from the original implementation, ensuring consistency with established benchmarks. For GOLEM and GRAN-DAG, we also use the default setting of gCastle (<a href="#ref-42">Zhang et al., 2021</a>). Note that the CSIvA model (<a href="#ref-13">Ke et al., 2023</a>) is also a closely related method, but it is not compared due to the unavailability of its relevant codes and its requirement for interventional data as input. The original implementation of SLdisco <a href="#ref-28">Petersen et al. (2023)</a> was developed in R. To enhance compatibility with our data generation and evaluation workflows, we reimplemented the model using PyTorch. The original AVICI model (<a href="#ref-21">Lorch et al., 2022</a>) does not support discrete data. Therefore, we use an embedding layer to replace its first linear layer when using AVICI on discrete data.

**Synthetic Data.** We randomly generate random graphs from multiple random graph models. For continuous data, following previous work (<a href="#ref-21">Lorch et al., 2022</a>), Erdős-Rényi (ER) and Scale-free (SF) are utilized as the training graph distribution {{< math >}}$p(G)${{< /math >}}. The degree of training graphs in our experiments varies randomly among 1, 2, and 3. For testing graph distributions, Watts-Strogatz (WS) and Stochastic Block Model (SBM) are used, with parameters consistent with those in the previous paper (<a href="#ref-21">Lorch et al., 2022</a>). All synthetic graphs for continuous data contain 30 nodes. The lattice dimension of Watts-Strogatz (WS) graphs is sampled from {{< math >}}$\{2, 3\}${{< /math >}}, yielding an average degree of about 4.92. The average degrees of Stochastic Block Model (SBM) graphs are set at 2, following the settings in the aforementioned paper. For discrete data, 11-node graphs are used. SF is utilized as the training graph distribution {{< math >}}$p(G)${{< /math >}} and ER is used for testing. The synthetic training data is generated in real-time, and the training process does not use the same data repeatedly. All synthetic test datasets contain 100 graphs, and the average values of the metrics on the 100 graphs are reported to comprehensively reflect the performance.

For the forward sampling process from graph to continuous data, both the linear Gaussian mechanism and general nonlinear mechanism are applied. Concretely, the Random Fourier Function mechanism is used for the general nonlinear data following the previous paper (<a href="#ref-21">Lorch et al., 2022</a>).

In synthesizing discrete datasets, the Bernoulli distribution is used following previous papers (<a href="#ref-6">Dai et al., 2023</a>; <a href="#ref-22">Ma et al., 2022</a>).

{{< figgrid caption="**Figure A5**: Illustration of the architecture comparison of Node-Edge models, SiCL-no-PF and SiCL." >}}
paire/abl.png | 100
{{< /figgrid >}}


**More Implementation Details and Computational Resources.** The two network modules, i.e., the SPN and VPN, are optimized by Adam optimizer with default hyperparameters. Following previous work (<a href="#ref-21">Lorch et al., 2022</a>), the training batch size is set as 20. All classic algorithms are run on an AMD EPYC 7V13 CPU, and DNN-based methods are run on Nvidia 1080Ti, A40 and A100 GPUs. Training SiCL on a 30-node training set with batch size 20 needs about 60GB memory, and training on a 11-node training set needs about 20GB memory. The learning rate is {{< math >}}$3 \times 10^{-4}${{< /math >}}. The batch size is 15, and the DNN models are trained for {{< math >}}$1.2 \times 10^{5}${{< /math >}} batches by default.

**Table A5**: **General comparison of SiCL and other methods**. The average performance results in three runs are provided for SiCL method. GES takes more than 24 hours per graph on WS-L-G, and SLdicso is unsuitable on non-linear-Gaussian data, hence the results are not included.

| Dataset | Method | Skeleton Prediction s-F1 ↑ | Skeleton Prediction s-Acc. ↑ | Skeleton Prediction s-AUC ↑ | Skeleton Prediction s-AUPRC ↑ | CPDAG Prediction v-F1 ↑ | CPDAG Prediction o-F1 ↑ | CPDAG Prediction SHD ↓ |
|---|---|---|---|---|---|---|---|---|
| WS-L-G | PC | 30.4 | 65.6 | N/A | N/A | 15.6 | 16.0 | 170.4 |
| | NOTEARS | 33.3 | 65.1 | N/A | N/A | 27.9 | 31.5 | 159.8 |
| | DAG-GNN | 35.5 | 55.4 | N/A | N/A | 32.2 | 32.7 | 193.7 |
| | GRAN-DAG | 16.6 | 62.1 | N/A | N/A | 11.7 | 11.7 | 170.1 |
| | GOLEM | 30.0 | 63.4 | N/A | N/A | 15.8 | 19.3 | 172.7 |
| | SLdisco | 0.1 | 66.0 | 50.2 | 34.6 | 0.0 | 0.1 | 147.9 |
| | AVICI | 39.9 | 74.0 | 71.5 | 62.2 | 28.2 | 35.8 | 119.2 |
| | SiCL | **44.7** | **75.3** | **73.7** | **65.4** | **32.0** | **38.5** | **116.1** |
| SBM-L-G | PC | 58.8 | 90.0 | N/A | N/A | 34.8 | 35.9 | 56.4 |
| | GES | 70.8 | 89.4 | N/A | N/A | 53.9 | 55.0 | 60.3 |
| | NOTEARS | 80.1 | 94.5 | N/A | N/A | 76.2 | 77.8 | 26.7 |
| | DAG-GNN | 66.2 | 87.4 | N/A | N/A | 60.3 | 62.5 | 61.0 |
| | GRAN-GAG | 22.6 | 85.9 | N/A | N/A | 13.8 | 14.4 | 64.7 |
| | GOLEM | 68.5 | 88.5 | N/A | N/A | 63.5 | 65.2 | 55.1 |
| | SLdisco | 1.9 | 85.7 | 56.3 | 17.6 | 0.9 | 1.2 | 62.6 |
| | AVICI | 84.3 | 96.2 | 98.1 | 92.7 | 79.1 | 81.6 | 17.7 |
| | SiCL | **85.8** | **96.4** | **98.3** | **93.4** | **80.6** | **82.7** | **17.1** |
| WS-RFF-G | PC | 36.1 | 69.9 | N/A | N/A | 14.8 | 16.1 | 156.9 |
| | GES | 41.7 | 66.6 | N/A | N/A | 21.1 | 23.6 | 174.1 |
| | NOTEARS | 37.7 | 64.6 | N/A | N/A | 30.9 | 33.4 | 164.4 |
| | DAG-GNN | 33.2 | 65.4 | N/A | N/A | 27.0 | 28.9 | 161.1 |
| | GRAN-DAG | 4.7 | 66.7 | N/A | N/A | 0.8 | 1.1 | 146.9 |
| | GOLEM | 27.6 | 62.4 | N/A | N/A | 13.8 | 17.7 | 175.8 |
| | AVICI | 47.7 | 75.9 | 76.3 | 67.6 | 38.7 | 45.2 | 110.6 |
| | SiCL | **51.8** | **77.4** | **81.1** | **72.9** | **40.3** | **46.3** | **107.0** |
| SBM-RFF-G | PC | 57.5 | 89.3 | N/A | N/A | 32.7 | 34.2 | 60.9 |
| | GES | 56.5 | 84.9 | N/A | N/A | 37.0 | 38.0 | 82.4 |
| | NOTEARS | 55.6 | 86.2 | N/A | N/A | 46.5 | 48.5 | 66.3 |
| | DAG-GNN | 47.1 | 82.1 | N/A | N/A | 39.0 | 40.6 | 86.2 |
| | GRAN-DAG | 17.4 | 87.4 | N/A | N/A | 3.2 | 3.8 | 58.2 |
| | GOLEM | 31.1 | 75.7 | N/A | N/A | 23.0 | 24.8 | 112.0 |
| | AVICI | 76.6 | 94.5 | 95.4 | 85.7 | 69.3 | 72.7 | 27.2 |
| | SiCL | **82.1** | **95.7** | **97.1** | **90.7** | **75.7** | **78.0** | **21.9** |
| ER-CPT-MC | PC | 82.2 | 83.0 | N/A | N/A | 39.2 | 40.6 | 16.4 |
| | GES | 82.1 | 81.8 | N/A | N/A | 40.4 | 42.4 | 17.1 |
| | NOTEARS | 16.7 | 74.8 | N/A | N/A | 0.2 | 0.6 | 16.1 |
| | DAG-GNN | 24.8 | 73.5 | N/A | N/A | 3.4 | 3.7 | 15.9 |
| | GRAN-DAG | 40.8 | 77.0 | N/A | N/A | 6.8 | 7.3 | 15.6 |
| | GOLEM | 37.6 | 66.4 | N/A | N/A | 4.6 | 9.3 | 21.9 |
| | AVICI | 76.9 | 88.4 | 93.5 | 87.9 | 56.6 | 57.6 | 10.2 |
| | SiCL | **84.2** | **90.1** | **96.6** | **94.0** | **58.3** | **59.9** | **10.1** |

**Table A6**: Full ablation study results.

| Dataset | Method | s-F1 ↑ | s-Acc. ↑ | s-AUC ↑ | s-AUPRC ↑ | v-F1 ↑ | o-F1 ↑ | SHD ↓ |
|---|---|---|---|---|---|---|---|---|
| WS-L-G | SiCL-Node-Edge | 39.9 | 74.0 | 71.5 | 62.2 | 28.2 | 35.8 | 119.2 |
| | SiCL-no-PF | 42.4 | 74.4 | 72.8 | 63.5 | 30.5 | 37.9 | 118.4 |
| | SiCL | **44.7** | **75.3** | **73.7** | **65.4** | **32.0** | **38.5** | **116.1** |
| SBM-L-G | SiCL-Node-Edge | 84.3 | 96.2 | 98.1 | 92.7 | 79.1 | 81.6 | 17.7 |
| | SiCL-No-PF | 85.5 | **96.4** | **98.3** | 93.3 | 79.4 | 82.2 | 17.3 |
| | SiCL | **85.8** | **96.4** | **98.3** | **93.4** | **80.6** | **82.7** | **17.1** |

## A8. Extra Experimental Results

{{< figgrid caption="**Figure A6**: Variation trends of the test performance of the V-structure Prediction Network on WS-LG and SBM-LG during training." >}}
paire/v_struc_ws_font.png | 45 | **(a)** WS-LG
paire/v_struc_sbm_font.png | 45 | **(b)** SBM-LG
{{< /figgrid >}}


### A8.1. Effectiveness of V-structure Prediction Network

Fig. A6 illustrates the test performance trends of the v-structure prediction model on SBM and WS random graphs during the training process. In this model, the feature extractor {{< math >}}$FE${{< /math >}} is fine-tuned from the skeleton prediction model. The performance increases rapidly and achieves a relatively high level after just a few initial epochs. This suggests that our v-structure prediction network is capable to predict v-structures, and indicates that the pre-trained pairwise features from the skeleton prediction model are both effective and generalizable.

### A8.2. More Evidence on Effectiveness of Pairwise Representation

To further support the effectiveness of using pairwise representation, we present additional experimental results on different training datasets and test datasets, including ER-L-G, SF-L-G, ER-RFF-G, and SF-RFF-G.

For models, we compare SiCL with a variant without pairwise representation, i.e., SiCL-no-PF.

The results are provided in Tab. A7. Models with pairwise representation ourperform the corresponding baseline models under almost all comparisons, further verifying the effectiveness of using pairwise representation in models.

**Table A7**: More performance comparison on the effectiveness of pairwise representation.

| Training Dataset | Test Dataset | Method | s-F1↑ | s-AUC↑ | s-AUPRC↑ | s-Acc.↑ |
|---|---|---|---|---|---|---|
| ER-L-G | ER-L-G | SiCL-no-PF | 75.7 | 84.6 | 83.1 | 78.5 |
| ER-L-G | ER-L-G | SiCL | 80.2 | 89.6 | 90.1 | 82.3 |
| ER-L-G | SF-L-G | SiCL-no-PF | 74.9 | 92.5 | 87.3 | 84.1 |
| ER-L-G | SF-L-G | SiCL | 79.0 | 96.0 | 93.7 | 87.0 |
| ER-L-G | ER-RFF-G | SiCL-no-PF | 49.5 | 60.5 | 49.1 | 58.8 |
| ER-L-G | ER-RFF-G | SiCL | 51.0 | 67.0 | 57.6 | 65.2 |
| ER-L-G | SF-RFF-G | SiCL-no-PF | 40.4 | 57.9 | 38.9 | 57.5 |
| ER-L-G | SF-RFF-G | SiCL | 46.4 | 71.4 | 53.7 | 69.0 |
| SF-L-G | ER-L-G | SiCL-no-PF | 64.6 | 77.3 | 68.7 | 70.7 |
| SF-L-G | ER-L-G | SiCL | 68.0 | 82.1 | 76.4 | 74.3 |
| SF-L-G | SF-L-G | SiCL-no-PF | 88.5 | 96.7 | 95.0 | 91.2 |
| SF-L-G | SF-L-G | SiCL | 89.7 | 97.9 | 97.0 | 92.4 |
| SF-L-G | ER-RFF-G | SiCL-no-PF | 44.3 | 62.3 | 50.9 | 58.4 |
| SF-L-G | ER-RFF-G | SiCL | 47.0 | 66.2 | 55.8 | 63.3 |
| SF-L-G | SF-RFF-G | SiCL-no-PF | 48.1 | 71.6 | 53.9 | 65.8 |
| SF-L-G | SF-RFF-G | SiCL | 56.0 | 79.6 | 64.8 | 74.2 |
| ER-RFF-G | ER-L-G | SiCL-no-PF | 64.0 | 73.3 | 65.8 | 67.3 |
| ER-RFF-G | ER-L-G | SiCL | 72.0 | 82.0 | 81.1 | 75.2 |
| ER-RFF-G | SF-L-G | SiCL-no-PF | 58.1 | 79.0 | 66.8 | 72.8 |
| ER-RFF-G | SF-L-G | SiCL | 70.1 | 88.0 | 83.6 | 80.9 |
| ER-RFF-G | ER-RFF-G | SiCL-no-PF | 63.2 | 74.3 | 67.7 | 71.0 |
| ER-RFF-G | ER-RFF-G | SiCL | 74.8 | 85.7 | 84.5 | 79.7 |
| ER-RFF-G | SF-RFF-G | SiCL-no-PF | 56.3 | 78.3 | 65.9 | 75.0 |
| ER-RFF-G | SF-RFF-G | SiCL | 68.2 | 87.0 | 81.5 | 82.1 |
| SF-RFF-G | ER-L-G | SiCL-no-PF | 60.3 | 71.2 | 58.7 | 64.7 |
| SF-RFF-G | ER-L-G | SiCL | 65.6 | 78.0 | 72.5 | 70.5 |
| SF-RFF-G | SF-L-G | SiCL-no-PF | 73.6 | 90.5 | 82.9 | 81.0 |
| SF-RFF-G | SF-L-G | SiCL | 79.1 | 94.2 | 90.0 | 85.5 |
| SF-RFF-G | ER-RFF-G | SiCL-no-PF | 57.7 | 71.4 | 60.7 | 66.8 |
| SF-RFF-G | ER-RFF-G | SiCL | 67.2 | 80.5 | 75.8 | 73.9 |
| SF-RFF-G | SF-RFF-G | SiCL-no-PF | 74.8 | 90.2 | 82.4 | 83.5 |
| SF-RFF-G | SF-RFF-G | SiCL | 80.4 | 94.2 | 90.5 | 87.3 |

### A8.3. Additional Comparison on DAG Prediction

We provide an additional comparison with the AVICI baseline on the DAG prediction task. Since SiCL predicts CPDAGs and does not directly produce DAG predictions, we corrected the DAG predictions from AVICI using the edge directions inferred from the CPDAGs predicted by SiCL. The results, summarized in the Table A8, demonstrate that incorporating CPDAG-inferred edge directions improves the DAG prediction metrics. This further confirms the effectiveness and generality of our approach, even in tasks focused on DAG metrics.

**Table A8**: Additional Comparison on DAG Prediction

| Method | Dataset | F1 Score↑ | AUC↑ | AUPRC↑ | Acc.↑ |
|---|---|---|---|---|---|
| AVICI | WS-L-G | 38.4 | 86.3 | 57.7 | 85.9 |
| SiCL-Corrected AVICI | WS-L-G | 35.8 | 87.2 | 60.5 | 86.2 |
| AVICI | SBM-L-G | 78.1 | 95.8 | 80.5 | 97.3 |
| SiCL-Corrected AVICI | SBM-L-G | 81.3 | 98.7 | 90.8 | 97.8 |

### A8.4. Comparison with Autoregressive models on Inference Time Costs

To validate that the autoregressive models have a relatively high time costs due to the quadratic number of inference runs w.r.t. number of variables, we reproduce the network architecture of a representative autoregressive model, i.e., CSIvA (<a href="#ref-13">Ke et al., 2023</a>), and compare SiCL with it.
We use the same random input for both the models with increasing number of variables.
The results are provided in Fig. A7.
The time costs of the autoregressive model show a fast increasing trend and are much more than costs of SiCL, validating the correctness of our analysis.

{{< figgrid caption="**Figure A7**: Comparison between an autoregressive model and SiCL on inference time costs." >}}
paire/inference_time_costs_auto_font.png | 50
{{< /figgrid >}}


### A8.5. Training Data Diversity and Model Generalization

We present experimental evidence that highlights the significant contribution of training data diversity to the model's generalization capabilities, even when applied to out-of-distribution (OOD) datasets.
To illustrate this, we train one SiCL model on a combined dataset of both SF and ER, and another solely on the SF dataset.
The comparative performance of these models is detailed in Tab. A9.
The model trained on the combined ER and SF datasets exhibited markedly better performance, not only on the ER dataset but also on the other two OOD datasets, with only a marginal decrease in performance on the SF dataset.
These findings suggest that enhancing the diversity of the training data correspondingly improves the model’s ability to generalize and maintain robust performance across novel OOD datasets.

**Table A9**: Comparison of SiCL models with different training data diversity on skeleton prediction.

**(a)** Model trained on both ER and SF

| Test Dataset | s-F1↑ | s-AUC↑ | s-AUPRC↑ | s-Acc.↑ |
|---|---|---|---|---|
| WS-L-G | 36.3 | 70.6 | 60.6 | 73.3 |
| SBM-L-G | 78.1 | 96.8 | 88.1 | 94.8 |
| ER-L-G | 80.7 | 96.0 | 89.2 | 94.7 |
| SF-L-G | 84.7 | 98.5 | 93.6 | 95.5 |

**(b)** Model trained on SF

| Test Dataset | s-F1↑ | s-AUC↑ | s-AUPRC↑ | s-Acc.↑ |
|---|---|---|---|---|
| WS-L-G | 40.1 | 63.0 | 46.1 | 63.5 |
| SBM-L-G | 64.3 | 91.7 | 72.9 | 90.9 |
| ER-L-G | 67.1 | 90.4 | 73.9 | 90.8 |
| SF-L-G | 87.8 | 98.9 | 95.3 | 96.1 |

### A8.6. Varying Amount of Training Graphs

We present an analysis of how varying the amount of the training graphs influences performance on the skeleton prediction task. The results, depicted in Fig. A8, illustrate a clear trend: model performance improves in tandem with the expansion of the training dataset. This trend underscores the potential of our method to achieve even greater accuracy given a more extensive dataset.

**Figure A8**: Model performance with varying amount of training graphs.

{{< figgrid caption="**Figure A8**: Model performance with varying amount of training graphs." >}}
paire/wstrainingsize_font.png | 45 | **(a)** WS dataset
paire/sbmtrainingsize_font.png | 45 | **(b)** SBM dataset
{{< /figgrid >}}


### A8.7. Varying Sample Size

We assess SiCL across various quantities of observational samples per graph during testing (100, 200, ..., 1000). The outcomes for both the skeleton prediction task and the CPDAG prediction task are depicted in Fig. A9. It is evident that the model's performance enhances with the augmentation of sample size. These consistent upward trends suggest that SiCL exhibits stability and is not overly sensitive to changes in sample size.

**Figure A9**: Variation trends of performance with varying sample sizes.

{{< figgrid caption="**Figure A9**: Variation trends of performance with varying sample sizes." >}}
paire/ws1_font.png | 44 | **(a)** Variation trends of skeleton predicton task performance on WS graph with varying sample sizes.
paire/ws2_font.png | 44 | **(b)** Variation trends of CPDAG predicton task performance on WS graph with varying sample sizes.
paire/sbm1_font.png | 44 | **(c)** Variation trends of skeleton predicton task performance on SBM graph with varying sample sizes.
paire/sbm2_font.png | 44 | **(d)** Variation trends of CPDAG predicton task performance on SBM graph with varying sample sizes.
{{< /figgrid >}}


### A8.8. Varying Edge Density

We evaluate SiCL over a range of edge densities in the test graphs, utilizing the SBM dataset, as it allows for the direct setting of average edge densities. The findings are presented in Fig. A10. It's apparent that the task is becomes more difficult as edge densities increase. However, the performance decline is not abrupt, indicating that SiCL's performance remains relatively stable across various edge densities, thereby confirming its versatility.

**Figure A10**: Variation trends of performance with varying edge densities.

{{< figgrid caption="**Figure A10**: Variation trends of performance with varying edge densities." >}}
paire/sbmskeletondensity_font.png | 45 | **(a)** Variation trends of skeleton predicton task performance on SBM graph with varying edge densities.
paire/sbmcpdagdensity_font.png | 45 | **(b)** Variation trends of CPDAG predicton task performance on SBM graph with varying edge densities.
{{< /figgrid >}}


### A8.9. Generality on Testing Graph Sizes

We offer an analytical perspective on the performance of the SiCL model when applied to larger WS-L-G graphs. It is important to highlight that the models were initially trained on graphs comprising 30 vertices, positioning this task within an out-of-distribution setting in terms of graph size. To establish a point of reference, we have included results from the PC algorithm as a baseline comparison. These findings can be examined in Tab. A10. Despite the OOD conditions, SiCL maintains robust performance, reinforcing its scalability and the model's general applicability across varying graph sizes.

**Table A10**: Performance comparison with varying amounts of graph sizes.

| Metric / Size | s-F1↑ 50 | s-F1↑ 70 | s-F1↑ 100 | v-F1↑ 50 | v-F1↑ 70 | v-F1↑ 100 | o-F1↑ 50 | o-F1↑ 70 | o-F1↑ 100 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PC | 17.7 | 14.8 | 10.6 | 6.4 | 5.0 | 3.7 | 7.0 | 5.6 | 4.0 |
| SiCL | **41.6** | **37.4** | **28.3** | **34.9** | **30.7** | **22.6** | **37.9** | **33.7** | **24.8** |

### A8.10. Acyclicity

**Table A11**: Count of cycles in the CPDAG predictions without post-processing of removing cycles.

| Dataset | WS-L-G | SBM-L-G |
| --- | --- | --- |
| Rate of Graphs with Cycles | 0.66 ± 0.66 % | 0.00 ± 0.00 % |

We provide an empirical evidence supporting of the rarity of cycles in the predictions. The experimental data presented in Tab. A11 corroborates that cycles are infrequently observed in the predicted CPDAGs, even though without any post-processing on removing cycles.

## References

<span id="ref-1"></span>Andersson, S. A., Madigan, D., and Perlman, M. D. (1997). A Characterization of Markov Equivalence Classes for Acyclic Digraphs. *The Annals of Statistics*, 25(2):505–541.

<span id="ref-2"></span>Charpentier, B., Kibler, S., and Günnemann, S. (2022). Differentiable DAG Sampling. In *ICLR*.

<span id="ref-3"></span>Chickering, D. M. (2002). Optimal Structure Identification with Greedy Search. *Journal of Machine Learning Research*, 3(3):507–554.

<span id="ref-4"></span>Colombo, D., Maathuis, M. H., et al. (2014). Order-Independent Constraint-Based Causal Structure Learning. *Journal of Machine Learning Research*, 15(1):3741–3782.

<span id="ref-5"></span>Cussens, J. (2011). Bayesian Network Learning With Cutting Planes. In *UAI*.

<span id="ref-6"></span>Dai, H., Ding, R., Jiang, Y., Han, S., and Zhang, D. (2023). ML4C: Seeing Causality through Latent Vicinity. In *SDM*.

<span id="ref-7"></span>Frydenberg, M. (1990). The Chain Graph Markov Property. *Scandinavian Journal of Statistics*, 17(4):333–353.

<span id="ref-8"></span>Geffner, T., Antoran, J., Foster, A., Gong, W., Ma, C., Kiciman, E., Sharma, A., Lamb, A., Kukla, M., Pawlowski, N., Allamanis, M., and Zhang, C. (2022). Deep End-to-end Causal Inference. In *NeurIPS 2022 Workshop on Causality for Real-world Impact*.

<span id="ref-9"></span>Glymour, C., Zhang, K., and Spirtes, P. (2019). Review of Causal Discovery Methods Based on Graphical Models. *Frontiers in Genetics*, 10:524.

<span id="ref-10"></span>Hoyer, P., Janzing, D., Mooij, J. M., Peters, J., and Schölkopf, B. (2008). Nonlinear Causal Discovery with Additive Noise Models. In *NeurIPS*.

<span id="ref-11"></span>Immer, A., Schultheiss, C., Vogt, J. E., Schölkopf, B., Bühlmann, P., and Marx, A. (2023). On the Identifiability and Estimation of Causal Location-Scale Noise Models. In *ICML*.

<span id="ref-12"></span>Janzing, D., Mooij, J., Zhang, K., Lemeire, J., Zscheischler, J., Daniušis, P., Steudel, B., and Schölkopf, B. (2012). Information-Geometric Approach to Inferring Causal Directions. *Artificial Intelligence*, 182:1–31.

<span id="ref-13"></span>Ke, N. R., Chiappa, S., Wang, J. X., Bornschein, J., Goyal, A., Rey, M., Weber, T., Botvinick, M., Mozer, M. C., and Rezende, D. J. (2023). Learning to Induce Causal Structure. In *ICLR*.

<span id="ref-14"></span>Koller, D. and Friedman, N. (2009). *Probabilistic Graphical Models: Principles and Techniques*. MIT press.

<span id="ref-15"></span>Lachapelle, S., Brouillard, P., Deleu, T., and Lacoste-Julien, S. (2020). Gradient-Based Neural DAG Learning. In *ICLR*.

<span id="ref-16"></span>Lauritzen, S. L. (1996). *Graphical Models*, volume 17. Clarendon Press.

<span id="ref-17"></span>Le, T. D., Hoang, T., Li, J., Liu, L., Liu, H., and Hu, S. (2016). A Fast PC Algorithm for High Dimensional Causal Discovery With Multi-Core PCs. *IEEE/ACM Transactions on Computational Biology and Bioinformatics*, 16(5):1483–1495.

<span id="ref-18"></span>Lippe, P., Cohen, T., and Gavves, E. (2022). Efficient neural causal discovery without acyclicity constraints. In *ICLR*.

<span id="ref-19"></span>Lopez-Paz, D., Muandet, K., and Recht, B. (2015). The Randomized Causation Coefficient. *Journal of Machine Learning Research*, 16(90):2901–2907.

<span id="ref-20"></span>Lopez-Paz, D., Nishihara, R., Chintala, S., Scholkopf, B., and Bottou, L. (2017). Discovering Causal Signals in Images. In *CVPR*.

<span id="ref-21"></span>Lorch, L., Sussex, S., Rothfuss, J., Krause, A., and Schölkopf, B. (2022). Amortized Inference for Causal Structure Learning. In *NeurIPS*.

<span id="ref-22"></span>Ma, P., Ding, R., Dai, H., Jiang, Y., Wang, S., Han, S., and Zhang, D. (2022). ML4S: Learning Causal Skeleton from Vicinal Graphs. In *KDD*.

<span id="ref-23"></span>Meek, C. (1995a). Causal Inference and Causal Explanation with Background Knowledge. In *UAI*, pages 403–410.

<span id="ref-24"></span>Meek, C. (1995b). Strong Completeness and Faithfulness in Bayesian Networks. In *UAI*.

<span id="ref-25"></span>Ng, I., Ghassami, A., and Zhang, K. (2020). On the Role of Sparsity and DAG Constraints for Learning Linear DAGs. In *NeurIPS*, pages 17943–17954.

<span id="ref-26"></span>Pearl, J. (2009). *Causality*. Cambridge university press.

<span id="ref-27"></span>Peters, J., Mooij, J. M., Janzing, D., and Schölkopf, B. (2014). Causal Discovery With Continuous Additive Noise Models. *Journal of Machine Learning Research*, 15(1):2009–2053.

<span id="ref-28"></span>Petersen, A. H., Ramsey, J., Ekstrøm, C. T., and Spirtes, P. (2023). Causal Discovery for Observational Sciences Using Supervised Machine Learning. *Journal of Data Science*, 21(2):255–280.

<span id="ref-29"></span>Ramsey, J., Zhang, J., and Spirtes, P. L. (2012). Adjacency-Faithfulness and Conservative Causal Inference. *arXiv preprint arXiv:1206.6843*.

<span id="ref-30"></span>Sachs, K., Perez, O., Pe'er, D., Lauffenburger, D. A., and Nolan, G. P. (2005). Causal Protein-Signaling Networks Derived From Multiparameter Single-Cell Data. *Science*, 308(5721):523–529.

<span id="ref-31"></span>Shimizu, S., Inazumi, T., Sogawa, Y., Hyvarinen, A., Kawahara, Y., Washio, T., Hoyer, P. O., Bollen, K., and Hoyer, P. (2011). DirectLiNGAM: A Direct Method for Learning a Linear Non-Gaussian Structural Equation Model. *Journal of Machine Learning Research*, 12(33):1225–1248.

<span id="ref-32"></span>Spirtes, P., Glymour, C., and Scheines, R. (2001). *Causation, Prediction, and Search*. MIT press.

<span id="ref-33"></span>Spirtes, P., Glymour, C. N., Scheines, R., and Heckerman, D. (2000). *Causation, prediction, and search*. MIT press.

<span id="ref-34"></span>Tsagris, M. (2019). Bayesian Network Learning with the PC Algorithm: An Improved and Correct Variation. *Applied Artificial Intelligence*, 33(2):101–123.

<span id="ref-35"></span>Varambally, S., Ma, Y., and Yu, R. (2024). Discovering Mixtures of Structural Causal Models from Time Series Data. In *ICML*.

<span id="ref-36"></span>Verma, T. and Pearl, J. (1990). Equivalence and Synthesis of Causal Models. In *UAI*.

<span id="ref-37"></span>Vowels, M. J., Camgoz, N. C., and Bowden, R. (2022). D’Ya Like Dags? A Survey on Structure Learning and Causal Discovery. *ACM Computing Surveys*, 55(4):1–36.

<span id="ref-38"></span>Weinberger, N. (2018). Faithfulness, Coordination and Causal Coincidences. *Erkenntnis*, 83(2):113–133.

<span id="ref-39"></span>Yu, K., Li, J., and Liu, L. (2016). A Review on Algorithms for Constraint-based Causal Discovery. *arXiv preprint arXiv:1611.03977*.

<span id="ref-40"></span>Yu, Y., Chen, J., Gao, T., and Yu, M. (2019). DAG-GNN: DAG Structure Learning with Graph Neural Networks. In *ICML*.

<span id="ref-41"></span>Zhang, K. and Hyvärinen, A. (2009). On the Identifiability of the Post-Nonlinear Causal Model. In *UAI*.

<span id="ref-42"></span>Zhang, K., Zhu, S., Kalander, M., Ng, I., Ye, J., Chen, Z., and Pan, L. (2021). gCastle: A Python Toolbox for Causal Discovery. *arXiv preprint arXiv:2111.15155*.

<span id="ref-43"></span>Zheng, X., Aragam, B., Ravikumar, P. K., and Xing, E. P. (2018). Dags With No Tears: Continuous Optimization for Structure Learning. In *NeurIPS*.

<span id="ref-44"></span>Zheng, Y., Huang, B., Chen, W., Ramsey, J., Gong, M., Cai, R., Shimizu, S., Spirtes, P., and Zhang, K. (2024). Causal-learn: Causal discovery in python. *Journal of Machine Learning Research*, 25(60):1–8.

<span id="ref-45"></span>Zhu, S., Ng, I., and Chen, Z. (2020). Causal Discovery with Reinforcement Learning. In *ICLR*.

---

## About this page

This is a web transcription of the paper's full text, produced with **Claude Opus 5 + ultracode**. The prose follows the original word for word; equations, tables, figures and numbers are reproduced as published. **It is provided for reference only, with no guarantee of accuracy — the original PDF is authoritative.** Where this page and the PDF disagree, the PDF is correct.

- Original PDF: [https://arxiv.org/pdf/2502.10883](https://arxiv.org/pdf/2502.10883)
- Paper page: [Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning]({{< relref "/publication/paire" >}})
- Chinese translation: [学习可识别的结构有助于避免基于深度神经网络的有监督因果学习中的偏差]({{< relref "/publication/paire-cn" >}})
- Code: [https://github.com/microsoft/reliableAI/tree/main/causal-kit/SiCL](https://github.com/microsoft/reliableAI/tree/main/causal-kit/SiCL)
