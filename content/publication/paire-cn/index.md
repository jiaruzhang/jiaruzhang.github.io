---
title: "学习可识别的结构有助于避免基于深度神经网络的有监督因果学习中的偏差"
subtitle: "Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning"

summary: "学习可识别的结构有助于避免基于深度神经网络的有监督因果学习中的偏差（AISTATS 2025）论文中文全文翻译，含全部公式、表格与插图。"

date: '2025-01-18T00:00:00Z'
publishDate: '2025-01-18T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/paire/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/paire/cn/'

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
本页是论文 **《Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning》**（AISTATS 2025）的中文全文翻译，正文、公式、表格与插图均与原文一一对应。
译文仅供参考，如需引用请以[英文原文 PDF](https://arxiv.org/pdf/2502.10883) 为准（[论文主页]({{< relref "/publication/paire" >}})）。也可查看[英文全文网页版]({{< relref "/publication/paire-en" >}})。
{{% /callout %}}

{{< toc title="目录" >}}

Jiaru Zhang, Rui Ding, Qiang Fu, Huang Bojun, Zizhen Deng, Yang Hua, Haibing Guan, Shi Han, Dongmei Zhang

> Jiaru Zhang（jiaruzhang@sjtu.edu.cn），Shanghai Jiao Tong University。该工作在其于微软亚洲研究院实习期间完成。
> Rui Ding（juding@microsoft.com），Microsoft。通讯作者。
> Qiang Fu（qifu@microsoft.com），Microsoft。
> Huang Bojun（bojhuang@gmail.com），Sony Research。
> Zizhen Deng（dengzizhen557@outlook.com），Peking University。
> Yang Hua（y.hua@qub.ac.uk），Queen’s University Belfast。
> Haibing Guan（hbguan@sjtu.edu.cn），Shanghai Jiao Tong University。
> Shi Han（shihan@microsoft.com），Microsoft。
> Dongmei Zhang（dongmeiz@microsoft.com），Microsoft。

## 摘要

因果发现是一项结构化预测任务，其目标是根据变量的数据样本预测变量之间的因果关系。监督式因果学习（Supervised Causal Learning, SCL）是该领域中一个新兴的范式。现有的基于深度神经网络（Deep Neural Network, DNN）的方法普遍采用「节点-边」方法：模型首先为每个变量节点计算一个嵌入向量，然后利用这些变量级表示，并行且独立地对每一条有向因果边进行预测。本文首先表明，这种架构存在某种系统性偏差，且无论模型规模和数据规模如何都无法缓解。随后我们提出监督式可识别因果学习（Supervised identifiable Causal Learning, SiCL），这是一种基于 DNN 的 SCL 方法，它预测一个骨架矩阵以及一个 v-张量（v-tensor，一个表示 v-结构的三阶张量）。根据马尔可夫等价类（Markov Equivalence Class, MEC）理论，在标准 MEC 设定下，骨架与 v-结构都是*可识别的*因果结构，因此关于骨架和 v-结构的预测不会受到因果发现中可识别性限制的影响；由此 SiCL 得以避免「节点-边」架构中的系统性偏差，并使因果发现的相合估计量成为可能。此外，SiCL 还配备了一个专门设计的成对编码器模块，其中包含一个单向注意力层，用以同时建模节点对的内部关系与外部关系。在合成基准与真实世界基准上的实验结果表明，SiCL 显著优于其他基于 DNN 的 SCL 方法。

## 1. 引言

因果发现旨在从观测数据样本中推断因果结构。
监督式因果学习（Supervised Causal Learning, SCL）(<a href="#ref-6">Dai et al., 2023</a>; <a href="#ref-13">Ke et al., 2023</a>; <a href="#ref-22">Ma et al., 2022</a>) 是该领域中一个新兴的范式。其基本思想是把因果发现视为一项*结构化预测*任务，并利用监督学习技术训练一个预测模型。
在训练阶段，会生成一个训练数据集，其中包含各种各样的因果机制及其对应的数据样本。
随后训练该预测模型，使其以这样的数据样本作为输入，并输出关于该数据样本背后因果机制的预测。

与传统的基于规则的方法或无监督方法 (<a href="#ref-9">Glymour et al., 2019</a>) 相比，SCL 方法展现出了很强的实证性能 (<a href="#ref-6">Dai et al., 2023</a>; <a href="#ref-22">Ma et al., 2022</a>)，以及对样本量与分布偏移的鲁棒性 (<a href="#ref-13">Ke et al., 2023</a>; <a href="#ref-21">Lorch et al., 2022</a>)。

基于深度神经网络（Deep Neural Network, DNN）的 SCL 采用 DNN 作为预测模型。它支持端到端训练，从而无需人工特征工程。此外，它能够有效处理连续型与离散型两种数据类型，并且可以学习潜在表示。
由 <a href="#ref-21">Lorch et al. (2022)</a>首次提出的一种特定 DNN 架构，在近期基于 DNN 的 SCL 工作中尤为流行。该模型首先把给定的数据样本转换成一组节点级特征向量，每个向量表示一个单独的变量（对应于相关因果图中的一个节点）。
基于这些节点级特征，模型随后输出一个加权邻接矩阵 {{< math >}}$A${{< /math >}}，其中 {{< math >}}$A_{ij}\in[0,1]${{< /math >}} 是有向边 {{< math >}}$i \rightarrow j${{< /math >}} 的估计概率（意味着 {{< math >}}$i${{< /math >}} 是 {{< math >}}$j${{< /math >}} 的直接原因）。

最后，所推断的因果图 {{< math >}}$G${{< /math >}} 的邻接矩阵是通过对 {{< math >}}$A${{< /math >}} 进行伯努利采样得到的，其中每个元素 {{< math >}}$G_{ij} \in \{0,1\}${{< /math >}} 都按照概率 {{< math >}}$A_{ij}${{< /math >}} *独立地*采样。
为方便起见，我们把这样的模型架构称为「节点-边」架构，因为其表示是针对单个节点学习的，而概率是针对单条有向边估计并采样的。

尽管「节点-边」方法广受欢迎并取得了令人鼓舞的结果 (<a href="#ref-21">Lorch et al., 2022</a>; <a href="#ref-45">Zhu et al., 2020</a>; <a href="#ref-2">Charpentier et al., 2022</a>; <a href="#ref-35">Varambally et al., 2024</a>)，我们仍指出它存在两点局限：

第一，「节点-边」架构在所推断的因果关系中引入了一种根本性偏差。具体而言，给定一个观测数据样本 {{< math >}}$D${{< /math >}}，一条有向因果边 {{< math >}}$i \rightarrow j${{< /math >}} 的存在性可能*必然*依赖于其他边的存在性。但现有的「节点-边」模型对每条边分别独立地进行预测，因此这类模型给出的概率预测 {{< math >}}$A_{ij}${{< /math >}} 只以输入样本 {{< math >}}$D${{< /math >}} 为条件，而不以 {{< math >}}$A${{< /math >}} 中其他元素的采样结果为条件，从而未能在其概率估计中捕捉至关重要的边间依赖。

举一个简单的例子：一个同时保留 {{< math >}}$G_1: X\rightarrow T \rightarrow Y${{< /math >}} 与 {{< math >}}$G_2: X\leftarrow T \leftarrow Y${{< /math >}} 两种可能性的「节点-边」模型，必然会以非零的概率输出边 {{< math >}}$X\rightarrow T${{< /math >}} 和 {{< math >}}$T \leftarrow Y${{< /math >}}，因而无法排除 {{< math >}}$G_3: X\rightarrow T \leftarrow Y${{< /math >}} 的可能性；然而对于与 {{< math >}}$G_1${{< /math >}} 和 {{< math >}}$G_2${{< /math >}} 相容的数据样本 {{< math >}}$D${{< /math >}} 而言，{{< math >}}$G_3${{< /math >}} 不可能是其背后的真实因果图 (<a href="#ref-36">Verma and Pearl, 1990</a>)。
至关重要的是，在一般情况下，仅凭观测数据无法区分 {{< math >}}$G_1${{< /math >}} 与 {{< math >}}$G_2${{< /math >}} (<a href="#ref-1">Andersson et al., 1997</a>; <a href="#ref-24">Meek, 1995b</a>)。
这意味着，任何「节点-边」模型若要保持可靠性，就必须（在观测到与二者中任一相容的数据样本时）同时保留 {{< math >}}$G_1${{< /math >}} 与 {{< math >}}$G_2${{< /math >}} 的可能性；而另一方面，这又导致它以不可避免的错误概率输出不可能出现的图 {{< math >}}$G_3${{< /math >}}。

第二，「节点-边」架构没有显式地表示关于节点对的特征，而我们认为这类特征对于基于观测数据的因果发现至关重要。
例如，只有当节点对 {{< math >}}$\langle X, Y\rangle${{< /math >}} 表现出*持久依赖性* (<a href="#ref-22">Ma et al., 2022</a>; <a href="#ref-33">Spirtes et al., 2000</a>) 时，因果边 {{< math >}}$X\rightarrow Y${{< /math >}} 才可能存在；所谓持久依赖性，是指无论以其他变量的哪个子集为条件，{{< math >}}$X${{< /math >}} 与 {{< math >}}$Y${{< /math >}} 都保持统计依赖。再举一例，对于因果有向无环图（Directed Acyclic Graph, DAG），要确定一个持久依赖的节点对 {{< math >}}$\langle X, Y\rangle${{< /math >}} 之间的因果方向，一个充分条件是 {{< math >}}$X${{< /math >}} 与 {{< math >}}$Y${{< /math >}} 表现出*定向不对称性*，即存在第三个变量 {{< math >}}$Z${{< /math >}}，使得 {{< math >}}$X${{< /math >}} 与 {{< math >}}$Z${{< /math >}} 持久依赖，而 {{< math >}}$Y${{< /math >}} 在以某个变量集 {{< math >}}$\mathbf{S}\not\ni X${{< /math >}} 为条件时可以与 {{< math >}}$Z${{< /math >}} 独立（或反之亦然）。像持久依赖性或定向不对称性这样的特征，本质上是一个节点对的集体属性，而不属于任何单个节点自身。

为了解决这些局限，本文提出一种新颖的基于 DNN 的 SCL 方法，称为监督式可识别因果学习（Supervised Identifiable Causal Learning, SiCL）。
SiCL 中的神经网络并不试图预测有向边的概率，而是试图预测一个骨架矩阵以及一个 v-张量（v-tensor，一个表示 v-结构的三阶张量）。
根据因果发现的马尔可夫等价类（Markov Equivalence Class, MEC）理论，在标准 MEC 设定下，骨架与 v-结构都是*可识别的*因果结构（而有向边不是），因此关于骨架和 v-结构的预测不会受到（不）可识别性限制的影响。
借助这一洞见，我们受理论启发的 DNN 架构完全避免了前文所讨论的边预测模型中的系统性偏差，并使因果发现能够使用*相合的*神经估计器（注：回顾一下，如果一个统计估计量在给定无限数据时收敛到真实值，则称该估计量是*相合的*）。
此外，SiCL 还配备了一个专门设计的成对编码器模块，其中包含一个单向注意力层。
由于该层同时以节点特征和节点对特征作为输入，它能够建模节点对的内部关系与外部关系。

在合成基准与真实世界基准上的实验结果表明，SiCL 能够有效解决上述两点局限，所得到的 SiCL 方案显著优于其他基于 DNN 的 SCL 方法：在真实世界的 Sachs 数据上，就 SHD（Structural Hamming Distance，结构汉明距离）而言取得了超过 50% 的性能提升。代码已公开发布于 <https://github.com/microsoft/reliableAI/tree/main/causal-kit/SiCL>。

## 2. 背景与相关工作

因果图模型由定义在多个随机变量上的联合概率分布 {{< math >}}$P${{< /math >}} 和一个 DAG {{< math >}}$G${{< /math >}} 共同确定。{{< math >}}$G${{< /math >}} 中的每个节点 {{< math >}}$X_i${{< /math >}} 表示 {{< math >}}$P${{< /math >}} 中的一个变量，而有向边 {{< math >}}$X_i \rightarrow X_j${{< /math >}} 表示从 {{< math >}}$X_i${{< /math >}} 到 {{< math >}}$X_j${{< /math >}} 的直接因果关系。因果发现任务通常要求从 {{< math >}}$P${{< /math >}} 的一个独立同分布（i.i.d.）样本出发，对 {{< math >}}$G${{< /math >}} 进行推断。

然而，因果发现存在一个众所周知的可识别性限制。一般而言，因果 DAG 至多只能识别到一个等价类。在标准假设设定下对这一可识别性限制的研究，催生了业已确立的 MEC 理论 (<a href="#ref-7">Frydenberg, 1990</a>; <a href="#ref-36">Verma and Pearl, 1990</a>)。如果某个因果特征的取值在标准 MEC 假设设定下于等价类内部保持不变，我们就称该特征是 *MEC-可识别的*。已知这类 MEC-可识别特征包括骨架和 v-结构集合，下面我们对二者作简要介绍。

定义在数据分布 {{< math >}}$P${{< /math >}} 上的*骨架* {{< math >}}$E${{< /math >}} 是这样一个无向图：{{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 之间存在一条边，当且仅当 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 在 {{< math >}}$P${{< /math >}} 中始终依赖，即 {{< math >}}$\forall Z \subseteq\left\{X_1, X_2, \cdots, X_d\right\} \backslash \left\{X_i, X_j \right\}${{< /math >}}，都有 {{< math >}}$X_i \nperp X_j | Z${{< /math >}}。在温和的假设下（例如 {{< math >}}$P${{< /math >}} 对 DAG {{< math >}}$G${{< /math >}} 满足马尔可夫性且相对于 {{< math >}}$G${{< /math >}} 是忠实的；详见附录第 A1.1 节），骨架与 DAG {{< math >}}$G${{< /math >}} 所对应的无向图相同 (<a href="#ref-33">Spirtes et al., 2000</a>)。

如果在（{{< math >}}$G${{< /math >}} 的骨架）中 {{< math >}}$X${{< /math >}} 与 {{< math >}}$Y${{< /math >}} 都与 {{< math >}}$T${{< /math >}} 相邻但彼此不相邻，则变量三元组 {{< math >}}$\langle X, T, Y \rangle${{< /math >}} 称为一个*无屏蔽三元组*（Unshielded Triple, UT）。若这两条边的方向分别由 {{< math >}}$X${{< /math >}} 和 {{< math >}}$Y${{< /math >}} 指向 {{< math >}}$T${{< /math >}}，它就成为一个 *v-结构*，记作 {{< math >}}$X \rightarrow T \leftarrow Y${{< /math >}}。

两个图是马尔可夫等价的，当且仅当它们具有相同的骨架和 v-结构。*马尔可夫等价类*（Markov Equivalence Class, MEC）可以用一个同时包含有向边和无向边的*完全部分有向无环图*（Completed Partially Directed Acyclic Graph, CPDAG）来表示。我们用 {{< math >}}$CPDAG(G)${{< /math >}} 表示由 {{< math >}}$G${{< /math >}} 导出的 CPDAG。

根据马尔可夫完备性定理 (<a href="#ref-24">Meek, 1995b</a>)，除非引入额外的假设（见下方注记），我们只能把因果图识别到其 MEC，即 CPDAG。这意味着 {{< math >}}$CPDAG(G)${{< /math >}} 中每一条（无）有向边都指示着一个（不）可识别的因果关系。

**注记：** 基于 MEC 的可识别性理论适用于一般情形设定，即在考虑所有可能的分布 {{< math >}}$P${{< /math >}} 的情况下成立。已知*如果*我们假设数据服从某些特殊的分布类，例如线性非高斯分布、加性噪声模型、后非线性模型或位置-尺度模型 (<a href="#ref-27">Peters et al., 2014</a>; <a href="#ref-31">Shimizu et al., 2011</a>; <a href="#ref-41">Zhang and Hyvärinen, 2009</a>; <a href="#ref-11">Immer et al., 2023</a>)，这一可识别性限制是可以被打破的（即 CPDAG 中的一条无向边可以被定向）。这些假设在实践中有时难以验证，因此本文考虑的是一般情形设定。关于可识别性与因果假设的更多讨论见附录第 A2 节。

### 2.1. 相关工作

在传统的因果发现方法中，与我们工作最相关的是基于约束的方法。它们的目标是找出与变量间条件独立性约束相一致的 DAG。这类方法先识别骨架，然后基于 v-结构的识别进行定向 (<a href="#ref-39">Yu et al., 2016</a>)。其输出是一个表示 MEC 的 CPDAG。该类别中的代表性算法包括 PC (<a href="#ref-33">Spirtes et al., 2000</a>)，以及诸如 Conservative-PC (<a href="#ref-29">Ramsey et al., 2012</a>)、PC-stable (<a href="#ref-4">Colombo et al., 2014</a>) 和 Parallel-PC (<a href="#ref-17">Le et al., 2016</a>) 等各种变体。与基于约束的方法相比，我们的方法和它们都建立在用于估计骨架与 v-结构的 MEC 理论原理之上。然而，传统方法依赖于基于显式约束的符号推理，而我们则采用 DNN 来捕捉与这些约束紧密关联的关键因果信息。

基于评分的方法旨在在组合约束下，依据预先定义的评分函数寻找最优的 DAG。这类方法采用特定的优化流程，例如前向-后向搜索的 GES (<a href="#ref-3">Chickering, 2002</a>)、爬山法 (<a href="#ref-14">Koller and Friedman, 2009</a>) 以及整数规划 (<a href="#ref-5">Cussens, 2011</a>)。连续优化方法则把离散的搜索过程转化为一个连续等式约束。NOTEARS (<a href="#ref-43">Zheng et al., 2018</a>) 将无环性约束表述为连续等式约束，并被 DAG-GNN (<a href="#ref-40">Yu et al., 2019</a>)、DECI (<a href="#ref-8">Geffner et al., 2022</a>) 进一步扩展以支持非线性因果关系。DECI (<a href="#ref-8">Geffner et al., 2022</a>) 是一种基于流的模型，可以在非线性加性噪声数据上同时完成因果发现与推理。

最近，ENCO (<a href="#ref-18">Lippe et al., 2022</a>) 作为一种连续优化方法被提出，其中边的定向被建模为一个单独的参数以维持无环性。如果能够获得对所有变量的干预，它可以保证收敛到正确的图。RL-BIC (<a href="#ref-45">Zhu et al., 2020</a>) 利用强化学习来搜索最优的 DAG。这些方法都可以视为无监督的，因为它们并不使用带有真实因果关系的额外数据集。关于这一文献脉络的全面梳理，我们建议参阅 <a href="#ref-9">Glymour et al. (2019)</a>; <a href="#ref-37">Vowels et al. (2022)</a>。

SCL 起步于在函数因果模型的形式化框架下对二变量情形中的边进行定向。RCC (<a href="#ref-19">Lopez-Paz et al., 2015</a>) 和 NCC (<a href="#ref-20">Lopez-Paz et al., 2017</a>) 等方法已经优于 ANM (<a href="#ref-10">Hoyer et al., 2008</a>) 或 IGCI (<a href="#ref-12">Janzing et al., 2012</a>) 之类的无监督方法。对于多变量情形，ML4S (<a href="#ref-22">Ma et al., 2022</a>) 提出了一种专门针对骨架学习的监督式方法。作为对 ML4S 的补充，ML4C (<a href="#ref-6">Dai et al., 2023</a>) 以数据和骨架二者作为输入，把无屏蔽三元组分类为 v-结构或非 v-结构。<a href="#ref-28">Petersen et al. (2023)</a>提出了 SLdisco 方法，利用 SCL 途径来解决 PC 与 GES 的一些局限。

基于 DNN 的 SCL 已成为实现端到端因果学习的一种重要方法。这一方向上的两项代表性工作，即 AVICI (<a href="#ref-21">Lorch et al., 2022</a>) 和 CSIvA (<a href="#ref-13">Ke et al., 2023</a>)，引入了交替注意力机制，以实现跨样本与跨变量的置换不变性。这两种方法都为每个节点学习一个单独的表示，并用它来预测有向边。其中，AVICI 考虑的是从观测数据预测 DAG 的任务，且恰恰采用了「节点-边」架构，因而受到第 1 节所讨论问题的困扰。另一方面，CSIvA 需要额外的干预数据作为输入才能识别完整的 DAG，并采用了一种自回归的 DNN 架构，其中的边通过多次推理运行被逐条顺序预测。因此，这种自回归方法由于所需的模型运行次数是平方级数量的（关于所考虑的变量数量），会带来非常高的推理开销，我们在附录第 A8.4 节中通过实验对此进行了验证。相比之下，本文提出的方法只需要运行一次 DNN 模型。除此之外，我们的方法在成对嵌入向量的使用上也不同于 AVICI 和 CSIvA。

## 3. 「节点-边」架构的局限性

「节点-边」架构十分常见，文献中已广泛采用它来生成输出的 DAG {{< math >}}$G${{< /math >}} (<a href="#ref-21">Lorch et al., 2022</a>; <a href="#ref-45">Zhu et al., 2020</a>; <a href="#ref-2">Charpentier et al., 2022</a>; <a href="#ref-35">Varambally et al., 2024</a>)。

在该架构中，DAG 中的每个元素 {{< math >}}$G_{ij}${{< /math >}} 都是从邻接矩阵 {{< math >}}$A${{< /math >}} 的对应元素 {{< math >}}$A_{ij}${{< /math >}} 独立采样得到的。该元素 {{< math >}}$A_{ij}${{< /math >}} 表示 {{< math >}}$i${{< /math >}} 是 {{< math >}}$j${{< /math >}} 的直接原因的概率。
我们引入一个仅含三个变量 {{< math >}}$X${{< /math >}}、{{< math >}}$Y${{< /math >}} 和 {{< math >}}$T${{< /math >}} 的简单而有效的示例设定，以揭示其局限性。

考虑一个以相等概率从两个因果模型生成 DAG 的模拟器：在模型 1 中，因果图为 {{< math >}}$G_1: X \rightarrow T \rightarrow Y${{< /math >}}，各变量满足 {{< math >}}$X \sim \mathcal{N} (0, 1)${{< /math >}}、{{< math >}}$T = X + \mathcal{N}(0, 1)${{< /math >}}、{{< math >}}$Y = T + \mathcal{N}(0, 1)${{< /math >}}。
在模型 2 中，因果图为 {{< math >}}$G_2: X \leftarrow T \leftarrow Y${{< /math >}}，各变量满足 {{< math >}}$ Y = \mathcal{N}(0, 3)${{< /math >}}、{{< math >}}$T = \frac{2}{3}Y + \mathcal{N}(0, \frac{2}{3})${{< /math >}}、{{< math >}}$X = 0.5T + \mathcal{N}(0, 0.5)${{< /math >}}。
在这种情况下，来自这两个因果模型的数据样本服从相同的联合分布，这使得 {{< math >}}$G_1${{< /math >}} 与 {{< math >}}$G_2${{< /math >}} 在本质上无法区分（仅凭观测数据样本）。

更重要的是，当把完全定向的因果 DAG 用作学习目标时（正如「节点-边」方法所做的那样），一个经过最优训练的神经网络会对 {{< math >}}$X - T${{< /math >}} 和 {{< math >}}$T - Y${{< /math >}} 这两条边的方向都预测出 0.5 的概率。
其结果是，图采样的输出会以 0.25 的概率成为 {{< math >}}$X \rightarrow T \leftarrow Y${{< /math >}}（见附录中的图 A4）。

这一错误概率的根源在于：对边 {{< math >}}$X \rightarrow T${{< /math >}} 的伯努利采样并不以边 {{< math >}}$T \leftarrow Y${{< /math >}} 的采样结果为条件。因此，即使 DNN 已经完美地建模了给定输入数据下每条边的*边缘概率*（对其他边求边缘），这一偏差依然无法避免。

我们进一步发现，0.25 还不是最坏情况下的错误率。
形式化地，对于图集合上的一个分布 {{< math >}}$Q${{< /math >}}，我们将各条边从边缘分布中独立采样所得到的图分布记为 {{< math >}}$M(Q)${{< /math >}}，即对任意因果边 {{< math >}}$e_1${{< /math >}} 与 {{< math >}}$e_2${{< /math >}}，有 {{< math >}}$P_{G\sim Q}(e_1 \in G) = P_{G\sim M(Q)}(e_1 \in G) = P_{G\sim M(Q)}(e_1 \in G | e_2 \in G)${{< /math >}}。
一般而言，在来自分布 {{< math >}}$Q${{< /math >}} 的数据样本 {{< math >}}$D${{< /math >}} 上经过最优训练的「节点-边」模型，本质上学到的是预测 {{< math >}}$M(Q)${{< /math >}}（在测试时给定相同的数据样本 {{< math >}}$D${{< /math >}} 的情况下）。
下面的命题表明，对于具有星形骨架的因果图，从边缘分布 {{< math >}}$M(Q)${{< /math >}} 中采样得到的图有 26.42% 的概率是错误的。

**命题 3.1**：设 {{< math >}}$\mathcal{G}_n${{< /math >}} 为具有 {{< math >}}$n+1${{< /math >}} 个节点的图的集合，其中存在一个中心节点 {{< math >}}$y${{< /math >}}，满足：(1) 其余每个节点都与 {{< math >}}$y${{< /math >}} 相连，(2) 其余节点之间没有边，(3) 至多有一条边指向 {{< math >}}$y${{< /math >}}。
我们有

{{< math >}}
$$
\sup_n \max_{Q} P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 
1 - \frac{2}{e} \approx 0.2642. \tag{1}
$$
{{< /math >}}

证明见附录第 A6 节。

这表明，一个预测边的神经网络即使被完美训练，也可能受到 0.2642 这一不可避免的错误率的困扰。

相比之下，预测骨架与 v-结构的模型在标准假设下具有相合性的理论渐近保证。
详细内容、证明及相关讨论见附录第 A1 节。

## 4. SiCL 方法

鉴于前文讨论的种种局限，我们在本节提出一种新的基于 DNN 的 SCL 方法，命名为 **SiCL**（**S**upervised **i**dentifiable **C**ausal **L**earning，监督式可识别因果学习）。

### 4.1. 整体工作流

{{< figgrid caption="**图 1**：SiCL 的推理流程。" >}}
paire/new_workflow.png | 100
{{< /figgrid >}}


遵循标准的基于 DNN 的 SCL 范式，核心推理过程由一个 DNN 实现。该 DNN 以一个矩阵 {{< math >}}$D\in\mathbb{R}^{n \times d}${{< /math >}} 所编码的数据样本作为输入，其中 {{< math >}}$d${{< /math >}} 为可观测变量的数量，{{< math >}}$n${{< /math >}} 为观测的条数。与以往的「节点-边」方法不同，SiCL 方法并不用 DNN 直接预测因果图，而是转而预测因果图的骨架与 v-结构，二者正是前文所述的 *MEC-可识别的*因果结构。

具体而言，我们的 DNN 输出两个对象：（1）一个骨架预测矩阵 {{< math >}}$S \in [0,1]^{d \times d}${{< /math >}}，其中 {{< math >}}$S_{ij}${{< /math >}} 刻画了在给定输入数据样本 {{< math >}}$D${{< /math >}} 的条件下，*无向*边 {{< math >}}$X_i - X_j${{< /math >}} 存在的条件概率；（2）一个 v-结构预测张量 {{< math >}}$V \in [0,1]^{d\times d\times d}${{< /math >}}，其中 {{< math >}}$V_{ijk}${{< /math >}} 刻画了同样在给定 {{< math >}}$D${{< /math >}} 的条件下，v-结构分量 {{< math >}}$X_j \rightarrow X_i \leftarrow X_k${{< /math >}} 存在的条件概率。在我们的实现中，{{< math >}}$S${{< /math >}} 与 {{< math >}}$V${{< /math >}} 分别由两个独立的子网络生成，它们被称为骨架预测网络（Skeleton Predictor Network, SPN）与 v-结构预测网络（V-structure Predictor Network, VPN）。

为进一步解决「节点-边」模型只具备节点级特征这一局限，我们提出为 SPN 和 VPN 配备成对编码器模块，以显式地捕捉节点对级特征。

基于骨架预测矩阵 {{< math >}}$S${{< /math >}} 与 v-结构预测张量 {{< math >}}$V${{< /math >}}，我们推断出因果图的骨架与 v-结构；由这二者即可确定唯一的 CPDAG。该 CPDAG 编码了一个马尔可夫等价类，我们可以从中挑选一个图实例作为对因果 DAG 的预测（如有需要）。

图 1 给出了 SiCL 整体推理工作流的示意图，该工作流的伪代码见附录中的算法 A1。

SPN 与 VPN 的参数按照标准的监督学习流程训练。在我们的实现中，我们只使用合成训练数据，这类数据相对容易获得，却往往能在真实世界的任务负载上带来很强的性能(<a href="#ref-13">Ke et al., 2023</a>)。

下文中，我们将详细阐述 SiCL 方法所采用的 DNN 架构、学习目标以及后处理流程。

### 4.2. 特征提取

**输入处理与节点特征编码器。** 给定输入数据样本，即一个矩阵 {{< math >}}$D\in\mathbb{R}^{n \times d}${{< /math >}}，输入处理模块对连续型输入数据包含一个线性层，对离散型输入数据则包含一个嵌入层，从而得到每条观测 {{< math >}}$l${{< /math >}} 中每个节点 {{< math >}}$i${{< /math >}} 的原始节点特征 {{< math >}}$\mathcal{F}^{raw}_{il}${{< /math >}}。

在此之后，我们使用一个节点特征编码器将原始节点特征进一步处理为最终的节点特征 {{< math >}}$\mathcal{F}_{il}${{< /math >}}。与以往的论文(<a href="#ref-13">Ke et al., 2023</a>; <a href="#ref-21">Lorch et al., 2022</a>)类似，该节点特征编码器是一个类 Transformer 网络，由交替作用于观测维度和节点维度上的注意力层构成；由于注意力操作固有的对称性，它天然地在变量维度和数据维度上都保持置换等变性。因篇幅所限，关于节点特征编码器的更多细节见附录第 A4 节。

**成对编码器。** 给定全部 {{< math >}}$d${{< /math >}} 个节点的节点特征 {{< math >}}$\mathcal{F} \in \mathbb{R}^{d\times n \times h}${{< /math >}}，成对编码器的目标是用 {{< math >}}$d^2${{< /math >}} 个成对特征来编码它们之间的成对关系，这些成对特征表示为一个张量 {{< math >}}$\mathcal{P} \in \mathbb{R}^{d\times d \times n \times h}${{< /math >}}，其中 {{< math >}}$\mathcal{P}_{ijl} \in \mathbb{R}^{h}${{< /math >}} 是对应于观测 {{< math >}}$l${{< /math >}} 中节点对 {{< math >}}$(i,j)${{< /math >}} 的成对特征。

正如第 1 节所论证的，要捕捉持久依赖性与定向不对称性，节点对的「内部」信息（即成对关系）与「外部」信息（例如条件分离集这一上下文）二者都是必需的。我们的成对编码器模块通过节点特征拼接以及 MLP 的非线性映射来建模内部关系。另一方面，我们在成对编码器内部使用注意力操作来捕捉上下文关系（包括持久依赖性与定向不对称性）。

更具体地说，成对编码器模块由以下几个部分组成（示意图见附录图 A3）：

1. *成对特征初始化。* 第一步是对每一对节点，把来自前一个节点特征编码器模块的节点特征拼接起来。随后，我们使用一个三层 MLP 把每个拼接后的向量 {{< math >}}$\mathcal{P}_{ijl} \in \mathbb{R}^{2h}${{< /math >}} 转换为 {{< math >}}$h${{< /math >}} 维的原始成对特征，即 {{< math >}}$\mathcal{P}_{ijl}^1 =  \mathrm{MLP}([\mathcal{F}_{il}; \mathcal{F}_{jl}])${{< /math >}}。该设计旨在捕捉存在于节点对内部的复杂关系。
2. *单向多头注意力。* 为了建模外部信息，我们采用一种注意力机制：其中查询由前述 {{< math >}}$d^2${{< /math >}} 个 {{< math >}}$h${{< /math >}} 维原始成对特征构成，而键和值则由 {{< math >}}$d${{< /math >}} 个单独节点的 {{< math >}}$h${{< /math >}} 维特征构成，即 {{< math >}}$\mathcal{P}^2 = \mathrm{MultiHeadAttention}(\mathcal{P}^1, \mathcal{F}, \mathcal{F})${{< /math >}}。注意，该注意力操作是单向的，也就是说我们只计算从原始成对特征 {{< math >}}$\mathcal{P}^1${{< /math >}} 到节点特征 {{< math >}}$F${{< /math >}} 的交叉注意力。这样设计是为了在捕捉节点对级与节点级信息（如第 1 节所讨论的，二者对于建模因果性都至关重要）的同时，维持合理的计算开销。
3. *最终处理。* 遵循被广泛采用的 Transformer 架构，我们在上一部分之后引入残差结构和 dropout 层，即 {{< math >}}$\mathcal{P}^3 = \mathrm{Norm}(\mathcal{P}^1 + \mathcal{P}^2)${{< /math >}}。最后，我们再引入一个三层 MLP，以进一步捕捉输入嵌入之间的复杂模式与非线性关系，并更有效地处理来自注意力机制的信息：{{< math >}}$\mathcal{P} = \mathrm{Norm}(\mathrm{MLP}(\mathcal{P}^3) + \mathcal{P}^3)${{< /math >}}。

   由此得到最终的成对特征张量 {{< math >}}$\mathcal{P} \in \mathbb{R}^{d \times d \times n \times h}${{< /math >}}。

### 4.3. 学习目标

如上文所述，我们的学习目标是骨架与 v-结构集合的组合，二者共同表示一个 MEC。我们针对这两个目标分别训练两个独立的神经（子）网络。

**骨架预测。** 由于节点对之间的持久依赖性决定了骨架中边的存在与否，成对特征天然地与骨架中的边相对应。因此，对于骨架学习任务，我们首先在观测维度上使用一个最大池化层，为每一对节点得到单个向量 {{< math >}}$\mathcal{S}_{ij} \in \mathbb{R}^{h}${{< /math >}}，即 {{< math >}}$\mathcal{S}_{ij} = \max_{k} \mathcal{P}_{ijk}${{< /math >}}。然后，再施加一个线性层和一个 Sigmoid 函数，把成对特征映射为对边的最终预测，即 {{< math >}}$S_{ij} = \mathrm{Sigmoid}(\mathrm{Linear}(\mathcal{S}_{ij}))${{< /math >}}。我们的学习标签，即表示骨架的无向图，可以简单地通过把 DAG {{< math >}}$G${{< /math >}} 的邻接矩阵与其转置 {{< math >}}$G^T${{< /math >}} 相加而算得。

因此，骨架预测任务的学习目标可以表述为 {{< math >}}$\min \mathcal{L}(S, G + G^T)${{< /math >}}，其中 {{< math >}}$\mathcal{L}${{< /math >}} 是常用的二元交叉熵损失函数。

**v-结构预测。**

当存在 {{< math >}}$\mathbf{S}${{< /math >}} 使得 {{< math >}}$X_k \notin \mathbf{S}${{< /math >}} 且 {{< math >}}$X_i \perp X_j | \mathbf{S}${{< /math >}} 时，UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} 就是一个 v-结构。受此启发，我们在观测维度上做最大池化之后，把节点对 {{< math >}}$\langle X_i, X_j \rangle${{< /math >}} 对应的成对特征与 {{< math >}}$X_k${{< /math >}} 的节点特征拼接起来，作为每个 UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} 的特征，即 {{< math >}}$\mathcal{U}_{kij} = [\max_l \mathcal{P}_{ijl};\max_l \mathcal{F}_{kl}]${{< /math >}}。之后，我们使用一个带 Sigmoid 函数的三层 MLP 来预测所有 UT 中 v-结构的存在与否，即 {{< math >}}$\mathcal{U}_{kij} = \mathrm{Sigmoid}(\mathrm{MLP}( \mathcal{U}_{kij}))${{< /math >}}。给定一个含 {{< math >}}$d${{< /math >}} 个节点的数据样本，它输出一个形状为 {{< math >}}$\mathbb{R}^{d \times d \times d}${{< /math >}} 的三阶张量，即 v-张量，对应于 v-结构存在与否的预测。v-张量的标签可由 {{< math >}}$\mathcal{V}_{kij} = G_{ik} G_{jk} (1 - G_{ij})(1 - G_{ji})${{< /math >}} 得到，其中 {{< math >}}$\mathcal{V}_{kij}${{< /math >}} 表示 v-结构 {{< math >}}$X_i \rightarrow X_k \leftarrow X_j${{< /math >}} 的存在与否。因此，v-结构预测任务的学习目标可以表述为 {{< math >}}$\min \mathcal{L}_{UT}(\mathcal{U}, \mathcal{V})${{< /math >}}，其中 {{< math >}}$\mathcal{L}_{UT}${{< /math >}} 是由 UT 掩码的二元交叉熵损失，也就是说我们只在有效的 UT 上计算该损失。在我们当前的实现中，特征编码器的参数是从骨架预测任务微调而来的，因为待分类的 UT 是从预测出的骨架中获得的，而骨架预测可以看作一个通用的预训练任务。

需要注意的是，如第 3 节末尾附近所提到的，采用我们这套学习目标的神经网络在渐近意义下具有正确性的理论保证。

### 4.4. 后处理

尽管我们的方法在理论上保证了渐近正确性，但在实践中预测出的 v-结构之间偶尔仍可能出现冲突。因此，在后处理阶段，我们遵循以往工作(<a href="#ref-6">Dai et al., 2023</a>)，采用一种直接的启发式方法来消解预测出的 v-结构之间可能存在的冲突与环。之后，我们使用改进版的 Meek 规则(<a href="#ref-23">Meek, 1995a</a>; <a href="#ref-34">Tsagris, 2019</a>)来得到其他 MEC-可识别边，且不引入额外的环。把骨架预测器模型给出的骨架与所有 MEC-可识别的边方向结合起来，我们就得到了 CPDAG 预测。

我们在附录第 A3 节给出了后处理过程的更详细描述。值得注意的是，我们当前的后处理设计是非常保守的，而且该模块在我们的整个框架中也并非必需；更多讨论与证据见附录第 A3 节。

## 5. 实验

**表 1**：**SiCL 与其他方法的总体比较**。SiCL 报告三次运行的平均性能结果。GES 在 WS-L-G 上每张图耗时超过 24 小时。SLdisco 不适用于非线性高斯数据。所有指标上的完整结果见附录表 A5。

| 方法 | WS-L-G s-F1↑ | WS-L-G o-F1↑ | SBM-L-G s-F1↑ | SBM-L-G o-F1↑ | WS-RFF-G s-F1↑ | WS-RFF-G o-F1↑ | SBM-RFF-G s-F1↑ | SBM-RFF-G o-F1↑ | ER-CPT-MC s-F1↑ | ER-CPT-MC o-F1↑ |
|---|---|---|---|---|---|---|---|---|---|---|
| PC | 30.4 | 16.0 | 58.8 | 35.9 | 36.1 | 16.1 | 57.5 | 34.2 | 82.2 | 40.6 |
| GES | * | * | 70.8 | 55.0 | 41.7 | 23.6 | 56.5 | 38.0 | 82.1 | 42.4 |
| NOTEARS | 33.3 | 31.5 | 80.1 | 77.8 | 37.7 | 33.4 | 55.6 | 48.5 | 16.7 | 0.6 |
| DAG-GNN | 35.5 | 32.7 | 66.2 | 62.5 | 33.2 | 28.9 | 47.1 | 40.6 | 24.8 | 3.7 |
| GRAN-DAG | 16.6 | 11.7 | 22.6 | 14.4 | 4.7 | 1.1 | 17.4 | 3.8 | 40.8 | 7.3 |
| GOLEM | 30.0 | 19.3 | 68.5 | 65.2 | 27.6 | 17.7 | 41.1 | 24.8 | 37.6 | 9.3 |
| SLdisco | 0.1 | 0.1 | 1.9 | 1.2 | * | * | * | * | * | * |
| AVICI | 39.9 | 35.8 | 84.3 | 81.6 | 47.7 | 45.2 | 76.6 | 72.7 | 76.9 | 57.6 |
| SiCL | **44.7** | **38.5** | **85.8** | **82.7** | **51.8** | **46.3** | **82.1** | **78.0** | **84.2** | **59.9** |

本节中，我们报告 SiCL 在合成基准与真实世界基准上的性能，随后给出消融实验。因篇幅所限，关于时间开销、通用性和无环性的更多结果与讨论移至附录第 A8 节。

### 5.1. 实验设计

**指标。** 我们通过以下两个任务来刻画一个因果发现方法的性能：

*骨架预测*：给定包含 d 个变量的数据样本 D，对于每一个变量对，我们希望推断它们之间是否存在直接因果关系。这里使用标准指标 **s-F1**（**skeleton-F1** 的缩写），它把骨架预测视为在 {{< math >}}$d(d-1)/2${{< /math >}} 个变量对上的二分类任务。为完整起见，我们也报告分类准确率结果。对于具有概率化输出的方法，我们还测量 AUC 和 AUPRC 分数。

*CPDAG 预测*：给定包含 d 个变量的数据样本 D，对于每一对变量，我们的目标是判定它们之间是否存在直接因果关系。若存在，则进一步评估其因果方向是否是 MEC-可识别的；若是，我们再尝试推断具体的因果方向。由于该任务同时涉及有向边与无向边的预测，我们使用 **SHD**（Structural Hamming Distance，结构汉明距离）来度量真实 CPDAG 与推断得到的 CPDAG 之间的差异。除此之外，我们还测量推断得到的 CPDAG 的有向子图的 **o-F1**（**orientation-F1** 的缩写）（与真实 CPDAG 的有向子图相比较），该指标着重刻画推断方法在识别 *MEC-可识别* 因果边上的定向能力。

最后，我们计算 **v-F1** 分数：该 F1 分数以真实 CPDAG 中的 v-结构集合（从所有有序三元组中选出）作为正例，以推断得到的 CPDAG 中的 v-结构作为正预测。

**测试数据。** 一个测试实例由真实因果图 G、每个变量 {{< math >}}$X_i${{< /math >}} 对应的结构方程 {{< math >}}$f_i${{< /math >}} 与噪声变量 {{< math >}}$\epsilon_i${{< /math >}}，以及一个独立同分布（i.i.d.）样本 D 组成。我们在实验中使用两类测试实例：

*解析型实例*：其中 G 从 DAG 分布 {{< math >}}$\mathcal{G}${{< /math >}} 中采样，{{< math >}}$\{f_i,\epsilon_i\}${{< /math >}} 则从结构方程分布 {{< math >}}$\mathcal{F}${{< /math >}} 和噪声元分布 {{< math >}}$\mathcal{N}${{< /math >}} 中采样。

对于 {{< math >}}$\mathcal{G}${{< /math >}}，我们考虑三种随机图分布：沃茨-斯特罗加茨（Watts-Strogatz, WS）模型、随机块模型（Stochastic Block Model, SBM）、埃尔德什-雷尼（Erdős-Rényi, ER）模型；对于 {{< math >}}$\mathcal{F}${{< /math >}}，我们考虑三种：随机线性（random linear, L）、随机傅里叶特征（Random Fourier Features, RFF）以及条件概率表（conditional probability table, CPT）。对于连续数据，{{< math >}}$\mathcal{N}${{< /math >}} 是高斯分布上的均匀分布；对于离散数据，{{< math >}}$\mathcal{N}${{< /math >}} 是多项类别分布上的狄利克雷分布。

我们考察五种测试实例组合：**WS-L-G**、**SBM-L-G**、**WS-RFF-G**、**SBM-RFF-G** 和 **ER-CPT-MC**。

*真实世界实例*：我们使用经典数据集 **Sachs** 来评估方法在真实世界场景中的性能。该数据集包含一份记录了 853 个人类免疫系统细胞中 11 种磷酸化蛋白浓度水平的数据样本，以及 <a href="#ref-30">Sachs et al. (2005)</a>基于专家共识与生物学文献在这 11 个变量上确定的因果图。

**算法。** 作为基线方法，我们与一系列具有代表性的无监督方法进行比较，包括 **PC**（使用 <a href="#ref-17">Le et al. (2016)</a>提出的较新的 Parallel-PC 变体）、**GES** (<a href="#ref-3">Chickering, 2002</a>)、**NOTEARS** (<a href="#ref-43">Zheng et al., 2018</a>)、**GOLEM** (<a href="#ref-25">Ng et al., 2020</a>)、**DAG-GNN** (<a href="#ref-40">Yu et al., 2019</a>)、**GRAN-DAG** (<a href="#ref-15">Lachapelle et al., 2020</a>)、**SLdisco** (<a href="#ref-28">Petersen et al., 2023</a>)，以及被视为当前最先进方法的基于 DNN 的 SCL 方法 **AVICI** (<a href="#ref-21">Lorch et al., 2022</a>)。

对于我们的方法，除了第 4 节所描述的完整 **SiCL** 实现之外，我们还实现了 **SiCL-Node-Edge** 与 **SiCL-no-PF**：前者使用节点特征来预测因果图，可视为等价于 AVICI；后者跳过成对特征提取，直接使用节点级特征来预测骨架和 v-张量（见附录图 A5）。

值得注意的是，SiCL 包含 2.8M 参数，而 SiCL-Node-Edge 与 SiCL-no-PF 包含 3.2M 参数，这是因为 SiCL 的节点特征编码器层数更少，以消除规模差异带来的潜在偏差。

对于基于 DNN 的 SCL 方法，其 DNN 使用合成数据训练，其中因果图服从埃尔德什-雷尼（Erdős-Rényi, ER）模型和无标度（Scale-Free, SF）模型，结构方程与噪声变量则服从与相应测试数据相同的分布类型。因此，训练时与测试时因果图分布之间的差异，在一定程度上有助于检验 SiCL 在**分布外**（Out-of-Distribution, **OOD**）设定下的通用性。

实验设定的更多细节见附录第 A7 节。

### 5.2. 合成数据集上的结果

我们在骨架预测任务和 CPDAG 预测任务上，对 SiCL 与各种基线方法进行了全面比较。skeleton-F1 和 orientation-F1 这两个指标的主要结果见表 1，全部指标上的结果见附录表 A5。

在连续数据上，基于 DNN 的 SCL 方法（即 AVICI 和 SiCL）相较于传统方法展现出稳定且明显的优势。SiCL 在骨架预测任务和 CPDAG 预测任务上都始终优于其他方法。另一方面，在离散数据 ER-CPT-MC 上，一些无监督方法取得了与基于 DNN 的 SCL 方法相当的性能。尽管如此，我们提出的 SiCL 仍是表现最佳者，这进一步证实了它在求解因果学习问题上的优越性。

### 5.3. 真实世界数据集上的结果

**表 2**：在 Sachs 数据集上的比较。

| 方法 | 骨架预测 s-F1↑ | 骨架预测 s-Acc.↑ | CPDAG 预测 SHD↓ | CPDAG 预测 #v-struc.↓ |
|---|---|---|---|---|
| PC | 68.6 | 80.0 | 19 | 12 |
| GES | 70.6 | 81.8 | 19 | 8 |
| DAG-GNN | 21.1 | 72.7 | 15 | **0** |
| NOTEARS | 11.1 | 70.9 | 16 | **0** |
| GRAN-DAG | 45.5 | 78.2 | 12 | **0** |
| GOLEM | 36.4 | 74.5 | 14 | **0** |
| AVICI | 66.7 | 83.5 | 18 | 14 |
| SiCL | **71.4** | **86.8** | **6** | **0** |

为评估 SiCL 的实际适用性，我们使用真实世界数据集 Sachs 进行了比较。我们采用从 bnlearn 库（注：<https://www.bnlearn.com/）获取的离散化> Sachs 数据。基于 DNN 的 SCL 方法是在随机合成图上训练的，因此这同时也是一个 **OOD** 预测任务。结果见表 2。

在骨架预测任务上，SiCL 表现最佳，不过优势幅度不大（通常比次优方法高 1~3 分）。在 CPDAG 预测任务上，SiCL 显著优于其他所有方法（相较于第二好的方法，将 SHD 从 12 降至 6）。有意思的是，Sachs 基准的真实因果 DAG 实际上不含任何 v-结构，因此任何被预测出来的 v-结构都是错误。我们看到，那些在骨架预测上与 SiCL 具有竞争力的方法（AVICI、PC、GES）在 Sachs 数据上错误地预测出了大量 v-结构，而 SiCL 则正确地预测出零个 v-结构。

**表 3**：SiCL 各组件的消融实验。完整指标见附录表 A6。

| 方法 | WS-L-G s-F1↑ | WS-L-G o-F1↑ | SBM-L-G s-F1↑ | SBM-L-G o-F1↑ |
|---|---|---|---|---|
| SiCL-Node-Edge | 39.9 | 35.8 | 84.3 | 81.6 |
| SiCL-no-PF | 42.4 | 37.9 | 85.5 | 82.2 |
| SiCL | **44.7** | **38.5** | **85.8** | **82.7** |

### 5.4. 消融实验

{{< figgrid caption="**图 2**：在一个构造数据集上，随观测样本数增加，SiCL-Node-Edge 与 SiCL-no-PF 的 o-F1 变化趋势对比。" >}}
paire/Cmp_on_of1_font.png | 100
{{< /figgrid >}}


**学习可识别结构的有效性。** 正如第 4.3 节所讨论的，SiCL 关注的是学习 MEC-可识别的因果结构，而不是直接学习邻接矩阵。为了验证这一想法的有效性，我们将 SiCL-Node-Edge 与 SiCL-no-PF 进行对比。这两个模型采用相似的节点特征编码器架构，但学习目标不同：SiCL-Node-Edge 预测邻接矩阵，而 SiCL-no-PF 预测骨架与 v-张量。结果如表 3 所示。SiCL-no-PF 在骨架预测和 CPDAG 预测两项任务上始终表现出更高的性能。这一观察印证了我们的理论结论：学习可识别的因果结构对于提升整体性能既是必要的，也是有益的。

为了进一步凸显学习可识别因果结构的重要性（尤其是在渐近意义下），我们在一个专门构造的数据集上开展了对比分析，该数据集包含六个节点，构成一个独立的 v-结构和一个 UT。图 2 表明，SiCL-Node-Edge 给出的 CPDAG 预测在定向 F1 分数上存在一个不可避免的误差，并且不会随着观测样本的增加而改善。相比之下，SiCL-no-PF 的预测达到了完美的准确率，这印证了学习可识别因果结构的价值。

**成对表示的有效性。** 为了评估成对表示的有效性，我们将完整版的 SiCL 与缺少成对特征的变体（SiCL-no-PF）进行对比。如表 3 所示，完整版 SiCL 在骨架预测和 CPDAG 预测两项任务上都始终优于 SiCL-no-PF。值得注意的是，我们有意将完整版 SiCL 的模型规模（2.8M 参数）设置得小于 SiCL-no-PF（3.2M 参数），以避免成对特征编码器模块带来的模型复杂度提升所可能产生的任何优势。在这种情况下仍然观察到的性能提升，凸显了成对特征在识别因果结构中的关键作用。此外，我们还在更多样化的设定下进行了进一步的对比，结果详见附录第 A7 节。这些结果显示出更为显著的、有利于 SiCL 的改进，进一步强调了成对表示在因果发现中的重要性。

## 6. 结论

我们提出了 SiCL，一种新颖的基于 DNN 的 SCL 方法，其设计目标是预测相应的骨架和一组 v-结构。我们表明，这样的设计不会受到当前架构中存在的（不）可识别性限制的影响。此外，SiCL 还配备了一个成对编码器模块，用以显式地建模节点对之间的关系。实验结果验证了这些想法的有效性。

本文还引出了若干有趣的开放问题。所提出的 DNN 模型工作在经典 MEC 理论下的标准设定中，其中骨架和 v-结构就是可识别结构。遵循同样的原则，探索如何在其他假设设定下学习别的可识别因果结构，会是一个有趣的未来工作方向。由于 DNN 本身的复杂性，如何解释我们模型的决策机制仍是一个开放问题。因此，未来工作可以考虑探究网络内部是如何做出决策的，并为传统方法提供一些启发。此外，所提出的成对编码器模块需要 {{< math >}}$O(d^3)${{< /math >}} 的计算复杂度，这可能会限制其目前在节点数量巨大的场景中的应用。未来工作可以着眼于简化这些操作，或探索复杂度更低的特征（例如低秩特征），以降低整体计算开销。

## 检查清单

1. 对于所提出的全部模型与算法，请检查你是否包含：
   1. 对数学设定、假设、算法和/或模型的清晰描述。[是]
   2. 对任一算法的性质与复杂度（时间、空间、样本量）的分析。[是]
   3. （可选）匿名化的源代码，并说明全部依赖项，包括外部库。[是]

2. 对于任何理论性论断，请检查你是否包含：
   1. 对所有理论结果的完整假设集合的陈述。[是]
   2. 对所有理论结果的完整证明。[是]
   3. 对任何假设的清晰解释。[是]

3. 对于呈现实证结果的全部图和表，请检查你是否包含：
   1. 复现主要实验结果所需的代码、数据与说明（放在补充材料中或以 URL 形式提供）。[是]
   2. 全部训练细节（例如，数据划分、超参数及其选取方式）。[是]
   3. 对具体度量或统计量以及误差棒的清晰定义（例如，关于多次运行实验时的随机种子）。[是]
   4. 对所用计算基础设施的描述。（例如，GPU 类型、内部集群或云服务提供商。）[是]

4. 如果你使用了已有资源（例如代码、数据、模型），或整理/发布了新的资源，请检查你是否包含：
   1. 如果你的工作使用了已有资源，对其创建者的引用。[是]
   2. 如适用，这些资源的许可证信息。[不适用]
   3. 如适用，放在补充材料中或以 URL 形式提供的新资源。[不适用]
   4. 关于数据提供者/整理者同意情况的信息。[是]
   5. 如适用，对敏感内容的讨论，例如个人可识别信息或冒犯性内容。[不适用]

5. 如果你使用了众包或开展了涉及人类受试者的研究，请检查你是否包含：
   1. 提供给参与者的完整指导语文本与截图。[不适用]
   2. 对参与者潜在风险的描述，如适用还应附上 Institutional Review Board (IRB) 批准的链接。[不适用]
   3. 支付给参与者的估计时薪，以及用于参与者报酬的总支出。[不适用]

**算法 A1**：用于预测因果结构的 SiCL 工作流

```
Procedure INFERENCE(data, target)
    用节点编码器计算节点特征
    按第 4.2 节用成对编码器计算成对特征
    if target 为 skeleton then
        按第 4.3 节计算骨架
    else
        按第 4.3 节计算 v-结构
    end if
End Procedure

Procedure TRAINING_PHASE()
    skeleton_predictor <- init_skeleton_predictor()
    采样图以及相应的数据
    用 INFERENCE(data, skeleton) 训练骨架预测器
    用 INFERENCE(data, v-structure) 训练 v-结构预测器，其中特征编码器从骨架预测器微调而来
End Procedure

Procedure TESTING_PHASE(test_data)
    用训练好的骨架预测器计算预测的骨架
    用训练好的 v-结构预测器计算预测的 v-结构
    将预测的骨架与 v-结构结合，得到预测的 CPDAG
End Procedure
```

{{< figgrid caption="**图 A3**：成对编码器模块示意图。在第 ① 部分，模块初始化原始的成对特征；在第 ② 部分，用单向注意力机制融合来自节点特征与成对特征的信息；在第 ③ 部分，用一个 MLP 加残差连接得到最终的成对特征。" >}}
paire/pairwiseencoder.png | 100
{{< /figgrid >}}


## A1. 理论保证

在本节中，我们深入探讨关于所提模型相对于样本量的渐近正确性的理论分析。第 A1.1 节给出与所研究问题相关的核心定义与假设。随后，从第 A1.2 节到第 A1.3 节，我们严格论证了神经网络模型的渐近正确性。最后，在第 A1.4 节中，我们详细讨论神经网络模型在实践中的优势与优越性。

### A1.1. 定义与假设

正如第 2 节所述，一个因果图模型由定义在 {{< math >}}$d${{< /math >}} 个随机变量 {{< math >}}$X_1, X_2, \cdots, X_{d}${{< /math >}} 上的联合概率分布 {{< math >}}$P${{< /math >}}，以及一个具有 {{< math >}}$d${{< /math >}} 个顶点（分别表示这 {{< math >}}$d${{< /math >}} 个变量）的 DAG {{< math >}}$G${{< /math >}} 共同定义。
一个观测数据集 {{< math >}}$D${{< /math >}} 由 {{< math >}}$n${{< /math >}} 行记录和 {{< math >}}$d${{< /math >}} 列构成，表示从 {{< math >}}$P${{< /math >}} 中独立同分布（i.i.d.）抽取的 {{< math >}}$n${{< /math >}} 个实例。
在本工作中，我们假设因果充分性成立：

**假设 A1.1（因果充分性）**：图中任何变量都不存在潜在共同原因。

此外，我们假设数据分布 {{< math >}}$P${{< /math >}} 对 DAG {{< math >}}$G${{< /math >}} 满足马尔可夫性：

**假设 A1.2（马尔可夫分解性质）**：给定一个联合概率分布 {{< math >}}$P${{< /math >}} 和一个 DAG {{< math >}}$G, P${{< /math >}}，若 {{< math >}}$P:=${{< /math >}} {{< math >}}$P\left(X_1, X_2, \cdots, X_d\right)=\prod_{i=1}^d P\left(X_i \mid \mathrm{pa}_i^G\right)${{< /math >}}，则称 {{< math >}}$P${{< /math >}} 关于 {{< math >}}$G${{< /math >}} 满足马尔可夫分解性质，其中 {{< math >}}$\mathrm{pa}_i^G${{< /math >}} 是 {{< math >}}$X_i${{< /math >}} 在 {{< math >}}$G${{< /math >}} 中的父节点集。

值得注意的是，马尔可夫分解性质等价于全局马尔可夫性（Global Markov Property, GMP）(<a href="#ref-16">Lauritzen, 1996</a>)，其表述为

**定义 A1.1（全局马尔可夫性（GMP））**：若 {{< math >}}$X \perp_G Y|Z \Rightarrow X \perp Y| Z${{< /math >}}，则称 {{< math >}}$P${{< /math >}} 关于 DAG {{< math >}}$G${{< /math >}} 满足 GMP（或称满足马尔可夫性）。这里 {{< math >}}$\perp_G${{< /math >}} 表示 d-分离，{{< math >}}$\perp${{< /math >}} 表示统计独立性。

GMP 表明，图 {{< math >}}$G${{< /math >}} 中的任何 d-分离都蕴含分布 {{< math >}}$P${{< /math >}} 中的条件独立性。我们进一步假设 {{< math >}}$P${{< /math >}} 相对于 {{< math >}}$G${{< /math >}} 是忠实的：

**假设 A1.3（忠实性）**：若 {{< math >}}$X \perp Y\left|Z \Rightarrow X \perp_G Y\right| Z${{< /math >}}，则分布 {{< math >}}$P${{< /math >}} 相对于 DAG {{< math >}}$G${{< /math >}} 是忠实的。

**定义 A1.2（标准假设）**：若假设 A1.1 至 A1.3 全部成立，则称我们的设定满足标准假设。

我们将骨架、无屏蔽三元组（Unshielded Triple, UT）以及 v-结构的定义重述如下。

**定义 A1.3（骨架）**：定义在数据分布 {{< math >}}$P${{< /math >}} 上的骨架 {{< math >}}$E${{< /math >}} 是一个无向图，其中 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 之间存在一条边，当且仅当 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 在 {{< math >}}$P${{< /math >}} 中始终相互依赖，即 {{< math >}}$\forall Z \subseteq\left\{X_1, X_2, \cdots, X_d\right\} \backslash \left\{X_i, X_j \right\}${{< /math >}}，都有 {{< math >}}$X_i \nperp X_j | Z${{< /math >}}。

在我们的假设下，骨架与 {{< math >}}$G${{< /math >}} 所对应的无向图相同 (<a href="#ref-33">Spirtes et al., 2000</a>)。

**定义 A1.4（无屏蔽三元组（UT）与 v-结构）**：若在 DAG {{< math >}}$G${{< /math >}} 或其对应的骨架中，{{< math >}}$X${{< /math >}} 与 {{< math >}}$Y${{< /math >}} 都与 {{< math >}}$T${{< /math >}} 相邻但彼此不相邻，则变量三元组 {{< math >}}$X, T, Y${{< /math >}} 称为一个无屏蔽三元组（UT），记作 {{< math >}}$\langle X, T, Y \rangle${{< /math >}}。
若在 {{< math >}}$G${{< /math >}} 中这两条边的方向都是从 {{< math >}}$X${{< /math >}} 和 {{< math >}}$Y${{< /math >}} 指向 {{< math >}}$T${{< /math >}}，则它成为一个 v-结构，记作 {{< math >}}$X \rightarrow T \leftarrow Y${{< /math >}}。

我们给出分离集的定义如下：

**定义 A1.5（分离集）**：对于节点对 {{< math >}}$X_i${{< /math >}} 和 {{< math >}}$X_j${{< /math >}}，若 {{< math >}}$X_i \perp X_j | Z ${{< /math >}}，则节点集 {{< math >}}$Z${{< /math >}} 是一个分离集。在忠实性假设下，分离集 {{< math >}}$Z${{< /math >}} 是邻近范围内将 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} d-分离的一个变量子集。

最后，我们假设在我们的设定下，神经网络可以用作通用逼近器。

**假设 A1.4（通用逼近能力）**：在我们的设定下，神经网络模型可以被训练成以任意精度逼近某个函数。

### A1.2. 骨架学习

在本节中，我们通过构造一个完美模型、再用神经网络对其进行逼近，来证明神经网络在骨架预测任务上的渐近正确性。
为了描述上的方便与简洁，我们将骨架预测器定义如下。

**定义 A1.6（骨架预测器）**：给定观测数据 {{< math >}}$D${{< /math >}}，骨架预测器是一个以观测数据 {{< math >}}$D${{< /math >}} 为定义域的谓词函数，用于预测每一对顶点之间的邻接关系。

现在我们将 <a href="#ref-22">Ma et al. (2022)</a>中的注记重述为如下命题。
它通过把 PC (<a href="#ref-33">Spirtes et al., 2000</a>) 的骨架预测步骤视为一个骨架预测器（该步骤已被证明是可靠且完备的），证明了完美骨架预测器的存在性。

**命题 A1.1（完美骨架预测器的存在性）**：存在一个骨架预测器，在 {{< math >}}$D${{< /math >}} 中样本充足时总能给出正确的骨架。

**证明**：我们把 PC (<a href="#ref-33">Spirtes et al., 2000</a>) 视为一个骨架预测器，据此构造由两部分组成的骨架预测器 {{< math >}}$SP${{< /math >}}。
在第一部分中，它为每一对节点 {{< math >}}$X_i${{< /math >}} 和 {{< math >}}$X_j${{< /math >}} 提取一个成对特征 {{< math >}}$\boldsymbol{x}_{i j}${{< /math >}}：

{{< math >}}
$$
\boldsymbol{x}_{i j}=\min _{Z \subseteq V \backslash\left\{X_i, X_j\right\}}\left\{X_i \sim X_j \mid Z\right\}, \tag{2}
$$
{{< /math >}}

其中 {{< math >}}$\left\{X_i \sim X_j \mid Z\right\} \in [0, 1] ${{< /math >}} 是一个标量值，用于度量在给定节点子集 {{< math >}}$Z${{< /math >}} 的条件下 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 之间的条件依赖度。

因此，{{< math >}}$\boldsymbol{x}_{i j} &gt; 0${{< /math >}} 表明这两个节点之间存在持久依赖性。

在第二部分中，它基于 {{< math >}}$\boldsymbol{x}_{i j}${{< /math >}} 预测邻接关系：

{{< math >}}
$$
\left(X_i,  X_j\right)= \begin{cases} 1 \text { (adjacent) } & \boldsymbol{x}_{i j} \neq 0 \\ 0 \text { (non-adjacent) } & \boldsymbol{x}_{i j} = 0\end{cases} \tag{3}
$$
{{< /math >}}

下面我们通过证明不存在假阳性预测和假阴性预测，来说明 {{< math >}}$SP${{< /math >}} 总能给出正确的骨架。这里，假阳性预测指 {{< math >}}$SP${{< /math >}} 把不相邻的节点对预测为相邻，假阴性预测指 {{< math >}}$SP${{< /math >}} 把相邻的节点对预测为不相邻。

- **假阳性。** 假设 {{< math >}}$X_i, X_j${{< /math >}} 不相邻。在马尔可夫假设下，存在一个节点集合 {{< math >}}$Z${{< /math >}} 使得 {{< math >}}$\left\{X_i \sim X_j \mid Z\right\} = 0${{< /math >}}，因而 {{< math >}}$\boldsymbol{x}_{ij} = 0${{< /math >}}。根据式 (3)，{{< math >}}$SP${{< /math >}} 总会把它们预测为不相邻。
- **假阴性**。假设 {{< math >}}$X_i, X_j${{< /math >}} 相邻。在忠实性假设下，对任意 {{< math >}}$Z \in V \backslash \left\{X_i, X_j\right\}, \left\{X_i \sim X_j \mid Z\right\} &gt; 0${{< /math >}}，这意味着 {{< math >}}$\boldsymbol{x}_{ij} &gt; 0${{< /math >}}。因此，{{< math >}}$SP${{< /math >}} 总会把它们预测为相邻。

因此，在马尔可夫假设和忠实性假设下，{{< math >}}$SP${{< /math >}} 绝不会给出任何假阳性预测或假阴性预测，也就是说，它总能给出正确的骨架。

有了完美骨架预测器的存在性，我们便可在我们的假设下证明神经网络模型在样本充足时的正确性。

**定理 A1.1**：在标准假设以及神经网络可用作通用逼近器的假设（假设 A1.4）下，
存在一个神经网络模型，在 {{< math >}}$D${{< /math >}} 中样本充足时总能预测出正确的骨架。

**证明**：由命题 A1.1 可知，存在一个完美骨架预测器能够预测出正确的骨架。
因此，根据假设 A1.4，可以训练一个神经网络模型来逼近该完美骨架预测，从而预测出正确的骨架。

### A1.3. 定向学习

与第 A1.2 节的整体思路类似，在本节中我们通过构造一个完美模型、再用神经网络对其进行逼近，来证明神经网络在 v-结构预测任务上的渐近正确性。

**定义 A1.7（v-结构预测器）**：给定来自一个顶点集为 {{< math >}}$V = \{X_1, \dots, X_p\}${{< /math >}} 的 {{< math >}}$BN${{< /math >}} 的、样本充足的观测数据 {{< math >}}$D${{< /math >}}，v-结构预测器是一个以观测数据 {{< math >}}$D${{< /math >}} 为定义域的谓词函数，用于预测每个无屏蔽三元组是否构成 v-结构。

下面的命题通过把 PC (<a href="#ref-33">Spirtes et al., 2000</a>) 的定向步骤视为一个 v-结构预测器，证明了完美 v-结构预测器的存在性。

**命题 A1.2（完美 v-结构预测器的存在性）**：在马尔可夫假设和忠实性假设下，存在骨架预测器总能给出正确的骨架。

**证明**：我们把 PC (<a href="#ref-33">Spirtes et al., 2000</a>) 视为一个 v-结构预测器，据此构造由两部分组成的 v-结构预测器 {{< math >}}$VP${{< /math >}}。

在第一部分中，它为每个 UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} 提取一个布尔特征 {{< math >}}$\boldsymbol{z}_{kij}${{< /math >}}：

{{< math >}}
$$
\boldsymbol{z}_{kij} = (X_k \in Z), \text{ where } Z \text{ is called as a sepset, i.e. } X_i\perp Y_j | Z.  \tag{4}
$$
{{< /math >}}

需要注意的是，分离集 {{< math >}}$Z${{< /math >}} 总是存在的，因为 UT 的分离集总是存在的（参见 <a href="#ref-6">Dai et al. (2023)</a>中的引理 4.1）。

在第二部分中，它基于 {{< math >}}$\boldsymbol{z}_{ijk}${{< /math >}} 预测 v-结构：

{{< math >}}
$$
\langle X_i, X_k, X_j\rangle = \begin{cases} 0 \text { (not v-structure) } & \boldsymbol{z}_{k i j } = True \\ 1 \text { (v-structure) } & \boldsymbol{z}_{k i j } = False\end{cases} \tag{5}
$$
{{< /math >}}

下面我们证明 {{< math >}}$VP${{< /math >}} 总能给出正确的 v-结构预测。
根据 <a href="#ref-33">Spirtes et al. (2000)</a>第 410 页的定理 5.1，在忠实性成立且样本充足的前提下，若一个 UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} 是 v-结构，则 {{< math >}}$X_k${{< /math >}} 不属于 {{< math >}}$(X_i, X_j)${{< /math >}} 的任何分离集；若一个 UT {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} 不是 v-结构，则 {{< math >}}$X_k${{< /math >}} 属于 {{< math >}}$(X_i, X_j)${{< /math >}} 的每一个分离集。因此，{{< math >}}$\boldsymbol{z}_{kij} = False${{< /math >}} 当且仅当 {{< math >}}$X_k${{< /math >}} 不在 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 的任何分离集中，即 {{< math >}}$\langle X_i, X_k, X_j \rangle${{< /math >}} 是一个 v-结构。

有了完美 v-结构预测器的存在性，我们便可在我们的假设下证明神经网络模型在样本充足时的正确性。

**定理 A1.2**：在标准假设以及神经网络可用作通用逼近器的假设（假设 A1.4）下，存在一个神经网络模型，在 {{< math >}}$D${{< /math >}} 中样本充足时总能预测出正确的 v-结构。

**证明**：由命题 A1.1 可知，存在一个完美骨架预测器能够预测出正确的 v-结构。
因此，根据假设 A1.4，可以训练一个神经网络模型来逼近该完美 v-结构预测，从而预测出正确的 v-结构。

### A1.4. 讨论

在上述各节中，我们通过构造理论上完美的预测器，证明了神经网络模型的渐近正确性。这两个预测器都由两部分组成：提供特征 {{< math >}}$\boldsymbol{x}_{ij}${{< /math >}} 与 {{< math >}}$\boldsymbol{z}_{ijk}${{< /math >}} 的特征提取器，以及对邻接关系和 v-结构的最终预测器。尽管它们在样本充足时具有正确性的理论保证，但值得注意的是，它们难以在实践中应用。例如，为了得到式 (2) 中的 {{< math >}}$\boldsymbol{x}_{ij}${{< /math >}}，我们需要针对每一个节点子集 {{< math >}}$Z \subseteq V \backslash\left\{X_i, X_j\right\}${{< /math >}} 计算 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 之间的条件依赖度。姑且不论 {{< math >}}$Z${{< /math >}} 的数量本身就呈阶乘复杂度，主要问题在于：当 {{< math >}}$Z${{< /math >}} 相对较大时，由于维度灾难，很难找到足够的样本来计算该条件依赖度。这一困难极大地阻碍了所构造的完美预测器在实际场景中的应用。

一些已有方法可以被理解为构造了更加实用的预测器。Majority-PC (MPC) (<a href="#ref-4">Colombo et al., 2014</a>) 通过将式 (4) – (5) 修改为如下形式，在有限样本上取得了更好的性能：

{{< math >}}
$$
\boldsymbol{z}_{kij} = \frac{\left|\left\{ (X_k, Z) | \{ X_i \sim X_j | Z\} = 0 \wedge X_k \in Z \right\}\right|}{\left|\left\{ Z | \{ X_i \sim X_j | Z\} = 0 \right\}\right|}, \tag{6}
$$
{{< /math >}}

以及

{{< math >}}
$$
\left\langle X_i, X_k, X_j\right\rangle= \begin{cases}0 \text { (not v-structure) } &amp; \boldsymbol{z}_{i j k} &gt; 0.5 \\ 1(\mathrm{v} \text {-structure) } &amp; \boldsymbol{z}_{i j k} \leq 0.5,\end{cases} \tag{7}
$$
{{< /math >}}

其中 {{< math >}}$\left\{X_i \sim X_j \mid Z\right\} \in [0, 1]${{< /math >}} 是一个标量值，用于度量在给定节点子集 {{< math >}}$Z${{< /math >}} 时 {{< math >}}$X_i${{< /math >}} 与 {{< math >}}$X_j${{< /math >}} 之间的条件依赖度，而 {{< math >}}$|\cdot|${{< /math >}} 表示一个集合的基数。由于其分类机制更为复杂，它在经验上取得了更好的性能。然而，从机器学习的角度来看，PC 与 MPC 这两种预测器所使用的特征都相对简单。作为 SCL 方法，ML4S (<a href="#ref-22">Ma et al., 2022</a>) 与 ML4C (<a href="#ref-6">Dai et al., 2023</a>) 通过人工特征工程以及利用强大的机器学习模型进行分类，提供了更为系统的特征化。

尽管这些方法展现出更强的实际效果，但其人工特征工程过程十分复杂。在本文中，我们利用神经网络作为通用逼近器，来学习对可识别因果结构的预测。这不仅简化了流程，还有可能发掘出数据中更细微、更复杂而人工方法可能忽略的模式。值得注意的是，使用神经网络进行 SCL 所带来的好处在其他工作中也有讨论，如 SLdisco (<a href="#ref-28">Petersen et al., 2023</a>) 与 CSIvA (<a href="#ref-13">Ke et al., 2023</a>) 中所提到的那样。

**算法 A2**：后处理

```
输入：加权骨架矩阵 S，加权 v-张量 U，骨架的阈值 τ_s，v-结构的阈值 τ_v
输出：预测的定向边集合 oriEdges，预测的骨架 skeleton

Step 1:
    // 通过阈值化得到预测的骨架。
    skeleton = {(i, j) | max(S_ij, S_ji) > τ_s}
    // 通过阈值化得到原始 v-结构 vstructs_raw。
    vstructs_raw = {(i, j, k) | (i, j) ∈ skeleton 且 (i, k) ∈ skeleton 且 (j, k) ∉ skeleton 且 max(U_ijk, U_ikj) > τ_v}

Step 2:
    // v-结构冲突消解：遵循 (Dai et al., 2023) 的做法，若存在另一个与之冲突且预测分数更高的 v-结构，则丢弃当前 v-结构。
    vstructs = {(i, j, k) ∈ vstructs_raw | 对任意 (i', j', k') ∈ vstructs_raw，(i' ≠ k 且 i' ≠ j) 或 (k' ≠ i 且 j' ≠ i) 或 U_i'j'k' < U_ijk}

Step 3:
    // 从 vstructs 得到预测的有向边。
    oriEdges_raw = {(j, i) | 存在 k，使得 (i, j, k) ∈ vstructs}
    // 为每条边设定分数，取包含该边的 v-结构中的最高分数。
    设定 {p_ij}，使得对于 v ∈ oriEdges_raw 且 v ∋ (i, j)，有 p_ij = max_v U_v。
    // 若存在任何环，则移除每个环中分数最小的边。
    oriEdges = {(i, j) ∈ oriEdges_raw | 对任意环 C，(i, j) ∉ C 或 (存在 (i', j') ∈ C，使得 p_ij > p_i'j')}

Step 4:
    // Meek 规则：向 oriEdges 中添加满足如下条件的有向边：(1) 引入这些边不会导致环或新的 v-结构；(2) 添加方向相反的边则必然导致环或新的 v-结构。
    oriEdges = oriEdges ∪ {(i, j) ∈ skeleton | (i, j) 符合 Meek 规则}
```

## A2. 关于可识别性与因果假设的更多讨论

**倡导在所有设定下都学习可识别结构。**
在本文中，出于展示的目的，我们不得不在一个具体的设定下、针对具体的可识别因果结构开展工作。
然而，我们想强调的是，可识别性这一概念本身及其在 SCL 中的种种影响，实际上是一个普遍性的问题，它并不怎么受制于「哪些因果结构在哪些假设下是可识别的」这一问题。
一个简单的事实是：在某些情形下因果边无法被识别 – 无论在该情形下有什么特征是可识别的 – 这种可识别性限制都会对 SCL 产生普遍的影响。
除非因果图/因果边本身变得完全不变/可识别（这是一种重要但显然并非普遍成立的特例），否则可识别性限制的存在就会给一种流行的 SCL 模型架构（即「节点-边」架构）带来根本性偏差，而这种偏差根本无法通过更大的模型或更多的数据来缓解。
这种「可识别性限制导致学习误差」的效应正是本文的主要论点，我们倡导设计专注于学习可识别特征（无论这些特征具体是什么）的神经网络。
换言之，完全可以去研究另一种设定，在其中另一类特征是可识别的；在那种情况下，我们同样会倡导去学习那种特征，而不是 v-结构。
例如，如果我们假设标准 MEC 假设，并且假设不存在因果分叉与 v-结构，那么可识别因果结构就变成了某种链结构。
在这种情况下，人们可能希望设计预测因果链的神经网络。

**标准假设的合理性。**
在本文中，我们选择了经典 MEC 理论下的标准设定，其中骨架与 v-结构是可识别结构。
该设定包含了马尔可夫条件与忠实性条件的假设。
与那些针对特定场景的假设（例如与某一特定数据生成过程绑定的假设）不同，这些假设是关于因果性的经典假设，常常被当作关于世界某些普遍方面的「公设」加以采纳。例如，

- <a href="#ref-26">Pearl (2009)</a>认为，稳定性（忠实性）源自参数之间严格等式约束在自然情况下的不可能性，这与因果机制的自主性是一致的。
- <a href="#ref-32">Spirtes et al. (2001)</a>通过指出因果路径在自然条件下的精确抵消极不可能发生，支持了因果忠实性条件（Causal Faithfulness Condition, CFC）。
- <a href="#ref-38">Weinberger (2018)</a>进一步强化了这一论证，提出导致 CFC 被违反的巧合是罕见的且缺乏解释力，从而进一步论证了在一般建模框架内采纳该条件的合理性。

这些考量凸显了上述假设的合理性与通用性，使它们成为我们分析中的自然选择。

## A3. 后处理的细节与讨论

为了全面清晰起见，我们在算法 A2 中给出了后处理算法的明确流程。

**讨论。** 值得注意的是，我们在后处理中的设计尽可能地保守。事实上，我们只是沿用深度学习中的惯例（即阈值化）来获得骨架和初始的 v-结构集合。随后，我们遵循基于约束的因果发现方法中的惯例来导出最终的定向边。因此，我们并未在精细设计上投入大量精力，也无意强调我们工作流中的这一环节。

冲突与环并非 SiCL 独有；事实上，它们是所有诸如 PC 这类基于约束的算法都会遇到的常见问题。此外，值得注意的是，如果网络表现完美，它们就绝不会出现。因此，v-结构的冲突消解与环的移除被设计为回退机制，用以确保我们工作流的可靠性，而非我们方法的核心要素。为了说明这一点，我们实验了一个相反的变体（直觉上这是一个糟糕的选择），它优先丢弃预测概率较高的那个 v-结构。如表 A4 所示，该变体与其对应方法之间的结果差异极小，这支持了我们的观点：冲突消解过程在我们的工作流中意义有限。另一方面，表 A11 中给出的实验结果凸显了预测中出现环的情况非常罕见，进一步印证了环移除组件的非必要性。

**表 A4**：所采用的冲突消解方法与一个相反变体之间的 o-F1 对比。

| 冲突消解方法 | WS-L-G | SBM-L-G |
|---|---|---|
| 原始冲突消解 | **41.1** | **83.3** |
| 相反冲突消解 | 40.7 | 83.2 |

## A4. 节点特征编码器的细节

受先前方法 (<a href="#ref-21">Lorch et al., 2022</a>; <a href="#ref-13">Ke et al., 2023</a>) 的启发，我们采用了一种类 Transformer 架构作为节点特征编码器，它交替地在观测维度或节点维度上使用注意力层。
具体而言，对于对应于 d 个节点和 n 条观测的原始节点特征 {{< math >}}$\mathcal{F} \in \mathbb{R}^{d \times n \times h}${{< /math >}}，我们的目标是同时捕捉不同节点之间以及不同观测之间的相关性。
因此，我们交替地在观测维度和节点维度上使用两个 transformer 编码器层：

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

上述操作会重复多次，以进行充分的特征编码。
它最终得到节点特征张量 {{< math >}}$\mathcal{F} \in \mathcal{R}^{d \times \times h}${{< /math >}}。

## A5. 第 3 节中案例研究的图示

图 A4 给出了第 3 节中「节点-边」方法案例研究的图示。
它清楚地表明，具有两种不同参数化形式的观测数据服从相同的联合分布：

{{< math >}}
$$
P(\left[X, Y, T\right]) =\mathcal{N}\left([0,0,0],\left[\begin{array}{lll}1 & 1 & 1 \\ 1 & 3 & 2 \\ 1 & 2 & 2\end{array}\right]\right). \tag{9}
$$
{{< /math >}}

因此，来自这两个 DAG 的观测数据集在本质上是无法区分的。

{{< figgrid caption="**图 A4**：用于凸显 Node-Edge 方法局限性的问题设定。*建议彩色查看。*" >}}
paire/ps.png | 90
{{< /figgrid >}}


## A6. 命题 3.1 的证明与讨论

我们首先以更详细的形式重述命题 3.1，并给出证明。

**命题 A6.1**：设 {{< math >}}$\mathcal{G}_n${{< /math >}} 为由 {{< math >}}$n+1${{< /math >}} 个节点构成的图的集合，其中存在一个中心节点 {{< math >}}$y${{< /math >}}，满足：(1) 其他每个节点都与 {{< math >}}$y${{< /math >}} 相连；(2) 其他节点之间没有边；(3) 至多有一条边指向 {{< math >}}$y${{< /math >}}。
对于 {{< math >}}$\mathcal{G}_n${{< /math >}} 上的任意分布 {{< math >}}$Q${{< /math >}}，设 {{< math >}}$M(Q)${{< /math >}} 为 {{< math >}}$\mathcal{G}_n${{< /math >}} 上的另一个分布，使得对任意因果边 {{< math >}}$e, e'${{< /math >}} 都有 {{< math >}}$P_{G\sim Q}(e \in G) = P_{G\sim M(Q)}(e \in G) = P_{G\sim M(Q)}(e \in G | e' \in G)${{< /math >}}。我们有

{{< math >}}
$$
\max_{Q} P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 
1 - \frac{2n-1}{n-1}(1 - \frac{1}{n})^n. \tag{10}
$$
{{< /math >}}

作为推论，我们有

{{< math >}}
$$
\sup_n \max_{Q} P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 
1 - \frac{2}{e} \approx 0.2642, \tag{11}
$$
{{< /math >}}

**证明**：将除中心节点之外的其他节点记为 {{< math >}}$x_i${{< /math >}}，其中 {{< math >}}$i \in \left\{1, 2, \dots, n\right\}${{< /math >}}。
在我们的设定下，集合 {{< math >}}$\mathcal{G}_n${{< /math >}} 包含 {{< math >}}$n + 1${{< /math >}} 个具有相同骨架且不含 v-结构的 DAG：{{< math >}}$G_0: y \rightarrow x_i${{< /math >}} 对所有 {{< math >}}$x_i${{< /math >}} 成立；以及 {{< math >}}$G_i: y \rightarrow x_j${{< /math >}} 对所有 {{< math >}}$x_j \neq x_i${{< /math >}} 成立，同时含有 {{< math >}}$x_i \rightarrow y${{< /math >}}。
将从 {{< math >}}$\mathcal{G}_n${{< /math >}} 中采样得到 DAG {{< math >}}$G_i${{< /math >}} 的概率记为 {{< math >}}$P_i${{< /math >}}。
因此，边 {{< math >}}$y \rightarrow x_i${{< /math >}} 的边缘概率为 {{< math >}}$1 - P_i${{< /math >}}。

若 {{< math >}}$\exists i${{< /math >}} 使得 {{< math >}}$P_i = 1${{< /math >}}，这意味着 {{< math >}}$\mathcal{G}_n${{< /math >}} 中只包含 DAG {{< math >}}$G_i${{< /math >}}。因此，{{< math >}}$M(Q)${{< /math >}} 等价于 {{< math >}}$Q${{< /math >}}，我们有 {{< math >}}$P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 0${{< /math >}}。

若 {{< math >}}$\forall i${{< /math >}} 都有 {{< math >}}$P_i &lt; 1${{< /math >}}，记 {{< math >}}$Q_i = 1 - P_i${{< /math >}}，并记 {{< math >}}$P(v)${{< /math >}} 为 {{< math >}}$G${{< /math >}} 不含 v-结构的概率。换言之，{{< math >}}$P(v) = P_{G \sim M(Q)}(G \in \mathcal{G}_n)${{< /math >}}。我们有

{{< math >}}
$$
\begin{aligned}
P(v) = \prod_{i=1}^n Q_i + \sum_{j=1}^n \frac{\prod_i^n Q_i}{Q_j} (1 - Q_j)
&= ( \prod_{i=1}^n Q_i) \cdot (1 + \sum_{j=1}^{n} \frac{1-Q_j}{Q_j}).
\end{aligned}
\tag{12}
$$
{{< /math >}}

由于 {{< math >}}$P(v)${{< /math >}} 是一个概率，我们有 {{< math >}}$P(v) &gt; 0${{< /math >}}。记函数

{{< math >}}
$$
f(Q_1, Q_2, \dots, Q_n) = \log P(v) = \sum_{i=1}^n \log Q_i + \log (1 + \sum_{j=1}^n \frac{1-Q_j}{Q_j}), \tag{13}
$$
{{< /math >}}

我们希望在满足 {{< math >}}$\sum_i Q_i \geq n - 1${{< /math >}} 且 {{< math >}}$Q_i \in (0, 1]${{< /math >}} 的条件下求它的最小值。

定义其拉格朗日函数

{{< math >}}
$$
L(Q_1, Q_2, \dots, Q_n, \lambda) = f + \lambda (n-1-\sum_i Q_i). \tag{14}
$$
{{< /math >}}

我们有

{{< math >}}
$$
\frac{\partial L}{\partial \lambda} = n - 1 - \sum_i Q_i, \tag{15}
$$
{{< /math >}}

以及

{{< math >}}
$$
\frac{\partial L}{\partial Q_i} = \frac{1}{Q_i}(1 - \frac{1}{Q_i(1-n + \sum_{k=1}^{n}\frac{1}{Q_k})}) - \lambda. \tag{16}
$$
{{< /math >}}

下面我们来求 {{< math >}}$L(Q_1, Q_2, \dots, Q_n, \lambda)${{< /math >}} 的极值。

**(1)** 若 {{< math >}}$\lambda = 0${{< /math >}}，则对 {{< math >}}$\forall i${{< /math >}} 有 {{< math >}}$\frac{\partial f}{\partial Q_i} = 0${{< /math >}}，于是

{{< math >}}
$$
\forall i, Q_i = \frac{1}{(1 - n + \sum_{k=1}^{n} \frac{1}{Q_k})}. \tag{17}
$$
{{< /math >}}

这表明 {{< math >}}$\forall i, Q_i = 1${{< /math >}}，因此 {{< math >}}$f = 0${{< /math >}} 且 {{< math >}}$P(v) = 1${{< /math >}}。

**(2)** 若 {{< math >}}$\lambda \neq 0${{< /math >}}，{{< math >}}$\exists i${{< /math >}}，则我们有对 {{< math >}}$\forall i${{< /math >}}，{{< math >}}$\frac{\partial f}{\partial Q_i} = \lambda${{< /math >}} 且 {{< math >}}$\sum_{i=1}^n = n - 1${{< /math >}}。换言之，我们有

{{< math >}}
$$
\forall i, j, \frac{\partial f}{\partial Q_i} = \frac{\partial f}{\partial Q_j} = \lambda. \tag{18}
$$
{{< /math >}}

定义函数

{{< math >}}
$$
h(Q_i) = \frac{\partial f}{\partial Q_i} = \frac{1}{Q_i}(1 - \frac{1}{Q_i(1-n + \sum_{k=1}^{n}\frac{1}{Q_k})}). \tag{19}
$$
{{< /math >}}

我们可以将该函数改写为

{{< math >}}
$$
h(Q_i) = \frac{1}{Q_i}(1 - \frac{1}{1 + AQ_i}), \tag{20}
$$
{{< /math >}}

其中 {{< math >}}$A = 1 - n + \sum_{k\neq i} \frac{1}{Q_k} \geq 1 - n + \frac{(n-1)^2}{n-1-Q_i} &gt; 0${{< /math >}}。
因此，{{< math >}}$h(x)${{< /math >}} 在其定义域内是一个单调函数。

这表明 {{< math >}}$\forall i, j${{< /math >}}，{{< math >}}$Q_i = Q_j = \frac{n-1}{n}${{< /math >}}，此时 {{< math >}}$P(v) = \frac{2n-1}{n-1}(1 - \frac{1}{n})^n${{< /math >}}。

下面我们列出 {{< math >}}$f${{< /math >}} 的边界点。

**(1)** {{< math >}}$\forall i${{< /math >}}，{{< math >}}$Q_i = 1${{< /math >}}，此即第一个极值点。

**(2)** {{< math >}}$\exists i${{< /math >}} 使得 {{< math >}}$Q_i${{< /math >}} 趋近于 0。由于约束 {{< math >}}$\sum Q_i \geq n - 1${{< /math >}}，其他各个 {{< math >}}$Q${{< /math >}} 都趋近于 1。我们有 {{< math >}}$\lim_{Q_i \rightarrow 0} f = 0${{< /math >}} 且 {{< math >}}$P(v) = 1${{< /math >}}。

综上，函数 {{< math >}}$f${{< /math >}} 的最大值点为 {{< math >}}$\forall i${{< /math >}}，{{< math >}}$Q_i = \frac{n - 1}{n}${{< /math >}}，此时

{{< math >}}
$$
P(v) = \frac{2n-1}{n-1}(1 - \frac{1}{n})^n, \tag{21}
$$
{{< /math >}}

以及

{{< math >}}
$$
P_{G \sim M(Q)}(G \notin \mathcal{G}_n) = 1- P(v) = 1 - \frac{2n-1}{n-1}(1 - \frac{1}{n})^n. \tag{22}
$$
{{< /math >}}

**讨论**：值得注意的是，{{< math >}}$\mathcal{G}_n${{< /math >}} 恰好就是 {{< math >}}$\mathcal{G}_n${{< /math >}} 中任意一个图的 MEC。
因此，{{< math >}}$P_{G \sim M(Q)}(G \notin \mathcal{G}_n)${{< /math >}} 表示从 {{< math >}}$M(Q)${{< /math >}} 中采样得到的图不正确的概率。
这表明，即便「节点-边」模型已被完美训练以预测 {{< math >}}$M(Q)${{< /math >}}，它仍可能存在 0.2642 这一不可避免的错误率。

## A7. 实验设置

**基线方法。** 为了展示所提出框架的有效性与优越性，我们从多个类别中选取了若干具有代表性的基线方法进行比较。PC 算法是一种基于条件独立性检验的经典基于约束的因果发现算法，这里选用其带并行化优化的版本 (<a href="#ref-17">Le et al., 2016</a>)。GES 是一种经典的基于评分的贪婪等价搜索算法，也被纳入比较 (<a href="#ref-3">Chickering, 2002</a>)。对于连续优化方法，我们与 NOTEARS (<a href="#ref-43">Zheng et al., 2018</a>)（一种有代表性的基于梯度的优化方法）以及 GOLEM (<a href="#ref-25">Ng et al., 2020</a>)（被视为 NOTEARS 的更高效变体）进行比较。对于基于神经网络的优化算法，我们与 DAG-GNN (<a href="#ref-40">Yu et al., 2019</a>)（一种基于图神经网络的优化算法）以及 GRAN-DAG（一种使用神经网络建模的基于梯度的算法 (<a href="#ref-15">Lachapelle et al., 2020</a>)）进行比较。对于基于 DNN 的 SCL 方法，我们与 AVICI 进行比较，它是与我们最相关的工作，并被视为当前最先进的方法 (<a href="#ref-21">Lorch et al., 2022</a>)。

**实现细节。** 除 SCL 方法（即 SLdisco 与 AVICI）外，各基线方法均采用 gCastle (<a href="#ref-42">Zhang et al., 2021</a>) 中的实现。对于 PC 算法，我们在条件独立性检验中采用 Fisher-Z 变换，显著性阈值取 0.05，这是统计分析以及当前 PC 实现中的常见选择 (<a href="#ref-42">Zhang et al., 2021</a>; <a href="#ref-44">Zheng et al., 2024</a>)。在 GES 实验中，我们的图选择准则是高斯贝叶斯信息准则（Bayesian Information Criterion, BIC），具体为 {{< math >}}$l_\infty${{< /math >}} 惩罚高斯似然评分。该准则在原论文 (<a href="#ref-3">Chickering, 2002</a>) 中即被使用，并且至今仍是文献中广受青睐的变体。对于 NOTEARS，我们遵循官方实现的设置，将其配置为最多 100 个对偶上升步，边丢弃阈值为 0.3。对于缺乏特定默认设置的超参数（例如 L1 惩罚与损失函数类型），我们默认采用 gCastle <a href="#ref-42">Zhang et al. (2021)</a>所使用的设置，即 L1 惩罚取 0.1、损失函数取 L2 损失。对于 DAG-GNN，我们直接采用其原始实现中的超参数设置，以保证与既有基准的一致性。对于 GOLEM 与 GRAN-DAG，我们同样使用 gCastle (<a href="#ref-42">Zhang et al., 2021</a>) 的默认设置。需要注意的是，CSIvA 模型 (<a href="#ref-13">Ke et al., 2023</a>) 也是一种密切相关的方法，但由于其相关代码无法获取，且它要求以干预数据作为输入，因此未纳入比较。SLdisco <a href="#ref-28">Petersen et al. (2023)</a>的原始实现是用 R 开发的。为了更好地兼容我们的数据生成与评估工作流，我们使用 PyTorch 重新实现了该模型。原始的 AVICI 模型 (<a href="#ref-21">Lorch et al., 2022</a>) 不支持离散数据。因此，在离散数据上使用 AVICI 时，我们用一个嵌入层替换了它的第一个线性层。

**合成数据。** 我们从多种随机图模型中随机生成随机图。对于连续数据，遵循先前工作 (<a href="#ref-21">Lorch et al., 2022</a>)，采用埃尔德什-雷尼（Erdős-Rényi, ER）模型与无标度（Scale-Free, SF）模型作为训练图分布 {{< math >}}$p(G)${{< /math >}}。在我们的实验中，训练图的度在 1、2、3 之间随机变化。对于测试图分布，使用沃茨-斯特罗加茨（Watts-Strogatz, WS）模型与随机块模型（Stochastic Block Model, SBM），其参数与先前论文 (<a href="#ref-21">Lorch et al., 2022</a>) 中的设置保持一致。用于连续数据的所有合成图均包含 30 个节点。沃茨-斯特罗加茨（WS）图的格点维度从 {{< math >}}$\{2, 3\}${{< /math >}} 中采样，由此得到约 4.92 的平均度。随机块模型（SBM）图的平均度设为 2，同样遵循上述论文中的设置。对于离散数据，使用 11 节点的图。SF 用作训练图分布 {{< math >}}$p(G)${{< /math >}}，ER 用于测试。合成训练数据是实时生成的，训练过程不会重复使用相同的数据。所有合成测试数据集均包含 100 个图，我们报告这 100 个图上各指标的平均值，以全面反映性能。

在从图到连续数据的前向采样过程中，线性高斯机制与一般非线性机制都被采用。具体而言，遵循先前论文 (<a href="#ref-21">Lorch et al., 2022</a>)，一般非线性数据使用随机傅里叶函数机制生成。

在合成离散数据集时，遵循先前论文 (<a href="#ref-6">Dai et al., 2023</a>; <a href="#ref-22">Ma et al., 2022</a>)，使用伯努利分布。

{{< figgrid caption="**图 A5**：Node-Edge 模型、SiCL-no-PF 与 SiCL 的架构对比示意图。" >}}
paire/abl.png | 100
{{< /figgrid >}}


**更多实现细节与计算资源。** 两个网络模块（即 SPN 与 VPN）使用 Adam 优化器以默认超参数进行优化。遵循先前工作 (<a href="#ref-21">Lorch et al., 2022</a>)，训练批大小设为 20。所有经典算法在一块 AMD EPYC 7V13 CPU 上运行，基于 DNN 的方法在 Nvidia 1080Ti、A40 和 A100 GPU 上运行。在 30 节点训练集上以批大小 20 训练 SiCL 大约需要 60GB 内存，在 11 节点训练集上训练大约需要 20GB 内存。学习率为 {{< math >}}$3 \times 10^{-4}${{< /math >}}。批大小为 15，DNN 模型默认训练 {{< math >}}$1.2 \times 10^{5}${{< /math >}} 个批次。

**表 A5**：**SiCL 与其他方法的总体比较**。SiCL 方法给出的是三次运行的平均性能结果。GES 在 WS-L-G 上每个图耗时超过 24 小时，而 SLdisco 不适用于非线性高斯数据，因此这些结果未被列入。

| 数据集 | 方法 | 骨架预测 s-F1 ↑ | 骨架预测 s-Acc. ↑ | 骨架预测 s-AUC ↑ | 骨架预测 s-AUPRC ↑ | CPDAG 预测 v-F1 ↑ | CPDAG 预测 o-F1 ↑ | CPDAG 预测 SHD ↓ |
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
| | GRAN-DAG | 22.6 | 85.9 | N/A | N/A | 13.8 | 14.4 | 64.7 |
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

**表 A6**：完整的消融实验结果。

| 数据集 | 方法 | s-F1 ↑ | s-Acc. ↑ | s-AUC ↑ | s-AUPRC ↑ | v-F1 ↑ | o-F1 ↑ | SHD ↓ |
|---|---|---|---|---|---|---|---|---|
| WS-L-G | SiCL-Node-Edge | 39.9 | 74.0 | 71.5 | 62.2 | 28.2 | 35.8 | 119.2 |
| | SiCL-no-PF | 42.4 | 74.4 | 72.8 | 63.5 | 30.5 | 37.9 | 118.4 |
| | SiCL | **44.7** | **75.3** | **73.7** | **65.4** | **32.0** | **38.5** | **116.1** |
| SBM-L-G | SiCL-Node-Edge | 84.3 | 96.2 | 98.1 | 92.7 | 79.1 | 81.6 | 17.7 |
| | SiCL-no-PF | 85.5 | **96.4** | **98.3** | 93.3 | 79.4 | 82.2 | 17.3 |
| | SiCL | **85.8** | **96.4** | **98.3** | **93.4** | **80.6** | **82.7** | **17.1** |

## A8. 额外实验结果

{{< figgrid caption="**图 A6**：v-结构预测网络在 WS-LG 与 SBM-LG 上训练过程中测试性能的变化趋势。" >}}
paire/v_struc_ws_font.png | 45 | **(a)** WS-LG
paire/v_struc_sbm_font.png | 45 | **(b)** SBM-LG
{{< /figgrid >}}


### A8.1. v-结构预测网络的有效性

图 A6 展示了 v-结构预测模型在训练过程中于 SBM 和 WS 随机图上的测试性能变化趋势。在该模型中，特征提取器 {{< math >}}$FE${{< /math >}} 是从骨架预测模型微调而来的。性能上升迅速，仅经过最初几个轮次就达到了相对较高的水平。这说明我们的 v-结构预测网络有能力预测 v-结构，并表明来自骨架预测模型的预训练成对特征既有效又具备泛化能力。

### A8.2. 关于成对表示有效性的更多证据

为了进一步支持使用成对表示的有效性，我们在不同的训练数据集与测试数据集上给出额外的实验结果，包括 ER-L-G、SF-L-G、ER-RFF-G 和 SF-RFF-G。

在模型方面，我们将 SiCL 与一个不使用成对表示的变体（即 SiCL-no-PF）进行比较。

结果见表 A7。在几乎所有对比中，带有成对表示的模型都优于相应的基线模型，进一步验证了在模型中使用成对表示的有效性。

**表 A7**：关于成对表示有效性的更多性能对比。

| 训练数据集 | 测试数据集 | 方法 | s-F1↑ | s-AUC↑ | s-AUPRC↑ | s-Acc.↑ |
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

### A8.3. 关于 DAG 预测的额外对比

我们给出在 DAG 预测任务上与 AVICI 基线方法的一项额外对比。由于 SiCL 预测的是 CPDAG，并不直接产生 DAG 预测，我们用从 SiCL 所预测的 CPDAG 中推断出的边方向来修正 AVICI 的 DAG 预测。表 A8 汇总的结果表明，引入由 CPDAG 推断出的边方向能够改善 DAG 预测的各项指标。这进一步确认了我们方法的有效性与通用性，即便是在以 DAG 指标为核心的任务上也是如此。

**表 A8**：关于 DAG 预测的额外对比。

| 方法 | 数据集 | F1 分数↑ | AUC↑ | AUPRC↑ | Acc.↑ |
|---|---|---|---|---|---|
| AVICI | WS-L-G | 38.4 | 86.3 | 57.7 | 85.9 |
| SiCL-Corrected AVICI | WS-L-G | 35.8 | 87.2 | 60.5 | 86.2 |
| AVICI | SBM-L-G | 78.1 | 95.8 | 80.5 | 97.3 |
| SiCL-Corrected AVICI | SBM-L-G | 81.3 | 98.7 | 90.8 | 97.8 |

### A8.4. 与自回归模型在推理时间开销上的对比

为验证自回归模型由于其推理运行次数关于变量数量呈平方级数量而具有相对较高的时间开销，我们复现了一个有代表性的自回归模型（即 CSIvA (<a href="#ref-13">Ke et al., 2023</a>)）的网络架构，并将 SiCL 与之进行比较。
我们对两个模型使用相同的随机输入，并不断增加变量数量。
结果见图 A7。
自回归模型的时间开销呈现快速增长的趋势，且远高于 SiCL 的开销，验证了我们分析的正确性。

{{< figgrid caption="**图 A7**：自回归模型与 SiCL 在推理时间开销上的对比。" >}}
paire/inference_time_costs_auto_font.png | 50
{{< /figgrid >}}


### A8.5. 训练数据多样性与模型泛化能力

我们给出实验证据，凸显训练数据多样性对模型泛化能力的显著贡献，即便是应用于分布外（OOD）数据集时也是如此。
为说明这一点，我们训练了两个 SiCL 模型：一个在 SF 与 ER 的合并数据集上训练，另一个仅在 SF 数据集上训练。
这两个模型的性能对比详见表 A9。
在 ER 与 SF 合并数据集上训练的模型表现出明显更好的性能，不仅在 ER 数据集上如此，在另外两个 OOD 数据集上同样如此，而在 SF 数据集上的性能仅有轻微下降。
这些发现表明，提升训练数据的多样性会相应地改善模型的泛化能力，并使其在全新的 OOD 数据集上保持稳健的性能。

**表 A9**：不同训练数据多样性的 SiCL 模型在骨架预测上的对比。

**(a) 在 ER 与 SF 上共同训练的模型**

| 测试数据集 | s-F1↑ | s-AUC↑ | s-AUPRC↑ | s-Acc.↑ |
|---|---|---|---|---|
| WS-L-G | 36.3 | 70.6 | 60.6 | 73.3 |
| SBM-L-G | 78.1 | 96.8 | 88.1 | 94.8 |
| ER-L-G | 80.7 | 96.0 | 89.2 | 94.7 |
| SF-L-G | 84.7 | 98.5 | 93.6 | 95.5 |

**(b) 在 SF 上训练的模型**

| 测试数据集 | s-F1↑ | s-AUC↑ | s-AUPRC↑ | s-Acc.↑ |
|---|---|---|---|---|
| WS-L-G | 40.1 | 63.0 | 46.1 | 63.5 |
| SBM-L-G | 64.3 | 91.7 | 72.9 | 90.9 |
| ER-L-G | 67.1 | 90.4 | 73.9 | 90.8 |
| SF-L-G | 87.8 | 98.9 | 95.3 | 96.1 |

### A8.6. 训练图数量的变化

我们分析了训练图数量的变化对骨架预测任务性能的影响。图 A8 所示的结果呈现出一个清晰的趋势：模型性能随训练数据集规模的扩大而提升。这一趋势凸显了我们的方法在获得更大规模数据集时达到更高准确率的潜力。

{{< figgrid caption="**图 A8**：训练图数量变化时的模型性能。" >}}
paire/wstrainingsize_font.png | 45 | **(a)** WS 数据集
paire/sbmtrainingsize_font.png | 45 | **(b)** SBM 数据集
{{< /figgrid >}}


### A8.7. 样本量的变化

我们在测试时评估了 SiCL 在每张图具有不同数量观测样本（100、200、……、1000）情况下的表现。骨架预测任务与 CPDAG 预测任务的结果如图 A9 所示。可以明显看出，模型性能随样本量的增大而提升。这些一致的上升趋势表明，SiCL 表现出良好的稳定性，对样本量的变化并不过分敏感。

{{< figgrid caption="**图 A9**：性能随样本量变化的趋势。" >}}
paire/ws1_font.png | 44 | **(a)** WS 图上骨架预测任务性能随样本量变化的趋势。
paire/ws2_font.png | 44 | **(b)** WS 图上 CPDAG 预测任务性能随样本量变化的趋势。
paire/sbm1_font.png | 44 | **(c)** SBM 图上骨架预测任务性能随样本量变化的趋势。
paire/sbm2_font.png | 44 | **(d)** SBM 图上 CPDAG 预测任务性能随样本量变化的趋势。
{{< /figgrid >}}


### A8.8. 边密度的变化

我们在一系列不同边密度的测试图上评估了 SiCL，采用的是 SBM 数据集，因为它允许直接设定平均边密度。结果如图 A10 所示。显然，随着边密度的增加，任务变得更加困难。然而，性能的下降并不突兀，这说明 SiCL 的性能在各种边密度下保持相对稳定，从而印证了它的通用性。

{{< figgrid caption="**图 A10**：性能随边密度变化的趋势。" >}}
paire/sbmskeletondensity_font.png | 45 | **(a)** SBM 图上骨架预测任务性能随边密度变化的趋势。
paire/sbmcpdagdensity_font.png | 45 | **(b)** SBM 图上 CPDAG 预测任务性能随边密度变化的趋势。
{{< /figgrid >}}


### A8.9. 在测试图规模上的通用性

我们从分析的角度考察了 SiCL 模型应用于更大的 WS-L-G 图时的性能。
需要强调的是，这些模型最初是在包含 30 个顶点的图上训练的，因此就图规模而言，该任务处于分布外（OOD）设定之下。
为了提供一个参照，我们纳入了 PC 算法的结果作为基线方法进行对比。
这些结果可见于表 A10。
尽管处于 OOD 条件下，SiCL 依然保持稳健的性能，这进一步印证了它的可扩展性以及模型在不同图规模下的普遍适用性。

**表 A10**：不同图规模下的性能对比。

| 指标 / 规模 | s-F1↑ / 50 | s-F1↑ / 70 | s-F1↑ / 100 | v-F1↑ / 50 | v-F1↑ / 70 | v-F1↑ / 100 | o-F1↑ / 50 | o-F1↑ / 70 | o-F1↑ / 100 |
|---|---|---|---|---|---|---|---|---|---|
| PC | 17.7 | 14.8 | 10.6 | 6.4 | 5.0 | 3.7 | 7.0 | 5.6 | 4.0 |
| SiCL | **41.6** | **37.4** | **28.3** | **34.9** | **30.7** | **22.6** | **37.9** | **33.7** | **24.8** |

### A8.10. 无环性

我们提供了实证证据来支持环在预测结果中十分罕见这一点。表 A11 中给出的实验数据证实，即使不做任何移除环的后处理，在预测得到的 CPDAG 中也很少观察到环。

**表 A11**：在不做移除环后处理的情况下，CPDAG 预测结果中环的统计。

| 数据集 | WS-L-G | SBM-L-G |
|---|---|---|
| 含环图的比例 | 0.66 ± 0.66 % | 0.00 ± 0.00 % |

## 参考文献

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

## 译者说明

本页中文译文由 **Claude Opus 5 + ultracode** 翻译，**仅供参考，对内容正确性不作保证**。译文力求与原文逐段对应，公式、表格与数值均按原文照录；专有名词（数据集名、模型名、评价指标、人名）保留英文原文。若中英文表述存在出入，一律以英文原文为准。

- 英文原文 PDF：[https://arxiv.org/pdf/2502.10883](https://arxiv.org/pdf/2502.10883)
- 英文全文（网页版）：[Full Text]({{< relref "/publication/paire-en" >}})
- 论文主页：[Learning Identifiable Structures Helps Avoid Bias in DNN-based Supervised Causal Learning]({{< relref "/publication/paire" >}})
- 代码仓库：[https://github.com/microsoft/reliableAI/tree/main/causal-kit/SiCL](https://github.com/microsoft/reliableAI/tree/main/causal-kit/SiCL)
