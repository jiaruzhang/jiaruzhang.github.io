---
title: "基于掩码视觉-语言-动作扩散的高效可解释端到端自动驾驶"
subtitle: "Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion"

summary: "基于掩码视觉-语言-动作扩散的高效可解释端到端自动驾驶（IROS 2026）论文中文全文翻译，含全部公式、表格与插图。"

date: '2026-09-27T00:00:00Z'
publishDate: '2026-07-16T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/mvlad-ad/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/mvlad-ad/cn/'

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
本页是论文 **《Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion》**（IROS 2026）的中文全文翻译，正文、公式、表格与插图均与原文一一对应。
译文仅供参考，如需引用请以[英文原文 PDF](/publication/mvlad-ad/mvlad-ad.pdf) 为准（[论文主页]({{< relref "/publication/mvlad-ad" >}})）。也可查看[英文全文网页版]({{< relref "/publication/mvlad-ad-en" >}})。
{{% /callout %}}

{{< toc title="目录" >}}

Jiaru Zhang, Manav Gagvani, Can Cui, Juntong Peng, Ruqi Zhang, and Ziran Wang

> J. Zhang、M. Gagvani 与 C. Cui 对本工作贡献相同。
>
> 作者单位为普渡大学（Purdue University），West Lafayette, IN 47907, USA。
> 通讯作者：Jiaru Zhang，邮箱：jiaru@purdue.edu。
>
> 本工作使用了普渡大学的 Anvil <a href="#ref-1">[1]</a>，其算力来自 Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) 计划 <a href="#ref-2">[2]</a>的 CIS251316 号分配额度；该计划由 National Science Foundation 资助项目 #2138259、#2138286、#2138307、#2137603 和 #2138296 支持。

## 摘要

大语言模型（Large Language Model, LLM）与视觉语言模型（Vision-Language Model, VLM）已成为端到端自动驾驶颇具前景的候选方案。然而，这类模型通常在推理时延、动作精度与可解释性方面面临挑战。
现有的自回归方法受困于缓慢的逐词元生成，而先前基于扩散的规划器往往依赖冗长的通用语言词元，缺乏显式的几何结构。
在本工作中，我们提出面向自动驾驶的掩码视觉-语言-动作扩散（Masked Vision-Language-Action Diffusion for Autonomous Driving, MVLAD-AD），这是一个新颖的框架，旨在借助掩码视觉-语言-动作扩散模型弥合高效规划与语义可解释性之间的鸿沟。
与那些把动作强行塞进语言空间的方法不同，我们引入了一种离散动作词元化策略，从真实世界的驾驶分布中构建出一个由运动学可行路径点组成的紧凑码本。
此外，我们提出几何感知嵌入学习，以确保隐空间中的嵌入能够逼近物理几何度量。
最后，我们引入动作优先解码策略，优先完成轨迹生成。
在 nuScenes 及其衍生基准上的大量实验表明，MVLAD-AD 取得了更优的效率，并在规划精度上超越了最先进的自回归与扩散基线方法，同时能够提供高保真且可解释的语义推理。
代码已开源：<https://github.com/lan-qing/MVLAD-AD>。

## I. 引言

自动驾驶的范式正在从模块化流水线转向端到端学习系统，由一个统一的模型直接把原始传感器输入映射为驾驶决策。
然而，传统的端到端模型往往表现为「黑箱」，可解释性有限，在复杂场景中泛化能力也较差。
近年来，大语言模型（Large Language Model, LLM）与视觉语言模型（Vision-Language Model, VLM）成为自动驾驶颇具前景的候选方案，能够对复杂交通场景以及人机交互进行推理。
通过把驾驶表述为语言建模问题，这类模型得以利用预训练获得的丰富世界知识，从而提升自动驾驶性能 <a href="#ref-3">[3]</a>, <a href="#ref-4">[4]</a>, <a href="#ref-5">[5]</a>。

尽管前景可观，当前面向自动驾驶的 LLM/VLM 模型仍面临三大挑战：推理时延、动作精度与可解释性。
现有方法大多依赖自回归生成。
自回归生成虽然强大，但如图 1(A) 所示，逐词元的推理（前向）过程对时延敏感的自动驾驶而言慢得难以接受。
此外，在语言空间中处理连续动作会得到冗长的词元表示，描述精确的轨迹需要过长的序列长度，限制了规划框架的推理效率。
最后，现有模型往往难以在给出规划的同时给出连贯的语义推理。依赖独立的事后解释模块，常常无法使语义推理与驾驶动作保持一致。

扩散语言模型作为一种强大的非自回归替代方案已经出现，它支持并行解码，可缓解推理时延瓶颈 <a href="#ref-6">[6]</a>。
在自动驾驶领域，ViLaD <a href="#ref-7">[7]</a>率先把这一范式用于高效规划，并在规划精度与推理速度两方面都取得了显著提升。然而，如图 1(B) 所示，ViLaD 依赖冗长的语言词元来表示连续轨迹，带来了表示冗余。
而且，它只预测驾驶决策，不给出任何语义推理解释。
这些因素同时限制了当前基于扩散语言模型的规划器的规划性能与透明度。

{{< figgrid caption="**图 1**：研究动机。基于 LLM/VLM 的自动驾驶范式对比。(A) 自回归 VLM 逐 token 顺序生成，延迟很高。(B) 标准的扩散语言模型可以并行生成，但仍工作在冗长的语言空间中。(C) 本文提出的视觉—语言—动作扩散模型引入了一个表达力强的码本，把连续动作映射为紧凑的离散 token。这一设计使动作空间与语言空间可以同时并行生成，大幅缩短序列长度，取得最快的推理速度。" >}}
mvlad-ad/vilad-vla-motivation.png | 100
{{< /figgrid >}}


与此同时，视觉-语言-动作（Vision-Language-Action, VLA）模型 <a href="#ref-8">[8]</a>, <a href="#ref-9">[9]</a>在具身智能中把感知、推理与控制统一到单一骨干网络内，取得了显著成功。
通过把控制信号作为语言词表中的专用词元来整合，VLA 促成了深层的语义接地，使物理机动动作能够与高层语言推理对齐。
然而，标准 VLA 架构主要是为机器人操作任务量身定制的，如何最好地把 VLA 建模用于端到端自动驾驶仍是一个开放问题。
若简单粗暴地对驾驶动作空间做离散化，会导致动作词元数量爆炸，使搜索空间难以处理，并阻碍高效学习。

在本工作中，我们提出面向自动驾驶的掩码视觉-语言-动作扩散（Masked Vision-Language-Action Diffusion for Autonomous Driving, MVLAD-AD），这是一个旨在同时实现高效规划与语义可解释性的全新框架。
与那些把连续动作强行塞进冗长语言空间的方法相比，我们引入了一种离散动作词元化策略，从真实世界驾驶分布中构建一个由运动学可行路径点组成的紧凑码本，从而有效压缩动作搜索空间。
我们进一步提出几何感知嵌入学习，使隐空间的嵌入与几何度量相对齐。
这些组件被集成到一个掩码 VLA 扩散 Transformer 中，用以建模驾驶动作与语言解释的联合概率。
为化解时延与可解释性之间的矛盾，我们采用动作优先解码策略，在推理时优先生成轨迹。
在 nuScenes 及其衍生推理数据集（包括 Nu-X 与 nuScenes-QA）上的大量实验表明，MVLAD-AD 在基于 VLM 的规划器中取得了很强的规划性能，同时还能提供高保真、可解释的语义推理。

综上，我们的贡献如下：

- 我们提出 MVLAD-AD，一个全新的端到端掩码 VLA 扩散框架，在保留语义推理能力的同时实现了高效的端到端自动驾驶。
- 我们设计了离散动作词元化，把轨迹映射为紧凑的动作词元；并提出几何感知嵌入学习，在嵌入空间中强制保持度量一致性。我们还进一步引入动作优先解码策略，以实现低时延规划。
- 我们开展了大量实验，结果表明 MVLAD-AD 在 nuScenes 规划基准上优于各基线方法，同时保持更优的推理速度，并能生成连贯、与物理相符的语言解释。

## II. 相关工作

### II-A. 基于大语言模型的端到端自动驾驶

端到端系统是自动驾驶领域新近兴起的一种范式，它把传感输入与自车状态直接映射为规划轨迹和／或底层控制动作 <a href="#ref-10">[10]</a>。受 LLM 显著进展的推动，近期研究开始探索用于端到端自动驾驶的生成式与自回归形式化建模。Chen 等人 <a href="#ref-11">[11]</a>采用自回归生成策略，顺序预测未来的场景表示以及相应的控制动作。沿着这一范式，Huang 等人 <a href="#ref-12">[12]</a>进一步把自回归建模扩展到 GPT 风格的架构，将未来驾驶场景表示并预测为词元序列。

随着多模态基础模型（通常为 VLM）的快速发展，它们在端到端自动驾驶中的应用也显著加速。早期工作验证了把以语言为中心的模型直接引入闭环驾驶系统的可行性。Shao 等人 <a href="#ref-13">[13]</a>提出了最早在闭环设置下运行的多模态 LLM 驱动端到端驾驶框架之一。他们的系统以导航目标、多模态传感观测以及辅助文本指令作为输入，直接输出底层控制指令。类似地，Wang 等人 <a href="#ref-14">[14]</a>探索了把 LLM 集成进自动驾驶流水线的方法，将基于语言的决策与高层车辆控制对齐，从而支持多模态输入并提供可解释的决策依据。

除了直接生成控制量之外，另一条并行的研究路线聚焦于利用基础模型的推理能力来支撑决策与场景理解。Xu 等人 <a href="#ref-5">[5]</a>提出了一个基于问答的驾驶框架，把驾驶决策形式化为结构化查询。在此基础上，Sima 等人 <a href="#ref-15">[15]</a>提出了基于图的视觉问答（visual question answering, VQA）范式，其中逻辑上相互依赖的问答对刻画了交通场景中复杂的关系推理。为了进一步增强显式的推理监督，Wang 等人 <a href="#ref-16">[16]</a>构建了一个思维链（chain-of-thought, CoT）驾驶数据集，同时标注中间推理步骤与最终决策，并提出了一个向推理过程中注入多个任务专用可学习查询的基线方法。随后，Hwang 等人 <a href="#ref-17">[17]</a>与 Xing 等人 <a href="#ref-18">[18]</a>引入了更为完整的多模态推理系统，它们在多样化的驾驶环境中展现出很强的泛化能力、鲁棒性与可扩展性。然而，大多数方法都建立在自回归生成范式之上，由于顺序解码和从左到右的生成模式，推理（前向）过程相对缓慢；同时它们主要通过只预测下一步的动作或帧来建模短期动态。

### II-B. 可解释的自动驾驶

长期以来，可解释性一直被视为基于学习的自动驾驶系统的一项关键挑战。随着深度神经网络越来越多地被用于感知、决策与控制，其「黑箱」特性使人们难以理解特定驾驶动作为何被产生。这种透明度的缺失直接影响系统验证、安全保障、信任建立与人机协作，从而催生了大量面向自动驾驶的可解释人工智能（explainable artificial intelligence, XAI）方法研究 <a href="#ref-19">[19]</a>, <a href="#ref-20">[20]</a>。

一大类可解释性方法关注模型无关的解释技术 <a href="#ref-21">[21]</a>, <a href="#ref-22">[22]</a>, <a href="#ref-23">[23]</a>。这些方法通过估计特征相关性或归因来解释单个预测，而不需要访问模型的内部结构。与此并行，也有一些模型相关的技术被提出用于分析内部表示，包括基于梯度的归因 <a href="#ref-24">[24]</a>、显著性可视化 <a href="#ref-25">[25]</a>以及基于注意力的解释 <a href="#ref-26">[26]</a>。

在自动驾驶领域内，注意力机制被广泛用于可视化驾驶场景中对控制决策具有强烈影响的空间区域 <a href="#ref-27">[27]</a>。为了进一步增强可解释性，基于注意力的驾驶模型被与自然语言生成相结合，使系统能够同时输出控制动作以及相应的、以视觉观测为依据的文本解释 <a href="#ref-28">[28]</a>。这一研究路线随后通过引入语言结构与解码约束得到扩展，从而为驾驶行为生成更连贯、信息量更丰富的文本解说 <a href="#ref-29">[29]</a>。然而，已有研究表明，仅靠基于注意力的解释不足以忠实反映模型的语义推理过程 <a href="#ref-30">[30]</a>。这一局限促成了混合式的可解释性策略，它们联合利用注意力权重、编码特征、梯度与类激活信号，以更好地逼近基于 Transformer 的架构背后的决策逻辑 <a href="#ref-31">[31]</a>。

近来，LLM 与 VLM 为自动驾驶的可解释性带来了范式转变，它们把语言作为表达感知、推理与决策依据的显式媒介。例如，DriveLM <a href="#ref-15">[15]</a>这类语言驱动的驾驶系统，以及基于 LLM 的闭环驾驶框架 <a href="#ref-13">[13]</a>, <a href="#ref-14">[14]</a>, <a href="#ref-32">[32]</a>, <a href="#ref-33">[33]</a>, <a href="#ref-34">[34]</a>，都能给出文本解释，阐明车辆观测到了什么以及为何采取特定动作。基于问答的形式化建模进一步实现了交互式的可解释性，使模型能够回答关于驾驶场景与决策的结构化查询 <a href="#ref-5">[5]</a>。更为完整的多模态推理框架，包括 EMMA <a href="#ref-17">[17]</a>与 OpenEMMA <a href="#ref-18">[18]</a>，则表明以语言为中心的解释能够泛化到多样化的驾驶场景，同时提升鲁棒性与透明度。

## III. 方法

{{< figgrid caption="**图 2**：MVLAD-AD 总览。(A) 离散动作分词与几何感知嵌入学习：我们从真实数据中构建紧凑的驾驶动作码本，并通过软分配与几何一致性目标学习具有几何感知能力的嵌入空间。(B) 统一的掩码 VLA 扩散：视觉、指令、动作与推理 token 被统一到同一个序列中做掩码生成式建模，“[M]”表示被掩蔽的 token。(C) 训练与推理优化：训练阶段采用两阶段学习策略；推理阶段引入动作优先的解码策略，优先生成轨迹以降低延迟，同时保证推理解释高度忠实于驾驶动作。" >}}
mvlad-ad/vilad-vla-new.png | 100
{{< /figgrid >}}


### III-A. 框架概览

如图 2 所示，MVLAD-AD 将端到端自动驾驶形式化为一个*条件掩码生成式建模*问题。我们从异构模态出发，构建一个统一的离散词元序列 {{< math >}}$\mathbf{x} = [\mathbf{x}^c; \mathbf{x}^g]${{< /math >}}。其中，条件部分 {{< math >}}$\mathbf{x}^c${{< /math >}} 汇聚了多视角视觉词元 {{< math >}}$\mathbf{x}^v${{< /math >}} 与文本指令词元 {{< math >}}$\mathbf{x}^i${{< /math >}}。目标生成部分 {{< math >}}$\mathbf{x}^g${{< /math >}} 则拼接了表示未来轨迹的离散动作词元 {{< math >}}$\mathbf{x}^a${{< /math >}} 与解释该决策的推理词元 {{< math >}}$\mathbf{x}^r${{< /math >}}。我们的目标是通过逆转一个并行的掩码过程 <a href="#ref-6">[6]</a>，学习条件分布 {{< math >}}$p_\theta(\mathbf{x}^g \mid \mathbf{x}^c)${{< /math >}}。

**多模态输入编码。** 为了处理这些异构输入，我们采用专门的编码器把所有信号投影到共享的嵌入空间中，从而保证跨模态的语义对齐：

- **视觉：** 多视角相机图像由预训练的视觉编码器编码，并经由一个可学习的 MLP 投影器映射到 Transformer 的维度。
- **指令：** 文本指令由标准文本词元化器进行词元化并转换为嵌入，用以让模型以特定的驾驶任务为条件。
- **动作：** 我们使用第 III-B 节中详述的离散动作词元化器，把连续路径点映射为离散动作词元；并使用第 III-C 节中详述的几何感知嵌入学习来获得它们的嵌入。
- **推理：** 与指令类似，推理内容同样按文本进行词元化与嵌入。

**统一序列建模。** 与将规划与解释相互隔离的模块化流水线不同，MVLAD-AD 把这些嵌入拼接为单一序列 {{< math >}}$\mathbf{X} = [\mathbf{X}^v; \mathbf{X}^i; \mathbf{X}^a; \mathbf{X}^r]${{< /math >}}，其中 {{< math >}}$\mathbf{X}^v${{< /math >}}、{{< math >}}$\mathbf{X}^i${{< /math >}}、{{< math >}}$\mathbf{X}^a${{< /math >}} 和 {{< math >}}$\mathbf{X}^r${{< /math >}} 分别对应视觉特征、文本指令、动作词元与推理词元的嵌入。这种统一表示使模型能够利用双向注意力机制，在全局上下文中捕捉视觉观测、文本指令、车辆动作与语义推理之间的相互依赖关系。

**掩码生成式 Transformer。** MVLAD-AD 的核心是一个用于掩码扩散生成式建模的、基于 Transformer 的预测器。如第 III-D 节所述，在联合学习动作与推理的训练策略下，该 Transformer 以统一序列为输入，其中 {{< math >}}$\mathbf{x}^a${{< /math >}} 与 {{< math >}}$\mathbf{x}^r${{< /math >}} 的一部分被掩码，模型学习重建这些被掩码的词元。该架构有效地建模了联合分布 {{< math >}}$p_\theta(\mathbf{x}^a, \mathbf{x}^r \mid \mathbf{x}^v, \mathbf{x}^i)${{< /math >}}，从而灵活地支持高效的并行规划与推理生成，具体见第 III-E 节。

### III-B. 驾驶动作词元化

为了弥合连续轨迹规划与离散语言生成之间的模态鸿沟，我们提出一种离散动作词元化策略。在该策略中，轨迹被建模为离散驾驶动作词元的序列。具体而言，我们将轨迹表示为自车局部坐标系下的一系列路径点。令 {{< math >}}$\mathbf{w} = (x, y) \in \mathbb{R}^2${{< /math >}} 表示单个路径点，它刻画了车辆在某个未来时刻相对于自车的位置；令 {{< math >}}$\mathcal{D} = \{\mathbf{w}_1, \mathbf{w}_2, \dots, \mathbf{w}_M\}${{< /math >}} 表示从真实世界驾驶数据集中收集到的全部有效路径点构成的集合，其中 {{< math >}}$M${{< /math >}} 是训练语料中跨所有时域的路径点总数。我们的目标是构建一个由 {{< math >}}$N${{< /math >}} 个代表性路径点组成的紧凑码本 {{< math >}}$\mathcal{C} = \{\mathbf{c}_1, \mathbf{c}_2, \dots, \mathbf{c}_N\}${{< /math >}}，使得任意连续路径点 {{< math >}}$\mathbf{w} \in \mathcal{D}${{< /math >}} 都能以最小误差由某个代表性聚类中心 {{< math >}}$\mathbf{c} \in \mathcal{C}${{< /math >}} 近似。这可以形式化为一个最小化簇内平方和的优化问题，即 {{< math >}}$\mathcal{J} = \sum_{i=1}^{M} \min_{\mathbf{c} \in \mathcal{C}} \| \mathbf{w}_i - \mathbf{c} \|_2^2${{< /math >}}，

其中 {{< math >}}$\| \cdot \|_2${{< /math >}} 表示欧氏距离。借助标准的 K-Means 求解器，我们得到最优的空间聚类中心集合 {{< math >}}$\mathcal{C}${{< /math >}}，它们即为我们的离散动作词元。

在训练与推理过程中，我们使用一个量化器 {{< math >}}$Q(\cdot)${{< /math >}} 把任意连续的预测路径点 {{< math >}}$\hat{\mathbf{w}}${{< /math >}} 映射到码本中最近聚类中心的索引：

{{< math >}}
$$
k = \arg\min_{j \in \{1, \dots, N\}} \| \hat{\mathbf{w}} - \mathbf{c}_j \|_2. \tag{1}
$$
{{< /math >}}

这一离散化把轨迹生成转化为一系列 N 类分类任务，从而将输出空间约束在物理可行的空间基元之上。

### III-C. 几何感知嵌入学习

虽然动作词元化提供了一个紧凑的词元码本，但如果把这些词元视为彼此独立的类别索引并使用随机初始化嵌入，就会丢弃轨迹空间中固有的丰富度量信息。为了让扩散语言模型能够对动作的物理属性进行推理，我们提出一个预训练阶段来学习几何感知嵌入空间。令 {{< math >}}$\mathbf{E} \in \mathbb{R}^{N \times D}${{< /math >}} 表示这 {{< math >}}$N${{< /math >}} 个动作词元的可学习嵌入矩阵。我们希望优化 {{< math >}}$\mathbf{E}${{< /math >}}，使隐空间中的欧氏距离能够近似物理空间中的几何距离。

**软分配与重建。** 为了稳定优化过程并弥合连续输入与离散词元之间的差距，我们在训练中采用温度缩放的软分配机制。给定一个真值路径点 {{< math >}}$\mathbf{w}${{< /math >}}，我们不执行硬查表，而是依据它到码本 {{< math >}}$\mathcal{C}${{< /math >}} 中最近的前 {{< math >}}$K${{< /math >}} 个聚类中心的距离，计算加权嵌入 {{< math >}}$\mathbf{z}${{< /math >}}：

{{< math >}}
$$
\mathbf{z} = \sum_{j \in \mathcal{N}_K(\mathbf{w})} \frac{\exp(-\|\mathbf{w} - \mathbf{c}_j\|_2 / \tau)}{\sum_{l \in \mathcal{N}_K(\mathbf{w})} \exp(-\|\mathbf{w} - \mathbf{c}_l\|_2 / \tau)} \mathbf{E}_j, \tag{2}
$$
{{< /math >}}

其中 {{< math >}}$\mathcal{N}_K(\mathbf{w})${{< /math >}} 表示 {{< math >}}$K${{< /math >}} 个最近聚类中心的索引集合，{{< math >}}$\tau${{< /math >}} 为温度参数。随后，我们使用一个轻量级 MLP 解码器 {{< math >}}$D_\phi: \mathbb{R}^D \to \mathbb{R}^2${{< /math >}} 重建原始坐标，并最小化重建损失 {{< math >}}$\mathcal{L}_{\text{recon}} = \| D_\phi(\mathbf{z}) - \mathbf{w} \|_2^2${{< /math >}}。

**度量对齐目标。** 为了显式地施加几何结构，我们引入两个辅助损失：

1. **几何一致性损失。** 我们要求嵌入空间中的成对距离与物理距离相关联。对于一批样本对 {{< math >}}$(\mathbf{w}_i, \mathbf{w}_j)${{< /math >}}，我们最小化二者归一化距离之间的差异：

   {{< math >}}
   $$
   \mathcal{L}_{\text{geom}} = \mathbb{E}_{i,j} \left[ \left( \frac{\| \mathbf{z}_i - \mathbf{z}_j \|_2}{\bar{d}_z} - \frac{\| \mathbf{w}_i - \mathbf{w}_j \|_2}{\bar{d}_w} \right)^2 \right], \tag{3}
   $$
   {{< /math >}}

   其中 {{< math >}}$\bar{d}_z${{< /math >}} 与 {{< math >}}$\bar{d}_w${{< /math >}} 是该批次内的成对距离中位数，充当鲁棒的缩放因子。

2. **对比聚类损失。** 我们施加有监督对比损失来构造隐空间。对于锚点 {{< math >}}$i${{< /math >}} 及其归一化嵌入 {{< math >}}$\tilde{\mathbf{z}}_i${{< /math >}}，令 {{< math >}}$P(i)${{< /math >}} 表示该批次中被分配到同一聚类中心索引的其他点的集合。该损失将正样本对拉近，同时把负样本推远：

   {{< math >}}
   $$
   \begin{aligned}
           \mathcal{L}_{\text{contra}} = \sum_{i \in \mathcal{B}} \frac{-1}{|P(i)|} \sum_{p \in P(i)} \log \frac{\exp(\tilde{\mathbf{z}}_i \cdot \tilde{\mathbf{z}}_p / \tau_{\text{con}})}{\sum\limits_{a \in A(i)} \exp(\tilde{\mathbf{z}}_i \cdot \tilde{\mathbf{z}}_a / \tau_{\text{con}})},
   \end{aligned} \tag{4}
   $$
   {{< /math >}}

   其中 {{< math >}}$A(i)${{< /math >}} 是该批次中所有其他索引的集合，{{< math >}}$\tau_{\text{con}}${{< /math >}} 为对比温度。

**优化与课程策略。** 最终的目标函数是上述三项损失的加权和。我们采用课程学习策略，以平缓地完成从连续表示到离散表示的过渡。参数 {{< math >}}$K${{< /math >}} 在训练过程中从 16 逐步衰减到 1。这使模型在早期阶段能够通过软插值捕捉局部流形结构，并最终收敛到硬离散映射，从而得到嵌入矩阵 {{< math >}}$\mathbf{E}${{< /math >}}。

### III-D. 训练流程

训练一个统一的 VLA 模型需要学习两种截然不同的能力：精确的规划与高层的语义推理。遵循标准的掩码扩散形式化表述 <a href="#ref-6">[6]</a>，基础训练目标是在被掩码词元上计算的负对数似然：

{{< math >}}
$$
\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, \mathbf{x}_0} \left[ - \sum_{i \in \mathcal{M}_t} \log p_\theta(x_{0,i} | \mathbf{x}_t) \right], \tag{5}
$$
{{< /math >}}

其中 {{< math >}}$\mathcal{M}_t${{< /math >}} 表示第 t 步的掩码索引集合。由于第 III-B 节中引入的离散动作词元代表了一种没有预训练知识基础的新模态，学习它们的分布比标准的语言建模更具挑战性。为此，我们采用两阶段训练策略。

**阶段一：以动作为中心的预热。**
在第一阶段，我们把轨迹生成任务单独隔离出来，以初始化对这一全新动作模态的学习。具体而言，我们将推理词元子序列 {{< math >}}$\mathbf{X}^r${{< /math >}} 从输入序列中完全剔除，构造出一条缩短的序列 {{< math >}}$\mathbf{X}_{\text{stage1}} = [\mathbf{X}^v; \mathbf{X}^i; \mathbf{X}^a]${{< /math >}}。我们对一部分动作词元索引施加掩码，并训练模型仅以视觉上下文与指令上下文为条件来重建它们。这一聚焦的目标迫使模型学习物理动力学以及动作码本的几何结构，从而在不受语言生成任务干扰的情况下建立起鲁棒的运动先验。

**阶段二：VLA 联合微调。** 在第二阶段，我们引入推理任务，此时统一序列同时包含动作词元子序列 {{< math >}}$\mathbf{X}^a${{< /math >}} 与推理词元子序列 {{< math >}}$\mathbf{X}^r${{< /math >}}。我们同时对动作词元索引和推理词元索引施加掩码，训练模型生成完整的输出序列。该阶段将物理动作与语义解释对齐，并利用阶段一中建立的运动先验。

### III-E. 推理流程

为了在低时延规划与语义可解释性这两项相互冲突的需求之间取得平衡，我们采用动作优先解码策略。标准的掩码扩散方法通常采用全局的基于置信度的采样调度，即*整条*序列中置信度分数最高的词元被优先解掩码。然而，这种不加约束的做法可能会以混杂的顺序生成文本词元与动作词元，从而推迟动作的可用时间。

作为替代，我们施加一种模态受限的解掩码策略，优先完成轨迹的确定。设 {{< math >}}$\mathbf{x}_t${{< /math >}} 表示扩散步 t 上的序列，其中同时包含动作词元位置与推理词元位置。在每次迭代中，Transformer 预测器为*所有*被掩码的词元输出概率分布 {{< math >}}$p_\theta(\mathbf{x}_0 \mid \mathbf{x}_t)${{< /math >}}。尽管模型会同时预测整条序列，但在规划被完全确定之前，我们将解掩码候选集严格限制在动作索引之内。具体而言，我们为所有被掩码的动作词元位置 j 计算置信度分数 {{< math >}}$u_j = \max_{k} p_\theta(x_j = k \mid \mathbf{x}_t)${{< /math >}}。随后我们选出置信度分数最高的那部分动作词元进行解掩码，即用预测得到的词元 ID 替换 {{< math >}}$[\texttt{M}]${{< /math >}}，同时让所有推理词元保持在掩码状态。

该策略带来两项关键收益。其一，在最初的若干步中把解码预算完全集中在 {{< math >}}$N_a${{< /math >}} 个动作词元上（其中 {{< math >}}$N_a${{< /math >}} 远小于文本长度），因此相比等待整条序列收敛，轨迹能够被显著更快地最终确定并交付执行。另一方面，一旦动作词元被完全解掩码，它们便作为固定的、完全可观测的条件，服务于后续推理词元的生成。这隐式地强制了语义一致性，因为生成的解释是以一个确定性的未来规划为条件的。

## IV. 实验

我们在高效规划与语言推理两方面对所提出的 MVLAD-AD 框架进行评估。为保证对比的全面性，我们与大量基线方法进行了比较，既包括开源的最先进模型，也包括具有代表性的商用大语言模型。

**数据集与指标。**
我们主要使用 nuScenes 数据集 <a href="#ref-35">[35]</a>进行规划评估。
具体而言，出于效率考虑，我们所有的实验与基线方法对比均在单帧设置下进行。
遵循标准评测协议，我们报告 1 s、2 s、3 s 时域上的 L2 位移误差，以及平均 L2 误差。对于推理任务，我们采用两个专门的数据集：Nu-X <a href="#ref-36">[36]</a>，一个聚焦于解释驾驶决策的数据集；以及 nuScenes-QA <a href="#ref-37">[37]</a>，一个面向自动驾驶场景的大规模视觉问答基准。
对于 Nu-X，我们采用标准的自然语言生成指标，包括 BLEU-4、METEOR、ROUGE-L 和 CIDEr，用以衡量生成解释与真值之间的语义对齐程度。
对于 nuScenes-QA，我们遵循已有文献 <a href="#ref-36">[36]</a>, <a href="#ref-38">[38]</a>的做法，报告零跳（H0）、单跳（H1）以及总体问题（All）上的准确率。

**基线方法。**
我们按照任务类型与模型架构对基线方法进行分类：

*1）规划基线方法：* 我们将 MVLAD-AD 与一系列具有代表性的、基于 VLM 的端到端驾驶方法进行比较。对于自回归 VLM，我们纳入了在不同骨干网络上微调得到的模型：LLaVA-1.6-Mistral-7B、Llama-3.2-11B-Vision-Instruct 和 Qwen2-VL-7B-Instruct。我们还与 DriveLM <a href="#ref-15">[15]</a>、OpenEMMA <a href="#ref-18">[18]</a>和 LightEMMA <a href="#ref-39">[39]</a>等框架中的最佳模型进行了比较。对于扩散式 VLM，我们纳入了 ViLaD <a href="#ref-7">[7]</a>，它作为直接的基线方法，用于验证我们的 VLA 联合建模与离散动作词元化的优势。此外，我们还与 UniAD <a href="#ref-40">[40]</a>这一具有代表性的端到端自动驾驶模型进行了对比。为保证公平比较，所报告的 UniAD 结果取自其单帧版本，评测方式与 <a href="#ref-15">[15]</a>中一致。

*2）解释基线方法：* 对于推理任务，我们与一系列专用模型进行比较，包括专门为驾驶推理设计的 Hint-AD <a href="#ref-36">[36]</a>、ALN-P3 <a href="#ref-38">[38]</a>和 TOD3Cap <a href="#ref-41">[41]</a>。我们还与若干具有代表性的闭源基础模型 GPT-4o 和 Gemini-1.5 做了基准对比，以评估我们这一专用的 7B 模型与工业界通用模型之间的差距。

**实现细节。**
MVLAD-AD 基于 PyTorch 框架实现。
我们使用预训练的 LLaDA 检查点 <a href="#ref-6">[6]</a>初始化模型权重。
为提升参数效率，我们采用秩为 r = 256 的低秩适配（Low-Rank Adaptation, LoRA）。
训练在 4 张 NVIDIA H100 GPU 上分布式进行，使用 `bfloat16` 精度，全局批大小为 32。
每个训练阶段包含 8 轮训练，约需 9 h 完成。
在评估阶段，所有推理实验均在单张 NVIDIA A100 GPU 上进行。

**表 I**：不同时域上的规划误差（m）与规划失败率（FR）。短横线表示该方法未报告相应结果。

| 方法 | 1 s (↓) | 2 s (↓) | 3 s (↓) | Avg (↓) | FR (↓) |
| --- | --- | --- | --- | --- | --- |
| LLaVA-1.6 | 0.91 | 2.50 | 3.44 | 2.28 | 55.25% |
| Llama-3.2 | 0.80 | 2.31 | 3.10 | 2.07 | 0.06% |
| Qwen2-VL | 1.32 | 2.94 | 3.98 | 2.74 | 0.03% |
| OpenEMMA | 1.45 | 3.21 | 3.76 | 2.81 | 16.11% |
| LightEMMA | - | - | 2.90 | 1.45 | **0.00%** |
| UniAD-Single | - | - | - | 1.80 | - |
| DriveLM | - | - | - | 1.39 | - |
| ViLaD | 0.81 | 1.93 | 2.69 | 1.81 | **0.00%** |
| MVLAD-AD | **0.70** | **1.31** | **2.34** | **1.28** | **0.00%** |

### IV-A. 规划对比

**扩散语言建模的优越性。** 首先，我们将基于扩散的架构 MVLAD-AD 与 ViLaD 同标准的自回归 VLM 基线方法进行比较。如表 I 所示，基于扩散的方法在所有时域上都稳定优于其它基线方法。具体而言，MVLAD-AD 取得了 1.28 m 的平均 L2 误差，相比基线方法显著降低了误差。我们认为这一差距源于扩散建模在规划任务上的固有优势：基于扩散的方法（MVLAD-AD 与 ViLaD）不受顺序预测的限制，能让模型更好地刻画驾驶行为的本质。

**VLA 建模的收益。** 在扩散框架内部，MVLAD-AD 进一步超越了先前最先进的方法 ViLaD，将平均 L2 误差从 1.81 m 降低到 1.28 m。这一性能提升验证了离散动作词元化的有效性。与仅在语言空间中运作的 ViLaD 不同，我们的框架将动作空间投影到一个由 N = 256 个代表性动作词元构成的紧凑码本上。因此，MVLAD-AD 执行的是在一组从真实世界数据中学到的驾驶动作上的分类任务。这降低了预测空间的复杂度，使模型能以更高的精度与稳定性锁定最优轨迹。

**系统鲁棒性。** 可靠性是机器人部署的前提条件。我们观察到，LLaVA-1.6 这类通用 VLM 存在很高的失败率（55.25%），这在很大程度上源于格式幻觉，即模型无法遵循规划所要求的严格输出语法。相比之下，MVLAD-AD 保持了 LightEMMA 与 ViLaD 的零失败优势，失败率为 0.00%。通过把动作空间映射到由合法轨迹聚类中心构成的固定码本，并采用结构化的解码过程，它在结构上保证了每一个生成的输出都对应一条合法轨迹。这一结构性约束消除了基于文本的规划器所固有的格式错误风险，保障了系统在下游执行中的鲁棒性。

{{< figgrid caption="**图 3**：在单张 NVIDIA A100 GPU 上，自回归方法与扩散式方法的规划推理时间对比。" >}}
mvlad-ad/inference_time_zoomed.png | 100
{{< /figgrid >}}


**推理效率。** 图 3 给出了推理时延的对比。基于扩散的架构借助并行解码，天然优于自回归基线方法。MVLAD-AD 则通过引入紧凑的 VLA 建模方案实现了进一步的加速。
我们的离散动作词元化不再通过冗余的文本词元来预测轨迹，而是把复杂的运动压缩为一段极短的运动基元序列。这有效缩短了规划所需的序列长度，显著降低了扩散去噪器的计算负载。
因此，MVLAD-AD 的推理时间为 1.72 s。相对于扩散基线方法 ViLaD 达到 1.6 倍加速比，相对于 LLaVA-1.6 达到 1.84 倍加速比，使 MVLAD-AD 成为端到端自动驾驶的一种高效解决方案。

### IV-B. 推理对比

**表 II**：Nu-X 上的性能对比

| 方法 | Nu-X CIDEr (↑) | Nu-X BLEU-4 (↑) | Nu-X METEOR (↑) | Nu-X ROUGE-L (↑) |
| --- | --- | --- | --- | --- |
| TOD3Cap | 14.5 | 2.45 | 10.5 | 23.0 |
| GPT-4o | 19.0 | 3.95 | 10.3 | 24.9 |
| Gemini-1.5 | 17.6 | 3.43 | 9.3 | 23.4 |
| Hint-AD | 22.4 | 4.18 | 13.2 | 27.6 |
| ALN-P3 | **28.6** | 5.59 | 14.7 | 35.2 |
| MVLAD-AD | 19.5 | **13.0** | **36.8** | **37.3** |

**表 III**：nuScenes-QA 上的性能对比

| 方法 | nuScenes-QA H0 (↑) | nuScenes-QA H1 (↑) | nuScenes-QA All (↑) |
| --- | --- | --- | --- |
| TOD3Cap | 53.0 | 45.1 | 49.0 |
| GPT-4o | 42.0 | 34.7 | 37.1 |
| Gemini-1.5 | 40.5 | 32.9 | 35.4 |
| Hint-AD | 55.4 | 48.0 | 50.5 |
| ALN-P3 | 57.1 | 50.9 | 52.9 |
| MVLAD-AD | **58.5** | **54.3** | **55.7** |

**Nu-X 上的驾驶解释。** 表 II 给出了在 Nu-X 数据集上的定量结果。相较于通用的 LVLM（例如 GPT-4o 与 Gemini-1.5）以及包括 Hint-AD 和 ALN-P3 在内的专用自动驾驶模型，MVLAD-AD 在大多数指标上都表现出更优的性能。值得注意的是，我们的方法取得了 13.0 的 BLEU-4 分数与 36.8 的 METEOR 分数，大幅超过先前最先进的方法 ALN-P3。
尽管 ALN-P3 取得了更高的 CIDEr 分数，但 MVLAD-AD 与 GPT-4o 这类通用模型的表现相当，后者同样因为生成的解释更加多样化而获得较低的 CIDEr 分数。
此外，我们在 BLEU-4 与 ROUGE-L 上的显著领先表明，MVLAD-AD 生成的描述在语义上更为丰富，并且在 n-gram 重叠意义下与参考描述对齐得更为精确。
这证实了我们的模型能够为复杂驾驶场景有效生成高质量的推理文本。

**nuScenes-QA 上的视觉问答。** 表 III 详细列出了在 nuScenes-QA 基准上的评测结果。MVLAD-AD 在总体划分上取得了 55.7% 的准确率，稳定地优于大规模商用模型与专用驾驶智能体。

这一性能飞跃表明，MVLAD-AD 捕捉到了驾驶场景中的复杂依赖关系，使模型能够以更高的精度回答关于交通动态的复杂问题。

### IV-C. 消融实验

**动作词表大小 N 的影响。** 离散码本的规模 N 在规划精度与分类任务的学习复杂度之间引入了一个根本性的权衡。由于我们的框架将轨迹规划表述为离散词元预测任务，训练损失主要反映分类难度，而 L2 误差衡量的是最终的规划性能。理论上，更大的 N 会减小量化误差，即连续路径点与其最近聚类中心之间的空间距离，从而提高重建保真度的上界。然而，词元数量过多又会增加分类任务的复杂度。

我们通过在词表大小 {{< math >}}$N \in \{128, 256, 384\}${{< /math >}} 下训练 MVLAD-AD 来考察这一权衡，结果如表 IV 所示。我们观察到，N = 256 取得了最佳平衡，达到了最低的规划误差 1.28 m。尤为关键的是，将 N 增大到 384 会使性能退化到 2.76 m，同时最终训练损失从 0.36 明显上升到 0.53。这表明，尽管理论精度更高，但由于要在密集的动作词元之间加以区分所带来的优化困难，模型难以收敛。反之，将 N 减小到 128 得到了最低的训练损失 0.32，说明分类任务变得更容易，但规划误差上升到 1.73 m。这证实了进一步缩小码本规模会造成量化瓶颈，从而限制物理精度，即便训练能够稳定收敛也无济于事。

**表 IV**：动作词表大小 N 的消融实验

| 动作词元数量 N | 最终训练损失 (↓) | L2 (m) 平均 (↓) |
| --- | --- | --- |
| 128 | 0.32 | 1.73 |
| 256（默认） | 0.36 | **1.28** |
| 384 | 0.53 | 2.76 |

**几何感知嵌入学习的有效性。** 我们考察几何感知嵌入学习对模型规划指标的影响。作为对照，我们移除该模块，改用随机初始化嵌入来训练模型。这带来了显著的性能下降，平均 L2 误差从 1.28 m 上升到 2.39 m。该对比证实，把动作词元当作彼此独立的类别索引会丢弃有用的度量信息，使模型难以学到有效的规划。通过施加几何一致性，我们的方法确保词元在隐空间中的距离与其物理位移相关联，从而使模型能够生成精确的轨迹。

**动作表示：路径点 vs. 位移。** 我们比较了将轨迹建模为绝对路径点与建模为相对位移 <a href="#ref-42">[42]</a>两种方式。如表 V 所示，位移表示几乎不影响规划精度（L2 误差仅从 1.28 m 略微上升到 1.30 m），却导致语义推理能力崩溃。采用位移的模型无法生成连贯的解释，CIDEr 分数骤降至 0.08，BLEU-4 也随之降低。这验证了我们采用绝对路径点的合理性。尽管位移能为短期规划提供足够的局部细节，但它们缺乏语义对齐所需的全局空间上下文。由于扩散是并行生成的，模型难以把彼此孤立的位移聚合起来以理解整体的机动动作。在缺少显式空间锚点的情况下，模型无法将物理动作映射到高层概念，从而导致可解释性明显崩塌。

**表 V**：动作表示的消融实验。C: CIDEr，B: BLEU-4，M: METEOR，R: ROUGE-L。

| 建模方式 | 规划 L2 (m) 平均 (↓) | Nu-X C (↑) | Nu-X B (↑) | Nu-X M (↑) | Nu-X R (↑) |
| --- | --- | --- | --- | --- | --- |
| 位移 | 1.30 | 0.08 | 5.66 | 23.1 | 26.4 |
| 路径点 | **1.28** | **19.5** | **13.0** | **36.8** | **37.3** |

## V. 结论

在本工作中，我们提出了 MVLAD-AD，这是一个用于端到端自动驾驶的统一框架，它在低时延、高精度的规划与语义可解释性之间取得了平衡。不同于常规的语言空间表示，我们的离散动作词元化策略成功地将连续轨迹规划转化为在运动学可行的运动基元上的鲁棒分类任务。此外，我们的几何感知嵌入弥合了语义推理与物理动力学之间的鸿沟，使系统能够生成高保真的轨迹以及有物理依据的解释。我们的动作优先解码策略进一步降低了推理（前向）过程中的规划时延。大量评估表明，MVLAD-AD 在 nuScenes 规划任务及相关语言基准上都取得了优异的性能，在规划精度与推理速度两方面均显著优于自回归基线方法，同时还能提供高质量的可解释语义推理。

## 参考文献

<span id="ref-1"></span>[1] X. C. Song, P. Smith, R. Kalyanam, X. Zhu, E. Adams, K. Colby, P. Finnegan, E. Gough, E. Hillery, R. Irvine, *et al.*, “Anvil - System Architecture and Experiences from Deployment and Early User Operations,” in *PEARC*, 2022.

<span id="ref-2"></span>[2] T. J. Boerner, S. Deems, T. R. Furlani, S. L. Knuth, and J. Towns, “ACCESS: Advancing Innovation: NSF’s Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support,” in *PEARC*, 2023.

<span id="ref-3"></span>[3] J. Mao, Y. Qian, J. Ye, H. Zhao, and Y. Wang, “GPT-Driver: Learning to Drive with GPT,” *arXiv preprint arXiv:2310.01415*, 2023.

<span id="ref-4"></span>[4] L. Wen, D. Fu, X. Li, X. Cai, T. MA, P. Cai, M. Dou, B. Shi, L. He, and Y. Qiao, “DiLu: A Knowledge-Driven Approach to Autonomous Driving with Large Language Models,” in *ICLR*, 2024.

<span id="ref-5"></span>[5] Z. Xu, Y. Zhang, E. Xie, Z. Zhao, Y. Guo, K.-Y. K. Wong, Z. Li, and H. Zhao, “DriveGPT4: Interpretable End-to-end Autonomous Driving via Large Language Model,” *IEEE Robotics and Automation Letters*, vol. 9, no. 10, 2024.

<span id="ref-6"></span>[6] S. Nie, F. Zhu, Z. You, X. Zhang, J. Ou, J. Hu, J. ZHOU, Y. Lin, J.-R. Wen, and C. Li, “Large Language Diffusion Models,” in *NeurIPS*, 2025.

<span id="ref-7"></span>[7] C. Cui, Y. Zhou, J. Peng, S.-Y. Park, Z. Yang, P. Sankaranarayanan, J. Zhang, R. Zhang, and Z. Wang, “ViLaD: A Large Vision Language Diffusion Framework for End-to-End Autonomous Driving,” *arXiv preprint arXiv:2508.12603*, 2025.

<span id="ref-8"></span>[8] M. J. Kim, K. Pertsch, S. Karamcheti, T. Xiao, A. Balakrishna, S. Nair, R. Rafailov, E. P. Foster, P. R. Sanketi, Q. Vuong, *et al.*, “OpenVLA: An Open-Source Vision-Language-Action Model,” in *CoRL*, 2024.

<span id="ref-9"></span>[9] B. Zitkovich, T. Yu, S. Xu, P. Xu, T. Xiao, F. Xia, J. Wu, P. Wohlhart, S. Welker, A. Wahid, *et al.*, “RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control,” in *CoRL*, 2023.

<span id="ref-10"></span>[10] L. Chen, P. Wu, K. Chitta, B. Jaeger, A. Geiger, and H. Li, “End-to-end Autonomous Driving: Challenges and Frontiers,” *IEEE Transactions on Pattern Analysis and Machine Intelligence*, vol. 46, no. 12, 2024.

<span id="ref-11"></span>[11] Y. Chen, Y. Wang, and Z. Zhang, “DrivingGPT: Unifying Driving World Modeling and Planning with Multi-modal Autoregressive Transformers,” in *ICCV*, 2025.

<span id="ref-12"></span>[12] X. Huang, E. M. Wolff, P. Vernaza, T. Phan-Minh, H. Chen, D. S. Hayden, M. Edmonds, B. Pierce, X. Chen, P. E. Jacob, *et al.*, “DriveGPT: Scaling Autoregressive Behavior Models for Driving,” *arXiv preprint arXiv:2412.14415*, 2024.

<span id="ref-13"></span>[13] H. Shao, Y. Hu, L. Wang, G. Song, S. L. Waslander, Y. Liu, and H. Li, “LMDrive: Closed-Loop End-to-End Driving with Large Language Models,” in *CVPR*, 2024.

<span id="ref-14"></span>[14] E. Cui, W. Wang, Z. Li, J. Xie, H. Zou, H. Deng, G. Luo, L. Lu, X. Zhu, and J. Dai, “DriveMLM: Aligning Multi-Modal Large Language Models with Behavioral Planning States for Autonomous Driving,” *Visual Intelligence*, vol. 3, no. 1, p. 22, 2025.

<span id="ref-15"></span>[15] C. Sima, K. Renz, K. Chitta, L. Chen, H. Zhang, C. Xie, J. Beißwenger, P. Luo, A. Geiger, and H. Li, “DriveLM: Driving with Graph Visual Question Answering,” in *ECCV*, 2024.

<span id="ref-16"></span>[16] T. Wang, E. Xie, R. Chu, Z. Li, and P. Luo, “DriveCoT: Integrating Chain-of-Thought Reasoning with End-to-End Driving,” *arXiv preprint arXiv:2403.16996*, 2024.

<span id="ref-17"></span>[17] J.-J. Hwang, R. Xu, H. Lin, W.-C. Hung, J. Ji, K. Choi, D. Huang, T. He, P. Covington, B. Sapp, Y. Zhou, J. Guo, D. Anguelov, and M. Tan, “EMMA: End-to-End Multimodal Model for Autonomous Driving,” *Transactions on Machine Learning Research*, 2025.

<span id="ref-18"></span>[18] S. Xing, C. Qian, Y. Wang, H. Hua, K. Tian, Y. Zhou, and Z. Tu, “OpenEMMA: Open-Source Multimodal Model for End-to-End Autonomous Driving,” in *WACV Workshops*, 2025.

<span id="ref-19"></span>[19] A. B. Arrieta, N. Díaz-Rodríguez, J. Del Ser, A. Bennetot, S. Tabik, A. Barbado, S. García, S. Gil-López, D. Molina, R. Benjamins, *et al.*, “Explainable Artificial Intelligence (XAI): Concepts, taxonomies, opportunities and challenges toward responsible AI,” *Information fusion*, vol. 58, pp. 82–115, 2020.

<span id="ref-20"></span>[20] D. Omeiza, H. Webb, M. Jirotka, and L. Kunze, “Explanations in Autonomous Driving: A Survey,” *IEEE Transactions on Intelligent Transportation Systems*, vol. 23, no. 8, pp. 10 142–10 162, 2021.

<span id="ref-21"></span>[21] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” in *NIPS*, 2017.

<span id="ref-22"></span>[22] A. Shrikumar, P. Greenside, and A. Kundaje, “Learning Important Features Through Propagating Activation Differences,” in *ICML*, 2017.

<span id="ref-23"></span>[23] M. T. Ribeiro, S. Singh, and C. Guestrin, “'Why Should I Trust You?': Explaining the Predictions of Any Classifier,” in *SIGKDD*, 2016.

<span id="ref-24"></span>[24] R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual Explanations From Deep Networks via Gradient-Based Localization,” in *ICCV*, 2017.

<span id="ref-25"></span>[25] K. Simonyan, A. Vedaldi, and A. Zisserman, “Deep Inside Convolutional Networks: Visualising Image Classification Models and Saliency Maps,” *arXiv preprint arXiv:1312.6034*, 2013.

<span id="ref-26"></span>[26] K. Xu, J. Ba, R. Kiros, K. Cho, A. Courville, R. Salakhudinov, R. Zemel, and Y. Bengio, “Show, Attend and Tell: Neural Image Caption Generation with Visual Attention,” in *ICML*, 2015.

<span id="ref-27"></span>[27] J. Kim and J. Canny, “Interpretable Learning for Self-Driving Cars by Visualizing Causal Attention,” in *ICCV*, 2017.

<span id="ref-28"></span>[28] J. Kim, A. Rohrbach, T. Darrell, J. Canny, and Z. Akata, “Textual Explanations for Self-Driving Vehicles,” in *ECCV*, 2018.

<span id="ref-29"></span>[29] M. A. Kühn, D. Omeiza, and L. Kunze, “Textual Explanations for Automated Commentary Driving,” in *IV*, 2023.

<span id="ref-30"></span>[30] S. Jain and B. C. Wallace, “Attention is not explanation,” in *NAACL*, 2019.

<span id="ref-31"></span>[31] Y. Qiang, D. Pan, C. Li, X. Li, R. Jang, and D. Zhu, “AttCAT: Explaining Transformers via Attentive Class Activation Tokens,” in *NeurIPS*, 2022.

<span id="ref-32"></span>[32] L. Chen, O. Sinavski, J. Hünermann, A. Karnsund, A. J. Willmott, D. Birch, D. Maund, and J. Shotton, “Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving,” in *ICRA*, 2024.

<span id="ref-33"></span>[33] C. Cui, Z. Yang, Y. Zhou, Y. Ma, J. Lu, L. Li, Y. Chen, J. Panchal, and Z. Wang, “Personalized Autonomous Driving with Large Language Models: Field Experiments,” in *ITSC*, 2024.

<span id="ref-34"></span>[34] C. Cui, Y. Ma, S.-Y. Park, Z. Yang, Y. Zhou, P. Liu, J. Lu, J. Peng, J. Zhang, R. Zhang, *et al.*, “LLM4AD: Large Language Models for Autonomous Driving—Concept, Review, Benchmark, Experiments, and Future Trends,” *Proceedings of the IEEE*, 2026.

<span id="ref-35"></span>[35] H. Caesar, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom, “nuScenes: A Multimodal Dataset for Autonomous Driving,” in *CVPR*, 2020.

<span id="ref-36"></span>[36] K. Ding, B. Chen, Y. Su, H.-a. Gao, B. Jin, C. Sima, X. Li, W. Zhang, P. Barsch, H. Li, *et al.*, “Hint-AD: Holistically Aligned Interpretability in End-to-End Autonomous Driving,” in *CoRL*, 2024.

<span id="ref-37"></span>[37] T. Qian, J. Chen, L. Zhuo, Y. Jiao, and Y.-G. Jiang, “NuScenes-QA: A Multi-Modal Visual Question Answering Benchmark for Autonomous Driving Scenario,” in *AAAI*, 2024.

<span id="ref-38"></span>[38] Y. Ma, B. Yaman, X. Ye, M. Yurt, J. Luo, A. Mallik, Z. Wang, and L. Ren, “ALN-P3: Unified Language Alignment for Perception, Prediction, and Planning in Autonomous Driving,” *arXiv preprint arXiv:2505.15158*, 2025.

<span id="ref-39"></span>[39] Z. Qiao, H. Li, Z. Cao, and H. X. Liu, “LightEMMA: Lightweight End-to-End Multimodal Model for Autonomous Driving,” *arXiv preprint arXiv:2505.00284*, 2025.

<span id="ref-40"></span>[40] Y. Hu, J. Yang, L. Chen, K. Li, C. Sima, X. Zhu, S. Chai, S. Du, T. Lin, W. Wang, *et al.*, “Planning-Oriented Autonomous Driving,” in *CVPR*, 2023.

<span id="ref-41"></span>[41] B. Jin, Y. Zheng, P. Li, W. Li, Y. Zheng, S. Hu, X. Liu, J. Zhu, Z. Yan, H. Sun, *et al.*, “TOD3Cap: Towards 3D Dense Captioning in Outdoor Scenes,” in *ECCV*, 2024.

<span id="ref-42"></span>[42] W. Tang, J. You, J. Liu, Z. Wang, R. Gan, Z. Huang, F. Wei, and B. Ran, “HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision-Language Models for Long-Tail Autonomous Driving,” *arXiv preprint arXiv:2602.00993*, 2026.

---

## 译者说明

本页中文译文由 **Claude Opus 5 + ultracode** 翻译，**仅供参考，对内容正确性不作保证**。译文力求与原文逐段对应，公式、表格与数值均按原文照录；专有名词（数据集名、模型名、评价指标、人名）保留英文原文。若中英文表述存在出入，一律以英文原文为准。

- 英文原文 PDF：[/publication/mvlad-ad/mvlad-ad.pdf](/publication/mvlad-ad/mvlad-ad.pdf)
- 英文全文（网页版）：[Full Text]({{< relref "/publication/mvlad-ad-en" >}})
- 论文主页：[Efficient and Explainable End-to-End Autonomous Driving via Masked Vision-Language-Action Diffusion]({{< relref "/publication/mvlad-ad" >}})
- 代码仓库：[https://github.com/lan-qing/MVLAD-AD](https://github.com/lan-qing/MVLAD-AD)
