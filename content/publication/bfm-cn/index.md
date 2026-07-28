---
title: "探究扩散模型少样本微调中的崩坏阶段并用贝叶斯神经网络加以缓解"
subtitle: "Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks"

summary: "探究扩散模型少样本微调中的崩坏阶段并用贝叶斯神经网络加以缓解（SIGKDD 2026）论文中文全文翻译，含全部公式、表格与插图。"

date: '2026-08-09T00:00:00Z'
publishDate: '2024-05-26T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/bfm/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/bfm/cn/'

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
本页是论文 **《Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks》**（SIGKDD 2026）的中文全文翻译，正文、公式、表格与插图均与原文一一对应。
译文仅供参考，如需引用请以[英文原文 PDF](https://arxiv.org/pdf/2405.19931) 为准（[论文主页]({{< relref "/publication/bfm" >}})）。也可查看[英文全文网页版]({{< relref "/publication/bfm-en" >}})。
{{% /callout %}}

{{< toc title="目录" >}}

Xiaoyu Wu, Jiaru Zhang, Yang Hua, Bohan Lyu, Hao Wang, Tao Song, and Haibing Guan

> Xiaoyu Wu (wuxiaoyu2000@sjtu.edu.cn) and Jiaru Zhang (jiaruzhang@sjtu.edu.cn), Shanghai Jiao Tong University, Shanghai, China. Both authors contributed equally to this research.
> Yang Hua (Y.Hua@qub.ac.uk), Queen’s University Belfast, Belfast, United Kingdom.
> Bohan Lyu (lvbh22@mails.tsinghua.edu.cn), Tsinghua University, Beijing, China.
> Hao Wang (hwang9@stevens.edu), Department of Electrical and Computer Engineering, Stevens Institute of Technology, Hoboken, New Jersey, United States.
> Tao Song (songt333@sjtu.edu.cn), Shanghai Jiao Tong University, Shanghai, China. Corresponding author.
> Haibing Guan (hbguan@sjtu.edu.cn), Shanghai Jiao Tong University, Shanghai, China.

## 摘要

扩散模型（Diffusion Models, DMs）的少样本微调是一项关键进展，它显著降低了训练开销，并使个性化 AI 应用成为可能。然而，我们在探究扩散模型的训练动态时观察到一个出乎意料的现象：在训练过程中，图像保真度起初不断提升，随后却意外地随着噪声图案的出现而恶化，直到后期伴随严重的过拟合才重新恢复。我们把生成噪声图案的这一阶段称为「崩坏阶段（corruption stage）」。为了理解崩坏阶段，我们首先对单样本微调场景进行启发式建模，随后将该建模扩展到更一般的情形。通过这一建模，我们确定了崩坏阶段的主要成因：少样本微调本身所固有的收窄的学习分布。为了解决这一问题，我们借助变分推断将贝叶斯神经网络（Bayesian Neural Networks, BNNs）应用于扩散模型，以隐式地拓宽学习到的分布，并指出 BNNs 的学习目标可以自然地视为扩散损失的期望以及与预训练扩散模型之间的进一步正则化。该方法与当前扩散模型的少样本微调方法高度兼容，且不引入任何额外推理开销。实验结果表明，我们的方法显著缓解了崩坏现象，并在物体驱动生成与主体驱动生成两类任务中提升了生成图像的保真度、质量与多样性。

**CCS Concepts:** • Computing methodologies → Machine learning; • Computing methodologies → Bayesian networks; • Computing methodologies → Neural networks.

**关键词：** 少样本微调；扩散模型；贝叶斯神经网络

## 1. 引言

近年来，扩散模型（Diffusion Models, DMs）的发展呈现出爆发式增长。这些模型在图像编辑 <a href="#ref-14">[14]</a>、视频编辑 <a href="#ref-31">[31]</a>等各类应用中展现出非凡的能力。其中尤为值得关注的是少样本微调方法 <a href="#ref-11">[11]</a>, <a href="#ref-22">[22]</a>, <a href="#ref-18">[18]</a>的出现：预训练模型基于一小组训练图像进行微调，从而实现个性化的生成。这类方法显著降低了训练中的显存与时间开销。此外，这些技术为根据特定主体或物体自适应地生成图像提供了强有力的工具，体现了个性化 AI 的理念，让人人都能用上 AI。近年来，这一创新甚至催生了若干社区，例如 civitai.com，其上拥有数以万计的检查点和数以百万计的下载量。

{{< figgrid caption="**图 1**：微调过程中使用与不使用 BNNs 时的图像保真度变化。训练迭代次数为零表示预训练扩散模型。我们用 DreamBooth 对 Stable Diffusion v1.5 微调了 5 次。" >}}
bfm/baseline_motivation.png | 49 | **(a)** 不使用 BNNs 的少样本微调过程。
bfm/bayes_motivation.png | 49 | **(b)** 使用 BNNs 的少样本微调过程。
{{< /figgrid >}}


尽管少样本微调方法在扩散模型中十分重要且被广泛使用，但在利用有限数据从一个大分布（即预训练扩散模型学习到的分布）迁移到一个小得多的分布（即微调后的扩散模型学习到的分布）时，这些方法往往表现不佳甚至完全失败 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>。我们首次识别出这些失败发生的时机与方式，并发现它们与一个反常现象相关：如图 1 所示，在微调过程中，生成图像与训练图像之间的相似度起初不断上升，随后却出人意料地下降，之后又再次上升。最终，扩散模型只能生成与训练图像完全相同的图像。值得注意的是，在相似度下降的这一阶段，我们观察到生成图像上出现了一些意料之外的噪声图案，因此我们将其命名为「崩坏阶段」（corruption stage）。

为了理解这一崩坏阶段，我们对少样本微调过程展开了进一步分析。具体而言，我们从对单样本情形（即微调过程中仅使用一张训练图像）的启发式建模出发，随后将该建模扩展到更一般的情形。该建模给出了生成图像中残留误差尺度的估计。我们进一步借助该建模说明，崩坏阶段是如何由少样本任务所固有的受限的学习到的分布而产生的。

基于上述分析，解决崩坏问题的方案应当聚焦于扩展学习到的分布。然而，在少样本微调中实现这种扩展仍然颇具挑战。传统的数据增强方法在应用于生成模型时，往往面临诸如泄漏 <a href="#ref-13">[13]</a>以及生成质量下降 <a href="#ref-6">[6]</a>等严重问题。受贝叶斯神经网络（Bayesian Neural Networks, BNNs）<a href="#ref-3">[3]</a>相关进展的启发，我们提出引入 BNNs，作为一种简洁而有效的策略来隐式地拓宽学习到的分布。我们进一步表明，其学习目标可以分解为扩散损失的期望与一个与预训练模型相关的额外正则化项。二者可以通过调整，在图像保真度与多样性之间达到权衡。我们的方法不引入任何额外推理开销，并且与扩散模型中已有的少样本微调方法具有良好的兼容性，包括 DreamBooth <a href="#ref-22">[22]</a>、LoRA <a href="#ref-11">[11]</a>和 OFT <a href="#ref-18">[18]</a>。实验表明，我们的方法显著缓解了崩坏问题，并在多种数据集、不同评价指标下大幅提升了各类少样本微调方法的性能。

总而言之，我们的主要贡献如下：

- 我们观察到扩散模型少样本微调过程中的一个反常现象：图像保真度先是提升，随后伴随噪声图案的出现而意外恶化，之后再次改善但伴有严重的过拟合。我们将噪声图案出现的这一阶段称为崩坏阶段。我们希望这一观察能够为未来关于扩散模型的研究提供启发。
- 我们为扩散模型的少样本微调过程提供了一种启发式建模，解释了崩坏阶段的出现与消失。借助该建模，我们指出主要问题源于少样本微调过程中扩散模型固有的受限的学习到的分布。
- 我们创新性地引入 BNNs 来拓宽学习到的分布，从而缓解此类崩坏。实验证实了其在不同评价指标上的改进效果，包括文本提示词保真度、图像保真度、生成多样性以及图像质量。

**开源与完整版本。** 代码已在 GitHub 上公开（注：<https://github.com/Nicholas0228/BNN-Finetuning-DMs>）。我们还在 arXiv 上维护了本文的扩展版本 <a href="#ref-29">[29]</a>，其中包含额外的实验与讨论。

## 2. 相关工作

### 2.1. 扩散模型与少样本微调

扩散模型（Diffusion Models, DMs）<a href="#ref-10">[10]</a>, <a href="#ref-24">[24]</a>, <a href="#ref-25">[25]</a>, <a href="#ref-26">[26]</a>是一类生成模型，它通过对初始从高斯分布中采样的变量逐步去噪来逼近数据分布。这类模型包含一个前向扩散过程和一个反向去噪过程。在前向过程中，所添加噪声 {{< math >}}$\varepsilon${{< /math >}} 的幅度随时间 {{< math >}}$t${{< /math >}} 递增，可由方程 {{< math >}}$x_t = \sqrt{\alpha_t}x_{0} + \sqrt{1-\alpha_t}\varepsilon${{< /math >}} 描述，其中 {{< math >}}$x_{0}${{< /math >}} 是给定的原始图像，时间 {{< math >}}$t${{< /math >}} 的取值范围在一般情形下为 {{< math >}}$\left\{1, \dots, 1000\right\}${{< /math >}}。反之，在反向过程中，扩散模型旨在用噪声预测模块 {{< math >}}$\epsilon_{\theta}${{< /math >}} 估计噪声，并随后将其从含噪图像 {{< math >}}$x_t${{< /math >}} 中去除。实际噪声与预测噪声之间的差异构成训练损失，记作扩散损失 {{< math >}}$\mathcal{L}_{DM}:= \mathbb{E}_{\varepsilon\sim \mathcal{N}(0,1), t}\left||\epsilon_{\theta}(x_{t}, t) - \varepsilon|\right|_{2}^{2}.${{< /math >}}

扩散模型中的少样本微调 <a href="#ref-7">[7]</a>, <a href="#ref-11">[11]</a>, <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>旨在用有限的一组图像对扩散模型进行个性化，从而便于创作定制化内容。
Gal 等人提出了一种技术，利用冻结的文生图模型嵌入空间中的新词元来捕捉所提供图像中呈现的概念 <a href="#ref-7">[7]</a>。
Ruiz 等人进一步提出了 DreamBooth，该方法微调扩散模型中的大部分参数，借助重建损失来捕捉输入图像更精确的细节，并以类别特定的保持损失确保与文本提示词的对齐 <a href="#ref-22">[22]</a>。
此外，Hu 等人提出了 LoRA，这是一种轻量级微调方法，通过插入待学习的低秩层，同时保持其余参数冻结 <a href="#ref-11">[11]</a>。
Qiu 等人提出了 OFT，该方法采用正交变换来提升生成质量 <a href="#ref-18">[18]</a>。尽管这些方法通常能够成功捕捉训练图像的细节，但它们都受到本文所观察到的崩坏阶段的困扰。

{{< figgrid caption="**图 2**：以 Dino 相似度衡量的、不同训练图像数量下少样本微调中图像保真度的变化。Dino 相似度越高表示图像保真度越好。随着训练图像数量增加，崩坏出现得更晚，严重程度也有所减轻。" >}}
bfm/baseline_05.png | 32 | **(a)** 在 1 张图像上微调。
bfm/baseline_0105.png | 32 | **(b)** 在 2 张图像上微调。
bfm/baseline_allimages.png | 32 | **(c)** 在 6 张图像上微调。
{{< /figgrid >}}


**图 2**：在不同训练图像数量下，少样本微调过程中图像保真度变化的示意图，以 Dino 相似度衡量。Dino 相似度越高表示图像保真度越好。随着训练图像数量的增加，崩坏出现得更晚，其严重程度也有所降低。

### 2.2. 贝叶斯神经网络

贝叶斯神经网络（Bayesian Neural Networks, BNNs）是一类随机神经网络，其特点是将参数视为随机变量而非固定值 <a href="#ref-3">[3]</a>, <a href="#ref-4">[4]</a>, <a href="#ref-17">[17]</a>。
其目标是在给定数据集 {{< math >}}$\mathcal{D}${{< /math >}} 的条件下，推断参数 {{< math >}}$\theta${{< /math >}} 的后验分布 {{< math >}}$P(\theta|\mathcal{D})${{< /math >}}。
这赋予了 BNNs 若干独特优势，例如能够对输出的分布进行建模、缓解过拟合以及增强模型可解释性 <a href="#ref-1">[1]</a>, <a href="#ref-12">[12]</a>。
BNNs 的一种常见变体是平均场变分贝叶斯神经网络，也被称为 *Bayes by Backprop*（反向传播贝叶斯），其中应用平均场变分推断来获得变分分布 {{< math >}}$Q_W(\theta)${{< /math >}}，以逼近后验分布 {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} <a href="#ref-3">[3]</a>。
近期研究表明，即使是仅将参数子集视为随机变量、而将其余参数保持为固定值的 BNN 模块，也能保留完整 BNNs 所具备的优势 <a href="#ref-8">[8]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-23">[23]</a>。
我们提出的方法可以看作是将 BNN 原理自然地推进到扩散模型的少样本微调之中。

## 3. 少样本微调中的崩坏阶段

在第 3.1 节中，我们首先给出关于扩散模型少样本微调过程中崩坏阶段的观察。在第 3.2 节中，为了更好地观察和理解与崩坏阶段相关的问题以及微调动态，我们提出一种以高斯分布作为近似的启发式建模。

为了应对刻画扩散模型微调过程动态的困难，我们采用了合理的简化，并通过一个具体案例的证据加以支撑。在第 3.3 节中，我们用所提出的建模解释崩坏阶段的出现与消失，并揭示受限的学习到的分布正是崩坏阶段的根本原因。

### 3.1. 观察

在本节中，我们考察扩散模型少样本微调过程中的性能变化。具体而言，我们使用 DreamBooth <a href="#ref-22">[22]</a>在不同数量的训练图像上微调 Stable Diffusion (SD) v1.5（注：<https://huggingface.co/runwayml/stable-diffusion-v1-5）> <a href="#ref-20">[20]</a>，并记录生成图像与训练图像之间的平均 Dino 相似度，作为图像保真度的度量 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>。

如图 2 所示，微调过程中图像保真度的变化并非单调的。

具体来说，少样本微调过程可以近似划分为以下几个阶段：

1. 在最初的若干次迭代中，图像保真度快速提升。

2. 随后，图像保真度出现**异常下降**。我们观察到，该阶段生成的图像中出现了不断增强的噪声图案，即崩坏阶段（corruption stage）的逐渐出现。

3. 再往后，生成保真度得以恢复，我们观察到生成图像上的崩坏图案逐步减弱，即崩坏阶段的逐渐消失。一旦在该阶段中崩坏完全消失，模型便进入一种接近「过拟合」的状态，此时它只能生成与训练图像完全一致的图像。因此，它失去了产生多样化图像的能力。

通过图 2a、图 2b 与图 2c 之间的对比可以注意到，当训练图像数量增加时，崩坏的起始时间被推迟，其严重程度也有所减轻。

### 3.2. 少样本微调的启发式建模

在本节中，我们首先对单样本微调的场景进行启发式建模，然后将其扩展到更一般的情形。该建模由一个具体案例上的证据加以支撑。

**扩散模型单样本微调的启发式建模。** 我们从一个有代表性的条件出发：数据集 {{< math >}}$\mathcal{D}${{< /math >}} 中只包含一张训练图像 {{< math >}}$x'${{< /math >}}。在该条件下，我们假设参数为 {{< math >}}$\theta${{< /math >}} 的微调后扩散模型把任意原始图像 {{< math >}}$x_0${{< /math >}} 与任意时间步 {{< math >}}$t${{< /math >}} 处的含噪图像（即 {{< math >}}$x_t${{< /math >}}）的联合分布建模为一个多元高斯分布 {{< math >}}$P_\theta(x_0, x_t)${{< /math >}}。具体而言，当模型仅用一张图像 {{< math >}}$x'${{< /math >}} 微调时，其关于 {{< math >}}$x_0${{< /math >}} 的边缘分布被近似为 {{< math >}}$P_\theta(x_0) \approx \mathcal{N}(x', \sigma_1^2)${{< /math >}}。

此外，由于含噪图像 {{< math >}}$x_t${{< /math >}} 是由 {{< math >}}$x_0${{< /math >}} 与单位高斯噪声 {{< math >}}$\epsilon${{< /math >}} 通过线性组合 {{< math >}}$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon${{< /math >}} 得到的，{{< math >}}$x_t${{< /math >}} 的边缘分布应当近似为 {{< math >}}$P_\theta(x_t) \approx \mathcal{N}(\sqrt{\alpha_t}x', \alpha_t \sigma_1^2 + (1 - \alpha_t))${{< /math >}}。值得注意的是，微调过程实际上是在缩小 {{< math >}}$P_\theta(x_t \mid x_0=x')${{< /math >}} 与 {{< math >}}$\mathcal{N}(\sqrt{\alpha_t} x', (1 - \alpha_t))${{< /math >}} 之间的 KL 散度，因此这两个分布在微调过程中应当越来越接近 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>。

在该建模下，扩散模型的核心任务——基于含噪图像 {{< math >}}$x_t${{< /math >}} 预测原始图像 {{< math >}}$x_0${{< /math >}}——可表示为 {{< math >}}$P_\theta(x_0|x_t)${{< /math >}}。我们发现，它实际上近似为一个同时与 {{< math >}}$x_{t}${{< /math >}} 和训练图像 {{< math >}}$x'${{< /math >}} 相关的高斯分布：

{{< math >}}
$$
\begin{aligned}
    P_\theta(x_0|x_t) \approx \mathcal{N}(x' + \delta_t(x_t, x'), \frac{(1-\alpha_t)}{\alpha_t \sigma_1^2 + (1 - \alpha_t)}), \\ \text{ where } \delta_t(x_t, x') =\frac{\sqrt{\alpha_t} \sigma_1^2}{\alpha_t \sigma_1^2 + (1 - \alpha_t)} (x_t - \sqrt{\alpha_t} x').
\end{aligned}
\tag{1}
$$
{{< /math >}}

在扩散模型看来最可能的 {{< math >}}$x_0${{< /math >}}，记为 {{< math >}}$\hat{x}_{0}${{< /math >}}，为：

{{< math >}}
$$
\hat{x}_{0} = \arg\max_{x_{0}}P_\theta(x_0|x_t) \approx x' + \delta_t(x_t, x'). \tag{2}
$$
{{< /math >}}

推导过程见附录第 A1 节。值得注意的是，误差项 {{< math >}}$\delta_t(x_t, x')${{< /math >}} 表示预测出的原始图像 {{< math >}}$\hat{x}_0${{< /math >}} 与训练图像 {{< math >}}$x'${{< /math >}} 之间的差异。直观上，{{< math >}}$\sigma_{1}${{< /math >}} 可以视为微调后的扩散模型重新生成训练样本 {{< math >}}$x'${{< /math >}} 的「置信度」。

上述建模的准确性受到训练迭代次数的影响。随着微调的推进，近似 {{< math >}}$P_\theta(x_0) \approx \mathcal{N}(x', \sigma_1^2)${{< /math >}} 会变得更加精确，该表述也更贴近真实场景。为了说明该建模中的一种极端情形，我们考虑 {{< math >}}$\sigma_1 = 0${{< /math >}}（即 {{< math >}}$\delta_t = 0${{< /math >}}）的场景。在该条件下，对于任意输入 {{< math >}}$x_t${{< /math >}}，扩散模型都会按照式 (2) 所描述的那样始终复现训练图像 {{< math >}}$x'${{< /math >}}。这表明在这种极端场景下，扩散模型完全丧失了其内在的去噪能力，转而只是重新生成训练图像。

在相反的极端情形下，即 {{< math >}}$\sigma_1 = +\infty${{< /math >}}（也就是 {{< math >}}$\delta_t = \frac{1}{\sqrt{\alpha_t}}x_t - x'${{< /math >}}），模型对 {{< math >}}$x_0${{< /math >}} 的预测恰好是 {{< math >}}$\frac{1}{\sqrt{\alpha_t}}x_t${{< /math >}}。这表明扩散模型完全丧失了生成图像的能力，此时扩散模型只是依据时间步 {{< math >}}$t${{< /math >}} 处的系数 {{< math >}}$\alpha_t${{< /math >}} 对 {{< math >}}$x_{t}${{< /math >}} 做了一次缩放，从而使 {{< math >}}$x_{t}${{< /math >}} 中的任何噪声也都残留在生成图像中。

**扩展到更一般的情形。** 我们进一步把建模扩展到 {{< math >}}$\mathcal{D}${{< /math >}} 包含多个训练样本的情形。具体而言，我们假设扩散模型关于原始图像 {{< math >}}$x_0${{< /math >}} 的学习到的分布（即 {{< math >}}$P_\theta(x_0)${{< /math >}}）是以一个图像集合 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 为中心的。在少样本微调下，随着训练的持续，{{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 会逐渐逼近训练数据集 {{< math >}}$\mathcal{D}${{< /math >}}。另一方面，对于预训练扩散模型而言，它们通常学习到一个大得多的流形，这可以理解为在一个足够大的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 上进行学习。对于所有这些面对含噪图像 {{< math >}}$x_{t}${{< /math >}} 的扩散模型，我们把它们的行为简化为：首先在 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 中找到一个样本 {{< math >}}$x^{*} \in \mathcal{I}_{\theta}${{< /math >}}，使误差项 {{< math >}}$\delta_t(x_{t}, x^{*})${{< /math >}} 最小化，然后依据式 (2) 估计出相应的 {{< math >}}$\hat{x}_0${{< /math >}}。在这一简化下，预训练扩散模型足够大的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 使其能够找到一个 {{< math >}}$x^{*}${{< /math >}}，让误差项 {{< math >}}$\delta_t(x_t, x^{*})${{< /math >}} 趋近于零，从而在绝大多数情况下避免了崩坏。

{{< figgrid caption="**图 3**：取 x<sub>t</sub> = 0、t = 1000 时，预训练与微调后扩散模型的去噪结果。由于输入不含噪声，预训练扩散模型基本不改变 x<sub>t</sub>。相反，在 1 张和 5 张图像上微调过的扩散模型都会把 x<sub>t</sub> 变换成与训练集 𝒟 中某个样本高度相似的图像。" >}}
bfm/zero_input_2.0.png | 100
{{< /figgrid >}}


**对该建模的支持。** 为了说明上述建模能够较好地近似实际场景，我们给出一个具体示例：把「含噪」图像设为 {{< math >}}$x_{t}=0${{< /math >}}，然后让预训练扩散模型和微调后的扩散模型分别对这张完全不含噪声的图像 {{< math >}}$x_{t}${{< /math >}} 进行去噪。这是一个特殊情形，因为 {{< math >}}$x_{t}${{< /math >}} 不含噪声，而一个正常发挥去噪器功能的典型扩散模型应当让它保持不变。按照我们的建模，这两类扩散模型都应当首先在各自的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 中找到一个样本 {{< math >}}$x^{*}${{< /math >}}。自然地，{{< math >}}$x_{t} = 0${{< /math >}} 位于预训练扩散模型的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 之中（更多证据参见附录第 A5 节）。因此，预训练扩散模型在去噪时应当几乎不改变这个 {{< math >}}$x_{t}=0${{< /math >}}。

相比之下，微调后的扩散模型的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 应当逐渐逼近训练数据集 {{< math >}}$\mathcal{D}${{< /math >}}。因此，微调后的扩散模型应当首先找到一个样本 {{< math >}}$x^{*} \in \mathcal{D}${{< /math >}}，然后依据式 (2) 把原始图像 {{< math >}}$x_0${{< /math >}} 预测为与 {{< math >}}$x^{*}${{< /math >}} 成比例的结果。

图 3 中给出的实验结果支持了我们的分析：可以观察到，预训练扩散模型并未大幅改变这个 {{< math >}}$x_{t} =0${{< /math >}}，而微调后的扩散模型则把这个 {{< math >}}$x_{t}${{< /math >}} 变换成了其训练数据集 {{< math >}}$\mathcal{D}${{< /math >}} 中的某个样本。这说明我们的建模与少样本微调中的真实场景是一致的。

### 3.3. 崩坏阶段的解释

{{< figgrid caption="**图 4**：DreamBooth 在不同训练迭代次数下估计得到的 σ<sub>1</sub>。" >}}
bfm/1.png | 60
{{< /figgrid >}}


本节中，我们依据前述关于扩散模型少样本微调过程的启发式建模来解释崩坏阶段。我们首先给出崩坏阶段出现的原因，并通过一个示例展示该问题的严重性。我们进一步说明：随着微调过程的持续，崩坏阶段为何会逐渐消失，进而导致「过拟合」。

**崩坏阶段的出现。** 正如式 (2) 所述，给定任意含噪图像 {{< math >}}$x_{t}${{< /math >}}，在经过一定的训练迭代次数后，微调后的扩散模型会预测出原始图像 {{< math >}}$\hat{x}_0=x^{*} + \delta_{t}${{< /math >}}，其中 {{< math >}}$x^{*} \in  \mathcal{I}_{\theta} \approx\mathcal{D}${{< /math >}}。误差项 {{< math >}}$\delta_{t}${{< /math >}} 的尺度与 {{< math >}}$\left\|x_t - \sqrt{\alpha_t} x^{*}\right\|_{2}${{< /math >}} 以及 {{< math >}}$\sigma_{1}${{< /math >}} 有关。我们基于从 {{< math >}}$\mathcal{N}(0, 1-\alpha_{t})${{< /math >}} 中采样得到的 {{< math >}}$x_{t}${{< /math >}} 来估计 {{< math >}}$\sigma_{1}${{< /math >}}，并在图 4 中给出了不同 {{< math >}}$t${{< /math >}} 上平均的 {{< math >}}$\sigma_{1,t}${{< /math >}}。结果表明，在中等迭代次数下 {{< math >}}$\sigma_{1}${{< /math >}} 仍保持在相对较高的水平，因而一旦 {{< math >}}$x_{t}${{< /math >}} 与 {{< math >}}$\sqrt{\alpha_t} x^{*}${{< /math >}} 不完全相同，就会产生显著的 {{< math >}}$\delta_{t}${{< /math >}}。

{{< figgrid caption="**图 5**：在 t = 100 的含噪图像 x<sub>t</sub> 的局部区域加入额外噪声 δ′ 后，预训练与微调后扩散模型的实验结果。预训练扩散模型能有效去除 δ′，生成高质量图像；微调后的扩散模型则无法消除 δ′，输出图像呈现崩坏模式。" >}}
bfm/additional_noise_2.0.jpg | 100
{{< /figgrid >}}


具体而言，对于仅有一张训练图像的情形，即 {{< math >}}$\mathcal{D} = \left\{ x' \right\}${{< /math >}}，我们将含噪图像设为 {{< math >}}$x_{t} = \sqrt{\alpha_{t}}x' + \sqrt{1-\alpha_{t}}\varepsilon +  \delta'${{< /math >}}，其中 {{< math >}}$\varepsilon\in\mathcal{N}(0,1)${{< /math >}}，而 {{< math >}}$\delta'${{< /math >}} 是引入到图像某一小块区域上的额外噪声。这个 {{< math >}}$\delta'${{< /math >}} 模拟了扩散模型的生成过程在某些 {{< math >}}$t${{< /math >}} 上不够准确的情形。我们进一步将时间变量设为 {{< math >}}$t=100${{< /math >}}，并对扩散模型微调 1000 次迭代，此时估计得到的 {{< math >}}$\sigma_{1}\approx4.8${{< /math >}}，如图 4 所示。根据式 (2) 与上述分析，我们可以计算出 {{< math >}}$\left\|\delta_{100}\right\|_{2}^{2}\approx 2.65\left\|\delta'\right\|_{2}^{2}${{< /math >}}。这意味着在该情形下，引入的额外噪声 {{< math >}}$\delta'${{< /math >}} 甚至被放大了，从而导致一个显著的误差项 {{< math >}}$\delta_{t}${{< /math >}}。图 5 展示了该设置下的实验结果，我们从中观察到一个显著的误差项 {{< math >}}$\delta_{t}${{< /math >}}，其形态与崩坏图案相似，这与上述分析是一致的。

**崩坏阶段的消失。** 然而，随着微调的持续进行，{{< math >}}$\sigma_{1}${{< /math >}} 会如图 4 所示不断下降，从而使预测误差 {{< math >}}$\delta_{t}${{< /math >}} 也随之减小。这表明崩坏阶段会消失，微调后的扩散模型逐渐进入只能严格重现训练图像 {{< math >}}$x' \in \mathcal{D}${{< /math >}} 的状态。这是一个典型的「过拟合」情形：微调后的扩散模型丧失了生成多样化输出的能力，因而变得不再可用。

总而言之，第 3.3 节中的分析表明：当扩散模型学习到的分布因 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 很小且标准差 {{< math >}}$\sigma_{1}${{< /math >}} 很高而受到严重限制时，崩坏阶段是如何发生的。

## 4. 将 BNNs 应用于少样本微调

### 4.1. 动机

基于我们的分析，崩坏阶段主要源于 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 较小所带来的受限的学习到的分布。受近期关于 BNNs 的工作 <a href="#ref-3">[3]</a>, <a href="#ref-8">[8]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-23">[23]</a>的启发——这些工作将参数 {{< math >}}$\theta${{< /math >}} 建模为随机变量——我们提出在扩散模型的少样本微调过程中应用 BNNs，作为一种简单而有效地扩展 {{< math >}}$\mathcal{I}_\theta${{< /math >}} 的方法。直观上，BNNs 的建模方式阻碍了扩散模型去学习训练数据集 {{< math >}}$\mathcal{D}${{< /math >}} 的精确分布。在不使用 BNNs 时，模型以高概率输出图像，聚焦于高置信度的情形；然而，使用 BNNs 训练的扩散模型会内在地生成一些较低概率的图像，迫使模型处理较低置信度的情形，从而阻碍扩散模型学习到精确的分布。因此，扩散模型被促使去学习一个更大且更鲁棒的分布，以应对这种随机性。

此外，微调过程中的采样随机性可以视为一种内在的数据增强。例如，在编码器–解码器结构视角下，编码器中的 BNNs 会在编码空间内引入扰动。这相当于编码空间层面的增强，最终在不损害图像质量的前提下通过增强鲁棒性与泛化性来影响解码得到的最终图像，从而隐式地扩展了相应的 {{< math >}}$\mathcal{I}_\theta${{< /math >}}。

### 4.2. 形式化

**建模。** BNNs 将参数 {{< math >}}$\theta${{< /math >}} 建模为随机变量。因此，带 BNN 的扩散模型学习到的分布为 {{< math >}}$P(x|\mathcal{D}) = \int P(x|\theta) P(\theta|\mathcal{D}) d\theta${{< /math >}}。具体而言，{{< math >}}$P(x|\theta)${{< /math >}} 是扩散模型所建模的图像分布，而 {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} 是在给定数据集 {{< math >}}$\mathcal{D}${{< /math >}} 下的后验参数分布。由于后验分布 {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} 是难以处理的，我们采用变分分布 {{< math >}}$Q_W (\theta)${{< /math >}} 来近似它。我们将每个参数 {{< math >}}$\theta${{< /math >}} 的变分分布建模为高斯分布：{{< math >}}$\theta \sim \mathcal{N}(\mu_{\theta}, \sigma_{\theta}^2)${{< /math >}}，其中 {{< math >}}$W = \left\{\mu_{\theta}, \sigma_{\theta}\right\}${{< /math >}} 为可训练参数。考虑到微调过程，我们用预训练扩散模型的相应参数（记为 {{< math >}}$\theta_0${{< /math >}}）来初始化期望项 {{< math >}}$\mu_{\theta}${{< /math >}}。遵循先前工作 <a href="#ref-3">[3]</a>，我们应用重参数化技巧来获取参数的梯度，详见附录第 A3 节。

**表 1**：使用 BNNs 微调在物体驱动生成与主体驱动生成下的性能。此处报告的是平均值，5 个不同随机种子之间的标准差见表 A6。

| 方法（物体驱动生成：DreamBooth 数据集） | Clip-T ↑ | Dino ↑ | Clip-I ↑ | Lpips ↑ | Clip-IQA ↑ | 方法（主体驱动生成：CelebA 数据集） | Clip-T ↑ | Dino ↑ | Clip-I ↑ | Lpips ↑ | Clip-IQA ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamBooth | 0.246 | 0.614 | 0.771 | 0.611 | 0.875 | DreamBooth | 0.186 | 0.642 | 0.723 | 0.511 | 0.789 |
| DreamBooth w/ BNNs | **0.256** | **0.633** | **0.785** | **0.640** | **0.893** | DreamBooth w/ BNNs | **0.205** | **0.696** | **0.757** | **0.515** | **0.811** |
| LoRA | 0.252 | 0.542 | 0.722 | 0.650 | 0.864 | LoRA | 0.216 | 0.602 | 0.656 | 0.644 | 0.804 |
| LoRA w/ BNNs | **0.261** | **0.618** | **0.769** | **0.678** | **0.890** | LoRA w/ BNNs | **0.227** | **0.604** | **0.663** | **0.688** | **0.824** |
| OFT | 0.233 | 0.649 | 0.786 | 0.629 | 0.861 | OFT | 0.164 | 0.675 | 0.728 | 0.549 | 0.784 |
| OFT w/ BNNs | **0.242** | **0.661** | **0.791** | **0.646** | **0.884** | OFT w/ BNNs | **0.185** | **0.696** | **0.743** | **0.570** | **0.798** |

**训练。** 在微调过程中，扩散模型通过最小化 Kullback–Leibler 散度（KL divergence）{{< math >}}$\mathrm{KL}(Q_W(\theta)|| P(\theta|\mathcal{D}))${{< /math >}} 进行训练，这等价于最小化

{{< math >}}
$$
\begin{aligned}
    \mathcal{L} &= - \int Q_W(\theta)\log \frac{P(\theta, \mathcal{D})}{Q_W(\theta)}  d\theta    \\ 
        &=  \mathbb{E}_{\theta\sim Q_W(\theta)} \underbrace{ -\log P(\mathcal{D}|\theta)}_{\mathcal{L}_{DM}}+\underbrace{\mathrm{KL}(Q_W(\theta) ||P(\theta))}_{\mathcal{L}_r}.
\end{aligned}
\tag{3}
$$
{{< /math >}}

遵循先前工作 <a href="#ref-33">[33]</a>，上述损失 {{< math >}}$\mathcal{L}${{< /math >}} 可以拆分为两项。在扩散模型中，第一项可以看作对训练数据集 {{< math >}}$\mathcal{D}${{< /math >}} 所建模的概率，它等价于第 2.1 节中给出的扩散损失 {{< math >}}$\mathcal{L}_{DM}${{< /math >}} 关于参数 {{< math >}}$\theta${{< /math >}} 的期望。第二项可以看作一个正则化项，用于限制变分分布 {{< math >}}$Q_W(\theta)${{< /math >}} 与先验分布 {{< math >}}$P(\theta)${{< /math >}} 之间的差异。我们将其命名为正则化损失 {{< math >}}$\mathcal{L}_r${{< /math >}}。在少样本微调中，我们认为预训练扩散模型天然地代表了先验信息，因此我们依据预训练扩散模型来设定先验分布 {{< math >}}$P(\theta)${{< /math >}}，即 {{< math >}}$P(\theta) = \mathcal{N}(\theta_0, \sigma^2)${{< /math >}}，其中 {{< math >}}$\sigma${{< /math >}} 是表示参数随机性的超参数。

在实践中，我们将学习目标形式化为 {{< math >}}$\mathcal{L}_{DM}${{< /math >}} 与 {{< math >}}$\mathcal{L}_r${{< /math >}} 的线性组合，并引入超参数 {{< math >}}$\lambda${{< /math >}}，即

{{< math >}}
$$
    W^* = \arg \min\limits_{W} \mathbb{E}_{\theta\sim Q_W(\theta)} \mathcal{L}_{DM} + \lambda \mathcal{L}_r. \tag{4}
$$
{{< /math >}}

训练过程总结于附录算法 1。经验上，我们发现仅使用 {{< math >}}$\mathbb{E}_{\theta\sim Q_W(\theta)} \mathcal{L}_{DM}${{< /math >}}，即把 {{< math >}}$\lambda${{< /math >}} 设为 0，就足以改善少样本微调。尽管如此，我们仍可以通过调整 {{< math >}}$\lambda${{< /math >}} 在生成多样性与图像保真度之间达成进一步的权衡。

**推理。** 在推理阶段，我们显式地将每个参数 {{< math >}}$\theta${{< /math >}} 替换为其均值 {{< math >}}$\mu_\theta${{< /math >}}，并像不带 BNNs 的扩散模型那样执行推理。这保证了在生产环境部署时，相比不带 BNNs 的微调后的扩散模型，我们不引入任何额外开销。

受先前关于 BNN 模块的方法 <a href="#ref-8">[8]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-23">[23]</a>的启发，我们在实践中只将参数的一个子集建模为随机变量，这降低了计算开销。使用 BNNs 微调扩散模型与现有的少样本微调方法兼容，包括 DreamBooth <a href="#ref-22">[22]</a>、LoRA <a href="#ref-11">[11]</a>和 OFT <a href="#ref-18">[18]</a>。更多细节见附录第 A4 节。

## 5. 实验

{{< figgrid caption="**图 6**：在主体驱动与物体驱动两类场景下，使用与不使用 BNNs 的少样本微调方法对比。图中同时给出按 Clip-I、Dino 和 Clip-IQA 衡量的最好情况与平均情况生成结果。挑选标准与更多可视化结果见附录第 A9 节与第 A12 节。" >}}
bfm/visualization_0.jpg | 99
{{< /figgrid >}}


我们将 BNNs 应用于不同任务下的多种少样本微调方法。对于物体驱动生成（微调后的扩散模型需要合成带有给定物体细节的图像），我们使用 DreamBooth <a href="#ref-22">[22]</a>数据集中全部 30 个类别，每个类别包含 4–6 张图像。对于主体驱动生成（微调后的扩散模型需要合成带有给定人物身份的图像），我们遵循已有研究 <a href="#ref-27">[27]</a>，从 CelebA-HQ <a href="#ref-16">[16]</a>中随机选取 30 个类别的图像，每个类别包含 5 张图像。大部分训练设置遵循已有方法 <a href="#ref-11">[11]</a>, <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>, <a href="#ref-27">[27]</a>。所有实验默认使用 5 个不同随机种子进行，此处报告的是平均性能。我们使用 Stable Diffusion v1.5（SD v1.5）作为默认的微调模型（注：<https://huggingface.co/runwayml/stable-diffusion-v1-5）。>

至于 BNNs，我们将默认的初始化标准差 {{< math >}}$\sigma_{\theta}${{< /math >}} 与先验标准差 {{< math >}}$\sigma${{< /math >}} 均设为 0.01。{{< math >}}$\lambda${{< /math >}} 默认设为 0。更多细节见附录第 A8.1 节。

遵循已有方法 <a href="#ref-11">[11]</a>, <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>，对于每个类别，我们微调一个扩散模型，并用各种提示词生成 100 张图像。这些生成图像被用于衡量不同少样本微调方法的性能。具体而言，我们使用 Clip-T <a href="#ref-19">[19]</a>衡量文本提示词保真度，使用 Clip-I <a href="#ref-9">[9]</a>和 Dino <a href="#ref-5">[5]</a>评估图像保真度，使用 Lpips <a href="#ref-34">[34]</a>评价生成多样性。此外，我们采用 Clip-IQA <a href="#ref-28">[28]</a>衡量无参考图像质量。关于这些评价指标的更多细节见附录第 A7 节。

### 5.1. 定量与可视化对比

我们在物体驱动生成与主体驱动生成两类任务下，将 BNNs 应用于不同的少样本微调方法。如表 1 与图 6 所示，BNNs 使所有少样本微调方法在文本提示词保真度（Clip-T）与图像保真度（Dino 和 Clip-I）上都获得了可观的提升。这些提升源于 BNNs 所带来的扩展后的学习到的分布，它使扩散模型更有能力根据不同提示词生成关于所学物体/主体的合理图像。BNNs 也大幅提升了无参考图像质量（Clip-IQA）。这主要是因为 BNNs 极大地减少了崩坏现象；由于崩坏图像在语义上与训练图像相距甚远，这也在一定程度上提升了图像保真度（Dino 和 Clip-I）。此外，我们观察到 BNNs 提升了生成多样性（Lpips）。我们认为这自然来自 BNNs 引入的随机性。

**表 2**：不同扩散模型下的性能。所有扩散模型均在 DreamBooth 数据集上用 DreamBooth 微调。

| 模型 | 方法 | Clip-T ↑ | Dino ↑ | Clip-I ↑ | Lpips ↑ | Clip-IQA ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| SD v1.5 | w/o BNNs | 0.246 | 0.614 | 0.771 | 0.611 | 0.875 |
| SD v1.5 | w/ BNNs | **0.256** | **0.633** | **0.785** | **0.640** | **0.893** |
| SD v1.4 | w/o BNNs | 0.248 | 0.594 | 0.762 | 0.618 | 0.872 |
| SD v1.4 | w/ BNNs | **0.249** | **0.620** | **0.777** | **0.656** | **0.895** |
| SD v2.0 | w/o BNNs | 0.240 | 0.563 | 0.739 | 0.604 | 0.875 |
| SD v2.0 | w/ BNNs | **0.248** | **0.610** | **0.764** | **0.649** | **0.925** |

### 5.2. 泛化性

在本节中，我们进一步表明 BNNs 可以应用于更广泛的场景并带来显著的性能提升，包括不同的扩散模型、不同的训练步数以及不同数量的训练图像。默认情况下，我们在 DreamBooth 上结合 BNNs 进行微调实验。

**不同的扩散模型。** 遵循已有工作 <a href="#ref-32">[32]</a>，我们在不同的扩散模型上进行实验。具体而言，除了默认的 SD v1.5，我们还在 SD v1.4 和 v2.0 <a href="#ref-20">[20]</a>上进行实验。训练细节见附录第 A8.2 节。表 2 表明，将 BNNs 应用于 SD 的不同扩散模型上，都能在多个评价指标上一致地改善少样本微调效果。

{{< figgrid caption="**图 7**：在 DreamBooth 上，每张图像采用不同训练迭代次数时，使用与不使用 BNNs 的性能对比。" >}}
bfm/generalization_on_training_steps_dino.png | 49 | **(a)** Dino
bfm/generalization_on_training_steps_clipiqa.png | 49 | **(b)** Clip-IQA
{{< /figgrid >}}


**训练迭代次数。** 图 7 显示，我们的方法一致地提升了图像质量（Clip-IQA）。当训练步数大于 {{< math >}}$100\times \rm{Num}${{< /math >}}（其中 {{< math >}}$\rm{Num}${{< /math >}} 表示微调时使用的图像数量）时，它同样提升了图像保真度（Dino）。在迭代次数较少时，模型会出现欠拟合。在这种情况下，由于 BNNs 鼓励模型学习一个更大的分布，它可能使欠拟合问题进一步加剧。这就导致了在 {{< math >}}$100\times \rm{Num}${{< /math >}} 处观察到的图像保真度（Dino）略有下降。

{{< figgrid caption="**图 8**：不同训练图像数量下的性能对比。" >}}
bfm/generalization_on_training_images_dino.png | 49 | **(a)** Dino
bfm/generalization_on_training_images_clipiqa.png | 49 | **(b)** Clip-IQA
{{< /figgrid >}}


{{< figgrid caption="**图 9**：关于不同 λ 与初始化 σ<sub>θ</sub> 的消融实验。" >}}
bfm/ablation_on_sigma_all.png | 49 | **(a)** 不同的初始化 σ<sub>θ</sub>
bfm/ablation_on_lambda_all.png | 49 | **(b)** 不同的 λ
{{< /figgrid >}}


**表 3**：在最优情形与平均情形两种场景下，有无 BNNs 的微调后扩散模型在各项评价指标上的用户研究结果。表中给出偏好有无 BNNs 微调后扩散模型所生成图像的用户百分比。

| 最优情形生成：方法 | 最优情形：主体保真度 | 最优情形：文本对齐度 | 最优情形：图像质量 | 平均情形生成：方法 | 平均情形：主体保真度 | 平均情形：文本对齐度 | 平均情形：图像质量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DreamBooth | 34.4% | 32.3% | 30.2% | DreamBooth | **50.5%** | 35.6% | 28.7% |
| DreamBooth w/ BNNs | **65.6%** | **67.7%** | **69.8%** | DreamBooth w/ BNNs | 49.5% | **64.4%** | **71.3%** |
| LoRA | 48.0% | 34.7% | 31.6% | LoRA | 27.5% | 26.5% | 24.5% |
| LoRA w/ BNNs | **52.0%** | **65.3%** | **68.4%** | LoRA w/ BNNs | **72.5%** | **73.5%** | **75.5%** |
| OFT | 30.6% | 34.7% | 40.8% | OFT | 41.1% | 26.8% | 39.3% |
| OFT w/ BNNs | **69.4%** | **65.3%** | **59.2%** | OFT w/ BNNs | **58.9%** | **73.2%** | **60.7%** |

**表 4**：将 BNNs 应用于扩散模型中不同层时的性能对比。所有实验均在 DreamBooth 数据集上使用 DreamBooth 微调完成。「N.A.」表示不应用 BNN。「CA」表示将 BNN 应用于交叉注意力模块。「UB」表示仅在上采样块上应用 BNNs。我们报告在一块 A100 GPU 上对每个类别微调时的显存开销与平均时间开销。

| 层 | Clip-T | Dino | Clip-I | Lpips | Clip-IQA | 显存 | 时间（秒） |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N.A. | 0.246 | 0.614 | 0.771 | 0.611 | 0.875 | 26.6 GB | 933 |
| Linear | 0.256 | 0.633 | 0.785 | 0.640 | 0.893 | 31.2 GB | 1107 |
| Linear (UB) | 0.259 | 0.614 | 0.776 | 0.646 | 0.890 | 29.0 GB | 1040 |
| Linear+CA | **0.272** | 0.611 | 0.760 | **0.672** | **0.897** | 32.5 GB | 1157 |
| LN+GN | 0.254 | **0.650** | **0.792** | 0.643 | 0.889 | 26.6 GB | 1088 |

**训练图像数量。** 我们还在不同训练图像数量下进行了实验。具体而言，我们使用 CelebA-HQ <a href="#ref-16">[16]</a>数据集，因为它每个类别包含足够多的图像。我们从 CelebA-HQ 数据集中随机选取五个类别，并在不同训练图像数量下开展实验。我们将训练迭代次数固定为 {{< math >}}$250\times\rm{Num}${{< /math >}}。

图 8 表明，我们的方法在不同训练图像数量下都一致地提升了图像保真度与图像质量。当训练图像更多时，这种提升更为明显。这主要是因为当训练图像数量 {{< math >}}$\rm{Num}${{< /math >}} 更大时，{{< math >}}$250\times\rm{Num}${{< /math >}} 通常会导致更严重的崩坏问题，因此 BNNs 通过扩展学习分布、缓解崩坏带来了更大的改善。

### 5.3. 消融实验

**初始化 {{< math >}}$\sigma_{\theta}${{< /math >}} 的尺度。** 初始化的标准差 {{< math >}}$\sigma_{\theta}${{< /math >}} 决定了微调过程中随机性的程度。我们在不同的初始化 {{< math >}}$\sigma_{\theta}${{< /math >}} 取值下应用 BNNs 进行实验。图 9a 中的实验结果表明，在初始化 {{< math >}}$\sigma_{\theta}${{< /math >}} 取值适中时，图像保真度（Dino）与图像质量（Clip-IQA）均得到提升。然而，当初始化 {{< math >}}$\sigma_{\theta}${{< /math >}} 过大时，扩散模型会崩溃，性能迅速下降。这表明在这种情形下引入的随机性过大，扩散模型几乎是在随机地更新。

**通过调整 {{< math >}}$\lambda${{< /math >}} 权衡多样性与保真度。** 式 (4) 表明 {{< math >}}$\lambda${{< /math >}} 控制着生成多样性与图像保真度之间的权衡。

如图 9b 所示，增大 {{< math >}}$\lambda${{< /math >}} 会带来生成多样性（Lpips）的提升，但代价是图像保真度（Dino）的下降。

**在何处应用 BNNs。** 如第 4.2 节所述，我们可以只将参数子集建模为随机变量，即只在扩散模型的一部分层上应用 BNNs。默认情况下，我们在 U-Net <a href="#ref-21">[21]</a>中除交叉注意力模块以外的所有线性层上应用 BNNs，并探究不同的选择如何影响性能与训练开销。

如表 4 所示，仅在 U-Net 的上采样块上应用 BNNs，扩散模型即可取得相对不错的性能，同时将被修改参数的比例降低到约 13.8%。我们还可以只在归一化层，即层归一化（Layer Normalization, LN）<a href="#ref-2">[2]</a>与组归一化（Group Normalization, GN）<a href="#ref-30">[30]</a>层上应用 BNNs，从而进一步降低训练开销。这将被修改参数的比例降至约 0.02%，同时仍保持相对较强的性能。

此外，当把 BNNs 应用于交叉注意力模块时，文本提示词保真度（Clip-T）显著提升，代价是图像保真度（Dino 与 Clip-I）的下降。直观上，这是因为输入图像只与有限的一组提示词相匹配，而在交叉注意力模块中应用 BNNs 会产生更为宽广的分布，从而匹配更多的提示词。

### 5.4. 用户研究

**用户研究设置。** 我们开展用户研究，以全面展示在扩散模型少样本微调中应用 BNNs 的优越性。具体而言，我们遵循已有方法 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>，对生成图像进行结构化人工评测，共有 101 位参与者。为此，我们使用了 DreamBooth 数据集与 CelebA-HQ 数据集。随后，我们在使用不同随机种子训练的五个模型上，为每个主体或物体使用从 25 条提示词中随机选取的一条提示词生成四张图像。为了进行全面比较，我们从生成图像中同时选取最优情形与平均情形（详见附录第 A9 节）。

每位参与者都需要比较由使用与不使用 BNNs 微调得到的模型所生成的图像对，涉及三种基线模型：DreamBooth、LoRA 和 OFT。每项任务包含三个二选一问题：

- **主体保真度：** 给定的两张图像中，哪一张所包含的主体或物体与下面的参考图像（来自训练数据集的一张图像）最相似？
- **文本对齐度：** 给定的两张图像中，哪一张与下面提供的文本描述（用于生成这些图像的提示词）最匹配？
- **图像质量：** 给定的两张图像中，哪一张表现出更高的图像质量？

**结果。** 结果呈现在表 3 中，其中给出了依据上述标准偏好每种方法（使用与不使用 BNNs）的参与者百分比。可以明显看出，无论是最优情形生成还是平均情形生成，使用 BNNs 的方法在几乎所有场景下都更受青睐。这种偏好在文本对齐度与整体图像质量方面尤为显著。

## 6. 结论

本文聚焦于扩散模型中的少样本微调，揭示了一个不寻常的「崩坏阶段」：图像保真度先是提升，随后因噪声图案而恶化，之后再度恢复。通过理论建模，我们将这一现象归因于少样本微调所固有的受限的学习到的分布。通过应用 BNNs 来拓宽学习到的分布，我们缓解了崩坏现象。在多种微调方法与数据集上的实验结果凸显了我们方法的通用性。

## 致谢

本工作得到国家重点研发计划（2022YFB4402102）与上海市可扩展计算与系统重点实验室的支持。（通讯作者：Tao Song）

## A1. 式 (2) 的推导

我们首先形式化地重述我们的假设。为方便起见，我们在假设与推导中使用等号而非约等号。

- 扩散模型将 {{< math >}}$x_0${{< /math >}} 与 {{< math >}}$x_t${{< /math >}} 的联合分布建模为一个多元高斯分布

  {{< math >}}
  $$
  \begin{aligned}
  &P_\theta([x_0, x_t]^T) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \\&=\mathcal{N}([x', \sqrt{\alpha_t}x']^T, \left[\begin{array}{cc}
  \sigma_1^2 & c \\
  c & \alpha_t \sigma_1^2 + (1 - \alpha_t)
  \end{array}\right]),
  \end{aligned}
  \tag{5}
  $$
  {{< /math >}}

  其中 {{< math >}}$c${{< /math >}} 表示 {{< math >}}$x_0${{< /math >}} 与 {{< math >}}$x_t${{< /math >}} 之间未知的协方差。

- 给定 {{< math >}}$x_0 = x'${{< /math >}} 时 {{< math >}}$x_t${{< /math >}} 的条件概率为

  {{< math >}}
  $$
  P_\theta(x_t \mid x_0=x') = \mathcal{N}(\sqrt{\alpha_t} x', (1 - \alpha_t)).
  \tag{6}
  $$
  {{< /math >}}

将 {{< math >}}$\boldsymbol{\Sigma}${{< /math >}} 的逆矩阵记为

{{< math >}}
$$
\boldsymbol{\Sigma}^{-1} = \left[\begin{array}{cc}
\lambda_{11} & \lambda_{12} \\
\lambda_{21} & \lambda_{22}
\end{array}\right]
= \frac{1}{|\boldsymbol{\Sigma}|}\left[\begin{array}{cc}
\alpha_t \sigma_1^2 + 1 - \alpha_t & -c \\
-c & \sigma_1^2
\end{array}\right],
\tag{7}
$$
{{< /math >}}

其中 {{< math >}}$|\boldsymbol{\Sigma}| = \sigma_1^2 (\alpha_t \sigma_1^2 + (1 - \alpha_t)) - c^2${{< /math >}} 是 {{< math >}}$\boldsymbol{\Sigma}${{< /math >}} 的行列式。根据联合高斯分布的性质，条件分布 {{< math >}}$P(x_t|x_0)${{< /math >}} 可以表示为

{{< math >}}
$$
P(x_t|x_0) = \mathcal{N}(\frac{\lambda_{22} \sqrt{\alpha_t x'} + \lambda_{12}x_0 - \lambda_{12}x' }{\lambda_{22}}, \frac{1}{\lambda_{22}}).
\tag{8}
$$
{{< /math >}}

根据式 (6)，

{{< math >}}
$$
\frac{1}{\lambda_{22}} = (1-\alpha_t),
\tag{9}
$$
{{< /math >}}

这意味着 {{< math >}}$c = \pm \sqrt{\alpha_t} \sigma_1^2${{< /math >}}。因此我们可知联合分布为

{{< math >}}
$$
\begin{aligned}
&P_\theta([x_0, x_t]^T) = \mathcal{N}(\boldsymbol{\mu}, \boldsymbol{\Sigma}) \\&=\mathcal{N}([x', \sqrt{\alpha_t}x']^T, \left[\begin{array}{cc}
\sigma_1^2 & \pm \sqrt{\alpha_t} \sigma_1^2 \\
\pm \sqrt{\alpha_t} \sigma_1^2 & \alpha_t \sigma_1^2 + (1 - \alpha_t)
\end{array}\right]).
\end{aligned}
\tag{10}
$$
{{< /math >}}

再次根据联合高斯分布的性质，我们有

{{< math >}}
$$
P(x_0 | x_t) = \mathcal{N}(x' \pm \frac{ \sqrt{\alpha_t} \sigma_1^2 (y - \sqrt{\alpha_t} x')}{\alpha_t \sigma_1^2 + (1 - \alpha_t)}, \frac{1 - \alpha_t}{\alpha_t \sigma_1^2 + 1 - \alpha_t}).
\tag{11}
$$
{{< /math >}}

在实践中，取正号更为合理，因为它表明预测图像 {{< math >}}$x_0${{< /math >}} 与输入含噪图像 {{< math >}}$x_t${{< /math >}} 的偏差是相互一致的。

因此，

{{< math >}}
$$
P(x_0 | x_t) = \mathcal{N}(x' + \frac{ \sqrt{\alpha_t} \sigma_1^2 (y - \sqrt{\alpha_t} x')}{\alpha_t \sigma_1^2 + (1 - \alpha_t)}, \frac{1 - \alpha_t}{\alpha_t \sigma_1^2 + 1 - \alpha_t}).
\tag{12}
$$
{{< /math >}}

## A2. 误差项在微调过程中趋于零的证明

本节中，我们给出一个直接的证明，说明当扩散损失趋于零时误差项也趋于零。

**证明**：正如先前工作所指出的 <a href="#ref-10">[10]</a>，扩散模型在给定分布上的概率以其证据下界（Evidence Lower Bound, ELBO）为下界：

{{< math >}}
$$
-\log P_{\theta}(x_{0}) \geq \mathbf{E}_{q}(x_{1:T} |x_0) \log \frac{ P_{\theta}(x_{0:T})}{Q(x_{1:T} |x_0)},
\tag{13}
$$
{{< /math >}}

并可简化为扩散损失：

{{< math >}}
$$
L_{DM}(x_0):=\mathbf{E}_{t, \epsilon\in N(0,1)} \Vert\varepsilon_{\theta} (\sqrt{\alpha_{t}}x_0 + \sqrt{1-\alpha_{t}}\epsilon, t) -\epsilon\Vert^{2}.
\tag{14}
$$
{{< /math >}}

微调方法的主要目标损失是直接最小化微调数据分布内数据的扩散损失。这可以理解为最小化 KL 散度 {{< math >}}$\mathbf{E}_{Q}(x'_{1:T} | x'_0) \log \frac{P_{\theta}(x'_{0:T})}{Q(x'_{1:T} | x'_0)}${{< /math >}}，其中 {{< math >}}$Q(x'_0)${{< /math >}} 表示微调数据分布。

在 {{< math >}}$L_{DM}(x'_0) \to 0${{< /math >}} 的最优条件下，我们有 {{< math >}}$P_{\theta}(x'_0) \to 1${{< /math >}} 且 {{< math >}}$P_{\theta}(x'_{t+1} | x'_{0}) \to Q(x'_{t+1} | x'_{0})${{< /math >}}。这意味着学习到的分布 {{< math >}}$P${{< /math >}} 与微调数据分布 {{< math >}}$Q${{< /math >}} 是一致的，因此我们有

{{< math >}}
$$
\arg\max P_\theta(x_0|x_t) = \arg\max Q(x_0|x_t) = x'.
\tag{15}
$$
{{< /math >}}

它对应于式 (1) 中的 {{< math >}}$\sigma_{1}=0${{< /math >}}，从而导致 {{< math >}}$\delta_{t}=0${{< /math >}}。

这从另一个侧面验证了我们建模的正确性，因而进一步支持了我们的理论分析。

## A3. 应用 BNNs 的细节

在微调中应用 BNNs 的训练过程总结于算法 1。
为了获得变分参数的梯度，即 {{< math >}}$W = \{\mu_\theta, \sigma_\theta\}${{< /math >}} 的梯度，我们采用 BNNs 中常用的重参数化技巧。

具体而言，我们首先为每个 {{< math >}}$\theta${{< /math >}} 采样一个单位高斯变量 {{< math >}}$\varepsilon_\theta${{< /math >}}，然后通过 {{< math >}}$\theta = \mu_\theta + \sigma_\theta \times \varepsilon_\theta${{< /math >}} 得到 {{< math >}}$\theta${{< /math >}} 的一个后验样本。
因此，梯度可以按如下方式计算：

{{< math >}}
$$
\begin{aligned}
\frac{\partial}{\partial W} \mathcal{L} &=
    \frac{\partial}{\partial W} \left[\mathbb{E}_{Q_W(\theta)}\mathcal{L}_{DM} + \mathcal{L}_r\right] \\ &= \mathbb{E}_{\varepsilon_\theta \sim \mathcal{N}(0,I)}\left[ \frac{\partial \mathcal{L}_{DM}}{\partial \theta} \frac{\partial \theta}{\partial W} + \frac{\partial \mathcal{L}_r}{\partial W}\right].
\end{aligned}
\tag{16}
$$
{{< /math >}}

详细推导可参阅先前工作 <a href="#ref-3">[3]</a>中的命题 1。

**算法 1**：使用 BNNs 微调扩散模型

```
输入：已初始化的变分参数 W = {μ_θ, σ_θ}，先验分布 P(θ) = N(θ_0, σ²)，
      微调数据集 D，微调迭代次数 N，超参数 λ
输出：微调后的变分参数 W = {μ_θ, σ_θ}

for i = 0 to N-1 do
    采样 ε_θ ~ N(0, I)。
    计算 θ = μ_θ + ε_θ ∘ σ_θ。
    采样 x ∈ D，t ~ U(1, 1000)，噪声 ε_t ~ N(0,1)
    计算 L_DM = ||ε_t - ε_θ(x_t, t)||²。
    计算 L_r = KL(P(θ) || N(μ_θ, σ_θ²))。
    计算 L = L_DM + λ L_r
    反向传播 L 并更新 μ_θ、σ_θ。
end for
```

## A4. 在不同少样本微调方法上应用 BNNs

**在 DreamBooth 上应用 BNNs。** DreamBooth 是一种全参数微调方法，也是主流的微调方法之一 <a href="#ref-22">[22]</a>。因此，DreamBooth 中的所有参数都可以被建模为 BNNs 的参数。

**在 LoRA 上应用 BNNs。** LoRA <a href="#ref-11">[11]</a>是一种经典的、轻量而有效的少样本微调方法。在 LoRA 层中，权重矩阵 {{< math >}}$\mathbf{W} \in \mathbb{R}^{d \times k}${{< /math >}} 被建模为预训练模型的固定权重与一个可训练的低秩分解之和：{{< math >}}$\mathbf{W} = \mathbf{W}_0 + \mathbf{B} \mathbf{A}${{< /math >}}，其中 {{< math >}}$\mathbf{W}_0 \in \mathbb{R}^{d \times k}, \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}${{< /math >}}，秩为 {{< math >}}$r${{< /math >}} <a href="#ref-11">[11]</a>。在我们的实现中，我们只将上矩阵 {{< math >}}$\mathbf{A}${{< /math >}} 转换为随机变量，而下矩阵 {{< math >}}$\mathbf{B}${{< /math >}} 仍保持为通常的可训练参数，以使 {{< math >}}$P(\mathbf{W})${{< /math >}} 成为高斯分布。这一设计在保持其有效性的同时，也降低了训练过程中额外的计算开销。

**在 OFT 上应用 BNNs。** OFT 是一种仅通过正交变换来调整权重的少样本微调方法 <a href="#ref-18">[18]</a>。在 OFT 层中，权重矩阵 {{< math >}}$\mathbf{W} \in \mathbb{R}^{d \times k}${{< /math >}} 被建模为 {{< math >}}$ \mathbf{W} = \mathbf{R} \mathbf{W}_0 ${{< /math >}}，其中 {{< math >}}$\mathbf{R}${{< /math >}} 通过 {{< math >}}$    \mathbf{R} = (\boldsymbol{I}+ 0.5 (\mathbf{Q} - \mathbf{Q}^T))(I-0.5 (\mathbf{Q} - \mathbf{Q}^T))^{-1}${{< /math >}} 被保证为正交矩阵，而 {{< math >}}$\mathbf{Q}${{< /math >}} 是原始 OFT 方法中的可训练参数。为保证正交性不被 BNNs 中的随机采样所破坏，我们只将可训练参数 {{< math >}}$\mathbf{Q}${{< /math >}} 转换为随机变量。

因此，经过上述变换后，{{< math >}}$\mathbf{R}${{< /math >}} 仍保持为正交矩阵，而 {{< math >}}$\mathbf{W}${{< /math >}} 仍保持为原始预训练权重 {{< math >}}$\mathbf{W}_0${{< /math >}} 的正交化变换。

**表 A5**：用于评测的提示词。[V] 表示特殊词元，[object] 表示物体的类型。

| 物体驱动生成所用提示词 | 主体驱动生成所用提示词 |
| --- | --- |
| a [V] [object] in the jungle | a photo of a [V] person wearing sunglasses |
| a [V] [object] in the snow | a photo of a [V] person with snowflakes in their hair |
| a [V] [object] on the beach | a photo of a [V] person with beachy hair waves |
| a [V] [object] on a cobblestone street | a photo of a [V] person wearing a beret |
| a [V] [object] on top of pink fabric | a photo of a [V] person with a neutral expression |
| a [V] [object] on top of a wooden floor | a photo of a [V] person with a contemplative look |
| a [V] [object] with a city in the background | a photo of a [V] person laughing heartily |
| a [V] [object] with a mountain in the background | a photo of a [V] person with an amused smile |
| a [V] [object] with a blue house in the background | a photo of a [V] person with forest green eyeshadow |
| a [V] [object] on top of a purple rug in a forest | a photo of a [V] person wearing a red hat |
| a [V] [object] wearing a red hat | a photo of a [V] person with a slight grin |
| a [V] [object] wearing a santa hat | a photo of a [V] person with a thoughtful gaze |
| a [V] [object] wearing a rainbow scarf | a photo of a [V] person wearing a black top hat |
| a [V] [object] wearing a black top hat and a monocle | a photo of a [V] person in a chef hat |
| a [V] [object] in a chef outfit | a photo of a [V] person in a firefighter helmet |
| a [V] [object] in a firefighter outfit | a photo of a [V] person in a police cap |
| a [V] [object] in a police outfit | a photo of a [V] person wearing pink glasses |
| a [V] [object] wearing pink glasses | a photo of a [V] person wearing a yellow headband |
| a [V] [object] wearing a yellow shirt | a photo of a [V] person in a purple wizard hat |
| a [V] [object] in a purple wizard outfit | a photo of a [V] person smiling |
| a red [V] [object] | a photo of a [V] person frowning |
| a purple [V] [object] | a photo of a [V] person looking surprised |
| a shiny [V] [object] | a photo of a [V] person winking |
| a wet [V] [object] | a photo of a [V] person yawning |
| a cube shaped [V] [object] | a photo of a [V] person laughing |

## A5. 验证实验的细节

**证明 {{< math >}}$x_{t}=0${{< /math >}} 位于预训练扩散模型的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 之内。** 我们给出两个证明，说明 {{< math >}}$x_{t}=0${{< /math >}} 位于预训练扩散模型的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 之内。这里使用 SD v1.5 作为预训练扩散模型。

- 如图 A10a 所示，我们使用提示词「a simple, solid gray image with no textures or variations」来生成图像，并观察到预训练扩散模型能够生成这类不含噪声的图像。
- 我们使用 diffusers 提供的 Img2Img Pipeline（注：<https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py>），且不提供任何提示词，即进行无条件生成。随后我们将输入图像设为一张空白图像，并将图生图强度设为 0.1，这意味着输入的含噪图像 {{< math >}}$x_{100}=\alpha_{100}\varepsilon${{< /math >}}，其中 {{< math >}}$\varepsilon\in \mathcal{N}(0,1)${{< /math >}}。如图 A10b 所示，我们同样可以观察到去噪结果完全不含噪声。

这两个结果都支持我们的论断：{{< math >}}$x_{t}=0${{< /math >}} 天然位于预训练扩散模型的 {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} 之内。

{{< figgrid caption="**图 A10**：证明 x<sub>t</sub> = 0 落在预训练扩散模型的 ℐ<sub>θ</sub> 之内。" >}}
bfm/show_zero_2.jpg | 48 | **(a)** 基于给定提示词生成的图像。
bfm/show_zero_1.png | 48 | **(b)** 以纯色图像为输入的 img2img 结果。
{{< /figgrid >}}


我们使用 DreamBooth 且不使用 PPL 损失 <a href="#ref-22">[22]</a>对一个 SD v1.5 进行微调。学习率固定为 {{< math >}}$5\times10^{-6}${{< /math >}}，并且只微调 U-Net <a href="#ref-21">[21]</a>。我们以 DreamBooth 数据集中的 backpack 类别为例，使用提示词「a [V] backpack」进行训练，其中「[V]」是特殊词元。我们设 {{< math >}}$x_{100}=0${{< /math >}}，并使用 diffusers 提供的 Img2Img Pipeline 展示一步去噪的结果。在去噪过程中，对于预训练扩散模型与微调后的扩散模型，提示词均固定为「a [V] backpack」。更多结果见图 A11。

{{< figgrid caption="**图 A11**：在不同扩散模型、不同推理步数下，输入 x<sub>t</sub> = 0 时生成图像的更多结果。" >}}
bfm/full_zero.jpg | 100
{{< /figgrid >}}


## A6. 对我们建模的补充支撑

为了进一步支撑我们在第 3.2 节中的建模，我们给出在不同 {{< math >}}$k${{< /math >}} 取值下 {{< math >}}$x_t = kx'${{< /math >}} 的更多结果。给定训练提示词时，根据我们在式 (2) 中的建模，微调后的扩散模型应当预测出原始图像 {{< math >}}$\hat{x_0} = (1 + \frac{\sqrt{\alpha_t} \sigma_1^2}{\alpha_t \sigma_1^2 + (1 - \alpha_t)} (k - \sqrt{\alpha_t}))x'${{< /math >}}，它是 {{< math >}}$x'${{< /math >}} 的一个缩放。

相比之下，预训练模型的生成结果不应与给定的训练样本 {{< math >}}$kx'${{< /math >}} 表现出类似的相关性，因为 {{< math >}}$kx'${{< /math >}} 既不属于预训练数据集，也不属于其训练分布 {{< math >}}$\mathcal{I}_\theta${{< /math >}}。

图 A12 所示的实验结果支持了我们的分析，从而进一步确认了我们建模的合理性。

{{< figgrid caption="**图 A12**：取 x<sub>t</sub> = kx′、k 取不同值时，预训练与微调后扩散模型的去噪结果。" >}}
bfm/merged_with_labels.png | 100
{{< /figgrid >}}


## A7. 评价指标

我们使用以下评价指标来稳健地衡量微调后的扩散模型在不同方面的表现：

**文本提示词保真度：** 遵循已有论文 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>，我们使用文本提示词与生成图像的 CLIP <a href="#ref-19">[19]</a>嵌入之间的平均相似度，记为 Clip-T。

**图像保真度：** 遵循已有论文 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>，我们计算生成图像与训练图像的 CLIP <a href="#ref-19">[19]</a>嵌入和 DINO <a href="#ref-5">[5]</a>嵌入之间的平均相似度，分别记为 Clip-I <a href="#ref-9">[9]</a>与 Dino。

**生成多样性：** 遵循已有论文 <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>，我们计算微调后的扩散模型所生成图像之间的平均 Lpips <a href="#ref-34">[34]</a>距离。

**图像质量：** 我们发现崩坏会大幅降低图像的视觉质量，使其无法使用。然而，全参考的图像质量评价指标无法完整地反映这类图像质量退化。因此，我们额外加入一个无参考图像评价指标来进行衡量。我们使用 Clip-IQA <a href="#ref-28">[28]</a>进行衡量，它是最先进的无参考图像质量评价指标之一。

## A8. 微调设置

**表 A6**：表 1 中所示结果的标准差。

| 方法（物体驱动生成：DreamBooth 数据集） | Clip-T | Dino | Clip-I | Lpips | Clip-IQA | 方法（主体驱动生成：CelebA 数据集） | Clip-T | Dino | Clip-I | Lpips | Clip-IQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamBooth | 0.0017 | 0.0057 | 0.0037 | 0.0051 | 0.0028 | DreamBooth | 0.0065 | 0.0179 | 0.0115 | 0.0087 | 0.0078 |
| DreamBooth w/ BNNs | 0.0016 | 0.0056 | 0.0035 | 0.0054 | 0.0023 | DreamBooth w/ BNNs | 0.0036 | 0.0102 | 0.0090 | 0.0191 | 0.0095 |
| LoRA | 0.0018 | 0.0087 | 0.0045 | 0.0048 | 0.0084 | LoRA | 0.0034 | 0.0085 | 0.0059 | 0.0060 | 0.0068 |
| LoRA w/ BNNs | 0.0032 | 0.0104 | 0.0064 | 0.0072 | 0.0021 | LoRA w/ BNNs | 0.0025 | 0.0049 | 0.0163 | 0.0078 | 0.0065 |
| OFT | 0.0030 | 0.0063 | 0.0041 | 0.0081 | 0.0045 | OFT | 0.0026 | 0.0194 | 0.0141 | 0.0130 | 0.0115 |
| OFT w/ BNNs | 0.0010 | 0.0028 | 0.0026 | 0.0062 | 0.0068 | OFT w/ BNNs | 0.0024 | 0.0078 | 0.0066 | 0.0061 | 0.0094 |

我们在一块 A100 GPU 上开展将 BNNs 应用于不同微调方法的实验。所有实验默认在全部 30 个类别、5 个不同随机种子下进行，并报告平均性能。我们在表 1 中的主要结果所对应的标准差见表 A6。

### A8.1. 少样本微调超参数

下面给出在我们的默认模型（即 SD v1.5）上各少样本微调方法的参数细节。我们用 {{< math >}}$\rm{Num}${{< /math >}} 表示用于训练的图像数量。

**Dreambooth**：我们使用 Diffusers 提供的训练脚本（注：<https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py）。训练过程中仅对> U-Net 进行微调。默认情况下，在 DreamBooth 数据集上训练步数设为 {{< math >}}$200\times\rm{Num}${{< /math >}}，在 CelebA 上设为 {{< math >}}$250\times\rm{Num}${{< /math >}}，学习率为 {{< math >}}$5 \times 10^{-6}${{< /math >}}。批大小设为 1，用于计算先验损失的类别图像数量默认为 {{< math >}}$200\times\rm{Num}${{< /math >}}。先验损失权重固定为 1.0。对于 DreamBooth 数据集，训练实例提示词为「a photo of a [V] {class prompt}」，其中 {class prompt} 指图像的类型（例如 dog、cat 等）。对于 CelebA 数据集，训练实例提示词为「a photo of a [V] person」。

**LoRA**：我们使用 Diffusers 提供的训练脚本（注：<https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py）。除学习率与训练步数分别调整为> {{< math >}}$1 \times 10^{-4}${{< /math >}} 和 {{< math >}}$400\times\rm{Num}${{< /math >}} 外，其余默认参数均与 Dreambooth (No Prior) 的情形保持一致。

**OFT**：我们使用作者提供的训练脚本（注：<https://github.com/Zeju1997/oft）。除学习率调整为> {{< math >}}$1 \times 10^{-4}${{< /math >}} 外，其余默认参数均与 Dreambooth (No Prior) 的情形保持一致。

对于所有涉及应用 BNNs 的实验，我们对学习步数、学习率、训练提示词以及其它超参数均保持默认设置。

在评测方面，每个训练检查点针对每条提示词生成四张图像，因此由 25 条不同提示词共得到 100 张图像。用于评测的提示词见表 A5，它们覆盖面广泛，可全面评估扩散模型的多样性与质量。

### A8.2. 不同架构扩散模型的训练设置

本节给出在 DreamBooth 下将 BNNs 应用于不同扩散模型时的训练设置。所有情形中学习率均固定为 {{< math >}}$5 \times 10^{-6}${{< /math >}}。SD v1.4 与 SD v2.0 的训练迭代次数分别设为 {{< math >}}$200 \times \mathrm{Num}${{< /math >}} 和 {{< math >}}$400 \times \mathrm{Num}${{< /math >}}。

对于在 SD v1.4 上应用 BNNs 的情形，我们将超参数设为 {{< math >}}$\lambda=0.1${{< /math >}}。其余所有超参数均设为默认值。

## A9. 最优情形生成与平均情形生成的对比

在实际场景中，用户通常会生成多张图像，并手动挑选最合适的一张来使用。因此，对最优情形的比较本质上评估的是所选图像的质量，而对平均情形的比较则评估的是挑选出一张合格图像的难度。相反，最差情形的场景可能实际意义较小，因为用户可以不断重新生成图像直到获得满意的结果，从而有效地规避了这类情形。

因此，本文着重展示最优情形与平均情形的生成结果。具体而言，我们首先使用 CLIP-T 和 Dino 筛选生成图像，保留排名前 90% 的图像，以确保其与提示词以及学习到的概念相符。随后，我们利用 CLIP-IQA 来识别出质量最高的图像和质量处于平均水平的图像。这一做法能够更全面地评估模型的性能。

尽管如此，为了完整起见，我们仍在图 A13 中给出了应用 BNNs 前后最差情形生成结果的一些对比。具体而言，我们展示的是使用 CLIP-IQA 评估得到的图像质量最低的那些图像。我们的实验覆盖了 DreamBooth 数据集中的 10 个随机类别，并以 DreamBooth 作为基线微调方法。结果表明，未使用 BNNs 微调的扩散模型通常会出现崩坏，而 BNNs 成功地缓解了这一现象，带来了更好的生成质量。

{{< figgrid caption="**图 A13**：按 CLIP-IQA 衡量的最差情况生成结果可视化。" >}}
bfm/stitched_image.jpg | 100
{{< /figgrid >}}


## A10. 局限性与未来工作

尽管应用 BNNs 的少样本微调扩散模型已展现出令人鼓舞的改进，本文同时也引出了若干有趣的开放问题。

首先，额外引入的随机性可能会使微调变慢。当扩散模型处于欠拟合状态时，这可能会降低生成质量。此外，当微调迭代次数不足时，模型学习图像中极为细致的图案的能力可能会有所下降。未来的工作可以聚焦于这些问题。

## A11. 更广泛的影响

本文致力于推进扩散模型的少样本微调技术，为在各种场景下创作个性化图像提供更有效的工具。正如本文所述，以往的少样本微调方法面临着崩坏现象。我们的方法利用 BNNs 解决了这一现象，并生成了质量更高的图像。

然而，该技术也存在被滥用的可能，恶意主体可能利用这些技术进行欺骗或误导。此类挑战凸显了在该领域持续探索的迫切必要性。个性化生成模型的发展及其符合伦理的应用不仅至关重要，也为未来研究提供了广阔空间。

## A12. 更多可视化结果

我们在图 A14 和图 A15 中展示了更多可视化结果。

{{< figgrid caption="**图 A14**：主体驱动与物体驱动场景下的更多可视化结果。" >}}
bfm/compress_visualization_1.jpg | 78
bfm/compress_visualization_2.jpg | 78
{{< /figgrid >}}


{{< figgrid caption="**图 A15**：主体驱动与物体驱动场景下的更多可视化结果。" >}}
bfm/compress_visualization_3.jpg | 80
{{< /figgrid >}}


## 参考文献

<span id="ref-1"></span>[1] Julyan Arbel, Konstantinos Pitas, Mariia Vladimirova, and Vincent Fortuin. 2023. A Primer on Bayesian Neural Networks: Review and Debates. *arXiv preprint arXiv:2309.16314* (2023).

<span id="ref-2"></span>[2] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. 2016. Layer Normalization. *arXiv preprint arXiv:1607.06450* (2016).

<span id="ref-3"></span>[3] Charles Blundell, Julien Cornebise, Koray Kavukcuoglu, and Daan Wierstra. 2015. Weight Uncertainty in Neural Network. In *ICML*.

<span id="ref-4"></span>[4] Wray L Buntine. 1991. Bayesian Backpropagation. *Complex Systems* 5 (1991), 603–643.

<span id="ref-5"></span>[5] Mathilde Caron, Hugo Touvron, Ishan Misra, Hervé Jégou, Julien Mairal, Piotr Bojanowski, and Armand Joulin. 2021. Emerging Properties in Self-supervised Vision Transformers. In *ICCV*.

<span id="ref-6"></span>[6] Giannis Daras, Kulin Shah, Yuval Dagan, Aravind Gollakota, Alex Dimakis, and Adam Klivans. 2023. Ambient Diffusion: Learning Clean Distributions from Corrupted Data. In *NeurIPS*.

<span id="ref-7"></span>[7] Rinon Gal, Yuval Alaluf, Yuval Atzmon, Or Patashnik, Amit H Bermano, Gal Chechik, and Daniel Cohen-Or. 2022. An Image Is Worth One Word: Personalizing Text-to-Image Generation Using Textual Inversion. *arXiv preprint arXiv:2208.01618* (2022).

<span id="ref-8"></span>[8] James Harrison, John Willes, and Jasper Snoek. 2023. Variational Bayesian Last Layers. In *AABI*.

<span id="ref-9"></span>[9] Jack Hessel, Ari Holtzman, Maxwell Forbes, Ronan Le Bras, and Yejin Choi. 2021. Clipscore: A Reference-free Evaluation Metric for Image Captioning. *arXiv preprint arXiv:2104.08718* (2021).

<span id="ref-10"></span>[10] Jonathan Ho, Ajay Jain, and Pieter Abbeel. 2020. Denoising Diffusion Probabilistic Models. In *NeurIPS*.

<span id="ref-11"></span>[11] Edward J Hu, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen, et al. 2021. LoRA: Low-Rank Adaptation of Large Language Models. In *ICLR*.

<span id="ref-12"></span>[12] Laurent Valentin Jospin, Hamid Laga, Farid Boussaid, Wray Buntine, and Mohammed Bennamoun. 2022. Hands-on Bayesian neural networks—A tutorial for deep learning users. *IEEE Computational Intelligence Magazine* 17, 2 (2022), 29–48.

<span id="ref-13"></span>[13] Tero Karras, Miika Aittala, Janne Hellsten, Samuli Laine, Jaakko Lehtinen, and Timo Aila. 2020. Training Generative Adversarial Networks with Limited Data. In *NeurIPS*.

<span id="ref-14"></span>[14] Bahjat Kawar, Shiran Zada, Oran Lang, Omer Tov, Huiwen Chang, Tali Dekel, Inbar Mosseri, and Michal Irani. 2022. Imagic: Text-Based Real Image Editing With Diffusion Models. *arXiv preprint arXiv:2210.09276* (2022).

<span id="ref-15"></span>[15] Agustinus Kristiadi, Matthias Hein, and Philipp Hennig. 2020. Being Bayesian, Even Just a Bit, Fixes Overconfidence in ReLU Networks. In *ICML*.

<span id="ref-16"></span>[16] Ziwei Liu, Ping Luo, Xiaogang Wang, and Xiaoou Tang. 2018. Large-scale celebfaces attributes (celeba) dataset. *Retrieved August* 15, 2018 (2018), 11.

<span id="ref-17"></span>[17] Radford M Neal. 2012. *Bayesian Learning for Neural Networks*. Vol. 118. Springer Science & Business Media.

<span id="ref-18"></span>[18] Zeju Qiu, Weiyang Liu, Haiwen Feng, Yuxuan Xue, Yao Feng, Zhen Liu, Dan Zhang, Adrian Weller, and Bernhard Schölkopf. 2023. Controlling Text-to-Image Diffusion by Orthogonal Finetuning. In *NeurIPS*.

<span id="ref-19"></span>[19] Alec Radford, Jong Wook Kim, Chris Hallacy, Aditya Ramesh, Gabriel Goh, Sandhini Agarwal, Girish Sastry, Amanda Askell, Pamela Mishkin, Jack Clark, et al. 2021. Learning Transferable Visual Models from Natural Language Supervision. In *ICML*.

<span id="ref-20"></span>[20] Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, and Björn Ommer. 2022. High-Resolution Image Synthesis With Latent Diffusion Models. In *CVPR*.

<span id="ref-21"></span>[21] Olaf Ronneberger, Philipp Fischer, and Thomas Brox. 2015. U-net: Convolutional Networks for Biomedical Image Segmentation. In *MICCAI*.

<span id="ref-22"></span>[22] Nataniel Ruiz, Yuanzhen Li, Varun Jampani, Yael Pritch, Michael Rubinstein, and Kfir Aberman. 2023. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. In *CVPR*.

<span id="ref-23"></span>[23] Mrinank Sharma, Sebastian Farquhar, Eric Nalisnick, and Tom Rainforth. 2023. Do Bayesian Neural Networks Need To Be Fully Stochastic?. In *AISTATS*.

<span id="ref-24"></span>[24] Jascha Sohl-Dickstein, Eric Weiss, Niru Maheswaranathan, and Surya Ganguli. 2015. Deep Unsupervised Learning Using Nonequilibrium Thermodynamics. In *ICML*.

<span id="ref-25"></span>[25] Yang Song and Stefano Ermon. 2019. Generative Modeling by Estimating Gradients of the Data Distribution. In *NeurIPS*.

<span id="ref-26"></span>[26] Yang Song, Jascha Sohl-Dickstein, Diederik P Kingma, Abhishek Kumar, Stefano Ermon, and Ben Poole. 2020. Score-based Generative Modeling through Stochastic Differential Equations. *arXiv preprint arXiv:2011.13456* (2020).

<span id="ref-27"></span>[27] Thanh Van Le, Hao Phung, Thuan Hoang Nguyen, Quan Dao, Ngoc Tran, and Anh Tran. 2023. Anti-DreamBooth: Protecting Users from Personalized Text-to-image Synthesis. In *ICCV*.

<span id="ref-28"></span>[28] Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. 2023. Exploring Clip for Assessing the Look and Feel of Images. In *AAAI*.

<span id="ref-29"></span>[29] Xiaoyu Wu, Jiaru Zhang, Yang Hua, Bohan Lyu, Hao Wang, Tao Song, and Haibing Guan. 2024. Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks. *arXiv preprint arXiv:2405.19931* (2024).

<span id="ref-30"></span>[30] Yuxin Wu and Kaiming He. 2018. Group Normalization. In *ECCV*.

<span id="ref-31"></span>[31] Ruihan Yang, Prakhar Srivastava, and Stephan Mandt. 2022. Diffusion Probabilistic Modeling for Video Generation. *arXiv preprint arXiv:2203.09481* (2022).

<span id="ref-32"></span>[32] Xiaoyu Ye, Hao Huang, Jiaqi An, and Yongtao Wang. 2024. DUAW: Data-free Universal Adversarial Watermark against Stable Diffusion Customization. In *ICLR 2024 Workshop on Secure and Trustworthy Large Language Models*. [https://openreview.net/forum?id=XYD342nKy8](https://openreview.net/forum?id=XYD342nKy8)

<span id="ref-33"></span>[33] Jiaru Zhang, Yang Hua, Tao Song, Hao Wang, Zhengui Xue, Ruhui Ma, and Haibing Guan. 2022. Improving Bayesian Neural Networks by Adversarial Sampling. In *AAAI*.

<span id="ref-34"></span>[34] Richard Zhang, Phillip Isola, Alexei A Efros, Eli Shechtman, and Oliver Wang. 2018. The Unreasonable Effectiveness of Deep Features as A Perceptual Metric. In *CVPR*.

---

## 译者说明

本页中文译文由 **Claude Opus 5 + ultracode** 翻译，**仅供参考，对内容正确性不作保证**。译文力求与原文逐段对应，公式、表格与数值均按原文照录；专有名词（数据集名、模型名、评价指标、人名）保留英文原文。若中英文表述存在出入，一律以英文原文为准。

- 英文原文 PDF：[https://arxiv.org/pdf/2405.19931](https://arxiv.org/pdf/2405.19931)
- 英文全文（网页版）：[Full Text]({{< relref "/publication/bfm-en" >}})
- 论文主页：[Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks]({{< relref "/publication/bfm" >}})
