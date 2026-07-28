---
title: "通过选择性 Jacobi 解码加速离散自回归标准化流的推理"
subtitle: "Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding"

summary: "通过选择性 Jacobi 解码加速离散自回归标准化流的推理（TMLR 2026，Featured Certification）论文中文全文翻译，含全部公式、表格与插图。"

date: '2026-05-13T00:00:00Z'
publishDate: '2026-05-13T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/sejd/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/sejd/cn/'

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
本页是论文 **《Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding》**（TMLR 2026，Featured Certification）的中文全文翻译，正文、公式、表格与插图均与原文一一对应。
译文仅供参考，如需引用请以[英文原文 PDF](https://openreview.net/pdf?id=xYATz9HpE7) 为准（[论文主页]({{< relref "/publication/sejd" >}})）。也可查看[英文全文网页版]({{< relref "/publication/sejd-en" >}})。
{{% /callout %}}

{{< toc title="目录" >}}

Jiaru Zhang、Juanwu Lu、Xiaoyu Wu、Ziran Wang、Ruqi Zhang

> Jiaru Zhang（jiaru@purdue.edu），物理人工智能研究所，普渡大学。
> Juanwu Lu（juanwu@purdue.edu），工程学院，普渡大学。
> Xiaoyu Wu（xw105@rice.edu），莱斯大学。
> Ziran Wang（ziran@purdue.edu），工程学院，普渡大学。
> Ruqi Zhang（ruqiz@purdue.edu），计算机科学系，普渡大学。
> 通讯作者：Jiaru Zhang。共同指导。

## 摘要

离散标准化流是一类颇具前景的生成模型，具有解析对数似然计算和端到端训练等优势。
然而，为确保可逆性以及可解析计算的 Jacobi 矩阵而施加的架构约束，限制了它们的表达能力与实际可用性。
近期的研究进展利用自回归建模，显著提升了表达能力与生成质量。
尽管如此，这类顺序建模在推理过程中天然地限制了并行计算，导致生成速度缓慢，进而阻碍了实际部署。
本文首先指出，推理中严格的顺序依赖对于生成高质量样本而言并非必需。
我们观察到，顺序建模中的子变量即使不严格地以全部前序子变量为条件，同样可以被近似。
此外，模型往往在第一层表现出较低的依赖冗余，而在后续层中表现出更高的冗余。
利用这些观察结果，我们提出选择性地使用 Jacobi 解码策略，通过并行迭代优化来加速其自回归推理。
理论分析表明该方法具有超线性收敛速率，并保证所需的迭代次数不会超过原始的顺序方法。
在多个数据集上的实证评估验证了我们的加速技术的通用性与有效性：在现代标准化流模型上实现了最高 4.7 倍的推理加速，同时保持了生成质量。

## 1. 引言

离散标准化流模型（*又称标准化流*）通过一系列映射来建模数据变量与潜在高斯变量之间的变换，其中映射函数被构造为由神经网络参数化的可逆函数，已成为一类颇具前景的生成模型。
这种方式允许使用单一损失函数进行端到端训练，从而确保编码与解码之间的一致性。
此外，该模型的结构便于计算解析对数似然。
这些优势吸引了研究界的广泛关注，使离散标准化流模型成为生成建模中一种颇具吸引力的工具 (<a href="#ref-5">Dinh et al., 2015</a>; <a href="#ref-6">Dinh et al., 2017</a>; <a href="#ref-9">Ho et al., 2019</a>; <a href="#ref-10">Kingma & Dhariwal, 2018</a>; <a href="#ref-20">Papamakarios et al., 2021</a>)。

尽管离散标准化流模型在理论上严谨，但由于可逆映射的要求以及计算 Jacobi 矩阵的需要所带来的不可避免的架构约束，它们在实践中往往表现出有限的生成能力。
例如，可逆函数的一种经典构造，即仿射耦合层，是将输入切分为两部分，输出则由分别经可训练网络处理后的结果拼接而成 (<a href="#ref-5">Dinh et al., 2015</a>; <a href="#ref-6">Dinh et al., 2017</a>; <a href="#ref-10">Kingma & Dhariwal, 2018</a>)。
这种形式确保了可逆函数与 Jacobi 矩阵均具有解析解，但同时也导致表达能力相对有限、可兼容的网络架构较少，从而影响生成样本的质量与实际应用。
为解决这一问题，近期提出的 TarFlow (<a href="#ref-31">Zhai et al., 2025</a>) 模型采用了受自回归标准化流 (<a href="#ref-11">Kingma et al., 2016</a>; <a href="#ref-19">Papamakarios et al., 2017</a>) 启发的分块自回归架构。
它将输入切分为一个更长的序列而非仅仅两部分，并使用掩码自回归变换来构造可逆函数。
这种自回归建模可以自然地融入因果视觉 Transformer 架构，后者提供了强大的表示能力，使 TarFlow 在密度估计与图像合成两方面均取得了最先进的性能。

然而，这类自回归顺序建模在赋予模型强大生成能力的同时，也带来了推理过程中高昂的并行计算复杂度。
具体而言，这种顺序构造的逆函数是一个自回归推理过程，意味着每个新的子变量都是顺序生成的，并依赖于此前已生成的全部子变量。
这限制了并行计算，从而降低了生成速度并制约了实际应用，先前工作 (<a href="#ref-31">Zhai et al., 2025</a>) 中亦指出了这一点。

在本工作中，我们首先指出，严格的顺序条件依赖在推理过程中包含大量随层而变的冗余依赖。
具体来说，我们观察到，即使没有最近的前序子变量，后续子变量仍然可以被近似。
此外，我们发现冗余的数量在不同层之间差异显著，后面的层比第一层表现出多得多的冗余。
受这些观察结果的启发，我们提出选择性地使用 Jacobi 解码方法来加速离散自回归标准化流模型的推理。
我们的方法利用并行迭代优化，快速收敛到高保真样本，既不需要额外训练，也无需修改原始模型架构，并且有超线性速率与有限步收敛的理论收敛保证作为支撑。
在 CIFAR-10、CIFAR-100 和 AFHQ (<a href="#ref-3">Choi et al., 2020</a>) 等多种数据集上的实验结果验证了所提加速方法的通用性与有效性。
代码已公开于 <https://github.com/lan-qing/SJD>。
总之，我们的贡献包括：

- **C1** 基于理论分析与实证观察，我们发现离散自回归标准化流的原始顺序推理中的严格依赖存在冗余，且冗余的数量在不同层之间各不相同。

- **C2** 我们提出了一种通过在推理中选择性地施加 Jacobi 解码来加速离散自回归标准化流推理的方法。理论分析表明该方法具有超线性收敛与有限步收敛保证。

- **C3** 我们开展了全面的实验来验证本文方法的有效性，在 CIFAR-10、CIFAR-100 和 AFHQ 等多种数据集上实现了最高 4.7 倍的速度提升，且对生成质量影响甚微。

## 2. 相关工作

**生成模型的推理加速。** 推理加速对于生成模型的实际应用至关重要，研究者已针对不同的生成模型提出了多种有效策略。对于扩散模型（Diffusion Models, DMs），改进的数值求解器 (<a href="#ref-7">Dockhorn et al., 2022</a>; <a href="#ref-15">Lu et al., 2022</a>; <a href="#ref-25">Song et al., 2021a</a>)、知识蒸馏 (<a href="#ref-22">Salimans & Ho, 2022</a>) 和一致性建模 (<a href="#ref-27">Song et al., 2023</a>) 等技术已显著缩短了采样时间。对于变分自编码器（Variational Autoencoder, VAE）和生成对抗网络（Generative Adversarial Network, GAN），网络剪枝 (<a href="#ref-14">Kumar et al., 2023</a>; <a href="#ref-24">Saxena et al., 2024</a>)、量化 (<a href="#ref-2">Andreev & Fritzler, 2022</a>) 和知识蒸馏 (<a href="#ref-1">Aguinaldo et al., 2019</a>; <a href="#ref-30">Yeo et al., 2024</a>) 等方法被广泛采用以提升推理效率。然而，这些方法通常利用的是这些模型所固有的特定架构性质或训练范式。因此，它们一般无法直接迁移到自回归标准化流的推理加速上。据我们所知，本文是首个探索标准化流模型推理加速的工作。

**Jacobi 解码。** 受非线性方程组求解研究 (<a href="#ref-17">Ortega & Rheinboldt, 2000</a>) 的启发，Jacobi 解码已成为加速神经网络推理的一种有前景的方法。它旨在把生成过程重新表述为求解方程组的迭代过程（通常被刻画为一个不动点问题），从而打破顺序依赖。<a href="#ref-26">Song et al. (2021b)</a>首次提出了将前馈计算解释为非线性方程组之解的理论框架，并揭示了 Jacobi 解码在 RNN 和 DenseNet 等网络上的巨大潜力。<a href="#ref-23">Santilli et al. (2023)</a>进一步证实了 Jacobi 解码在语言生成任务上的有效性。该方法此后又通过额外的微调得到改进，以保持所解码 token 的一致性 (<a href="#ref-12">Kou et al., 2024</a>)。在图像生成方面，<a href="#ref-28">Teng et al. (2025)</a>在自回归文本到图像生成模型上探索了将 Jacobi 解码与 token 接受的概率准则相结合的推理加速。尽管此前已被应用于语言生成与图像生成，这些方法普遍会遇到诸如质量退化 (<a href="#ref-28">Teng et al., 2025</a>) 以及在离散 token 空间中加速有限 (<a href="#ref-23">Santilli et al., 2023</a>) 等问题。

## 3. 方法

本节介绍选择性地施加 Jacobi 解码以加速离散自回归标准化流的方法。它通过打破子变量之间的依赖来提升推理效率。
我们首先介绍标准化流（第 3.1 节）。受顺序冗余及其深度异质性这两项观察的启发（第 3.2 节），我们提出利用 Jacobi 迭代进行并行推理（第 3.3 节）。我们证明该方法具有超线性收敛速率，并且能在有限步内收敛（第 3.4 节）。此外，我们提出一种仅对冗余较高的层施加并行 Jacobi 解码的策略，从而进一步提升效率（第 3.5 节）。为提高可读性，我们还在第 A 节给出了一张符号表。

### 3.1. 离散自回归标准化流

标准化流是一类生成模型，它借助变量变换法则 (<a href="#ref-5">Dinh et al., 2015</a>; <a href="#ref-21">Rezende & Mohamed, 2015</a>)，显式地学习潜在随机变量 {{< math >}}$\bm{z}${{< /math >}} 与数据 {{< math >}}$\bm{x}${{< /math >}} 之间的可微双射 {{< math >}}$\bm{x}=f(\bm{z})${{< /math >}}。最优变换由最大化观测数据的对数似然给出

{{< math >}}
$$
f^{\ast}\gets\underset{f}{\arg\max}\log{p}(\bm{x})=\log{p}(\bm{z})-\log\det{\mathbf{J}}_{f},
\tag{1}
$$
{{< /math >}}

其中 {{< math >}}${\mathbf{J}}_{f}${{< /math >}} 是 {{< math >}}$f${{< /math >}} 的 Jacobi 矩阵。为了便于刻画更复杂的数据分布 {{< math >}}$\bm{x}\sim{p_{\text{data}}(\bm{x})}${{< /math >}}，离散标准化流学习的不是单一双射，而是 {{< math >}}$K${{< /math >}} 个中间双射的级联，使得

{{< math >}}
$$
\bm{x}=f(\bm{z}_{K})\triangleq\left(f_{0}\circ{f_{1}}\circ\ldots\circ{f_{K-1}}\right)(\bm{z}_{K}).
\tag{2}
$$
{{< /math >}}

于是，最优的双射集合由下式给出

{{< math >}}
$$
f^{\ast}\gets\underset{f_{0},\ldots,f_{K-1}}{\arg\max}\log{p(\bm{z}_{K})}-\sum\limits_{k=0}^{K-1}\log\det{\mathbf{J}}_{f_{k}}.
\tag{3}
$$
{{< /math >}}

然而，为高维变量恰当地构造双射 {{< math >}}$f_{k}${{< /math >}} 并非易事。一种被广泛采用的方法是基于耦合的标准化流 (<a href="#ref-5">Dinh et al., 2015</a>)，它把高维的 {{< math >}}$\bm{z}_{k}${{< /math >}} 切分为一对子变量 {{< math >}}$\bm{z}_{k}=\left[\bm{z}_{k,1}, \bm{z}_{k,2}\right]^\intercal${{< /math >}}，并启发了一系列后续工作：这些工作通过构造诸如实值非体积保持变换（real-valued non-volume preserving, RealNVP）(<a href="#ref-6">Dinh et al., 2017</a>)、可逆 1×1 卷积 (<a href="#ref-10">Kingma & Dhariwal, 2018</a>) 以及自注意力层 (<a href="#ref-9">Ho et al., 2019</a>) 等基本模块，将神经网络结合进建模过程。

更近期的研究为离散标准化流提出了一种自回归范式，它把随机变量任意地切分为由 {{< math >}}$L${{< /math >}} 个子变量组成的序列，从而扩展了上述形式化表述 (<a href="#ref-11">Kingma et al., 2016</a>; <a href="#ref-19">Papamakarios et al., 2017</a>)。
此时，双射 {{< math >}}$f_{k}${{< /math >}} 可以定义为

{{< math >}}
$$
\begin{aligned}
    \bm{z}_{k+1}&amp;=
    f_{k}(\bm{z}_{k})=f_{k}\left(\begin{bmatrix}\bm{z}_{k,1} &amp; \bm{z}_{k,2} &amp; \ldots &amp; \bm{z}_{k,L}\end{bmatrix}^{\intercal}\right) \\
    &amp;\triangleq \begin{bmatrix}\boldsymbol{z}_{k,1} \\ \left(\boldsymbol{z}_{k,2}-g_{k}(\boldsymbol{z}_{k,&lt;2})\right)\odot\exp(s_{k}(\boldsymbol{z}_{k,&lt;2})) \\ \vdots \\ \left(\boldsymbol{z}_{k,L}-g_{k}(\boldsymbol{z}_{k,&lt;L})\right)\odot\exp(s_{k}(\boldsymbol{z}_{k,&lt;L}))\end{bmatrix},
\end{aligned}
\tag{4}
$$
{{< /math >}}

其中 {{< math >}}$\odot${{< /math >}} 表示 Hadamard 积，{{< math >}}$s_{k}(\cdot)${{< /math >}} 与 {{< math >}}$g_{k}(\cdot)${{< /math >}} 是训练过程中需要学习的函数。
需要注意的是，每次变换之后还会对 {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} 施加一次置换（例如逆序），以确保序列中所有位置都能经由整个流被变换，为简洁起见我们将其省略。
上述双射自然地给出一个逆变换：

{{< math >}}
$$
\begin{aligned}
\boldsymbol{z}_{k}
&amp;=f^{-1}_{k}(\boldsymbol{z}_{k+1})=f^{-1}_{k}\left(\begin{bmatrix}\bm{z}_{k+1,1} &amp; \ldots &amp; \bm{z}_{k+1,L}\end{bmatrix}^{\intercal}\right)\\
&amp;\triangleq\begin{bmatrix}\boldsymbol{z}_{k+1,1} \\ \boldsymbol{z}_{k+1, 2}\odot\exp(-s_{k}\left(\boldsymbol{z}_{k,&lt;2})\right) + g_{k}(\boldsymbol{z}_{k,&lt;2}) \\ \vdots \\ \boldsymbol{z}_{k+1,L}\odot\exp(-s_{k}\left(\boldsymbol{z}_{k,&lt;L})\right) + g_{k}(\boldsymbol{z}_{k,&lt;L})\end{bmatrix}.
\end{aligned}
\tag{5}
$$
{{< /math >}}

通过被集成到因果视觉 Transformer 等架构之中，离散自回归标准化流在密度估计与图像生成上已展现出优越性 (<a href="#ref-31">Zhai et al., 2025</a>)。然而，推理过程中出现了一个关键问题，即推理需要计算逆变换。如式 (5) 所示，生成 {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} 依赖于此前生成的全部 {{< math >}}$\boldsymbol{z}_{k,&lt;l}${{< /math >}}，这使并行计算无从进行，并导致采样缓慢，从而限制了实际应用场景。为解决这一问题，我们利用关于冗余的两项基本观察，具体见下一节。

### 3.2. 冗余

**顺序冗余。** 如式 (5) 所示，由逆变换所定义的生成过程揭示了每个子变量与其全部前序子变量之间的顺序依赖。
因此，原始的推理流程强制采用一种严格顺序的生成范式，显著限制了生成速度。
我们假设这种严格依赖是冗余的，对于图像这类具有内在空间局部性与连续性的数据而言尤其如此。
因此，即便没有来自全部前序子变量的精确且即时的信息，元素 {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} 也可能被合理地推得。

为验证我们的假设，我们在推理时采用一种直接的变换进行实验：把元素 {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} 的变换步骤修改为忽略序列中最近的 {{< math >}}$o${{< /math >}} 个前序元素的信息，即

{{< math >}}
$$
\boldsymbol{z}_{k,l}= \boldsymbol{z}_{k+1,l}\odot\exp(-s_{k}(\boldsymbol{z}_{k,&lt;(l-o)}))+g_{k}(\boldsymbol{z}_{k,&lt;(l-o)}),
\tag{6}
$$
{{< /math >}}

其中 {{< math >}}$l&gt;1${{< /math >}}，{{< math >}}$\boldsymbol{z}_{k,&lt;(l-o)}${{< /math >}} 是通过在注意力操作中掩蔽最近的 {{< math >}}$o${{< /math >}} 个前序子变量得到的。图 2 中给出的实验结果表明，虽然随着被移除的最近前序子变量增多，图像质量有所下降，但模型仍然能够生成有意义的图像。这支持了如下论断：严格的顺序依赖中包含着潜在可利用的冗余。

{{< figgrid caption="**图 1**：标准推理与「掩蔽最近 o = 5 个前序依赖」的推理，二者各层输出之间的余弦相似度与 L2 距离。o = 1 与 o = 2 的结果见图 A1。" >}}
sejd/obs_c100_5_dual_axis.png | 45 | **(a)** CIFAR-100
sejd/obs_afhq_5_dual_axis.png | 45 | **(b)** AFHQ
{{< /figgrid >}}


{{< figgrid caption="**图 2**：掩蔽了对最近 o 个子变量的依赖后的生成结果。模型仍能生成有意义的图像，说明借助并行计算实现加速具有潜在可行性。" >}}
sejd/samples_cifar-10_0.png | 22 | **(a)** CIFAR-10，真值
sejd/samples_cifar-10_1.png | 22 | **(b)** CIFAR-10，o = 1
sejd/samples_cifar-10_2.png | 22 | **(c)** CIFAR-10，o = 2
sejd/samples_cifar-10_5.png | 22 | **(d)** CIFAR-10，o = 5
sejd/samples_afhq_0.jpg | 22 | **(e)** AFHQ，真值
sejd/samples_afhq_1.jpg | 22 | **(f)** AFHQ，o = 1
sejd/samples_afhq_2.jpg | 22 | **(g)** AFHQ，o = 2
sejd/samples_afhq_5.jpg | 22 | **(h)** AFHQ，o = 5
{{< /figgrid >}}


**冗余的深度异质性。** 我们进一步考察在生成过程（{{< math >}}$\boldsymbol{z}_{K} \rightarrow \dots \rightarrow \boldsymbol{x}${{< /math >}}）中，顺序冗余的程度是否随层的不同而变化。从理论上讲，我们预期存在异质性：来自第一层的随机变量 {{< math >}}$\boldsymbol{z}_{K}${{< /math >}} 承担着从高斯噪声出发进行结构初始化的任务，往往对前序子变量表现出很强的依赖，因为纯噪声输入在理论上不含任何信息，因而更加依赖此前生成的 {{< math >}}$\boldsymbol{z}_{K,&lt;l}${{< /math >}} 所提供的上下文。
相反，后续的变换是在对前一次变换所输出的、信息丰富的 {{< math >}}$\boldsymbol{z}_{k+1,l}${{< /math >}} 进行精化，因而对此前生成的 {{< math >}}$\boldsymbol{z}_{k,&lt;l}${{< /math >}} 只具有较弱的顺序依赖。

为验证这一点，我们测量标准推理输出 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} 与按式 (6) 掩蔽最近 {{< math >}}$o${{< /math >}} 个依赖后所生成结果之间的余弦相似度与 L2 距离偏差。
如图 1 所示，第一层的偏差显著大于后续各层。
这一结果从经验上证实了第一层的低冗余，与我们的预期一致。
后续各层偏差极小，这与它们的精化作用相符——它们利用了信息丰富的输入以及已有的上下文结构。
这一观察促使我们探索针对生成过程的逐层特定优化。

### 3.3. 基于 Jacobi 迭代的并行推理

**算法 1**：用于 {{< math >}}$f_{k}${{< /math >}} 的 Jacobi 解码

```
输入：序列 z_{k+1}，函数 s_k(·) 与 g_k(·)，停止阈值 τ
输出：序列 z_k

初始化 z_k^0 = 0, t = 0
当 true 时，执行
    t ← t + 1,  z_{k,1}^t ← z_{k+1,1}
    对于 l = 2, ... L
    并行执行
        z_{k,l}^t ← z_{k+1,l} ⊙ exp(-s_k(z_{k,<l}^{t-1})) + g_k(z_{k,<l}^{t-1})
    结束 for 循环
    如果 ||z_k^t - z_k^{t-1}||_∞ < τ
        跳出
    结束 if 判断
结束 while 循环
z_k = z_k^t
```

我们关于顺序冗余的实证观察揭示了离散自回归标准化流推理中的一条关键性质：*后续元素 {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} 的生成对前序元素 {{< math >}}$\boldsymbol{z}_{k,&lt;l}${{< /math >}} 中的不精确性具有一定程度的鲁棒性*。这一观察表明，标准推理过程所强制施加的严格且完全收敛的顺序依赖可能含有冗余，对生成质量而言并非每一步都严格必要。这一发现促使我们探索能够利用该鲁棒性的并行计算策略。

为实现并行化，我们首先重新审视推理任务。如前所述，通过式 (5) 中定义的逆变换、从输入 {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} 生成目标序列 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}}，本质上要求找到满足全部自回归条件依赖的唯一解 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}}。它可以形式化地视为求解一个由 {{< math >}}$L${{< /math >}} 个非线性方程 {{< math >}}$\mathcal{F}_{l}${{< /math >}} 构成的方程组，其对 {{< math >}}$l=1,\ldots,L${{< /math >}} 隐式定义为

{{< math >}}
$$
\mathcal{F}_{l}(\boldsymbol{z}_{k,l},\boldsymbol{z}_{k,&lt;l},\boldsymbol{z}_{k+1,l})=0,
\tag{7}
$$
{{< /math >}}

其中 {{< math >}}$\mathcal{F}_{l}=0${{< /math >}} 表示在已知输入 {{< math >}}$\boldsymbol{z}_{k+1, l}${{< /math >}} 的条件下，由逆变换第 {{< math >}}$k${{< /math >}} 步所施加的约束条件。标准的顺序推理方法采用类 Gauss-Seidel 方法隐式地求解该方程组，其中 {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} 的计算依赖于紧邻其前、且已完全计算得到的值 {{< math >}}$\boldsymbol{z}_{k,1},\ldots,\boldsymbol{z}_{k,l-1}${{< /math >}}。

利用第 3.2 节的观察所揭示的并行性潜力，我们提出采用 Jacobi 解码方法来求解式 (7) 所定义的方程组。它不采用顺序更新，而是执行迭代式的并行更新。从初始估计 {{< math >}}$\boldsymbol{z}_{k}^{0}${{< /math >}} 出发，第 {{< math >}}$t+1${{< /math >}} 次迭代计算出新的估计 {{< math >}}$\boldsymbol{z}_{k}^{t+1}${{< /math >}}，其中每个元素 {{< math >}}$\boldsymbol{z}_{k, l}^{t+1}${{< /math >}} *仅*基于上一迭代点 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} 中的元素以及上一层的输出 {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} 计算得到：

{{< math >}}
$$
\begin{aligned}
\text{ For } &amp;l= 1,\dots,L, \text{solve for } \boldsymbol{z}_{k,l}^{t+1} \text{ from: } \\
&amp;\mathcal{F}_l(\boldsymbol{z}_{k,l}^{t+1}, \boldsymbol{z}_{k,&lt;l}^{t}, \boldsymbol{z}_{k+1,l}) = 0.
\end{aligned}
\tag{8}
$$
{{< /math >}}

由于一次迭代内每个 {{< math >}}$\boldsymbol{z}_{k,l}^{t+1}${{< /math >}} 的计算只依赖于已完成的上一次迭代所得的值 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}}，全部 {{< math >}}$L${{< /math >}} 个更新可以**并发地**计算，从而打破了顺序瓶颈。该迭代过程持续进行，直到满足合适的停止准则，例如相邻迭代点之差的范数 {{< math >}}$\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{t-1} \|${{< /math >}} 足够小。在离散自回归标准化流推理中应用 Jacobi 解码的完整流程总结于算法 1。

**备注 3.1**：Jacobi 解码技术已在其它生成建模场景中被探索过，例如语言模型与自回归图像生成 (<a href="#ref-23">Santilli et al., 2023</a>; <a href="#ref-26">Song et al., 2021b</a>)。然而，直接套用 Jacobi 解码往往收效有限，例如在语言模型上只带来微弱的加速 (<a href="#ref-12">Kou et al., 2024</a>)，或在图像合成中损害样本质量 (<a href="#ref-28">Teng et al., 2025</a>)。尽管存在这些不利的先例，我们假设离散自回归标准化流具备某些特性，能够缓解上述问题，从而使 Jacobi 迭代成为一种更可行的策略。实证观察到的冗余依赖表明，该系统可能容忍使用来自上一次迭代的、略不精确的信息 {{< math >}}$\boldsymbol{z}_{k,&lt;l}^{t}${{< /math >}}，而这正是实现 Jacobi 并行更新的基础。此外，求逆过程的确定性避免了随机模型中存在的累积采样误差，并且在连续空间中进行运算相比离散设定可能带来更平滑的迭代收敛。这些方面为在离散自回归标准化流中引入 Jacobi 解码提供了合理依据。

### 3.4. 收敛性

本节从理论上分析所提出的 Jacobi 迭代推理方法的收敛性，并指出两条关键性质：

- 在一定条件下，该迭代具有局部超线性收敛性，这意味着其在实践中收敛迅速。
- 由于离散自回归标准化流固有的三角依赖结构，该迭代可保证在不超过原始推理步数的迭代次数内收敛到精确解，从而给出了一个最坏情况界。

首先，我们分析局部收敛行为。在初始化恰当、接近真实解的假设下，Jacobi 迭代以超线性速率收敛。

{{< figgrid caption="**图 3**：AFHQ 上的可视化对比。本文方法将生成过程加速 4.5 倍，同时保持了生成内容的质量与保真度。CIFAR-10 与 CIFAR-100 上更多的可视化对比见图 A7 与图 A8。" >}}
sejd/samples.jpg | 49 | **(a)** 顺序推理
sejd/samples_appro_0.5.jpg | 49 | **(b)** 本文方法，加速 4.5 倍
{{< /figgrid >}}


**命题 3.1（超线性收敛速率）**：对于算法 1 中定义的迭代序列 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}}，{{< math >}}$\exists \delta &gt;0${{< /math >}}，使得对任意满足 {{< math >}}$||\boldsymbol{z}_{k}^0 - \boldsymbol{z}_{k}|| &lt; \delta${{< /math >}} 的 {{< math >}}$\forall \boldsymbol{z}_{k}^0${{< /math >}}，迭代序列 {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}${{< /math >}} 均收敛到 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}}，且具有超线性收敛速率，即

{{< math >}}
$$
\|\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}\| = o(\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}\|).
$$
{{< /math >}}

第 B 节给出了详细证明。除收敛速率之外，离散自回归标准化流推理的特定结构还提供了很强的全局保证。该任务等价于求解一个三角方程组，其中第 {{< math >}}$l${{< /math >}} 个未知子变量仅依赖于其前的未知量。<a href="#ref-23">Santilli et al. (2023)</a>; <a href="#ref-26">Song et al. (2021b)</a>也指出了这一结构性质，它保证对于长度为 {{< math >}}$L${{< /math >}} 的序列，Jacobi 方法至多经过 {{< math >}}$L${{< /math >}} 次迭代即可收敛到精确解。

**命题 3.2（有限步收敛保证）**：对于算法 1 中定义的迭代序列 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}}。将 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} 的序列长度记为 {{< math >}}$L${{< /math >}}，我们有 {{< math >}}$\boldsymbol{z}_{k}^{L} = \boldsymbol{z}_{k}${{< /math >}}。

之所以出现这种有限步收敛，是因为三角结构使得信息在更新过程中能够确定地沿序列传播。第 B 节给出了形式化的证明。因此，命题 3.2 给出了到达精确解所需计算步数的严格上界。

这些命题共同确立了 Jacobi 解码的内在效率。命题 3.1 表明误差以超线性速率下降。这说明收敛可以很快，因为每一次迭代（{{< math >}}$t &lt; L${{< /math >}}）都会使各自的误差获得越来越显著的下降 (<a href="#ref-4">Dennis & Schnabel, 1996</a>)。需要注意的是，尽管在理论分析中我们假设了接近真实解的初始化，但在各种初始化下收敛速度依然很快，如第 4.3 节所示。另一方面，命题 3.2 进一步表明，至多经过 {{< math >}}$L${{< /math >}} 次迭代即可保证收敛到精确解 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}}。这些结果印证了将 Jacobi 解码用于离散自回归标准化流快速推理的潜力。

**表 1**：顺序推理、均匀 Jacobi 解码（uniform Jacobi decoding, UJD）与本文方法的对比。下标表示三次运行中的最大偏差。

| 配置 数据集 | 配置 方法 | 生成速度 时间（s）↓ | 生成速度 加速比 ↑ | 生成质量 FID ↓ | 生成质量 CLIP-IQA ↑ | 生成质量 BRISQUE ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | 顺序推理 | 9.56<sub>±0.42</sub> | 1.0× | 9.71<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 56.35<sub>±0.21</sub> |
| CIFAR-10 | UJD | 3.92<sub>±0.09</sub> | 2.4× | 10.19<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 56.79<sub>±0.20</sub> |
| CIFAR-10 | **本文方法** | **2.63**<sub>±0.13</sub> | **3.6**× | 10.20<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 56.78<sub>±0.19</sub> |
| CIFAR-100 | 顺序推理 | 9.57<sub>±0.27</sub> | 1.0× | 8.22<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 57.75<sub>±0.12</sub> |
| CIFAR-100 | UJD | 3.30<sub>±0.10</sub> | 2.9× | 8.26<sub>±0.10</sub> | 0.35<sub>±0.00</sub> | 57.76<sub>±0.06</sub> |
| CIFAR-100 | **本文方法** | **2.04**<sub>±0.03</sub> | **4.7**× | 8.19<sub>±0.16</sub> | 0.35<sub>±0.00</sub> | 57.78<sub>±0.12</sub> |
| AFHQ | 顺序推理 | 186.28 | 1.0× | 15.42 | 0.63 | 15.55 |
| AFHQ | UJD | 219.24 | 0.8× | 15.44 | 0.63 | 15.44 |
| AFHQ | **本文方法** | **41.21** | **4.5**× | 15.44 | 0.63 | 15.56 |

### 3.5. 在何处使用 Jacobi 解码

虽然理论分析显示 Jacobi 迭代颇具前景，但均匀施加它会带来实际的权衡。具体而言，在一次迭代中，我们需要同时更新序列的全部 {{< math >}}$L${{< /math >}} 个元素，这相当于以内存换取原始顺序设定下每次迭代所需的时间——在原始顺序设定中，每次只更新一个序列元素。此外，现代 Transformer 架构中用于优化注意力算子的常用键值（key-value, KV）缓存 (<a href="#ref-18">Ott et al., 2019</a>) 无法直接用于 Jacobi 解码，因为被解码的子变量是近似值，且必须在每次迭代中更新。这些局限在依赖较强之处尤为突出，可能使得均匀 Jacobi 解码方法在这类场景中比经过优化的顺序解码还要慢。

受这一权衡以及第 3.2 节中给出的顺序冗余的深度异质性的启发，我们利用第一层（第 {{< math >}}$K${{< /math >}} 步）往往表现出更强依赖这一点，提出了我们的选择性分层处理策略。在带有 KV 缓存的现代 Transformer 架构上，该策略对依赖密集的第一层使用标准的顺序解码，因为在该层原始的顺序解码效果良好。随后，它对其余各层切换为并行的 Jacobi 迭代，因为这些层预期具有更高的冗余。这使得我们既能获得并行性带来的收益，又能避免第一层因依赖更强而产生的高额额外计算开销。这一应用方式旨在通过策略性地把 Jacobi 解码方法施加到冗余更高的层上，从而实现更有效的整体推理加速。

## 4. 实验

本节中，我们将所提方法应用于 TarFlow —— 一种最先进的离散自回归标准化流模型 (<a href="#ref-31">Zhai et al., 2025</a>)。
实验在 CIFAR-10 与 CIFAR-100 (<a href="#ref-13">Krizhevsky, 2009</a>) 上使用从零开始训练的模型进行，并在分辨率为 256×256 的 AFHQ (<a href="#ref-3">Choi et al., 2020</a>) 上使用官方发布的 TarFlow 检查点进行。

参与比较的基线方法包括标准的顺序推理（离散自回归标准化流的默认推理方法），以及均匀 Jacobi 解码（uniform Jacobi decoding, UJD）方法，后者对所有层一律施加 Jacobi 解码策略。
Jacobi 迭代的默认停止阈值 {{< math >}}$\tau${{< /math >}} 设为 {{< math >}}$0.5${{< /math >}}。
更多实验细节，包括网络架构与超参数，见第 E.1 节。

我们的主要评估同时覆盖计算效率与生成样本的质量。生成速度由每批次平均推理时间以及相对于顺序推理基线方法的整体加速比来衡量，二者均在两块 L40S GPU 上测得。对于生成质量，我们采用 Fréchet Inception 距离（Fréchet Inception Distance, FID）(<a href="#ref-8">Heusel et al., 2017</a>)，这是一个被广泛采用的指标，用于衡量生成图像分布与真实数据分布之间的感知相似度。对于感知质量，我们另外报告两个无参考指标：CLIP-IQA (<a href="#ref-29">Wang et al., 2023</a>)，它依据与 CLIP 嵌入的对齐程度来评估质量；以及 BRISQUE (<a href="#ref-16">Mittal et al., 2012</a>)，一种对常见失真敏感的盲图像质量评估器。

为了全面起见，我们还在规模更小的掩码自回归流（Masked Autoregressive Flow, MAF）(<a href="#ref-19">Papamakarios et al., 2017</a>) 上，针对图像生成与 Boltzmann 分布近似两类任务进行了测试，详见第 E.3 节。相关结果同样确认了我们的方法能取得显著加速，并进一步验证了其通用性。

### 4.1. 推理性能对比

我们将本文方法与标准的顺序推理基线方法以及 UJD 方法进行比较。如图 3 与第 E 节所示，对样本的目视检查确认我们的方法保持了高感知保真度，其输出在视觉上与原始的顺序生成结果相当。这一定性观察得到了表 1 中详细定量分析的印证。FID、CLIP-IQA 和 BRISQUE 等指标表明，在所有场景下，UJD 与我们的方法都在很大程度上保持了生成质量，相对于顺序推理基线方法仅有小幅退化。

然而，尽管 UJD 在规模较小的 CIFAR-10 与 CIFAR-100 数据集上展现出加速收益，它在更大的 AFHQ 数据集上却失效了，其推理速度反而慢于顺序推理基线方法。在 AFHQ 上表现较差的原因很可能在于更高的单次迭代计算开销，再加上可能更强的依赖关系，共同抵消了并行带来的增益。相反，我们的方法通过选择性地施加 Jacobi 迭代，在所有数据集上都一致地取得了可观的加速比。
值得注意的是，如表 1 所示，相较于顺序推理基线方法，我们的方法实现了最高 4.7 倍的加速，凸显了其有效性。这确认了所提出的选择性方案对于在不牺牲生成质量的前提下实现有效且可推广的加速至关重要。

### 4.2. 分析的验证

{{< figgrid caption="**图 4**：Jacobi 解码在前两个网络层上的收敛动态。所有层的完整结果见图 A2。图中展示误差（以当前迭代点与顺序推理输出之差的 ℓ<sub>2</sub> 范数衡量）随迭代次数的变化，可见整体收敛很快，而第 1 层的收敛明显更慢。" >}}
sejd/errs_afhq_layer0.png | 49 | **(a)** 第 1 层
sejd/errs_afhq_layer1.png | 49 | **(b)** 第 2 层
{{< /figgrid >}}


为了从实验上验证我们的理论分析与经验观察，我们分析了 Jacobi 迭代的收敛动态。图 4 绘制了在 AFHQ 数据集上，迭代过程中不同层的迭代点 {{< math >}}$\boldsymbol{z}_k^t${{< /math >}} 与顺序推理所得真值 {{< math >}}$\boldsymbol{z}_k${{< /math >}} 之差的 {{< math >}}$\ell_2${{< /math >}} 范数所度量的误差。作为参照，我们同时给出原始顺序推理的误差变化情况，其中按照默认实现，未被推理出的子变量在计算时被视作输入子变量 {{< math >}}$\boldsymbol{z}^{t+1}${{< /math >}}。结果清楚地表明，Jacobi 过程的误差迅速下降，往往在远少于理论最坏情况界 {{< math >}}$L${{< /math >}} 的迭代次数内就达到接近零的误差，为其快速收敛性质提供了经验支持。此外，图 4 还揭示了各层之间截然不同的收敛行为。通过 Jacobi 迭代，第一层对应的误差下降速度明显慢于后续层。
这直接验证了我们在第 3.2 节中关于初始层依赖更强的观察，并从经验上确认了选择性策略背后的合理性 —— 即把并行迭代施加于收敛更快的后续层。
这些结果确认了依赖关系在层间的差异，从而验证了我们方法的有效性。

### 4.3. 消融实验

{{< figgrid caption="**图 5**：关于停止阈值 τ 的消融实验：本文方法在不同 τ 取值下的 FID 分数与推理时间，体现了速度—质量之间的权衡。" >}}
sejd/ablation_cifar10.png | 32 | **(a)** CIFAR-10
sejd/ablation_cifar100.png | 32 | **(b)** CIFAR-100
sejd/ablation_afhq.png | 32 | **(c)** AFHQ
{{< /figgrid >}}


**τ 的影响。**
为了进一步理解停止阈值超参数 {{< math >}}$\tau${{< /math >}} 的影响，我们开展了一项消融实验。
我们改变 {{< math >}}$\tau${{< /math >}} 的取值，并测量由此得到的生成质量（以 FID 衡量）与推理时间。
展示这两个指标之间权衡关系的结果见图 5。正如预期的那样，增大阈值 {{< math >}}$\tau${{< /math >}} 会让并行迭代更早终止，从而显著减少整体推理时间。
然而，在停止之前允许相邻迭代点之间存在更大的差异，会导致生成结果不够精确。
这一点体现在 FID 分数上：随着 {{< math >}}$\tau${{< /math >}} 增大，FID 倾向于上升。
值得注意的是，结果显示当 {{< math >}}$\tau${{< /math >}} 的取值低于 1.0 时，FID 的上升相对平缓，而推理时间的下降依然可观。
这支持了如下结论：在适当选取 {{< math >}}$\tau${{< /math >}} 的情况下，我们的方法能有效提升生成速度，而对生成质量的影响很小。
{{< math >}}$\tau=0.5${{< /math >}} 始终提供了良好的平衡，在取得可观加速的同时，使生成质量接近基线方法。
因此，我们在本文其余所有实验中都采用 {{< math >}}$\tau=0.5${{< /math >}} 作为默认设置。

{{< figgrid caption="**图 6**：关于不同初始化方式的消融实验。" >}}
sejd/initialization_comparison.png | 90
{{< /figgrid >}}


**初始化的影响。**
我们对 {{< math >}}$\boldsymbol{z}_k^0${{< /math >}} 的多种初始化方法开展了消融实验，包括零初始化 {{< math >}}$\boldsymbol{z}_k^0=\mathbf{0}${{< /math >}}、标准正态初始化 {{< math >}}$\boldsymbol{z}_k^0 \sim \mathcal{N}(\mathbf{0}, I)${{< /math >}}，以及用上一层的输出进行初始化 {{< math >}}$\boldsymbol{z}_k^0=\boldsymbol{z}_{k+1}${{< /math >}}。图 6 中的实验结果表明，在各种初始化方式下加速性能保持相近，这支持了我们关于超线性收敛速率具有普遍性、且对具体初始化方法不敏感的分析。

## 5. 结论

本文首先观察到自回归标准化流模型中存在的依赖冗余，以及该冗余在不同层之间的显著差异。基于这一观察，我们提出将并行的 Jacobi 解码方法选择性地施加于依赖冗余较高的层，以此加速推理。理论分析证明了所提方法具有超线性收敛速率，并给出了所需迭代总次数的最坏情况保证。全面的实验验证了理论分析的正确性，并表明我们的方法在多种场景下均实现了显著的推理加速，提升了标准化流模型的实用价值。

## A. 符号

为提高可读性，我们在下面给出符号表。

**表 A1**：符号表。

| 符号 | 含义 |
| --- | --- |
| {{< math >}}$\boldsymbol{x}${{< /math >}} | 观测数据的向量。 |
| {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} | 离散标准化流第 {{< math >}}$k${{< /math >}} 步随机变量的向量。 |
| {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} | 第 {{< math >}}$k${{< /math >}} 步随机变量切分后的第 {{< math >}}$l${{< /math >}} 个子变量。 |
| {{< math >}}$\boldsymbol{z}_{k,l}^{t}${{< /math >}} | 第 {{< math >}}$t${{< /math >}} 个解码步时，第 {{< math >}}$k${{< /math >}} 步随机变量中的第 {{< math >}}$l${{< /math >}} 个子变量。 |
| {{< math >}}$\mathbf{J}_{f}${{< /math >}} | 函数 {{< math >}}$f${{< /math >}} 的 Jacobi 矩阵 |
| {{< math >}}$T${{< /math >}} | Jacobi 解码中的解码步数。 |
| {{< math >}}$L${{< /math >}} | 切分后随机变量中子变量的个数。 |
| {{< math >}}$K${{< /math >}} | 离散标准化流的步数。 |

## B. 理论证明

我们首先把 Jacobi 迭代映射形式化地重新定义为函数 {{< math >}}$F(\cdot)${{< /math >}}。

**定义 B.1（Jacobi 迭代映射）**：设 {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} 是给定的长度为 {{< math >}}$L${{< /math >}} 的向量序列，并设 {{< math >}}$s_{k}, g_{k}${{< /math >}} 为给定的函数。迭代映射 {{< math >}}$F${{< /math >}} 对 {{< math >}}$\boldsymbol{z}${{< /math >}} 按分量定义为：

{{< math >}}
$$
F(\boldsymbol{z})_{l} =
\begin{cases} 
  \boldsymbol{z}_{k+1,1} &amp; l = 1 \\
  \boldsymbol{z}_{k+1,l} \odot \exp\bigl(-s_{k}(\boldsymbol{z}_{&lt;l})\bigr) + g_k(\boldsymbol{z}_{&lt;l}) &amp; l=2,\ldots,L
\end{cases}
\tag{9}
$$
{{< /math >}}

其中 {{< math >}}$\boldsymbol{z}_{&lt;l} \coloneqq [z_{1}, \dots, z_{l-1}]^T${{< /math >}}。迭代序列由 {{< math >}}$\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t})${{< /math >}} 以初始值 {{< math >}}$\boldsymbol{z}^{0}_{k}${{< /math >}} 生成。

容易观察到，该迭代存在一个不动点 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}。例如，式 (5) 中顺序解码方法所输出的 {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} 就是一个不动点。此外，由于我们使用神经网络作为参数化函数 {{< math >}}$s_{k}${{< /math >}} 与 {{< math >}}$g_{k}${{< /math >}}，映射 {{< math >}}$F${{< /math >}} 是连续可微的。

基于以上观察，我们可以得到：从足够接近不动点的初始序列出发，迭代序列 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} 以至少超线性的收敛速率收敛到 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}，如下面的定理所示。

**命题 B.1（超线性收敛速率）**：存在 {{< math >}}$\delta&gt;0${{< /math >}}，使得只要 {{< math >}}$\left\|\boldsymbol{z}_{k}^{0} - \boldsymbol{z}_{k}^{\ast}\right\| &lt; \delta${{< /math >}}，迭代序列 {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}${{< /math >}} 就收敛到 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}，且至少具有超线性收敛速率。这意味着误差满足：

{{< math >}}
$$
\|\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}^{\ast}\| = o(\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\|) \quad \text{as } \boldsymbol{z}_{k}^{t} \to \boldsymbol{z}_{k}^{\ast}.
\tag{10}
$$
{{< /math >}}

**证明**：记 {{< math >}}$\boldsymbol{e}^{t} = \boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}${{< /math >}} 为第 {{< math >}}$t${{< /math >}} 次迭代处的近似误差。

迭代映射 {{< math >}}$F(\boldsymbol{z})${{< /math >}} 的结构（如前面的定义所详述：{{< math >}}$F(\boldsymbol{z})_{1} = \boldsymbol{z}_{k+1,1}${{< /math >}} 是常数，而对 {{< math >}}$l=2,\ldots,L${{< /math >}} 有 {{< math >}}$F(\boldsymbol{z})_{l} = F(\boldsymbol{z}_{&lt;l})${{< /math >}}）保证了对任意 {{< math >}}$\boldsymbol{z}${{< /math >}}，Jacobi 矩阵 {{< math >}}$\mathbf{J}_F(\boldsymbol{z})${{< /math >}} 都是严格下三角的。

这意味着 {{< math >}}$\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})${{< /math >}} 是严格下三角矩阵。因此，{{< math >}}$\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})${{< /math >}} 的所有特征值均为零。于是，不动点处 Jacobi 矩阵的谱半径为

{{< math >}}
$$
\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) = 0.
\tag{11}
$$
{{< /math >}}

由于 {{< math >}}$F${{< /math >}} 是连续可微的，Taylor 定理允许我们把 {{< math >}}$F(\boldsymbol{z}_{k}^{t})${{< /math >}} 在 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 附近展开。对于充分接近 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 的 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}}：

{{< math >}}
$$
F(\boldsymbol{z}_{k}^{t}) = F(\boldsymbol{z}_{k}^{\ast}) + \mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})(\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}) + o(\left\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\right\|).
\tag{12}
$$
{{< /math >}}

利用迭代的定义 {{< math >}}$\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t})${{< /math >}} 以及不动点处的性质 {{< math >}}$\boldsymbol{z}_{k}^{\ast} = F(\boldsymbol{z}_{k}^{\ast})${{< /math >}}，我们有：

{{< math >}}
$$
\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}^{\ast} = F(\boldsymbol{z}_{k}^{t}) - F(\boldsymbol{z}_{k}^{\ast}) = \mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})(\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}) + o(\left\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\right\|).
\tag{13}
$$
{{< /math >}}

由此得到一个误差传播动态：

{{< math >}}
$$
\boldsymbol{e}^{t+1} = \mathbf{J}_F(\boldsymbol{z}_{k}^{\ast}) \boldsymbol{e}^{t} + o(\|\boldsymbol{e}^{t}\|).
\tag{14}
$$
{{< /math >}}

关于迭代法的标准定理（例如已有工作 (<a href="#ref-17">Ortega & Rheinboldt, 2000</a>) 中与收敛的 Q-阶有关的结果，Q-超线性的讨论可参见其定理 10.1.4 附近的内容）指出：若一个迭代是收敛的，且其误差满足式 (14) 中的关系，则该收敛是 Q-超线性的当且仅当 {{< math >}}$\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) = 0${{< /math >}}。与此同时，条件 {{< math >}}$\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) &lt; 1${{< /math >}}（此处由于 {{< math >}}$\rho=0${{< /math >}} 而自然满足）保证了 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 是一个吸引点，因此对于充分接近 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 的 {{< math >}}$\boldsymbol{z}_{k}^{0}${{< /math >}}，序列 {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}${{< /math >}} 一定收敛到 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}。

既然 {{< math >}}$\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) = 0${{< /math >}}，所引用的收敛理论就直接蕴含该迭代是 Q-超线性的。按照定义，Q-超线性收敛意味着当 {{< math >}}$\boldsymbol{e}^{t} \to \boldsymbol{0}${{< /math >}} 时有 {{< math >}}$\left\|\boldsymbol{e}^{t+1}\right\| = o(\left\|\boldsymbol{e}^{t}\right\|)${{< /math >}}。因此，我们得出结论：

{{< math >}}
$$
\|\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}^{\ast}\| = o(\left\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\right\|) \quad \text{as } \boldsymbol{z}_{k}^{t} \to \boldsymbol{z}_{k}^{*}.
\tag{15}
$$
{{< /math >}}

这就表明收敛速率至少是超线性的。

**命题 B.2（有限步收敛保证）**：对于定义 B.1 中所定义的迭代映射 {{< math >}}$F${{< /math >}} 与迭代序列 {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}_{t\ge 0}${{< /math >}}，该迭代至多经过 {{< math >}}$L${{< /math >}} 步就收敛到不动点：

{{< math >}}
$$
\boldsymbol{z}_{k}^{t} = \boldsymbol{z}_{k}^{\ast} \quad \forall t\geq L. 
\tag{16}
$$
{{< /math >}}

**证明**：核心性质在于：输出的第 {{< math >}}$l${{< /math >}} 个分量 {{< math >}}$(F(\boldsymbol{z}))_{l}${{< /math >}} 只依赖于输入 {{< math >}}$\boldsymbol{z}${{< /math >}} 的前 {{< math >}}$l-1${{< /math >}} 个分量，即 {{< math >}}$\boldsymbol{z}_{&lt;l}${{< /math >}}。我们同样知道 {{< math >}}$(F(\boldsymbol{z}_{k}^{0}))_{1} = \boldsymbol{z}_{k+1,1} = \boldsymbol{z}_{k,1}^{\ast}${{< /math >}}。

我们将对迭代步 {{< math >}}$t${{< /math >}}（从 {{< math >}}$t=1${{< /math >}} 到 {{< math >}}$t=L${{< /math >}}）用数学归纳法证明：迭代点 {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} 的前 {{< math >}}$t${{< /math >}} 个分量与不动点 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 的相应分量相同。设 {{< math >}}$P(t)${{< /math >}} 为如下命题：

{{< math >}}
$$
P(t): \quad \boldsymbol{z}_{k,l}^{t} = \boldsymbol{z}_{k,l}^{\ast} \quad \forall{1 \le l \le t}. 
\tag{17}
$$
{{< /math >}}

容易验证命题 {{< math >}}$P(1)${{< /math >}} 成立。假设 {{< math >}}$P(t)${{< /math >}} 成立，即对所有 {{< math >}}$1 \le l \le t${{< /math >}} 有 {{< math >}}$\boldsymbol{z}_{k,l}^{t} = \boldsymbol{z}_{k,l}^{\ast}${{< /math >}}，我们希望证明 {{< math >}}$P(t+1)${{< /math >}} 也成立，即对所有 {{< math >}}$1 \le l \le t+1${{< /math >}} 有 {{< math >}}$\boldsymbol{z}_{k,l}^{t+1} = \boldsymbol{z}_{k,l}^{\ast}${{< /math >}}。

由于 {{< math >}}$\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t})${{< /math >}}：

- **对于 {{< math >}}$1 \le l \le t${{< /math >}}：** {{< math >}}$\boldsymbol{z}_{k,l}^{t+1} = (F(\boldsymbol{z}_{k}^{t}))_{l}${{< /math >}} 的计算只依赖于 {{< math >}}$\boldsymbol{z}_{k,&lt;l}^{t}${{< /math >}}。由于 {{< math >}}$j &lt; l \leq t${{< /math >}}，归纳假设 {{< math >}}$P(t)${{< /math >}} 蕴含这些分量满足 {{< math >}}$\boldsymbol{z}_{k,j}^{t} = \boldsymbol{z}_{k,j}^{\ast}${{< /math >}}。因此 {{< math >}}$\boldsymbol{z}_{k,&lt;l}^{t} = \boldsymbol{z}_{k,&lt;l}^{\ast}${{< /math >}}。又因为 {{< math >}}$(F(\cdot))_{l}${{< /math >}} 只依赖于这前 {{< math >}}$l-1${{< /math >}} 个分量，我们有

  {{< math >}}
  $$
  (F(\boldsymbol{z}_{k}^{t}))_{l} = (F(\boldsymbol{z}_{k}^{\ast}))_{l}.
  \tag{18}
  $$
  {{< /math >}}

  由于 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 是不动点，故 {{< math >}}$F(\boldsymbol{z}_{k}^{\ast}) = \boldsymbol{z}_{k}^{\ast}${{< /math >}}，于是我们有

  {{< math >}}
  $$
  \boldsymbol{z}_{k,l}^{t+1} = (F(\boldsymbol{z}_{k}^{\ast}))_{l} = \boldsymbol{z}_{k,l}^{\ast}.
  \tag{19}
  $$
  {{< /math >}}

- **对于 {{< math >}}$l = t+1${{< /math >}}：** {{< math >}}$\boldsymbol{z}_{k,t+1}^{t+1} = (F(\boldsymbol{z}_{k}^{t}))_{t+1}${{< /math >}} 的计算只依赖于 {{< math >}}$\boldsymbol{z}_{k,&lt;t+1}^{t}${{< /math >}}。该子向量中的分量为 {{< math >}}$\boldsymbol{z}_{k,j}^{t}${{< /math >}}，{{< math >}}$j = 1, \dots, t${{< /math >}}。根据归纳假设 {{< math >}}$P(t)${{< /math >}}，它们等于 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} 的相应分量。因此 {{< math >}}$\boldsymbol{z}_{k,&lt;t+1}^{t} = \boldsymbol{z}_{k,&lt;t+1}^{\ast}${{< /math >}}。又因为 {{< math >}}$(F(\cdot))_{t+1}${{< /math >}} 只依赖于前 {{< math >}}$t${{< /math >}} 个分量

  {{< math >}}
  $$
  (F(\boldsymbol{z}_{k}^{t}))_{t+1} = (F(\boldsymbol{z}_{k}^{\ast}))_{t+1}
  \tag{20}
  $$
  {{< /math >}}

  利用不动点性质：

  {{< math >}}
  $$
  \boldsymbol{z}_{k,t+1}^{t+1} = (F(\boldsymbol{z}_{k}^{\ast}))_{t+1} = \boldsymbol{z}_{k,t+1}^{\ast}.
  \tag{21}
  $$
  {{< /math >}}

  这说明第 {{< math >}}$(t+1)${{< /math >}} 个分量在第 {{< math >}}$t+1${{< /math >}} 步就变得正确了。

结合式 (19) 与式 (21)，我们证明了 {{< math >}}$\boldsymbol{z}_{k,l}^{t+1} = \boldsymbol{z}_{k,l}^{\ast},\forall{1 \le l \le t+1}${{< /math >}}。因此，{{< math >}}$P(t+1)${{< /math >}} 成立。

由数学归纳法可知，{{< math >}}$P(t)${{< /math >}} 对所有 {{< math >}}$t=1, \dots, L${{< /math >}} 都成立。特别地，{{< math >}}$P(L)${{< /math >}} 成立：

{{< math >}}
$$
\boldsymbol{z}_{k,l}^{L} = \boldsymbol{z}_{k,l}^{\ast} \quad \forall{1 \le l \le L}.
\tag{22}
$$
{{< /math >}}

这意味着整个向量在 {{< math >}}$L${{< /math >}} 步之后一定与不动点相同：

{{< math >}}
$$
\boldsymbol{z}_{k}^{L} = \boldsymbol{z}_{k}^{\ast}.
\tag{23}
$$
{{< /math >}}

假设对某个 {{< math >}}$t \ge L${{< /math >}} 有 {{< math >}}$\boldsymbol{z}_{k}^{t} = \boldsymbol{z}_{k}^{\ast}${{< /math >}}。那么在下一次迭代中：

{{< math >}}
$$
\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t}) = F(\boldsymbol{z}_{k}^{\ast}) = \boldsymbol{z}_{k}^{\ast}
\tag{24}
$$
{{< /math >}}

出于同样的原因，如果该序列在第 {{< math >}}$L${{< /math >}} 步到达 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}，那么在其后的所有步骤中都会保持在 {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}。因此，我们证明了

{{< math >}}
$$
\boldsymbol{z}_{k}^{t} = \boldsymbol{z}_{k}^{\ast} \quad \forall{t \geq L}.
\tag{25}
$$
{{< /math >}}

## C. 局限性与未来工作

尽管我们的方法在自回归标准化流模型的推理加速上展现出令人鼓舞的提升，它同时也凸显了若干开放问题。首先，虽然顺序冗余和深度冗余在训练好的模型中普遍存在，但这类冗余在部分训练或训练不足的模型中是否仍保持同等程度尚不清楚。这一不确定性可能会在模型欠拟合或处于训练早期阶段时影响方法的有效性。其次，本文只聚焦于推理阶段加速，尚未探索如何利用本文的原理来优化训练过程或指导神经网络架构设计。未来工作可以进一步研究所观察到的这些顺序冗余与深度冗余的本质。由此获得的洞见随后可用于指导模型训练策略并启发架构设计，从而有可能得到本身就更高效的模型。

## D. 内存复杂度分析

原始的顺序推理和我们的方法都具有 {{< math >}}$O(L^2)${{< /math >}} 的内存复杂度。原始 TarFlow 需要 {{< math >}}$O(L^2)${{< /math >}} 的内存，这是因为注意力机制作用在长度为 {{< math >}}$L${{< /math >}} 的序列上，而 KV 缓存并不会增加渐近复杂度。类似地，我们的方法也保持相同的 {{< math >}}$O(L^2)${{< /math >}} 复杂度，因为我们在每一步都对整个长度为 {{< math >}}$L${{< /math >}} 的序列执行注意力操作，且不需要在不同步骤之间共享注意力矩阵。

尽管渐近复杂度相同，实际的内存占用在实践中却有差别。在 AFHQ 上（批大小为 16），我们的方法仅占用 5.2GB 内存，而使用 KV 缓存的基线实现则需要 7.8GB。这一差异源于：基线为避免冗余计算而额外存储了 {{< math >}}$K${{< /math >}} 和 {{< math >}}$V${{< /math >}} 张量，而我们的方法通过并行处理来获得加速，无需这类存储开销。因此，尽管我们的方法并行处理 {{< math >}}$L${{< /math >}} 个输入，它仍表现出更好的内存效率。

## E. 补充实验

### E.1. TarFlow 上的实验细节

**模型细节。** 我们主要基线方法的网络架构取自 TarFlow 的公开实现（注：<https://github.com/apple/ml-tarflow>）。在 AFHQ 数据集上的实验中，我们使用 TarFlow 作者发布的预训练检查点。
受计算资源限制，在本文的工作范围内无法按照 TarFlow 的原始配置在 ImageNet 数据集上训练模型。
在 CIFAR 数据集上的实验中，我们基本遵循 TarFlow 提供的默认设置，仅作少量调整。
这些改动是为了更好地契合我们的实验目标，或是为了适应资源上的限制。
表 A2 汇总了各数据集的关键配置。

**表 A2**：各数据集的实验配置。

|  | CIFAR-10 | CIFAR-100 | AFHQ |
| --- | --- | --- | --- |
| 分辨率 | 32×32 | 32×32 | 256×256 |
| 图块大小 P | 2 | 2 | 8 |
| 序列长度 L | 256 | 256 | 1024 |
| 块数 K | 6 | 6 | 8 |
| 每块层数 | 6 | 6 | 8 |
| 隐藏维度 | 256 | 256 | 768 |
| 批大小 | 256 | 256 | 256 |

**评估细节。**
为估计生成速度，我们计算 10 次不同运行下的每批次平均耗时。
对于 Fréchet Inception 距离（Fréchet Inception Distance, FID）的估计，我们按照标准的 FID 定义，计算原始数据集与同等规模的生成数据集之间的距离。
为评估生成质量，我们采用 CLIP-IQA 和 BRISQUE 等指标，在与原始数据集规模相匹配的生成样本集合上计算各指标的平均分数。
这些做法涉及取平均和大样本量，因而能够保证结果稳定且具有代表性。
所采用的评估方法在多个批次上对结果取平均并使用大规模数据集，可以确保结果具有代表性。
这与文献中已确立的做法一致 (<a href="#ref-26">Song et al., 2021b</a>; <a href="#ref-28">Teng et al., 2025</a>)。
对于 CIFAR 数据集，我们还额外报告三次运行中的最大偏差。
该偏差的量级远小于不同方法之间观察到的性能差距，从而确认了我们对比结果的统计显著性以及所报告改进的有效性。

**推理细节。**
表 A3 报告了使用我们的选择性 Jacobi 解码进行推理时每层的平均迭代次数。第 1 层使用标准的顺序解码（L-1 步），其余各层使用 Jacobi 迭代。值得注意的是，几乎所有 Jacobi 层都在极少的迭代次数内收敛（通常为 4—7 次），远低于最坏情况界 L，这验证了我们的理论分析。CIFAR-10 上第 2 层相对较高的迭代次数与我们关于深度异质性的观察一致，即越靠近第一层的层往往表现出越强的顺序依赖。

**表 A3**：每层的平均 Jacobi 迭代次数（τ=0.5）。第 1 层使用顺序解码；其余各层使用 Jacobi 迭代。

| 层 | CIFAR-10 | CIFAR-100 | AFHQ |
| --- | --- | --- | --- |
| 1（顺序推理） | 255 | 255 | 1023 |
| 2（Jacobi） | 53.9 | 7.5 | 6.6 |
| 3（Jacobi） | 4.9 | 4.7 | 6.0 |
| 4（Jacobi） | 4.0 | 4.0 | 5.2 |
| 5（Jacobi） | 3.0 | 4.0 | 5.0 |
| 6（Jacobi） | 6.1 | 5.2 | 5.9 |
| 7（Jacobi） | — | — | 4.5 |
| 8（Jacobi） | — | — | 4.0 |

表 A4 给出了顺序推理与我们的方法各自的逐层运行时间分解。在顺序推理中，各层耗时大致相同。在 SJD 下，第 1 层（顺序解码）占据了总开销的主要部分，而每个 Jacobi 层只需其中很小一部分时间即可完成。这确认了加速来源于在冗余较高的层上用快速收敛的 Jacobi 迭代替代了代价高昂的顺序解码。

**表 A4**：顺序推理与 SJD 的逐层运行时间分解对比。「其它」包括自去噪、GPU 间通信以及噪声生成开销等。

| 层 | 顺序推理 时间 (s) | 顺序推理 % | SJD（本文方法） 时间 (s) | SJD（本文方法） % | Jacobi 迭代次数 |
| --- | --- | --- | --- | --- | --- |
| *CIFAR-10* |  |  |  |  |  |
| 1（顺序） | 1.45 | 14.8% | 1.50 | 55.7% | 255 |
| 2（Jacobi） | 1.45 | 14.8% | 0.69 | 25.7% | 53.9 |
| 3—6（Jacobi） | 5.82 | 59.4% | 0.23 | 8.6% | 3.0—6.1 |
| 其它 | 1.08 | 11.0% | 0.27 | 10.0% | — |
| **总计** | **9.79** | **100%** | **2.69** | **100%** | **3.6×** |
| *CIFAR-100* |  |  |  |  |  |
| 1（顺序） | 1.51 | 16.5% | 1.55 | 76.6% | 255 |
| 2（Jacobi） | 1.50 | 16.4% | 0.10 | 4.7% | 7.5 |
| 3—6（Jacobi） | 6.00 | 65.6% | 0.23 | 11.4% | 4.0—5.2 |
| 其它 | 0.15 | 1.6% | 0.15 | 7.2% | — |
| **总计** | **9.15** | **100%** | **2.02** | **100%** | **4.5×** |
| *AFHQ* |  |  |  |  |  |
| 1（顺序） | 21.60 | 12.5% | 21.46 | 52.0% | 1023 |
| 2（Jacobi） | 20.85 | 12.1% | 2.56 | 6.2% | 6.6 |
| 3—8（Jacobi） | 125.10 | 72.4% | 11.85 | 28.7% | 4.0—6.0 |
| 其它 | 5.35 | 3.1% | 5.38 | 13.1% | — |
| **总计** | **172.90** | **100%** | **41.25** | **100%** | **4.2×** |

### E.2. 图 1 与图 4 的完整结果。

我们分别在图 A1 与图 A2 中给出图 1 与图 4 的完整结果。

{{< figgrid caption="**图 A1**：标准推理与「分别掩蔽最近 o = 1、o = 2、o = 5 个前序依赖」的推理，二者各层输出之间的余弦相似度与 L2 距离。" >}}
sejd/obs_c100_1_dual_axis.png | 30 | **(a)** CIFAR-100，o = 1
sejd/obs_afhq_1_dual_axis.png | 30 | **(b)** AFHQ，o = 1
sejd/obs_c100_2_dual_axis.png | 30 | **(c)** CIFAR-100，o = 2
sejd/obs_afhq_2_dual_axis.png | 30 | **(d)** AFHQ，o = 2
sejd/obs_c100_5_dual_axis.png | 30 | **(e)** CIFAR-100，o = 5
sejd/obs_afhq_5_dual_axis.png | 30 | **(f)** AFHQ，o = 5
{{< /figgrid >}}


{{< figgrid caption="**图 A2**：Jacobi 解码在各网络层上的收敛动态。图中展示误差（以当前迭代点与顺序推理输出之差的 ℓ<sub>2</sub> 范数衡量）随迭代次数的变化，可见整体收敛很快，而第 1 层的收敛明显更慢。" >}}
sejd/errs_afhq_layer0.png | 22 | **(a)** 第 1 层
sejd/errs_afhq_layer1.png | 22 | **(b)** 第 2 层
sejd/errs_afhq_layer2.png | 22 | **(c)** 第 3 层
sejd/errs_afhq_layer3.png | 22 | **(d)** 第 4 层
sejd/errs_afhq_layer4.png | 22 | **(e)** 第 5 层
sejd/errs_afhq_layer5.png | 22 | **(f)** 第 6 层
sejd/errs_afhq_layer6.png | 22 | **(g)** 第 7 层
sejd/errs_afhq_layer7.png | 22 | **(h)** 第 8 层
{{< /figgrid >}}


### E.3. 在掩码自回归流上的实验

本节在图像生成与 Boltzmann 分布近似两类任务上测试掩码自回归流（Masked Autoregressive Flow, MAF）模型。我们采用 `nflows` 的实现（注：<https://github.com/bayesiains/nflows/blob/master/nflows/transforms/autoregressive.py>）。需要注意的是，KV 缓存并不适用于这种基于 MLP 的架构。因此，Jacobi 解码能够在所有层上都带来加速，于是我们选择在所有层上施加 Jacobi 解码，而非仅在非第一层上使用，从而获得更高的加速比。

**Boltzmann 分布近似。** 我们在一个 8 层的 MAF 上进行测试，该模型通过反向 KL 训练，用于近似二维 Ising 模型在高温（T=3.0）无序态下的 Boltzmann 分布。反向 KL 散度损失在 3000 轮训练中从初始值 -1111 下降到最终值 -1129。为评估性能，我们生成了 100,000 个样本，并将我们的方法与标准的顺序推理方法进行对比。结果表明，我们的方法在几乎不影响质量的前提下带来了显著的加速，如表 A5 所示。

**表 A5**：在 Boltzmann 分布近似任务上，MAF 的顺序推理与本文方法的对比。

| 方法 | 推理时间（s） | 每格点平均能量 | 平均绝对磁化强度 |
| --- | --- | --- | --- |
| 顺序推理 | 16.84 | 0.0005 | 0.0500 |
| 本文方法 | 1.07 | -0.0003 | 0.0498 |

接近零的能量与磁化强度数值与无序态物理相符，说明样本质量得到了保持。值得注意的是，我们实现了 15.7 倍的加速。该实验为本文方法的通用性提供了有力证据。

**图像生成。** 我们在一个于二值 MNIST 上训练的 8 层掩码自回归流（MAF）上开展了进一步实验。由于该模型的表达能力有限，其生成质量较低。但我们仍然观察到，本文方法与顺序推理所生成的图像质量非常接近，如图 A3 所示。生成 100 张图像时，本文方法仅需 15.24 秒，而原始的顺序推理方法需要 281.00 秒。这是高达 18.4 倍的显著加速，为本文方法的通用性提供了有力证据。

{{< figgrid caption="**图 A3**：二值 MNIST 上的可视化对比。" >}}
sejd/generations.png | 98 | **(a)** 顺序推理
sejd/generations_jacobi.png | 98 | **(b)** 本文方法，加速 18.4 倍
{{< /figgrid >}}


### E.4. 重建一致性评估

离散自回归标准化流的一个核心优势在于其严格可逆的架构，这使得完美重建成为可能。为了明确量化本文的并行迭代近似所引入的任何数值偏差，并验证模型可逆性得以保持，我们对重建一致性进行了评估。具体而言，我们使用标准的顺序前向传播，将原始数据集中的真实图像 {{< math >}}$\boldsymbol{x}${{< /math >}} 映射为精确的潜变量 {{< math >}}$\boldsymbol{z}${{< /math >}}，随后再使用我们的 SJD 将其重建回像素空间 {{< math >}}$\hat{\boldsymbol{x}}${{< /math >}}。

本文方法在原始输入 {{< math >}}$\boldsymbol{x}${{< /math >}} 与重建输出 {{< math >}}$\hat{\boldsymbol{x}}${{< /math >}} 之间取得了极低的均方误差（Mean Squared Error, MSE）：CIFAR-10 上为 0.00636，CIFAR-100 上为 0.00313，AFHQ 上为 0.00122（均在 SJD、{{< math >}}$\tau=0.5${{< /math >}} 下评估）。这些接近于零的数值误差表明，放松严格的顺序依赖所引入的偏差实际上可以忽略不计，并且并行迭代能够紧密收敛到精确的顺序推理解。这一定量精度还得到了目视检查的进一步印证。如图 A4、图 A5 和图 A6 所示，重建图像与原始图像在视觉上无法区分，没有可察觉的细节损失。定量与定性结果都有力地验证了：本文方法成功保持了基于流的模型所固有的严格双射一致性与高保真生成质量。

{{< figgrid caption="**图 A4**：CIFAR-10 测试集上的重建一致性。**上排**：原始真实图像。**下排**：用本文的 Selective Jacobi Decoding 重建的图像。重建结果在视觉上与原始输入无法区分。" >}}
sejd/cifar-10_orig_vs_recon_appro_0.5.png | 95
{{< /figgrid >}}


{{< figgrid caption="**图 A5**：CIFAR-100 测试集上的重建一致性。**上排**：原始真实图像。**下排**：用本文的 Selective Jacobi Decoding 重建的图像。" >}}
sejd/cifar-100_orig_vs_recon_appro_0.5.png | 95
{{< /figgrid >}}


{{< figgrid caption="**图 A6**：AFHQ 测试集上的重建一致性。**上排**：原始真实图像。**下排**：用本文的 Selective Jacobi Decoding 重建的图像。" >}}
sejd/afhq_orig_vs_recon_appro_0.5.jpg | 95
{{< /figgrid >}}


### E.5. 与 GAN 和扩散模型的对比分析

为了给本文所提加速方法的实用价值提供参照，我们在 CIFAR-10 数据集上与具有代表性的生成对抗网络（Generative Adversarial Network, GAN）和扩散模型进行了对比分析。对于 GAN 基线方法，我们使用官方实现从零开始训练 FastGAN (<a href="#ref-32">Zhong et al., 2020</a>)。对于扩散模型基线方法，我们评估了 DDIM (<a href="#ref-25">Song et al., 2021a</a>)，使用公开的 `google/ddpm-cifar10-32` 检查点并采用 20 个推理步，以建立可比的速度水平。

**表 A6**：在 CIFAR-10 数据集上，本文方法与 FastGAN、DDIM 的对比。

| **方法** | **推理时间（s）** ↓ | **FID** ↓ |
| --- | --- | --- |
| Fast GAN | 3.41 | 9.67 |
| DDIM（20 步） | 2.31 | 19.32 |
| 本文方法 | 2.63 | 10.20 |

如表 A6 所详述，本文方法在生成速度与质量之间展现出极具竞争力的平衡。它的推理速度快于 FastGAN，而在 FID 上仅有很小的权衡代价。虽然 20 步的 DDIM 略快一些，但其生成质量付出了严重的代价。与 20 步的 DDIM 相比，本文方法仅略慢一点，却取得了明显更好的生成质量。

### E.6. 更多可视化结果

我们在图 A7 和图 A8 中提供了 CIFAR-10 与 CIFAR-100 上更多的可视化实验结果。所有结果都一致地证实了本文方法对生成质量的影响很小。

{{< figgrid caption="**图 A7**：CIFAR-10 上的可视化对比。" >}}
sejd/cifar10base.png | 49 | **(a)** 顺序推理
sejd/cifar10sjd.png | 49 | **(b)** 本文方法，加速 3.6 倍
{{< /figgrid >}}


{{< figgrid caption="**图 A8**：CIFAR-100 上的可视化对比。" >}}
sejd/cifar100base.png | 49 | **(a)** 顺序推理
sejd/cifar100sjd.png | 49 | **(b)** 本文方法，加速 4.7 倍
{{< /figgrid >}}


## 参考文献

<span id="ref-1"></span>Angeline Aguinaldo, Ping-Yeh Chiang, Alex Gain, Ameya Patil, Kolten Pearson, and Soheil Feizi. Compressing GANs using Knowledge Distillation. *arXiv preprint arXiv:1902.00159*, 2019.

<span id="ref-2"></span>Pavel Andreev and Alexander Fritzler. Quantization of Generative Adversarial Networks for Efficient Inference: A Methodological Study. In *ICPR*, 2022.

<span id="ref-3"></span>Yunjey Choi, Youngjung Uh, Jaejun Yoo, and Jung-Woo Ha. StarGAN v2: Diverse Image Synthesis for Multiple Domains. In *CVPR*, 2020.

<span id="ref-4"></span>J. E. Dennis and Robert B. Schnabel. *Numerical Methods for Unconstrained Optimization and Nonlinear Equations (Classics in Applied Mathematics, 16)*. SIAM, 1996.

<span id="ref-5"></span>Laurent Dinh, David Krueger, and Yoshua Bengio. NICE: Non-linear Independent Components Estimation. In *ICLR Workshop*, 2015.

<span id="ref-6"></span>Laurent Dinh, Jascha Sohl-Dickstein, and Samy Bengio. Density Estimation using Real NVP. In *ICLR*, 2017.

<span id="ref-7"></span>Tim Dockhorn, Arash Vahdat, and Karsten Kreis. GENIE: Higher-Order Denoising Diffusion Solvers. In *NeurIPS*, 2022.

<span id="ref-8"></span>Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, and Sepp Hochreiter. GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium. In *NeurIPS*, volume 30, 2017.

<span id="ref-9"></span>Jonathan Ho, Xi Chen, Aravind Srinivas, Yan Duan, and Pieter Abbeel. Flow++: Improving flow-based generative models with variational dequantization and architecture design. In *ICML*, 2019.

<span id="ref-10"></span>Durk P Kingma and Prafulla Dhariwal. Glow: Generative Flow with Invertible 1x1 Convolutions. In *NeurIPS*, 2018.

<span id="ref-11"></span>Durk P Kingma, Tim Salimans, Rafal Jozefowicz, Xi Chen, Ilya Sutskever, and Max Welling. Improved Variational Inference with Inverse Autoregressive Flow. In *NeurIPS*, 2016.

<span id="ref-12"></span>Siqi Kou, Lanxiang Hu, Zhezhi He, Zhijie Deng, and Hao Zhang. CLLMs: Consistency Large Language Models. In *ICML*, 2024.

<span id="ref-13"></span>Alex Krizhevsky. Learning Multiple Layers of Features from Tiny Images. Technical report, University of Toronto, 2009.

<span id="ref-14"></span>Aman Kumar, Khushboo Anand, Shubham Mandloi, Ashutosh Mishra, Avinash Thakur, Neeraj Kasera, and AP Prathosh. CoroNetGAN: Controlled Pruning of GANs via Hypernetworks. In *ICCV Workshop*, 2023.

<span id="ref-15"></span>Cheng Lu, Yuhao Zhou, Fan Bao, Jianfei Chen, Chongxuan Li, and Jun Zhu. DPM-Solver: A Fast ODE Solver for Diffusion Probabilistic Model Sampling in Around 10 Steps. *NeurIPS*, 2022.

<span id="ref-16"></span>Anish Mittal, Anush Krishna Moorthy, and Alan Conrad Bovik. No-Reference Image Quality Assessment in the Spatial Domain. *IEEE Transactions on image processing*, 21 0 (12): 0 4695–4708, 2012.

<span id="ref-17"></span>James M Ortega and Werner C Rheinboldt. *Iterative Solution of Nonlinear Equations in Several Variables*. SIAM, 2000.

<span id="ref-18"></span>Myle Ott, Sergey Edunov, Alexei Baevski, Angela Fan, Sam Gross, Nathan Ng, David Grangier, and Michael Auli. FAIRSEQ: A Fast, Extensible Toolkit for Sequence Modeling. In *NAACL*, 2019.

<span id="ref-19"></span>George Papamakarios, Theo Pavlakou, and Iain Murray. Masked Autoregressive Flow for Density Estimation. In *NeurIPS*, 2017.

<span id="ref-20"></span>George Papamakarios, Eric Nalisnick, Danilo Jimenez Rezende, Shakir Mohamed, and Balaji Lakshminarayanan. Normalizing Flows for Probabilistic Modeling and Inference. *Journal of Machine Learning Research*, 22 0 (57): 0 1–64, 2021.

<span id="ref-21"></span>Danilo Rezende and Shakir Mohamed. Variational Inference with Normalizing Flows. In *ICML*, 2015.

<span id="ref-22"></span>Tim Salimans and Jonathan Ho. Progressive Distillation for Fast Sampling of Diffusion Models. In *ICLR*, 2022.

<span id="ref-23"></span>Andrea Santilli, Silvio Severino, Emilian Postolache, Valentino Maiorca, Michele Mancusi, Riccardo Marin, and Emanuele Rodola. Accelerating Transformer Inference for Translation via Parallel Decoding. In *ACL*, 2023.

<span id="ref-24"></span>Divya Saxena, Jiannong Cao, Jiahao Xu, and Tarun Kulshrestha. RG-GAN: Dynamic Regenerative Pruning for Data-Efficient Generative Adversarial Networks. In *AAAI*, 2024.

<span id="ref-25"></span>Jiaming Song, Chenlin Meng, and Stefano Ermon. Denoising Diffusion Implicit Models. In *ICLR*, 2021a.

<span id="ref-26"></span>Yang Song, Chenlin Meng, Renjie Liao, and Stefano Ermon. Accelerating Feedforward Computation via Parallel Nonlinear Equation Solving. In *ICML*, 2021b.

<span id="ref-27"></span>Yang Song, Prafulla Dhariwal, Mark Chen, and Ilya Sutskever. Consistency Models. In *ICLR*, 2023.

<span id="ref-28"></span>Yao Teng, Han Shi, Xian Liu, Xuefei Ning, Guohao Dai, Yu Wang, Zhenguo Li, and Xihui Liu. Accelerating Auto-regressive Text-to-Image Generation with Training-free Speculative Jacobi Decoding. In *ICLR*, 2025.

<span id="ref-29"></span>Jianyi Wang, Kelvin CK Chan, and Chen Change Loy. Exploring CLIP for Assessing the Look and Feel of Images. In *AAAI*, 2023.

<span id="ref-30"></span>Sangyeop Yeo, Yoojin Jang, and Jaejun Yoo. Nickel and Diming Your GAN: A Dual-Method Approach to Enhancing GAN Efficiency via Knowledge Distillation. In *ECCV*, 2024.

<span id="ref-31"></span>Shuangfei Zhai, Ruixiang Zhang, Preetum Nakkiran, David Berthelot, Jiatao Gu, Huangjie Zheng, Tianrong Chen, Miguel Ángel Bautista, Navdeep Jaitly, and Joshua M Susskind. Normalizing Flows are Capable Generative Models. In *ICML*, 2025.

<span id="ref-32"></span>Jiachen Zhong, Xuanqing Liu, and Cho-Jui Hsieh. Improving the Speed and Quality of GAN by Adversarial Training. *arXiv preprint arXiv:2008.03364*, 2020.

---

## 译者说明

本页中文译文由 **Claude Opus 5 + ultracode** 翻译，**仅供参考，对内容正确性不作保证**。译文力求与原文逐段对应，公式、表格与数值均按原文照录；专有名词（数据集名、模型名、评价指标、人名）保留英文原文。若中英文表述存在出入，一律以英文原文为准。

- 英文原文 PDF：[https://openreview.net/pdf?id=xYATz9HpE7](https://openreview.net/pdf?id=xYATz9HpE7)
- 英文全文（网页版）：[Full Text]({{< relref "/publication/sejd-en" >}})
- 论文主页：[Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding]({{< relref "/publication/sejd" >}})
- 代码仓库：[https://github.com/lan-qing/SJD](https://github.com/lan-qing/SJD)
