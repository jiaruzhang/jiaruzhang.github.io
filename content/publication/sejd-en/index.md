---
title: "Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding"
subtitle: "Full text — web transcription"

summary: "Web transcription of the full text of Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding (TMLR 2026, Featured Certification), with all equations, tables and figures."

date: '2026-05-13T00:00:00Z'
publishDate: '2026-05-13T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/sejd/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/sejd/en/'

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
This page is a web transcription of the full text of **“Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding”** (TMLR 2026, Featured Certification), reproduced here for easier reading, searching and linking.
It is provided **for reference only**; please read and cite the [original PDF](https://openreview.net/pdf?id=xYATz9HpE7), which is authoritative. A [Chinese translation]({{< relref "/publication/sejd-cn" >}}) is also available.
{{% /callout %}}

{{< toc >}}

Jiaru Zhang, Juanwu Lu, Xiaoyu Wu, Ziran Wang, and Ruqi Zhang

> Jiaru Zhang (jiaru@purdue.edu), Institute of Physical Artificial Intelligence, Purdue University.
> Juanwu Lu (juanwu@purdue.edu), College of Engineering, Purdue University.
> Xiaoyu Wu (xw105@rice.edu), Rice University.
> Ziran Wang (ziran@purdue.edu), College of Engineering, Purdue University.
> Ruqi Zhang (ruqiz@purdue.edu), Department of Computer Science, Purdue University.
> Correspondence to: Jiaru Zhang. Equal advising.

## Abstract

Discrete normalizing flows are promising generative models with advantages such as analytical log-likelihood computation and end-to-end training.
However, the architectural constraints to ensure invertibility and tractable Jacobian computation limit their expressive power and practical usability.
Recent advancements utilize autoregressive modeling, significantly enhancing expressive power and generation quality.
Nevertheless, such sequential modeling inherently restricts parallel computation during inference, leading to slow generation that impedes practical deployment.
In this paper, we first identify that strict sequential dependency in inference is unnecessary to generate high-quality samples.
We observe that sub-variables in sequential modeling can also be approximated without strictly conditioning on all preceding sub-variables.
Moreover, the models tend to exhibit low dependency redundancy in the initial layer and higher redundancy in subsequent layers.
Leveraging these observations, we propose to selectively use Jacobi decoding strategy that accelerates its autoregressive inference through parallel iterative optimization.
Theoretical analyses demonstrate the method's superlinear convergence rate and guarantee that the number of iterations required is no greater than the original sequential approach.
Empirical evaluations across multiple datasets validate the generality and effectiveness of our acceleration technique, achieving up to 4.7 times faster inference on modern normalizing flow models while preserving generation quality.

## 1. Introduction

Discrete normalizing flow models (*a.k.a. normalizing flows*) have emerged as promising generative models by modeling the transformation between a data variable and a latent Gaussian variable through a series of mappings, where the mapping functions are constructed as invertible functions parameterized by neural networks.
This approach allows for end-to-end training with a single loss function, ensuring consistency between encoding and decoding.
Moreover, the model's structure facilitates the computation of the analytical log-likelihood.
These benefits have attracted significant interest from the research community, positioning discrete normalizing flow models as a compelling tool for generative modeling (<a href="#ref-5">Dinh et al., 2015</a>; <a href="#ref-6">Dinh et al., 2017</a>; <a href="#ref-9">Ho et al., 2019</a>; <a href="#ref-10">Kingma & Dhariwal, 2018</a>; <a href="#ref-20">Papamakarios et al., 2021</a>).

Although discrete normalizing flow models are theoretically rigorous, they often exhibit limited generation capabilities in practice due to unavoidable architectural constraints imposed by the requirement for invertible mappings and the need to compute Jacobian matrices.
For example, a classical construction of the invertible function, i.e., the affine coupling layer, is to split the input into two parts, and the output is obtained by the concatenation of those processed separately by trainable networks (<a href="#ref-5">Dinh et al., 2015</a>; <a href="#ref-6">Dinh et al., 2017</a>; <a href="#ref-10">Kingma & Dhariwal, 2018</a>).
This formulation ensures the analytical solution of both the invertible function and the Jacobian matrix, but it also yields relatively limited expressive power and less compatible network architectures, which affect the quality of generated samples and practical applications.
To address this, the recently proposed TarFlow (<a href="#ref-31">Zhai et al., 2025</a>) model employs a block autoregressive architecture inspired by autoregressive normalizing flows (<a href="#ref-11">Kingma et al., 2016</a>; <a href="#ref-19">Papamakarios et al., 2017</a>).
It splits the input into a longer sequence rather than a couple and constructs the invertible function using masked autoregressive transformations.
This autoregressive modeling can be naturally incorporated into the causal vision transformer architecture, which provides a powerful representation, enabling TarFlow to achieve state-of-the-art performance in both density estimation and image synthesis.

However, while such autoregressive sequential modeling endows the model with powerful generative capabilities, it also incurs high parallel computational complexity during inference.
Concretely, the inverse function of the sequential construction is an autoregressive inference process, meaning that each new sub-variable is generated sequentially, relying on all of the previously generated sub-variables.
It limits parallel computation, thereby slowing generation speeds and restricting practical applications, as also noted in previous work (<a href="#ref-31">Zhai et al., 2025</a>).

In this work, we first identify that strict sequential conditioning contains significant, layer-varying redundant dependencies during inference.
Specifically, we observe that subsequent sub-variables can still be approximated without the nearest preceding sub-variables.
Moreover, we find that the amount of redundancy varies substantially across layers, with the later layers exhibiting substantially more redundancy than the first.
Motivated by these observations, we propose to selectively use the Jacobi decoding approach to accelerate the inference of discrete autoregressive normalizing flow models.
Our approach leverages parallel iterative optimization to achieve fast convergence to high-fidelity samples, without requiring additional training or modifications to the original model architecture, and is backed by theoretical convergence guarantees of superlinear speed and finite convergence.
Experimental results on diverse datasets, including CIFAR-10, CIFAR-100, and AFHQ (<a href="#ref-3">Choi et al., 2020</a>), verify the generality and effectiveness of the proposed acceleration approach.
The code is available at <https://github.com/lan-qing/SJD>.
In summary, our contributions include:

- **C1** We discover that strict dependency on the original sequential inference of discrete autoregressive normalizing flows contains redundancy, and the quantity of the redundancy varies across different layers, based on our theoretical analysis and empirical observation.

- **C2** We introduce an approach to accelerate inference in discrete autoregressive normalizing flows by selectively applying Jacobi decoding in inference. Theoretical analysis demonstrates superlinear convergence and a finite convergence guarantee.

- **C3** We conduct comprehensive experiments to validate the effectiveness of our approach, showing up to 4.7 times speed improvements with little impact on generation quality across diverse datasets, including CIFAR-10, CIFAR-100, and AFHQ.

## 2. Related Work

**Inference Acceleration of Generative Models.** Inference acceleration is critical for the practical application of generative models, and researchers have proposed various effective strategies for different generative models.
For Diffusion Models (DMs), techniques such as improved numerical solvers (<a href="#ref-7">Dockhorn et al., 2022</a>; <a href="#ref-15">Lu et al., 2022</a>; <a href="#ref-25">Song et al., 2021a</a>), knowledge distillation (<a href="#ref-22">Salimans & Ho, 2022</a>), and consistency modeling (<a href="#ref-27">Song et al., 2023</a>) have significantly reduced sampling times.
For Variational Autoencoders (VAEs) and Generative Adversarial Networks (GANs), methods like network pruning (<a href="#ref-14">Kumar et al., 2023</a>; <a href="#ref-24">Saxena et al., 2024</a>), quantization (<a href="#ref-2">Andreev & Fritzler, 2022</a>), and knowledge distillation (<a href="#ref-1">Aguinaldo et al., 2019</a>; <a href="#ref-30">Yeo et al., 2024</a>) are widely adopted to enhance inference efficiency.
These approaches, however, typically leverage specific architectural properties or training paradigms inherent to these models.
Consequently, they are generally not directly transferable to inference acceleration of autoregressive normalizing flows.
To the best of our knowledge, this work is the first to explore the acceleration of inference for normalizing flow models.

**Jacobi Decoding.** Inspired by the research on non-linear equation solutions (<a href="#ref-17">Ortega & Rheinboldt, 2000</a>),
Jacobi decoding has emerged as a promising approach to accelerating neural network inference.
It aims to break the sequential dependency by reformulating generation as an iterative process of solving a system of equations, often framed as a fixed-point problem.
<a href="#ref-26">Song et al. (2021b)</a> first developed a theoretical framework for interpreting feedforward computation as the solution of nonlinear equations and revealed the significant potential of Jacobi decoding for networks such as RNNs and DenseNets.
<a href="#ref-23">Santilli et al. (2023)</a> further confirms the effectiveness of Jacobi decoding on the language generation task.
It was further improved with additional fine-tuning to keep the consistency of decoded tokens (<a href="#ref-12">Kou et al., 2024</a>).
In image generation, <a href="#ref-28">Teng et al. (2025)</a> explored the inference acceleration by the combination of Jacobi decoding with a probabilistic criterion for token acceptance on autoregressive text-to-image generation models.
Although previously applied to language and image generation, these methods commonly encounter issues such as quality degradation (<a href="#ref-28">Teng et al., 2025</a>) and limited acceleration in discrete token spaces (<a href="#ref-23">Santilli et al., 2023</a>).

## 3. Methodology

This section introduces the method that selectively applies Jacobi decoding to accelerate discrete autoregressive normalizing flows. It improves inference efficiency by breaking dependencies among sub-variables.
We begin with an introduction to normalizing flows (Section 3.1). Motivated by observations of sequential redundancy and its depthwise heterogeneity (Section 3.2), we propose parallel inference using Jacobi iterations (Section 3.3). We show that this method converges superlinearly and in finite time (Section 3.4). In addition, we propose a strategy that applies parallel Jacobi decoding only to higher-redundancy layers, thereby further improving efficiency (Section 3.5). To promote readability, we also present a table of notations in Section A.

### 3.1. Discrete Autoregressive Normalizing Flow

Normalizing flow is a family of generative models that explicitly learn a differential bijection {{< math >}}$\bm{x}=f(\bm{z})${{< /math >}} between a latent random variable {{< math >}}$\bm{z}${{< /math >}} and data {{< math >}}$\bm{x}${{< /math >}} by leveraging the change of variable law (<a href="#ref-5">Dinh et al., 2015</a>; <a href="#ref-21">Rezende & Mohamed, 2015</a>). The optimal transformation is given by maximizing the log-likelihood of observed data

{{< math >}}
$$
f^{\ast}\gets\underset{f}{\arg\max}\log{p}(\bm{x})=\log{p}(\bm{z})-\log\det{\mathbf{J}}_{f},
\tag{1}
$$
{{< /math >}}

where {{< math >}}${\mathbf{J}}_{f}${{< /math >}} is the Jacobian matrix of {{< math >}}$f${{< /math >}}. To facilitate capturing of more complex data distribution {{< math >}}$\bm{x}\sim{p_{\text{data}}(\bm{x})}${{< /math >}}, a discrete normalizing flow learns not a single but a cascade of {{< math >}}$K${{< /math >}} intermediate bijections such that

{{< math >}}
$$
\bm{x}=f(\bm{z}_{K})\triangleq\left(f_{0}\circ{f_{1}}\circ\ldots\circ{f_{K-1}}\right)(\bm{z}_{K}).
\tag{2}
$$
{{< /math >}}

The optimal set of bijections is then given by

{{< math >}}
$$
f^{\ast}\gets\underset{f_{0},\ldots,f_{K-1}}{\arg\max}\log{p(\bm{z}_{K})}-\sum\limits_{k=0}^{K-1}\log\det{\mathbf{J}}_{f_{k}}.
\tag{3}
$$
{{< /math >}}

However, properly constructing the bijections {{< math >}}$f_{k}${{< /math >}} for high-dimensional variables is tricky. A commonly adopted method, coupling-based normalizing flow (<a href="#ref-5">Dinh et al., 2015</a>), splits the high-dimensional {{< math >}}$\bm{z}_{k}${{< /math >}} into a pair of sub-variables {{< math >}}$\bm{z}_{k}=\left[\bm{z}_{k,1}, \bm{z}_{k,2}\right]^\intercal${{< /math >}}, and has inspired consecutive works that combine neural networks for modeling by constructing building blocks such as the real-valued non-volume preserving (RealNVP) (<a href="#ref-6">Dinh et al., 2017</a>), invertible {{< math >}}$1\times 1${{< /math >}} convolution (<a href="#ref-10">Kingma & Dhariwal, 2018</a>), and self-attention layers (<a href="#ref-9">Ho et al., 2019</a>).

More recent studies propose an autoregressive paradigm for discrete normalizing flow, which extends the above formulation by arbitrarily splitting random variables into a sequence of {{< math >}}$L${{< /math >}} sub-variables (<a href="#ref-11">Kingma et al., 2016</a>; <a href="#ref-19">Papamakarios et al., 2017</a>).
Then, the bijections {{< math >}}$f_{k}${{< /math >}} can be defined by

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

where {{< math >}}$\odot${{< /math >}} denotes Hadamard product, {{< math >}}$s_{k}(\cdot)${{< /math >}} and {{< math >}}$g_{k}(\cdot)${{< /math >}} are functions to learn during training.
Note that a permutation (e.g., reversing) of {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} is also applied after each transformation to ensure all locations in the sequence can be transformed through the entire flow, while we omit it for simplicity.
The bijection above naturally yields an inverse:

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

Discrete autoregressive normalizing flows have shown superiority in density estimation and image generation (<a href="#ref-31">Zhai et al., 2025</a>) by integrating them into architectures such as the causal vision transformer. However, a critical issue arises during inference, which involves computing the inverse transformation. As suggested by (5), generating {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} depends on all previous generations {{< math >}}$\boldsymbol{z}_{k,&lt;l}${{< /math >}}, which prohibits parallel computation and leads to slow sampling that restricts practical use cases. To address this issue, we leverage two fundamental observations on redundancy, presented in the next section.

### 3.2. Redundancy

**Sequential Redundancy.** As shown in (5), the generation process defined by the inverse transformation reveals a sequential dependency between each sub-variable and all preceding sub-variables.
Consequently, the original inference procedure enforces a strictly sequential generation paradigm, significantly limiting generation speed.
We hypothesize that this strict dependence is redundant, particularly for data such as images, which exhibit inherent spatial locality and continuity.
Therefore, an element {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} might be reasonably inferred even without precise, up-to-the-moment information from all its predecessors.

To validate our hypothesis, we conducted experiments using a straightforward transformation during inference, where the transformation step for element {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} is modified to ignore information from the {{< math >}}$o${{< /math >}}-nearest preceding elements in the sequence by

{{< math >}}
$$
\boldsymbol{z}_{k,l}= \boldsymbol{z}_{k+1,l}\odot\exp(-s_{k}(\boldsymbol{z}_{k,&lt;(l-o)}))+g_{k}(\boldsymbol{z}_{k,&lt;(l-o)}),
\tag{6}
$$
{{< /math >}}

for {{< math >}}$l&gt;1${{< /math >}}, where {{< math >}}$\boldsymbol{z}_{k,&lt;(l-o)}${{< /math >}} is obtained by masking out the nearest {{< math >}}$o${{< /math >}} preceding sub-variables in the attention operation. The experimental results shown in Figure 2 indicate that while the image quality diminishes as more of these nearest preceding sub-variables are removed, the model is still capable of generating meaningful images. It supports the claim that strict sequential dependencies contain potentially exploitable redundancy.

{{< figgrid caption="**Figure 1**: Cosine similarities and L2 distances between layer outputs from standard inference and inference with o = 5 nearest preceding dependencies masked. Results of o = 1 and o = 2 are available in A1." >}}
sejd/obs_c100_5_dual_axis.png | 45 | **(a)** CIFAR-100
sejd/obs_afhq_5_dual_axis.png | 45 | **(b)** AFHQ
{{< /figgrid >}}


{{< figgrid caption="**Figure 2**: Generations where dependency on the nearest o sub-variables is masked. It can still generate meaningful images, indicating the potential feasibility of acceleration with parallel computing." >}}
sejd/samples_cifar-10_0.png | 22 | **(a)** CIFAR-10, groundtruth
sejd/samples_cifar-10_1.png | 22 | **(b)** CIFAR-10, o = 1
sejd/samples_cifar-10_2.png | 22 | **(c)** CIFAR-10, o = 2
sejd/samples_cifar-10_5.png | 22 | **(d)** CIFAR-10, o = 5
sejd/samples_afhq_0.jpg | 22 | **(e)** AFHQ, groundtruth
sejd/samples_afhq_1.jpg | 22 | **(f)** AFHQ, o = 1
sejd/samples_afhq_2.jpg | 22 | **(g)** AFHQ, o = 2
sejd/samples_afhq_5.jpg | 22 | **(h)** AFHQ, o = 5
{{< /figgrid >}}


**Depthwise Heterogeneity of Redundancy.** We further investigate whether the degree of sequential redundancy varies across different layers during the generation process ({{< math >}}$\boldsymbol{z}_{K} \rightarrow \dots \rightarrow \boldsymbol{x}${{< /math >}}). Theoretically, we expect heterogeneity: random variables {{< math >}}$\boldsymbol{z}_{K}${{< /math >}} from the first layer, which performs structure initiation from a Gaussian noise, tend to exhibit high dependency on preceding sub-variables, as the pure noise input contains theoretically no information and hence relies more on context given by preceding generations {{< math >}}$\boldsymbol{z}_{K,&lt;l}${{< /math >}}.
Conversely, subsequent transformations refine informative outputs from the previous transformation {{< math >}}$\boldsymbol{z}_{k+1,l}${{< /math >}}, leading to weak sequential dependency regarding preceding generations {{< math >}}$\boldsymbol{z}_{k,&lt;l}${{< /math >}}.

To verify this, we measure the cosine similarity and L2 distance deviation between the standard inference outputs {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} and those generated while masking the {{< math >}}$o${{< /math >}} nearest dependencies as shown in (6).
As shown in Figure 1, the deviation is significantly larger for the first layer compared to subsequent ones.
This result empirically confirms the low redundancy in the first layer, which is consistent with our expectation.
The minimal deviation in subsequent layers aligns with their refinement role, which leverages informative inputs and the existing contextual structure.
This observation motivates exploring layer-specific optimizations for the generation process.

### 3.3. Parallel Inference by Jacobi Iteration

**Algorithm 1**: Jacobi decoding for {{< math >}}$f_{k}${{< /math >}}

```
Input:  Sequence z_{k+1}, functions s_k(·) and g_k(·), stopping threshold τ
Output: Sequence z_k

Initialize z_k^0 = 0, t = 0
while true do
    t ← t + 1,  z_{k,1}^t ← z_{k+1,1}
    for l = 2, ... L
    do in parallel
        z_{k,l}^t ← z_{k+1,l} ⊙ exp(-s_k(z_{k,<l}^{t-1})) + g_k(z_{k,<l}^{t-1})
    end for
    if ||z_k^t - z_k^{t-1}||_∞ < τ
        break
    end if
end while
z_k = z_k^t
```

Our empirical observations of sequential redundancy revealed a key property of inference in discrete autoregressive normalizing flow: *the generation of subsequent element {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} exhibits a degree of robustness to inaccuracies in the preceding elements {{< math >}}$\boldsymbol{z}_{k,&lt;l}${{< /math >}}*. This observation suggests that the strict, fully converged sequential dependency enforced by the standard inference procedure might contain redundancies and is not strictly necessary at every step for generation quality. This finding motivates exploring parallel computation strategies that can exploit this robustness.

To enable parallelization, we first re-examine the inference task. As established, generating the target sequence {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} from the input {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} via the inverse transformation defined in (5) fundamentally requires finding the unique solution {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} that satisfies the entire set of autoregressive conditional dependencies. It can be formally viewed as solving a system of {{< math >}}$L${{< /math >}} non-linear equations {{< math >}}$\mathcal{F}_{l}${{< /math >}}, implicitly defined for {{< math >}}$l=1,\ldots,L${{< /math >}} as

{{< math >}}
$$
\mathcal{F}_{l}(\boldsymbol{z}_{k,l},\boldsymbol{z}_{k,&lt;l},\boldsymbol{z}_{k+1,l})=0,
\tag{7}
$$
{{< /math >}}

where {{< math >}}$\mathcal{F}_{l}=0${{< /math >}} represents the condition imposed by the {{< math >}}$k${{< /math >}}-th step of the inverse transform, given the known input {{< math >}}$\boldsymbol{z}_{k+1, l}${{< /math >}}. The standard sequential inference method implicitly solves this system using a Gauss-Seidel-like approach, where the computation of {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} relies on the immediately preceding, fully computed values {{< math >}}$\boldsymbol{z}_{k,1},\ldots,\boldsymbol{z}_{k,l-1}${{< /math >}}.

Leveraging the potential for parallelism indicated by our observations in Section 3.2, we propose employing the Jacobi decoding method to solve the system defined by (7). Instead of sequential updates, it performs iterative, parallel updates. Starting from an initial estimate {{< math >}}$\boldsymbol{z}_{k}^{0}${{< /math >}}, each iteration {{< math >}}$t+1${{< /math >}} computes a new estimate {{< math >}}$\boldsymbol{z}_{k}^{t+1}${{< /math >}} where every element {{< math >}}$\boldsymbol{z}_{k, l}^{t+1}${{< /math >}} is calculated based *only* on the elements from the previous iterate {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} and the output from previous layer {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}}:

{{< math >}}
$$
\begin{aligned}
\text{ For } &amp;l= 1,\dots,L, \text{solve for } \boldsymbol{z}_{k,l}^{t+1} \text{ from: } \\
&amp;\mathcal{F}_l(\boldsymbol{z}_{k,l}^{t+1}, \boldsymbol{z}_{k,&lt;l}^{t}, \boldsymbol{z}_{k+1,l}) = 0.
\end{aligned}
\tag{8}
$$
{{< /math >}}

Because the calculation of each {{< math >}}$\boldsymbol{z}_{k,l}^{t+1}${{< /math >}} within an iteration only depends on values from the completed previous iteration {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}}, all {{< math >}}$L${{< /math >}} updates can be computed **concurrently**, breaking the sequential bottleneck. This iterative process continues until a suitable stopping criterion is met, such as the norm of the difference between consecutive iterates {{< math >}}$\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{t-1} \|${{< /math >}} being sufficiently small. The whole process of applying Jacobi decoding in discrete autoregressive normalizing flow inference is summarized in Algorithm 1.

**Remark 3.1.** Jacobi decoding techniques have been explored in other generative modeling contexts, such as language models and autoregressive image generation (<a href="#ref-23">Santilli et al., 2023</a>; <a href="#ref-26">Song et al., 2021b</a>). However, the straightforward application of Jacobi decoding has often shown limited success, e.g., marginal speedups for language models (<a href="#ref-12">Kou et al., 2024</a>) or compromised sample quality in image synthesis (<a href="#ref-28">Teng et al., 2025</a>). Despite these challenging precedents, we hypothesize that discrete autoregressive normalizing flows possess characteristics that might mitigate such issues and make Jacobi iteration a more viable strategy. The empirically observed redundant dependencies suggest that the system might tolerate the use of slightly inaccurate information {{< math >}}$\boldsymbol{z}_{k,&lt;l}^{t}${{< /math >}} from the previous iteration, which is fundamental to enabling Jacobi's parallel updates. Moreover, the deterministic nature of inversion avoids the compounding sampling errors present in stochastic models, and operating in continuous spaces could enable smoother iterative convergence than in discrete settings. These points provide a rationale for incorporating Jacobi decoding for discrete autoregressive normalizing flows.

### 3.4. Convergence

In this section, we theoretically analyze the convergence of the proposed Jacobi iterative inference method and identify two crucial properties:

- The iteration exhibits local superlinear convergence under certain conditions, implying rapid convergence in practice.
- Due to the inherent triangular dependency structure of discrete autoregressive normalizing flow, the iteration is guaranteed to converge to the exact solution with no more than the number of steps of the original inference, providing a worst-case bound.

Firstly, we analyze the local convergence behavior. Assuming an appropriate initialization close to the true solution, the Jacobi iteration converges superlinearly.

{{< figgrid caption="**Figure 3**: Visualization comparison on AFHQ. Our method accelerates generation by 4.5 times while maintaining the quality and fidelity of the generated content. Additional visual comparisons on CIFAR-10 and CIFAR-100 are available in A7 and A8." >}}
sejd/samples.jpg | 49 | **(a)** Sequential
sejd/samples_appro_0.5.jpg | 49 | **(b)** Ours, 4.5 times acceleration
{{< /figgrid >}}


**Proposition 3.1 (Superlinear Convergence Speed).** For the iterative sequence {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} defined in Algorithm 1, {{< math >}}$\exists \delta &gt;0${{< /math >}}, s.t. {{< math >}}$\forall \boldsymbol{z}_{k}^0${{< /math >}} satisfies {{< math >}}$||\boldsymbol{z}_{k}^0 - \boldsymbol{z}_{k}|| &lt; \delta${{< /math >}}, the iterative sequence {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}${{< /math >}} converges to {{< math >}}$\boldsymbol{z}_{k}${{< /math >}}, with superlinear convergence rate, i.e.,

{{< math >}}
$$
\|\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}\| = o(\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}\|).
$$
{{< /math >}}

Section B provides the detailed proof. Beyond the convergence rate, the specific structure of discrete autoregressive normalizing flow inference provides a strong global guarantee. The task is equivalent to solving a triangular system of equations, where the {{< math >}}$l${{< /math >}}-th unknown sub-variable depends only on preceding unknowns. This structural property, also noted in <a href="#ref-23">Santilli et al. (2023)</a>; <a href="#ref-26">Song et al. (2021b)</a>, ensures that the Jacobi method converges to the exact solution in at most {{< math >}}$L${{< /math >}} iterations for a sequence with length {{< math >}}$L${{< /math >}}.

**Proposition 3.2 (Finite Convergence Guarantee).** For the iterative sequence {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} defined in Algorithm 1. Denoting the sequence length of {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} as {{< math >}}$L${{< /math >}}, we have {{< math >}}$\boldsymbol{z}_{k}^{L} = \boldsymbol{z}_{k}${{< /math >}}.

This finite convergence occurs because the triangular structure allows information to propagate definitively through the sequence during the updates. Section B presents a formal proof. Proposition 3.2 thus provides a strict upper bound on the computational steps needed to reach the exact solution.

These propositions jointly establish the inherent efficiency of Jacobi decoding. Proposition 3.1 reveals that the error reduces with a superlinear rate. This indicates the convergence can be fast, as each iteration {{< math >}}$t &lt; L${{< /math >}} yields an increasingly substantial reduction in their respective errors (<a href="#ref-4">Dennis & Schnabel, 1996</a>). Note that though we assume a close initialization in the theoretical analysis, the convergence speed remains fast under various initializations, as shown in Section 4.3. On the other hand, Proposition 3.2 further presents the convergence to the exact solution {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} is guaranteed in at most {{< math >}}$L${{< /math >}} iterations. These results confirm the potential of applying Jacobi decoding for fast inference in discrete autoregressive normalizing flows.

**Table 1**: Comparison of sequential inference, uniform Jacobi decoding, and our approach. The subscript indicates the maximum deviation in three runs.

| Configuration Dataset | Configuration Method | Generation Speed Time (s) ↓ | Generation Speed Speed Up ↑ | Generation Quality FID ↓ | Generation Quality CLIP-IQA ↑ | Generation Quality BRISQUE ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| CIFAR-10 | Sequential | 9.56<sub>±0.42</sub> | 1.0× | 9.71<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 56.35<sub>±0.21</sub> |
| CIFAR-10 | UJD | 3.92<sub>±0.09</sub> | 2.4× | 10.19<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 56.79<sub>±0.20</sub> |
| CIFAR-10 | **Ours** | **2.63**<sub>±0.13</sub> | **3.6**× | 10.20<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 56.78<sub>±0.19</sub> |
| CIFAR-100 | Sequential | 9.57<sub>±0.27</sub> | 1.0× | 8.22<sub>±0.03</sub> | 0.35<sub>±0.00</sub> | 57.75<sub>±0.12</sub> |
| CIFAR-100 | UJD | 3.30<sub>±0.10</sub> | 2.9× | 8.26<sub>±0.10</sub> | 0.35<sub>±0.00</sub> | 57.76<sub>±0.06</sub> |
| CIFAR-100 | **Ours** | **2.04**<sub>±0.03</sub> | **4.7**× | 8.19<sub>±0.16</sub> | 0.35<sub>±0.00</sub> | 57.78<sub>±0.12</sub> |
| AFHQ | Sequential | 186.28 | 1.0× | 15.42 | 0.63 | 15.55 |
| AFHQ | UJD | 219.24 | 0.8× | 15.44 | 0.63 | 15.44 |
| AFHQ | **Ours** | **41.21** | **4.5**× | 15.44 | 0.63 | 15.56 |

### 3.5. Where to Use Jacobi Decoding

While the theoretical analysis shows promise for Jacobi iteration, its uniform application involves practical trade-offs. Concretely, in one iteration, we need to update all {{< math >}}$L${{< /math >}} elements of the sequence simultaneously, which trades off memory for the time required per iteration in the original sequential setting, where only one sequence element is updated. Moreover, the commonly used key-value (KV) cache for optimized attention operators in modern transformer architectures (<a href="#ref-18">Ott et al., 2019</a>) is not directly applicable to Jacobi decoding, as the decoded sub-variables are approximated and must be updated at each iteration. These limitations are particularly relevant where dependencies are strong, potentially making the uniform Jacobi decoding approach slower than optimized sequential decoding in such scenarios.

Motivated by this trade-off and the depthwise heterogeneity of sequential redundancy presented in Section 3.2, we exploit that the first ({{< math >}}$K${{< /math >}}-th) layer often exhibits stronger dependencies and propose our selective layer processing strategy. On modern transformer architectures with a KV cache, it uses standard sequential decoding for the dependency-heavy first layer, where the original sequential decoding works well. Subsequently, it switches to the parallel Jacobi iteration for the remaining layers, where higher redundancy is expected. It enables the benefits of parallelism while avoiding the high additional computational costs in the first layer due to stronger dependencies. This application aims to achieve more effective overall inference acceleration by strategically applying the Jacobi decoding method to layers with higher redundancy.

## 4. Experiments

In this section, we apply our approach to TarFlow, a state-of-the-art discrete autoregressive normalizing flow model (<a href="#ref-31">Zhai et al., 2025</a>).
Experiments are conducted on CIFAR-10 and CIFAR-100 (<a href="#ref-13">Krizhevsky, 2009</a>) using models trained from scratch, and AFHQ (<a href="#ref-3">Choi et al., 2020</a>) with a resolution of 256×256 using the released TarFlow checkpoint.

The compared baselines include the standard sequential inference, the default inference method for discrete autoregressive normalizing flows, and the uniform Jacobi decoding (UJD) method, in which the Jacobi decoding strategy is applied to all layers.
The default stopping threshold {{< math >}}$\tau${{< /math >}} for Jacobi iterations is set as {{< math >}}$0.5${{< /math >}}.
More experimental details, including network architectures and hyperparameters, are available in Section E.1.

Our main evaluation covers both computational efficiency and the quality of the generated samples. Generation speed is quantified by the average inference time per batch and the overall speedup relative to the sequential baseline, both measured on two L40S GPUs. For generative quality, we utilize the Fréchet Inception Distance (FID) (<a href="#ref-8">Heusel et al., 2017</a>), a widely adopted metric that measures the perceptual similarity between the distribution of generated images and the real data distribution. For perceptual quality, we report two more no-reference metrics: CLIP-IQA (<a href="#ref-29">Wang et al., 2023</a>), which assesses quality based on alignment with CLIP embeddings, and BRISQUE (<a href="#ref-16">Mittal et al., 2012</a>), a blind image quality assessor sensitive to common distortions.

For comprehensiveness, we additionally test on smaller-scale Masked Autoregressive Flows (MAFs) (<a href="#ref-19">Papamakarios et al., 2017</a>) on both image generation and Boltzmann distribution approximation tasks, as detailed in Section E.3. The results also confirm that our method achieves significant acceleration and further verify its generality.

### 4.1. Comparisons on Inference

We evaluate our method against the standard sequential baseline and the UJD method. As illustrated in Figure 3 and Section E, visual inspection of samples confirms that our method maintains high perceptual fidelity, producing outputs visually comparable to the original sequential generations. This qualitative observation is confirmed by quantitative analysis detailed in Table 1. Metrics such as FID, CLIP-IQA, and BRISQUE indicate that generative quality is largely preserved with both UJD and our method across all scenarios, with only small degradation relative to the sequential baseline.

However, while UJD demonstrates acceleration benefits on the smaller CIFAR-10 and CIFAR-100 datasets, it fails on the larger AFHQ dataset, resulting in inference speed slower than the sequential baseline. This poorer performance on AFHQ is likely attributable to higher per-iteration computational costs, combined with potentially stronger dependencies that negate parallel gains. Conversely, our method consistently achieves substantial speedups across all datasets by selectively applying Jacobi iterations.
Notably, our method achieves up to a 4.7-times acceleration compared to the sequential baseline, as shown in Table 1, highlighting its effectiveness. This confirms that the proposed selective approach is crucial for achieving effective and generalizable acceleration without sacrificing generation quality.

### 4.2. Verification of Analysis

{{< figgrid caption="**Figure 4**: Convergence dynamics of Jacobi decoding across the first two network layers. Full results for all layers are shown in A2. The plot shows the variation in the error (measured by the ℓ<sub>2</sub> norm of the difference between the current iterate and the sequential output) over iterations, demonstrating fast overall convergence and the notably slower convergence of the first layer." >}}
sejd/errs_afhq_layer0.png | 49 | **(a)** Layer 1
sejd/errs_afhq_layer1.png | 49 | **(b)** Layer 2
{{< /figgrid >}}


To experimentally validate our theoretical analysis and empirical observations, we analyze the convergence dynamics of the Jacobi iterations. Figure 4 plots the error measured as the {{< math >}}$\ell_2${{< /math >}} norm of the difference between the iterate {{< math >}}$\boldsymbol{z}_k^t${{< /math >}} and the ground truth {{< math >}}$\boldsymbol{z}_k${{< /math >}} from sequential inference during the iteration process across different layers on the AFHQ dataset. As a reference, we also present the error variation of the original sequential inference, where the un-inferred sub-variables are regarded as the input sub-variables {{< math >}}$\boldsymbol{z}^{t+1}${{< /math >}} according to the default implementation for calculation. The results clearly show a rapid decrease in error for the Jacobi process, often reaching nearly zero error in substantially fewer iterations than the theoretical worst-case bound {{< math >}}$L${{< /math >}}, providing empirical support for its fast-convergence properties. Furthermore, Figure 4 reveals distinct convergence behavior across layers. The error associated with the first layer decreases noticeably more slowly via Jacobi iterations than that of subsequent layers.
This directly validates our observation in Section 3.2 of stronger dependencies in the initial layer and empirically confirms the rationale behind the selective strategy, which applies parallel iterations to the faster-converging later layers.
These results confirm the layer-wise differences in dependencies, thereby verifying the effectiveness of our method.

### 4.3. Ablation Study

{{< figgrid caption="**Figure 5**: Ablation study on the stopping threshold τ: FID scores and inference times for our method across different τ values, illustrating the speed-quality trade-off." >}}
sejd/ablation_cifar10.png | 32 | **(a)** CIFAR-10
sejd/ablation_cifar100.png | 32 | **(b)** CIFAR-100
sejd/ablation_afhq.png | 32 | **(c)** AFHQ
{{< /figgrid >}}


**Influence of τ.**
To further understand the impact of the stopping threshold hyperparameter {{< math >}}$\tau${{< /math >}}, we perform an ablation study.
We vary the value of {{< math >}}$\tau${{< /math >}} and measure the resulting generative quality by FID and inference time.
The results, illustrating the trade-off between these two metrics, are presented in Figure 5. As expected, increasing the threshold {{< math >}}$\tau${{< /math >}} allows parallel iterations to terminate earlier, thereby significantly reducing the overall inference time.
However, allowing larger differences between consecutive iterates before stopping can lead to a less precise generation.
This is reflected in the FID scores, which tend to increase as {{< math >}}$\tau${{< /math >}} becomes larger.
Notably, the results show that for values {{< math >}}$\tau${{< /math >}} below 1.0, the increase in FID is relatively gradual, while the reduction in inference time remains substantial.
This supports that, with an appropriately chosen {{< math >}}$\tau${{< /math >}}, our method effectively increases generation speed with only a minor impact on generation quality.
{{< math >}}$\tau=0.5${{< /math >}} consistently provides a favorable balance, achieving considerable acceleration while maintaining generative quality close to the baseline.
Therefore, we adopt {{< math >}}$\tau=0.5${{< /math >}} as the default setting for all other experiments presented in this paper.

{{< figgrid caption="**Figure 6**: Ablation study on different initializations." >}}
sejd/initialization_comparison.png | 90
{{< /figgrid >}}


**Influence of initialization.**
We perform an ablation study with various initialization methods of {{< math >}}$\boldsymbol{z}_k^0${{< /math >}}, including zero initialization {{< math >}}$\boldsymbol{z}_k^0=\mathbf{0}${{< /math >}}, standard normal initialization {{< math >}}$\boldsymbol{z}_k^0 \sim \mathcal{N}(\mathbf{0}, I)${{< /math >}}, and initialization with the output of previous layer {{< math >}}$\boldsymbol{z}_k^0=\boldsymbol{z}_{k+1}${{< /math >}}. Experimental results in Figure 6 show that the acceleration performance remains similar across various initializations, supporting our analysis of superlinear convergence speed as general and insensitive to specific initialization methods.

## 5. Conclusion

In this paper, we first observed the dependency redundancy within the autoregressive normalizing flow model and its significant variation across different layers. Based on this observation, we propose selectively applying the parallel Jacobi decoding method to layers with high dependency redundancy to accelerate inference. Theoretical analysis demonstrated the superlinear convergence of the proposed approach and provided a worst-case guarantee on the total number of required iterations. Comprehensive experiments verified the correctness of the theoretical analysis and demonstrated that our method achieves significant inference acceleration across multiple scenarios, enhancing the practical value of normalizing flow models.

## A. Notation

To promote readability, we present a table of notations below.

**Table A1**: Table of notations.

| Notation | Description |
| --- | --- |
| {{< math >}}$\boldsymbol{x}${{< /math >}} | Vector of observed data. |
| {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} | Vector of {{< math >}}$k${{< /math >}}-th step random variable of a discrete normalizing flow. |
| {{< math >}}$\boldsymbol{z}_{k,l}${{< /math >}} | {{< math >}}$l${{< /math >}}-th sub-variable in the split of {{< math >}}$k${{< /math >}}-th step random variable. |
| {{< math >}}$\boldsymbol{z}_{k,l}^{t}${{< /math >}} | {{< math >}}$l${{< /math >}}-th sub-variable in the {{< math >}}$k${{< /math >}}-th step random variable at {{< math >}}$t${{< /math >}}-th decoding step. |
| {{< math >}}$\mathbf{J}_{f}${{< /math >}} | Jacobian matrix of function {{< math >}}$f${{< /math >}} |
| {{< math >}}$T${{< /math >}} | Number of decoding steps in Jacobi decoding. |
| {{< math >}}$L${{< /math >}} | Number of sub-variables in the split random variable. |
| {{< math >}}$K${{< /math >}} | Number of steps in a discrete normalizing flow. |

## B. Theoretical Proofs

We first formally redefine our Jacobi iteration map as the function {{< math >}}$F(\cdot)${{< /math >}}.

**Definition B.1** (Jacobi Iteration Map). Let {{< math >}}$\boldsymbol{z}_{k+1}${{< /math >}} be a given vector sequence of length {{< math >}}$L${{< /math >}}, and let {{< math >}}$s_{k}, g_{k}${{< /math >}} be given functions. The iteration map {{< math >}}$F${{< /math >}} is defined component-wise for {{< math >}}$\boldsymbol{z}${{< /math >}} as:

{{< math >}}
$$
F(\boldsymbol{z})_{l} =
\begin{cases} 
  \boldsymbol{z}_{k+1,1} &amp; l = 1 \\
  \boldsymbol{z}_{k+1,l} \odot \exp\bigl(-s_{k}(\boldsymbol{z}_{&lt;l})\bigr) + g_k(\boldsymbol{z}_{&lt;l}) &amp; l=2,\ldots,L
\end{cases} \tag{9}
$$
{{< /math >}}

where {{< math >}}$\boldsymbol{z}_{&lt;l} \coloneqq [z_{1}, \dots, z_{l-1}]^T${{< /math >}}. The iterative sequence is generated by {{< math >}}$\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t})${{< /math >}} with initial {{< math >}}$\boldsymbol{z}^{0}_{k}${{< /math >}}.

It is easy to observe that there exists a fixed point {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} for this iteration. For example, the output {{< math >}}$\boldsymbol{z}_{k}${{< /math >}} from the sequential decoding approach in (5) is a fixed point.
Moreover, as we use neural networks as parameterized functions {{< math >}}$s_{k}${{< /math >}} and {{< math >}}$g_{k}${{< /math >}}, the map {{< math >}}$F${{< /math >}} is continuously differentiable.
Under these observations, we have the iterative sequence {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} converges to {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} with at least a superlinear convergence rate starting from a close initial sequence, as shown in the following theorem.

**Proposition B.1** (Superlinear Convergence Rate). There exists {{< math >}}$\delta&gt;0${{< /math >}} such that if {{< math >}}$\left\|\boldsymbol{z}_{k}^{0} - \boldsymbol{z}_{k}^{\ast}\right\| &lt; \delta${{< /math >}}, the iterative sequence {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}${{< /math >}} converges to {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} with at least a superlinear convergence rate. This means that the error satisfies:

{{< math >}}
$$
\|\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}^{\ast}\| = o(\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\|) \quad \text{as } \boldsymbol{z}_{k}^{t} \to \boldsymbol{z}_{k}^{\ast}. \tag{10}
$$
{{< /math >}}

**Proof.** Denote {{< math >}}$\boldsymbol{e}^{t} = \boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}${{< /math >}} the approximation error at iteration {{< math >}}$t${{< /math >}}.

The structure of the iteration map {{< math >}}$F(\boldsymbol{z})${{< /math >}} (as detailed in prior definitions: {{< math >}}$F(\boldsymbol{z})_{1} = \boldsymbol{z}_{k+1,1}${{< /math >}} is constant, and {{< math >}}$F(\boldsymbol{z})_{l} = F(\boldsymbol{z}_{&lt;l})${{< /math >}} for {{< math >}}$l=2,\ldots,L${{< /math >}}) ensures that the Jacobian matrix {{< math >}}$\mathbf{J}_F(\boldsymbol{z})${{< /math >}} is strictly lower triangular for any {{< math >}}$\boldsymbol{z}${{< /math >}}.
This implies that {{< math >}}$\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})${{< /math >}} is strictly lower triangular. Consequently, all eigenvalues of {{< math >}}$\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})${{< /math >}} are zero. Therefore, the spectral radius of the Jacobian at the fixed point is

{{< math >}}
$$
\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) = 0. \tag{11}
$$
{{< /math >}}

As {{< math >}}$F${{< /math >}} is continuously differentiable, Taylor's theorem allows us to expand {{< math >}}$F(\boldsymbol{z}_{k}^{t})${{< /math >}} around {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}. For {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} sufficiently close to {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}:

{{< math >}}
$$
F(\boldsymbol{z}_{k}^{t}) = F(\boldsymbol{z}_{k}^{\ast}) + \mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})(\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}) + o(\left\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\right\|). \tag{12}
$$
{{< /math >}}

Using the iteration definition {{< math >}}$\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t})${{< /math >}} and the property at fixed-point {{< math >}}$\boldsymbol{z}_{k}^{\ast} = F(\boldsymbol{z}_{k}^{\ast})${{< /math >}}, we have:

{{< math >}}
$$
\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}^{\ast} = F(\boldsymbol{z}_{k}^{t}) - F(\boldsymbol{z}_{k}^{\ast}) = \mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})(\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}) + o(\left\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\right\|). \tag{13}
$$
{{< /math >}}

This yields an error propagation dynamic:

{{< math >}}
$$
\boldsymbol{e}^{t+1} = \mathbf{J}_F(\boldsymbol{z}_{k}^{\ast}) \boldsymbol{e}^{t} + o(\|\boldsymbol{e}^{t}\|). \tag{14}
$$
{{< /math >}}

Standard theorems on iterative methods (e.g., results related to Q-order of convergence as found in previous work (<a href="#ref-17">Ortega & Rheinboldt, 2000</a>), see discussion around Theorem 10.1.4 for Q-superlinear) state that if an iteration converges. Its error satisfies the relationship in (14), then the convergence is Q-superlinear if and only if {{< math >}}$\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) = 0${{< /math >}}.
Meanwhile, the condition {{< math >}}$\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) &lt; 1${{< /math >}} (which is satisfied here since {{< math >}}$\rho=0${{< /math >}}) ensures that {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} is a point of attraction, so for {{< math >}}$\boldsymbol{z}_{k}^{0}${{< /math >}} sufficiently close to {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}, the sequence {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}${{< /math >}} is guaranteed to converge at {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}.

Given that {{< math >}}$\rho(\mathbf{J}_F(\boldsymbol{z}_{k}^{\ast})) = 0${{< /math >}}, the cited convergence theory directly implies that the iteration is Q-superlinear. By definition, Q-superlinear convergence means that {{< math >}}$\left\|\boldsymbol{e}^{t+1}\right\| = o(\left\|\boldsymbol{e}^{t}\right\|)${{< /math >}} as {{< math >}}$\boldsymbol{e}^{t} \to \boldsymbol{0}${{< /math >}}. Therefore, we conclude that:

{{< math >}}
$$
\|\boldsymbol{z}_{k}^{t+1} - \boldsymbol{z}_{k}^{\ast}\| = o(\left\|\boldsymbol{z}_{k}^{t} - \boldsymbol{z}_{k}^{\ast}\right\|) \quad \text{as } \boldsymbol{z}_{k}^{t} \to \boldsymbol{z}_{k}^{*}. \tag{15}
$$
{{< /math >}}

This demonstrates at least a superlinear convergence rate.

**Proposition B.2** (Finite Convergence Guarantee). For iteration map {{< math >}}$F${{< /math >}} and iterative sequence {{< math >}}$\{\boldsymbol{z}_{k}^{t}\}_{t\ge 0}${{< /math >}} defined in Definition B.1, the iteration converges to the fixed point in at most {{< math >}}$L${{< /math >}} steps:

{{< math >}}
$$
\boldsymbol{z}_{k}^{t} = \boldsymbol{z}_{k}^{\ast} \quad \forall t\geq L. \tag{16}
$$
{{< /math >}}

**Proof.** The core property is that the {{< math >}}$l${{< /math >}}-th component of the output, {{< math >}}$(F(\boldsymbol{z}))_{l}${{< /math >}}, depends only on the first {{< math >}}$l-1${{< /math >}} components of the input {{< math >}}$\boldsymbol{z}${{< /math >}}, specifically {{< math >}}$\boldsymbol{z}_{&lt;l}${{< /math >}}. We also know that {{< math >}}$(F(\boldsymbol{z}_{k}^{0}))_{1} = \boldsymbol{z}_{k+1,1} = \boldsymbol{z}_{k,1}^{\ast}${{< /math >}}.

We will prove by induction on the iteration step {{< math >}}$t${{< /math >}} (from {{< math >}}$t=1${{< /math >}} to {{< math >}}$t=L${{< /math >}}) that the first {{< math >}}$t${{< /math >}} components of the iterate {{< math >}}$\boldsymbol{z}_{k}^{t}${{< /math >}} match those of the fixed point {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}. Let {{< math >}}$P(t)${{< /math >}} be the statement:

{{< math >}}
$$
P(t): \quad \boldsymbol{z}_{k,l}^{t} = \boldsymbol{z}_{k,l}^{\ast} \quad \forall{1 \le l \le t}. \tag{17}
$$
{{< /math >}}

It is easy to check that the statement {{< math >}}$P(1)${{< /math >}} holds.
By assuming that {{< math >}}$P(t)${{< /math >}} holds, that is, {{< math >}}$\boldsymbol{z}_{k,l}^{t} = \boldsymbol{z}_{k,l}^{\ast}${{< /math >}} for all {{< math >}}$1 \le l \le t${{< /math >}}, we want to show that {{< math >}}$P(t+1)${{< /math >}} holds, meaning {{< math >}}$\boldsymbol{z}_{k,l}^{t+1} = \boldsymbol{z}_{k,l}^{\ast}${{< /math >}} for all {{< math >}}$1 \le l \le t+1${{< /math >}}.

Since {{< math >}}$\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t})${{< /math >}}:

- **For {{< math >}}$1 \le l \le t${{< /math >}}:**
    The calculation of {{< math >}}$\boldsymbol{z}_{k,l}^{t+1} = (F(\boldsymbol{z}_{k}^{t}))_{l}${{< /math >}} depends only on {{< math >}}$\boldsymbol{z}_{k,&lt;l}^{t}${{< /math >}}. Given {{< math >}}$j &lt; l \leq t${{< /math >}}, the inductive hypothesis {{< math >}}$P(t)${{< /math >}} implies {{< math >}}$\boldsymbol{z}_{k,j}^{t} = \boldsymbol{z}_{k,j}^{\ast}${{< /math >}} for these components. Thus, {{< math >}}$\boldsymbol{z}_{k,&lt;l}^{t} = \boldsymbol{z}_{k,&lt;l}^{\ast}${{< /math >}}. Because {{< math >}}$(F(\cdot))_{l}${{< /math >}} only depends on these first {{< math >}}$l-1${{< /math >}} components, we have

    {{< math >}}
    $$
    (F(\boldsymbol{z}_{k}^{t}))_{l} = (F(\boldsymbol{z}_{k}^{\ast}))_{l}. \tag{18}
    $$
    {{< /math >}}

    Since {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} is a fixed point hence {{< math >}}$F(\boldsymbol{z}_{k}^{\ast}) = \boldsymbol{z}_{k}^{\ast}${{< /math >}}, we have

    {{< math >}}
    $$
    \boldsymbol{z}_{k,l}^{t+1} = (F(\boldsymbol{z}_{k}^{\ast}))_{l} = \boldsymbol{z}_{k,l}^{\ast}. \tag{19}
    $$
    {{< /math >}}

- **For {{< math >}}$l = t+1${{< /math >}}:**
    The calculation of {{< math >}}$\boldsymbol{z}_{k,t+1}^{t+1} = (F(\boldsymbol{z}_{k}^{t}))_{t+1}${{< /math >}} depends only on {{< math >}}$\boldsymbol{z}_{k,&lt;t+1}^{t}${{< /math >}}. The components in this sub-vector are {{< math >}}$\boldsymbol{z}_{k,j}^{t}${{< /math >}} for {{< math >}}$j = 1, \dots, t${{< /math >}}. By the inductive hypothesis {{< math >}}$P(t)${{< /math >}}, these are equal to the corresponding components of {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}}. Therefore, {{< math >}}$\boldsymbol{z}_{k,&lt;t+1}^{t} = \boldsymbol{z}_{k,&lt;t+1}^{\ast}${{< /math >}}. Because {{< math >}}$(F(\cdot))_{t+1}${{< /math >}} only depends on the first {{< math >}}$t${{< /math >}} components

    {{< math >}}
    $$
    (F(\boldsymbol{z}_{k}^{t}))_{t+1} = (F(\boldsymbol{z}_{k}^{\ast}))_{t+1} \tag{20}
    $$
    {{< /math >}}

    Using the fixed-point property:

    {{< math >}}
    $$
    \boldsymbol{z}_{k,t+1}^{t+1} = (F(\boldsymbol{z}_{k}^{\ast}))_{t+1} = \boldsymbol{z}_{k,t+1}^{\ast}. \tag{21}
    $$
    {{< /math >}}

    This indicates that the {{< math >}}$(t+1)${{< /math >}}-th component becomes correct at step {{< math >}}$t+1${{< /math >}}.

Combining (19) and (21), we show that {{< math >}}$\boldsymbol{z}_{k,l}^{t+1} = \boldsymbol{z}_{k,l}^{\ast},\forall{1 \le l \le t+1}${{< /math >}}. Thus, {{< math >}}$P(t+1)${{< /math >}} holds.

By mathematical induction, {{< math >}}$P(t)${{< /math >}} holds for all {{< math >}}$t=1, \dots, L${{< /math >}}. In particular, {{< math >}}$P(L)${{< /math >}} holds:

{{< math >}}
$$
\boldsymbol{z}_{k,l}^{L} = \boldsymbol{z}_{k,l}^{\ast} \quad \forall{1 \le l \le L}. \tag{22}
$$
{{< /math >}}

This implies the entire vector is guaranteed to match the fixed point after {{< math >}}$L${{< /math >}} steps:

{{< math >}}
$$
\boldsymbol{z}_{k}^{L} = \boldsymbol{z}_{k}^{\ast}. \tag{23}
$$
{{< /math >}}

Assume {{< math >}}$\boldsymbol{z}_{k}^{t} = \boldsymbol{z}_{k}^{\ast}${{< /math >}} for some {{< math >}}$t \ge L${{< /math >}}. Then in the next iteration:

{{< math >}}
$$
\boldsymbol{z}_{k}^{t+1} = F(\boldsymbol{z}_{k}^{t}) = F(\boldsymbol{z}_{k}^{\ast}) = \boldsymbol{z}_{k}^{\ast} \tag{24}
$$
{{< /math >}}

For the same reason, if the sequence reaches {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} at step {{< math >}}$L${{< /math >}}, it remains at {{< math >}}$\boldsymbol{z}_{k}^{\ast}${{< /math >}} for all subsequent steps. Therefore, it is shown that

{{< math >}}
$$
\boldsymbol{z}_{k}^{t} = \boldsymbol{z}_{k}^{\ast} \quad \forall{t \geq L}. \tag{25}
$$
{{< /math >}}

## C. Limitations and Future Work

While our method demonstrates promising improvements in inference acceleration for autoregressive normalizing flow models, it also highlights several open problems.
First, although sequential and depthwise redundancy is commonly observed in trained models, it remains unknown whether this redundancy persists to the same extent in partially trained or undertrained models. This uncertainty might affect model effectiveness when models underfit or are in the early stages of training.
Second, this paper focuses exclusively on inference-time acceleration. The potential for leveraging our principles to optimize the training process or guide neural architecture design has not been explored.
Future work could further investigate the nature of these observed sequential and depthwise redundancies.
Such insights might then be leveraged to guide model training strategies and inform architectural design, potentially leading to models that are inherently more efficient.

## D. Analysis on Memory Complexity

Both the original sequential inference and our method have {{< math >}}$O(L^2)${{< /math >}} memory complexity. The original TarFlow requires {{< math >}}$O(L^2)${{< /math >}} memory due to the attention mechanism operating on L-length sequences, with KV cache not adding to the asymptotic complexity. Similarly, our method maintains the same {{< math >}}$O(L^2)${{< /math >}} complexity since we perform attention operations on the entire L-length sequence at each step, without requiring shared attention matrices across different steps.

While the asymptotic complexity is identical, the actual memory usage differs in practice. On AFHQ (batch size 16), our method uses only 5.2GB of memory compared to 7.8GB for the baseline implementation with KV cache. This difference arises because the baseline stores additional K and V tensors to avoid redundant computations, whereas our approach achieves acceleration through parallel processing without this storage overhead. Thus, our method demonstrates better memory efficiency despite processing L inputs in parallel.

## E. Additional Experiments

### E.1. Experimental Details on TarFlow

**Model Details.** The network architectures for our main baseline models are adopted from the publicly available implementation of TarFlow (Note: <https://github.com/apple/ml-tarflow>). For experiments on the AFHQ dataset, we utilize the pre-trained checkpoint released by the TarFlow authors.
Due to computational resource constraints, training models on the ImageNet dataset according to the original TarFlow configurations was not feasible within the scope of this work.
For experiments on the CIFAR datasets, we largely follow the default settings provided by TarFlow, with a few adjustments.
These modifications are implemented to better suit our experimental objectives or to accommodate resource limitations.
Table A2 summarizes the key configurations for each dataset.

**Table A2**: Experimental configuration for each dataset.

|  | CIFAR-10 | CIFAR-100 | AFHQ |
| --- | --- | --- | --- |
| Resolution | 32×32 | 32×32 | 256×256 |
| Patch size P | 2 | 2 | 8 |
| Sequence length L | 256 | 256 | 1024 |
| Number of blocks K | 6 | 6 | 8 |
| Layers per block | 6 | 6 | 8 |
| Hidden dimension | 256 | 256 | 768 |
| Batch size | 256 | 256 | 256 |

**Evaluation Details.**
To estimate generation speed, we compute the average time per batch across 10 distinct runs.
For Fréchet Inception Distance (FID) estimation, we compute the distance between the original dataset and a generated dataset of the same size, following the standard FID definition.
To evaluate generation quality, we use metrics such as CLIP-IQA and BRISQUE, computing the average score for each metric over the set of generated samples, matching the original dataset size.
These approaches, which involve averaging and large sample sizes, ensure stable, representative results.
The evaluation methods employed, which involve averaging results across multiple batches and utilizing extensive datasets, can ensure representative results.
It is consistent with established practices in the literature (<a href="#ref-26">Song et al., 2021b</a>; <a href="#ref-28">Teng et al., 2025</a>).
For CIFAR datasets, we additionally report the maximum deviation in three runs.
The magnitude of this deviation is considerably smaller than the performance differentials observed between methods, thereby affirming the statistical significance of our comparative results and the validity of the reported enhancements.

**Inference Details.**
Table A3 reports the average number of iterations per layer during inference with our Selective Jacobi Decoding. Layer 1 uses standard sequential decoding (L-1 steps), while the remaining layers use Jacobi iteration. Notably, almost all Jacobi layers converge in very few iterations (typically 4–7), far below the worst-case bound of L, validating our theoretical analysis. The relatively higher iteration count at Layer 2 on CIFAR-10 is consistent with our observation of depthwise heterogeneity, in which layers closer to the first layer tend to exhibit stronger sequential dependencies.

**Table A3**: Average number of Jacobi iterations per layer (τ=0.5). Layer 1 uses sequential decoding; remaining layers use Jacobi iteration.

| Layer | CIFAR-10 | CIFAR-100 | AFHQ |
| --- | --- | --- | --- |
| 1 (Sequential) | 255 | 255 | 1023 |
| 2 (Jacobi) | 53.9 | 7.5 | 6.6 |
| 3 (Jacobi) | 4.9 | 4.7 | 6.0 |
| 4 (Jacobi) | 4.0 | 4.0 | 5.2 |
| 5 (Jacobi) | 3.0 | 4.0 | 5.0 |
| 6 (Jacobi) | 6.1 | 5.2 | 5.9 |
| 7 (Jacobi) | — | — | 4.5 |
| 8 (Jacobi) | — | — | 4.0 |

Table A4 presents the per-layer runtime breakdown for both sequential inference and our method. In sequential inference, each layer takes approximately the same time. Under SJD, Layer 1 (sequential) dominates the total cost, while each Jacobi layer completes in a fraction of the time. This confirms that the acceleration stems from replacing expensive sequential decoding with fast-converging Jacobi iterations on layers with high redundancy.

**Table A4**: Per-layer runtime breakdown comparing sequential inference and SJD. “Other” includes self-denoising, inter-GPU communication, and noise generation overhead, etc.

| Layer | Sequential Time (s) | Sequential % | SJD (Ours) Time (s) | SJD (Ours) % | Jacobi Iters |
| --- | --- | --- | --- | --- | --- |
| *CIFAR-10* |  |  |  |  |  |
| 1 (Seq) | 1.45 | 14.8% | 1.50 | 55.7% | 255 |
| 2 (Jacobi) | 1.45 | 14.8% | 0.69 | 25.7% | 53.9 |
| 3–6 (Jacobi) | 5.82 | 59.4% | 0.23 | 8.6% | 3.0–6.1 |
| Other | 1.08 | 11.0% | 0.27 | 10.0% | — |
| **Total** | **9.79** | **100%** | **2.69** | **100%** | **3.6×** |
| *CIFAR-100* |  |  |  |  |  |
| 1 (Seq) | 1.51 | 16.5% | 1.55 | 76.6% | 255 |
| 2 (Jacobi) | 1.50 | 16.4% | 0.10 | 4.7% | 7.5 |
| 3–6 (Jacobi) | 6.00 | 65.6% | 0.23 | 11.4% | 4.0–5.2 |
| Other | 0.15 | 1.6% | 0.15 | 7.2% | — |
| **Total** | **9.15** | **100%** | **2.02** | **100%** | **4.5×** |
| *AFHQ* |  |  |  |  |  |
| 1 (Seq) | 21.60 | 12.5% | 21.46 | 52.0% | 1023 |
| 2 (Jacobi) | 20.85 | 12.1% | 2.56 | 6.2% | 6.6 |
| 3–8 (Jacobi) | 125.10 | 72.4% | 11.85 | 28.7% | 4.0–6.0 |
| Other | 5.35 | 3.1% | 5.38 | 13.1% | — |
| **Total** | **172.90** | **100%** | **41.25** | **100%** | **4.2×** |

### E.2. Full Results of Fig. 1 and Fig. 4.

We provide full results of Fig. 1 and Fig. 4 in Fig. A1 and Fig. A2, respectively.

{{< figgrid caption="**Figure A1**: Cosine similarities and L2 distances between layer outputs from standard inference and inference with o = 1, o = 2, and o = 5 nearest preceding dependencies masked." >}}
sejd/obs_c100_1_dual_axis.png | 30 | **(a)** CIFAR-100, o = 1
sejd/obs_afhq_1_dual_axis.png | 30 | **(b)** AFHQ, o = 1
sejd/obs_c100_2_dual_axis.png | 30 | **(c)** CIFAR-100, o = 2
sejd/obs_afhq_2_dual_axis.png | 30 | **(d)** AFHQ, o = 2
sejd/obs_c100_5_dual_axis.png | 30 | **(e)** CIFAR-100, o = 5
sejd/obs_afhq_5_dual_axis.png | 30 | **(f)** AFHQ, o = 5
{{< /figgrid >}}


{{< figgrid caption="**Figure A2**: Convergence dynamics of Jacobi decoding across network layers. The plot shows the variation in the error (measured by the ℓ<sub>2</sub> norm of the difference between the current iterate and the sequential output) over iterations, demonstrating fast overall convergence and the notably slower convergence of the first layer." >}}
sejd/errs_afhq_layer0.png | 22 | **(a)** Layer 1
sejd/errs_afhq_layer1.png | 22 | **(b)** Layer 2
sejd/errs_afhq_layer2.png | 22 | **(c)** Layer 3
sejd/errs_afhq_layer3.png | 22 | **(d)** Layer 4
sejd/errs_afhq_layer4.png | 22 | **(e)** Layer 5
sejd/errs_afhq_layer5.png | 22 | **(f)** Layer 6
sejd/errs_afhq_layer6.png | 22 | **(g)** Layer 7
sejd/errs_afhq_layer7.png | 22 | **(h)** Layer 8
{{< /figgrid >}}


### E.3. Experiments on Masked Autoregressive Flow

In this section, we test Masked Autoregressive Flow (MAF) models on both image generation and Boltzmann distribution approximation tasks.
The implementation from `nflows` (Note: <https://github.com/bayesiains/nflows/blob/master/nflows/transforms/autoregressive.py>) is used.
Note that KV-cache does not apply to this MLP-based architecture.
Therefore, Jacobi decoding could accelerate across all layers, so we select all layers for Jacobi decoding rather than using it only on non-first layers, yielding even higher acceleration.

**Boltzmann Distribution Approximation.** We test on an 8-layer MAF trained via reverse KL to approximate the Boltzmann distribution for a high-temperature (T=3.0) disordered state of a 2D Ising model. The reverse KL divergence loss is reduced from an initial value of -1111 to a final value of -1129 over 3000 epochs. To evaluate the performance, we generated 100,000 samples and compared our method against the standard sequential method. The results demonstrate a significant acceleration with negligible impact on quality, as shown in Table A5.

**Table A5**: Comparison between Sequential inference and our method of MAF on Boltzmann distribution approximation task.

| Method | Inference Time (s) | Average Energy / Site | Average Absolute Magnetization |
| --- | --- | --- | --- |
| Sequential | 16.84 | 0.0005 | 0.0500 |
| Ours | 1.07 | -0.0003 | 0.0498 |

The near-zero energy and magnetization values are consistent with disordered-state physics, confirming that sample quality is maintained. Notably, we achieve a 15.7x speedup. This experiment provides strong evidence of the general applicability of our method.

**Image Generation.**
We conduct further experiments on an 8-layer Masked Autoregressive Flow (MAF) trained on binary MNIST.
Due to the model's limited expressive power, the generative quality is low. However, we still observe that the generations from our method and sequential inference have very similar image quality, as shown in Fig. A3.
To generate 100 images, our method takes only 15.24 seconds, while the original sequential method required 281.00 seconds. This is a significant 18.4x acceleration, providing strong evidence of our method's general applicability.

{{< figgrid caption="**Figure A3**: Visualization comparison on binary MNIST." >}}
sejd/generations.png | 98 | **(a)** Sequential
sejd/generations_jacobi.png | 98 | **(b)** Ours, 18.4 times acceleration
{{< /figgrid >}}


### E.4. Evaluation of Reconstruction Consistency

A core advantage of discrete autoregressive normalizing flows is their strictly invertible architecture, which allows for perfect reconstruction. To explicitly quantify any numerical deviation introduced by our parallel iterative approximation and to verify the preservation of the model's invertibility, we evaluate the reconstruction consistency. Specifically, we map real images {{< math >}}$\boldsymbol{x}${{< /math >}} from the original dataset to the exact latent variables {{< math >}}$\boldsymbol{z}${{< /math >}} using the standard sequential forward pass, and subsequently reconstruct them back to the pixel space {{< math >}}$\hat{\boldsymbol{x}}${{< /math >}} using our SJD.

Our method achieves exceptionally low Mean Squared Error (MSE) scores between the original inputs {{< math >}}$\boldsymbol{x}${{< /math >}} and the reconstructed outputs {{< math >}}$\hat{\boldsymbol{x}}${{< /math >}}: 0.00636 on CIFAR-10, 0.00313 on CIFAR-100, and 0.00122 on AFHQ (all evaluated with SJD, {{< math >}}$\tau=0.5${{< /math >}}). These near-zero numerical errors confirm that the deviation introduced by relaxing the strict sequential dependency is virtually negligible, and that the parallel iterations converge tightly to the exact sequential solutions. This quantitative precision is further corroborated by visual inspection. As illustrated in Fig. A4, Fig. A5, and Fig. A6, the reconstructed images are visually indistinguishable from the original ones, with no perceptible loss of detail. Both the quantitative and qualitative results strongly verify that our method successfully preserves the strict bijective consistency and high-fidelity generation quality inherent to flow-based models.

{{< figgrid caption="**Figure A4**: Reconstruction consistency on the CIFAR-10 test set. **Top row:** Original real images. **Bottom row:** Reconstructed images using our Selective Jacobi Decoding. The reconstructions are visually indistinguishable from the original inputs." >}}
sejd/cifar-10_orig_vs_recon_appro_0.5.png | 95
{{< /figgrid >}}


{{< figgrid caption="**Figure A5**: Reconstruction consistency on the CIFAR-100 test set. **Top row:** Original real images. **Bottom row:** Reconstructed images using our Selective Jacobi Decoding." >}}
sejd/cifar-100_orig_vs_recon_appro_0.5.png | 95
{{< /figgrid >}}


{{< figgrid caption="**Figure A6**: Reconstruction consistency on the AFHQ test set. **Top row:** Original real images. **Bottom row:** Reconstructed images using our Selective Jacobi Decoding." >}}
sejd/afhq_orig_vs_recon_appro_0.5.jpg | 95
{{< /figgrid >}}


### E.5. Comparative Analysis with GAN and Diffusion Models

To contextualize the practical utility of our proposed acceleration method, we provide a comparative analysis against representative Generative Adversarial Networks (GANs) and Diffusion Models on the CIFAR-10 dataset. For the GAN baseline, we trained FastGAN (<a href="#ref-32">Zhong et al., 2020</a>) from scratch using the official implementation. For the diffusion baseline, we evaluated DDIM (<a href="#ref-25">Song et al., 2021a</a>) using the publicly available `google/ddpm-cifar10-32` checkpoint at 20 inference steps to establish a comparable speed profile.

**Table A6**: Comparison of our method against FastGAN and DDIM on the CIFAR-10 dataset.

| **Method** | **Inference Time (s)** ↓ | **FID** ↓ |
| --- | --- | --- |
| Fast GAN | 3.41 | 9.67 |
| DDIM (20 steps) | 2.31 | 19.32 |
| Ours | 2.63 | 10.20 |

As detailed in Table A6, our method demonstrates a highly competitive balance between generation speed and quality. It achieves faster inference than FastGAN with only a marginal trade-off in FID. While DDIM at 20 steps is slightly faster, it incurs a severe penalty to generation quality. Compared with DDIM at 20 steps, our method is only marginally slower but achieves substantially better generation quality.

### E.6. More visualized results

We provide more visualized experimental results on CIFAR-10 and CIFAR-100 in Fig. A7 and Fig. A8.
All results consistently confirm the little impact of our method on generation quality.

{{< figgrid caption="**Figure A7**: Visualization comparison on CIFAR-10." >}}
sejd/cifar10base.png | 49 | **(a)** Sequential
sejd/cifar10sjd.png | 49 | **(b)** Ours, 3.6 times acceleration
{{< /figgrid >}}


{{< figgrid caption="**Figure A8**: Visualization comparison on CIFAR-100." >}}
sejd/cifar100base.png | 49 | **(a)** Sequential
sejd/cifar100sjd.png | 49 | **(b)** Ours, 4.7 times acceleration
{{< /figgrid >}}


## References

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

## About this page

This is a web transcription of the paper's full text, produced with **Claude Opus 5 + ultracode**. The prose follows the original word for word; equations, tables, figures and numbers are reproduced as published. **It is provided for reference only, with no guarantee of accuracy — the original PDF is authoritative.** Where this page and the PDF disagree, the PDF is correct.

- Original PDF: [https://openreview.net/pdf?id=xYATz9HpE7](https://openreview.net/pdf?id=xYATz9HpE7)
- Paper page: [Accelerating Inference of Discrete Autoregressive Normalizing Flows by Selective Jacobi Decoding]({{< relref "/publication/sejd" >}})
- Chinese translation: [通过选择性 Jacobi 解码加速离散自回归标准化流的推理]({{< relref "/publication/sejd-cn" >}})
- Code: [https://github.com/lan-qing/SJD](https://github.com/lan-qing/SJD)
