---
title: "Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks"
subtitle: "Full text — web transcription"

summary: "Web transcription of the full text of Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks (SIGKDD 2026), with all equations, tables and figures."

date: '2026-08-09T00:00:00Z'
publishDate: '2024-05-26T00:00:00Z'

type: page
math: true

# 发布为论文页的下级路径。内容文件不能直接放进 content/publication/bfm/ ——
# 那是一个 leaf bundle，Hugo 不渲染其子目录里的 .md，所以用 url 覆盖输出路径。
url: '/publication/bfm/en/'

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
This page is a web transcription of the full text of **“Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks”** (SIGKDD 2026), reproduced here for easier reading, searching and linking.
It is provided **for reference only**; please read and cite the [original PDF](https://arxiv.org/pdf/2405.19931), which is authoritative. A [Chinese translation]({{< relref "/publication/bfm-cn" >}}) is also available.
{{% /callout %}}

{{< toc >}}

Xiaoyu Wu, Jiaru Zhang, Yang Hua, Bohan Lyu, Hao Wang, Tao Song, and Haibing Guan

> Xiaoyu Wu (wuxiaoyu2000@sjtu.edu.cn) and Jiaru Zhang (jiaruzhang@sjtu.edu.cn), Shanghai Jiao Tong University, Shanghai, China. Both authors contributed equally to this research.
> Yang Hua (Y.Hua@qub.ac.uk), Queen’s University Belfast, Belfast, United Kingdom.
> Bohan Lyu (lvbh22@mails.tsinghua.edu.cn), Tsinghua University, Beijing, China.
> Hao Wang (hwang9@stevens.edu), Department of Electrical and Computer Engineering, Stevens Institute of Technology, Hoboken, New Jersey, United States.
> Tao Song (songt333@sjtu.edu.cn), Shanghai Jiao Tong University, Shanghai, China. Corresponding author.
> Haibing Guan (hbguan@sjtu.edu.cn), Shanghai Jiao Tong University, Shanghai, China.

## Abstract

Few-shot fine-tuning of Diffusion Models (DMs) is a key advancement, significantly reducing training costs and enabling personalized AI applications. However, we explore the training dynamics of DMs and observe an unanticipated phenomenon: during the training process, image fidelity initially improves, then unexpectedly deteriorates with the emergence of noisy patterns, only to recover later with severe overfitting. We term the stage with generated noisy patterns as *corruption stage*. To understand this corruption stage, we begin by heuristically modeling the one-shot fine-tuning scenario, and then extend this modeling to more general cases. Through this modeling, we identify the primary cause of this corruption stage: a narrowed learning distribution inherent in the nature of few-shot fine-tuning. To tackle this, we apply Bayesian Neural Networks (BNNs) on DMs with variational inference to implicitly broaden the learned distribution, and present that the learning target of the BNNs can be naturally regarded as an expectation of the diffusion loss and a further regularization with the pretrained DMs. This approach is highly compatible with current few-shot fine-tuning methods in DMs and does not introduce any extra inference costs. Experimental results demonstrate that our method significantly mitigates corruption, and improves the fidelity, quality and diversity of the generated images in both object-driven and subject-driven generation tasks.

**CCS Concepts:** • Computing methodologies → Machine learning; • Computing methodologies → Bayesian networks; • Computing methodologies → Neural networks.

**Keywords:** Few-shot Fine-tuning; Diffusion Models; Bayesian Neural Networks

## 1. Introduction

Recent years have witnessed a surge in the development of Diffusion Models (DMs). These models have showcased extraordinary capabilities in various applications, such as image editing <a href="#ref-14">[14]</a> and video editing <a href="#ref-31">[31]</a>, among others. Particularly noteworthy is the advent of few-shot fine-tuning methods <a href="#ref-11">[11]</a>, <a href="#ref-22">[22]</a>, <a href="#ref-18">[18]</a>, in which a pretrained model is fine-tuned to personalize generation based on a small set of training images. These approaches have significantly reduced both memory and time costs in training. Moreover, these techniques offer powerful tools for adaptively generating images based on specific subjects or objects, embodying personalized AI and making AI accessible to everyone. In recent years, this innovation has even fostered the emergence of several communities, such as <civitai.com>, which boasts tens of thousands of checkpoints and millions of downloads.

{{< figgrid caption="**Figure 1**: Image fidelity variation during fine-tuning with and without BNNs. Zero training iteration indicates pretrained DMs. We fine-tune Stable Diffusion v1.5 with DreamBooth for 5 runs." >}}
bfm/baseline_motivation.png | 49 | **(a)** Few-shot fine-tuning process without BNNs.
bfm/bayes_motivation.png | 49 | **(b)** Few-shot fine-tuning process with BNNs.
{{< /figgrid >}}


**Figure 1**: Image fidelity variation during fine-tuning with and without BNNs. Zero training iteration indicates pretrained DMs. We fine-tune Stable Diffusion v1.5 with DreamBooth for 5 runs.

Despite the importance and widespread usage of few-shot fine-tuning methods in DMs, these methods often struggle or even fail when transferring from a large distribution (i.e., pretrained DMs' learned distribution) to a much smaller one (i.e., fine-tuned DMs' learned distribution) using limited data <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>.
We are the first to identify when and how these failures occur, and find that they are related to an unusual phenomenon:
As shown in Fig. 1, the similarity between the generated images and the training images initially increases during fine-tuning, but then unexpectedly decreases, before increasing once more.
Ultimately, the DMs are only capable of generating images identical to the training images.
Notably, on the stage with decreasing similarity, we observe there appear some unexpected noisy patterns on the generated images, therefore we name it as *corruption stage*.

To understand this corruption stage, we carry out further analysis on the few-shot fine-tuning process.
Specifically, we start from heuristic modeling a one-shot case, i.e.,
using only one training image during fine-tuning, and then extend the modeling to more general cases.
The modeling provides an estimation of the error scale remaining in the generated images.
We further show how corruption stages emerge due to
the limited learned distribution inherent in few-shot tasks with this modeling.

Based on the analysis above,
the solution to corruption should concentrate on expanding the learned distribution.
However, this expansion remains challenging in few-shot fine-tuning.
Traditional data augmentation methods, often face significant problems such as leakage <a href="#ref-13">[13]</a> and a reduction in generation quality <a href="#ref-6">[6]</a> when applied to generative models. Inspired by advancements in Bayesian Neural Networks (BNNs) <a href="#ref-3">[3]</a>, we propose to incorporate BNNs as a straightforward yet potent strategy to implicitly broaden the learned distribution.
We further present that its learning target can be decomposed into an expectation of the diffusion loss and an extra regularization term related to the pretrained model.
They can be adjusted to reach a trade-off between image fidelity and diversity.
Our method does not introduce any extra inference costs, and has good compatibility with existing few-shot fine-tuning methods in DMs, including DreamBooth <a href="#ref-22">[22]</a>, LoRA <a href="#ref-11">[11]</a>, and OFT <a href="#ref-18">[18]</a>.
Experiments demonstrate that our method significantly alleviates the corruption issues and substantially enhances the performance across various few-shot fine-tuning methods on diverse datasets under different metrics.

In summary, our main contributions are as follows:

- We observe an abnormal phenomenon during few-shot fine-tuning process on DMs: The image fidelity  first enhances, then unexpectedly worsens with the appearance of noisy patterns, before improving again but with severe overfitting. We refer to the phase where noisy patterns appear as the corruption stage.  We hope this observation could inform future research on DMs.
- We provide a heuristic modeling for few-shot fine-tuning process on DMs, explaining the emergence and disappearance of the corruption stage. With this modeling, we pinpoint that the main issue stems from the constrained learned distribution of DMs inherent in few-shot fine-tuning process.
- We innovatively incorporate BNNs to broaden the learned distribution, hence relieving such corruption. Experiments confirm its effectiveness in improving on different metrics, including text prompt fidelity, image fidelity, generation diversity, and image quality.

**Open-source and Full Version.** The code is available at GitHub (Note: <https://github.com/Nicholas0228/BNN-Finetuning-DMs>). We also maintain an extended version of this paper on arXiv <a href="#ref-29">[29]</a> with additional experiments and discussions.

## 2. Related Works

### 2.1. Diffusion Models and Few-shot Fine-tuning

Diffusion Models (DMs) <a href="#ref-10">[10]</a>, <a href="#ref-24">[24]</a>, <a href="#ref-25">[25]</a>, <a href="#ref-26">[26]</a> are generative models that approximate a data distribution through the gradual denoising of a variable initially sampled from a Gaussian distribution. These models involve a forward diffusion process and a backward denoising process. In the forward process, the extent of the addition of a noise {{< math >}}$\varepsilon${{< /math >}} increases over time {{< math >}}$t${{< /math >}}, as described by the equation {{< math >}}$x_t = \sqrt{\alpha_t}x_{0} + \sqrt{1-\alpha_t}\varepsilon${{< /math >}}, where {{< math >}}$x_{0}${{< /math >}} is a given original image and the range of time {{< math >}}$t${{< /math >}} is {{< math >}}$\left\{1, \dots, 1000\right\}${{< /math >}} in general cases. Conversely, in the backward process, the DMs aim to estimate the noise with a noise-prediction module {{< math >}}$\epsilon_{\theta}${{< /math >}} and subsequently remove it from the noisy image {{< math >}}$x_t${{< /math >}}. The discrepancy between the actual and predicted noise serves as the training loss, denoted as diffusion loss {{< math >}}$\mathcal{L}_{DM}:= \mathbb{E}_{\varepsilon\sim \mathcal{N}(0,1), t}\left||\epsilon_{\theta}(x_{t}, t) - \varepsilon|\right|_{2}^{2}.${{< /math >}}

Few-shot fine-tuning in DMs <a href="#ref-7">[7]</a>, <a href="#ref-11">[11]</a>, <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a> aims at personalizing DMs with a limited set of images, facilitating the creation of tailored content.
Gal *et. al.* introduced a technique that leverages new tokens within the embedding space of a frozen text-to-image model to capture the concepts presented in provided images <a href="#ref-7">[7]</a>.
Ruiz *et. al.* further proposed DreamBooth, which fine-tunes most parameters in DMs leveraging a reconstruction loss that captures more precise details of the input images, with a class-specific preservation loss ensuring the alignment with the textual prompts <a href="#ref-22">[22]</a>.
Additionally, Hu *et. al.* proposed LoRA, a lightweight fine-tuning approach that inserts low-rank layers to be learned while keeping other parameters frozen <a href="#ref-11">[11]</a>.
Qiu *et. al.* presented OFT, a method that employs orthogonal transformations to enhance the quality of generation <a href="#ref-18">[18]</a>. Although these methods generally succeed in capturing the details of training images, they suffer from the corruption stage observed in this paper.

{{< figgrid caption="**Figure 2**: Illustration of image fidelity variation in few-shot fine-tuning under different numbers of training images measured by Dino similarity. Higher Dino similarity indicates better image fidelity. As the number of training images increases, the corruption occurs later, and its severity is reduced." >}}
bfm/baseline_05.png | 32 | **(a)** Fine-tuning on 1 image.
bfm/baseline_0105.png | 32 | **(b)** Fine-tuning on 2 images.
bfm/baseline_allimages.png | 32 | **(c)** Fine-tuning on 6 images.
{{< /figgrid >}}


**Figure 2**: Illustration of image fidelity variation in few-shot fine-tuning under different numbers of training images measured by Dino similarity. Higher Dino similarity indicates better image fidelity. As the number of training images increases, the corruption occurs later, and its severity is reduced.

### 2.2. Bayesian Neural Networks

Bayesian Neural Networks (BNNs) are a type of stochastic neural networks characterized by treating the parameters as random variables rather than fixed values <a href="#ref-3">[3]</a>, <a href="#ref-4">[4]</a>, <a href="#ref-17">[17]</a>.
The objective is to infer the posterior distribution {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} for the parameters {{< math >}}$\theta${{< /math >}} given a dataset {{< math >}}$\mathcal{D}${{< /math >}}.
It endows BNNs with several distinct advantages, such as the capability to model the distributions for output, to mitigate overfitting, and to enhance model interpretability <a href="#ref-1">[1]</a>, <a href="#ref-12">[12]</a>.
One prevalent variant of BNNs is the mean-field variational BNN, also known as *Bayes by Backprop*, where the mean-field variational inference is applied to obtain the variational distribution {{< math >}}$Q_W(\theta)${{< /math >}} to approximate the posterior distribution {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} <a href="#ref-3">[3]</a>.
Recent studies demonstrated that even a BNN module, which treats only a subset of parameters as random variables while maintaining the rest as fixed, can retain the benefits associated with full BNNs <a href="#ref-8">[8]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-23">[23]</a>.
Our proposed method can be regarded as a natural progression of BNN principles applied to few-shot fine-tuning in DMs.

## 3. Corruption Stage in Few-shot Fine-tuning

In Sec. 3.1, we first present the observation on the corruption stage during few-shot fine-tuning of DMs. In Sec. 3.2, to better observe and understand the issues and fine-tuning dynamics with the corruption stage, we propose a heuristic modeling that uses Gaussian distribution as an approximation.

To address the challenges of modeling the dynamics of DMs during fine-tuning, we adopt reasonable simplifications, supported by evidence from a specific case. In Sec. 3.3, we explain the emergence and vanishing of the corruption stage by our modeling, and reveal the limited learned distribution is the root cause of the corruption stage.

### 3.1. Observation

In this section, we explore the performance variation during the few-shot fine-tuning process of DMs. Concretely, we fine-tune Stable Diffusion (SD) v1.5 (Note: <https://huggingface.co/runwayml/stable-diffusion-v1-5>) <a href="#ref-20">[20]</a> with DreamBooth <a href="#ref-22">[22]</a> on different number of training images, and record the average Dino similarity between the generated images and training images as a measure of the image fidelity <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>.

As shown in Fig. 2, the variation of the image fidelity is not monotonous during fine-tuning.

Specifically, the few-shot fine-tuning process can be approximately divided into the following phases:

1. In the first a few iterations, the image fidelity improves quickly.
2. Later, there is an **abnormal decrease** in the image fidelity.

    We observe the generated images in this phase present with increasing noisy patterns, i.e., the gradual emergence of the corruption stage.
3. Subsequently, the generation fidelity recovers, and we observe the corruption patterns on generated images progressively diminish, i.e. the gradual disappearance of the corruption stage. Once corruption completely diminishes in this phase, the model enters a state close to *overfitting*, where it can only generate images identical to the training images. As a result, it loses the ability to produce diverse images.

Through the comparisons between Fig. 2a, Fig. 2b and Fig. 2c, it is worth noting that when the number of training images increases, the onset of corruption is delayed and its severity is lessened.

### 3.2. Heuristic Modeling on Few-shot Fine-tuning

In this section, we begin by heuristically modeling one-shot fine-tuning scenarios and then extend it to more general cases. This modeling is supported by an evidence on a specific case.

**Heuristic Modeling for One-shot Fine-tuning on DMs.** We start with a representative condition where the dataset {{< math >}}$\mathcal{D}${{< /math >}} contains only one training image {{< math >}}$x'${{< /math >}}. Under this condition, we suppose the fine-tuned DMs with parameter {{< math >}}$\theta${{< /math >}} model the joint distribution of any original image {{< math >}}$x_0${{< /math >}} and any noisy image at time {{< math >}}$t${{< /math >}}, i.e., {{< math >}}$x_t${{< /math >}}, as a multivariate Gaussian distribution {{< math >}}$P_\theta(x_0, x_t)${{< /math >}}. Concretely, its marginal distribution of {{< math >}}$x_0${{< /math >}} is approximated as {{< math >}}$P_\theta(x_0) \approx \mathcal{N}(x', \sigma_1^2)${{< /math >}} when the model is fine-tuned with only one image {{< math >}}$x'${{< /math >}}.

Additionally, as the noisy image {{< math >}}$x_t${{< /math >}} is obtained by a linear combination {{< math >}}$x_t = \sqrt{\alpha_t} x_0 + \sqrt{1 - \alpha_t} \epsilon${{< /math >}} between {{< math >}}$x_0${{< /math >}} and a unit Gaussian noise {{< math >}}$\epsilon${{< /math >}}, the marginal distribution of {{< math >}}$x_t${{< /math >}} should approximate {{< math >}}$P_\theta(x_t) \approx \mathcal{N}(\sqrt{\alpha_t}x', \alpha_t \sigma_1^2 + (1 - \alpha_t))${{< /math >}}. Notably, the fine-tuning process in fact narrows the KL divergence between {{< math >}}$P_\theta(x_t \mid x_0=x')${{< /math >}} and {{< math >}}$\mathcal{N}(\sqrt{\alpha_t} x', (1 - \alpha_t))${{< /math >}}, thus these distributions should be increasingly close during fine-tuning <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>.

With this modeling, the main focus of the DMs, predicting the original image {{< math >}}$x_0${{< /math >}} based on a noisy image {{< math >}}$x_t${{< /math >}}, is represented as {{< math >}}$P_\theta(x_0|x_t)${{< /math >}}. We find that it in fact approximates a Gaussian distribution related to both {{< math >}}$x_{t}${{< /math >}} and the training image {{< math >}}$x'${{< /math >}}:

{{< math >}}
$$
\begin{aligned}
    P_\theta(x_0|x_t) \approx \mathcal{N}(x' + \delta_t(x_t, x'), \frac{(1-\alpha_t)}{\alpha_t \sigma_1^2 + (1 - \alpha_t)}), \\ \text{ where } \delta_t(x_t, x') =\frac{\sqrt{\alpha_t} \sigma_1^2}{\alpha_t \sigma_1^2 + (1 - \alpha_t)} (x_t - \sqrt{\alpha_t} x').
\end{aligned}
\tag{1}
$$
{{< /math >}}

The most possible {{< math >}}$x_0${{< /math >}} in the view of the DMs, {{< math >}}$\hat{x}_{0}${{< /math >}}, is:

{{< math >}}
$$
\hat{x}_{0} = \arg\max_{x_{0}}P_\theta(x_0|x_t) \approx x' + \delta_t(x_t, x').
\tag{2}
$$
{{< /math >}}

The derivation is provided at the Appendix Sec. A1. Notably, the error term {{< math >}}$\delta_t(x_t, x')${{< /math >}} represents the difference between the predicted original image {{< math >}}$\hat{x}_0${{< /math >}} and the training image {{< math >}}$x'${{< /math >}}. Intuitively, {{< math >}}$\sigma_{1}${{< /math >}} can be treated as the “confidence” of the fine-tuned DM regenerating the training sample {{< math >}}$x'${{< /math >}}.

The accuracy of the above modeling is influenced by the number of training iterations. As fine-tuning progresses, the approximation {{< math >}}$P_\theta(x_0) \approx \mathcal{N}(x', \sigma_1^2)${{< /math >}} becomes more precise, and the formulation more closely reflects real scenarios. To illustrate an extreme case in this modeling, we consider a scenario where {{< math >}}$\sigma_1 = 0${{< /math >}}, i.e., {{< math >}}$\delta_t = 0${{< /math >}}. Under this condition, for any input {{< math >}}$x_t${{< /math >}}, the DMs consistently reproduce the training image {{< math >}}$x'${{< /math >}} as described by Eq. (2). This indicates that the DMs fully lose the intrinsic denoising ability in this extreme scenario, only regenerating the training image instead.

In the opposite extreme, where {{< math >}}$\sigma_1 = +\infty${{< /math >}}, i.e., {{< math >}}$\delta_t = \frac{1}{\sqrt{\alpha_t}}x_t - x'${{< /math >}}, the model's prediction for {{< math >}}$x_0${{< /math >}} is exactly {{< math >}}$\frac{1}{\sqrt{\alpha_t}}x_t${{< /math >}}. This indicates that the DMs entirely lose the ability to generate images. Instead, DMs only rescale {{< math >}}$x_{t}${{< /math >}} based on the factor {{< math >}}$\alpha_t${{< /math >}} at the time step {{< math >}}$t${{< /math >}}, leading to any noise in {{< math >}}$x_{t}${{< /math >}} is also remaining in the generated image.

**Extension to More General Cases.** We further extend our modeling to the case where {{< math >}}$\mathcal{D}${{< /math >}} contains multiple training samples. In specific, we assume the learned distribution of the original image {{< math >}}$x_0${{< /math >}} of the DMs, i.e., {{< math >}}$P_\theta(x_0)${{< /math >}}, is centered with an image set {{< math >}}$\mathcal{I}_{\theta}${{< /math >}}. Under few-shot fine-tuning, as the training continues, {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} gradually approximates the training dataset {{< math >}}$\mathcal{D}${{< /math >}}. On the other hand, for the pretrained DMs, they typically learn a much larger manifold, which can be interpreted as learning with a sufficiently large {{< math >}}$\mathcal{I}_{\theta}${{< /math >}}. For all these DMs facing with noisy image {{< math >}}$x_{t}${{< /math >}}, we simplify their behavior as firstly finding a sample {{< math >}}$x^{*} \in \mathcal{I}_{\theta}${{< /math >}} to minimize the error term {{< math >}}$\delta_t(x_{t}, x^{*})${{< /math >}}, and then estimating the corresponding {{< math >}}$\hat{x}_0${{< /math >}} according to Eq. (2). With this simplification, the sufficiently large {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} of pretrained DMs enables them to find an {{< math >}}$x^{}${{< /math >}} such that the error term {{< math >}}$\delta_t(x_t, x^{})${{< /math >}} approaches zero, thereby preventing corruption in most instances.

{{< figgrid caption="**Figure 3**: Denoised images from pretrained and fine-tuned DMs using x<sub>t</sub> = 0 and t = 1000. The pretrained DMs do not largely change x<sub>t</sub> as it is free of noise. In contrast, both DMs fine-tuned on 1 and 5 images transform x<sub>t</sub> to make it closely resembles one sample within the training dataset 𝒟." >}}
bfm/zero_input_2.0.png | 100
{{< /figgrid >}}


**Support for the Modeling.** To support the above modeling closely approximates the practical scenarios, we present a specific example where we set the “noisy” image {{< math >}}$x_{t}=0${{< /math >}}, and then make both the pretrained and fine-tuned DMs denoise this image {{< math >}}$x_{t}${{< /math >}} which is entirely free of noise. This is a special case as {{< math >}}$x_{t}${{< /math >}} is free of noise and the typical DM should leave it unchanged to function effectively as a denoiser. According to our modeling, both DMs should first find one sample {{< math >}}$x^{*}${{< /math >}} within their own {{< math >}}$\mathcal{I}_{\theta}${{< /math >}}. Naturally, {{< math >}}$x_{t} = 0${{< /math >}} is within {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} of the pretrained DMs (See Appendix Sec. A5 for more evidence). Therefore, the pretrained DMs should leave this {{< math >}}$x_{t}=0${{< /math >}} almost unchanged during denoising.

In comparison, the {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} of the fine-tuned DMs should gradually approximate the training dataset {{< math >}}$\mathcal{D}${{< /math >}}. Therefore, the fine-tuned DMs should first find one sample {{< math >}}$x^{*} \in \mathcal{D}${{< /math >}}, and then predict the original image {{< math >}}$x_0${{< /math >}} as proportional to {{< math >}}$x^{*}${{< /math >}} according to Eq. (2).

Experimental results provided in Fig. 3 support our analysis, where we can observe the pretrained DMs do not largely change this {{< math >}}$x_{t} =0${{< /math >}}, but the fine-tuned DMs transform this {{< math >}}$x_{t}${{< /math >}} to one of the samples in its training dataset {{< math >}}$\mathcal{D}${{< /math >}}. This supports that our modeling aligns with real-world scenarios during few-shot fine-tuning.

### 3.3. Explanation of the Corruption Stage

{{< figgrid caption="**Figure 4**: Estimated σ<sub>1</sub> under different training iterations for DreamBooth." >}}
bfm/1.png | 60
{{< /figgrid >}}


In this section, we explain the corruption stage according to our heuristic modeling about few-shot fine-tuning process on DMs. We first present the reason for the emergence of the corruption stage, with an example showing the severity of such problem. We further demonstrate why the corruption stage gradually disappears, resulting in “overfitting” as the fine-tuning process continues.

**Emergence of the Corruption Stage.** As stated in Eq. (2), given any noisy image {{< math >}}$x_{t}${{< /math >}}, the fine-tuned DMs would predict the original image {{< math >}}$\hat{x}_0=x^{*} + \delta_{t}${{< /math >}}, where {{< math >}}$x^{*} \in  \mathcal{I}_{\theta} \approx\mathcal{D}${{< /math >}} after certain training iterations. The scale of the error term {{< math >}}$\delta_{t}${{< /math >}} is related to {{< math >}}$\left\|x_t - \sqrt{\alpha_t} x^{*}\right\|_{2}${{< /math >}} and {{< math >}}$\sigma_{1}${{< /math >}}. We estimate {{< math >}}$\sigma_{1}${{< /math >}} based on {{< math >}}$x_{t}${{< /math >}} sampled from {{< math >}}$\mathcal{N}(0, 1-\alpha_{t})${{< /math >}} and present the average {{< math >}}$\sigma_{1,t}${{< /math >}} among different {{< math >}}$t${{< /math >}}s in Fig. 4. It shows that the {{< math >}}$\sigma_{1}${{< /math >}} remains relatively high under moderate iterations, resulting in a significant {{< math >}}$\delta_{t}${{< /math >}} once {{< math >}}$x_{t}${{< /math >}} is not identical to {{< math >}}$\sqrt{\alpha_t} x^{*}${{< /math >}}.

{{< figgrid caption="**Figure 5**: Experimental results for pretrained and fine-tuned DMs with an additional noise δ′ in a small region of the noisy image x<sub>t</sub> at t = 100. The pretrained DMs effectively remove δ′, producing high-quality images. Conversely, the fine-tuned DMs fail to eliminate δ′, with output images showing corruption patterns." >}}
bfm/additional_noise_2.0.jpg | 100
{{< /figgrid >}}


Concretely, for the case with only one training image, i.e., {{< math >}}$\mathcal{D} = \left\{ x' \right\}${{< /math >}}, we set the noisy image {{< math >}}$x_{t} = \sqrt{\alpha_{t}}x' + \sqrt{1-\alpha_{t}}\varepsilon${{< /math >}} + {{< math >}}$\delta'${{< /math >}}, where {{< math >}}$\varepsilon\in\mathcal{N}(0,1)${{< /math >}} and {{< math >}}$\delta'${{< /math >}} is an additional noise introduced to a small region of the image. This {{< math >}}$\delta'${{< /math >}} simulates the case where the generating process of DMs is inaccurate in some {{< math >}}$t${{< /math >}}s. We further set the time variable {{< math >}}$t=100${{< /math >}}, and fine-tune DMs with 1000 iterations, with the estimated {{< math >}}$\sigma_{1}\approx4.8${{< /math >}} as shown in Fig. 4.
According to Eq. (2) and the analysis above, we can compute {{< math >}}$\left\|\delta_{100}\right\|_{2}^{2}\approx 2.65\left\|\delta'\right\|_{2}^{2}${{< /math >}}. It means the additional noise {{< math >}}$\delta'${{< /math >}} introduced is even expanded in this case, leading to a significant error term {{< math >}}$\delta_{t}${{< /math >}}. Fig. 5 shows the experimental results under this setting, where we observe a significant error term {{< math >}}$\delta_{t}${{< /math >}}, resembling a corruption pattern, and consisting with the above analysis.

**Vanishing of the Corruption Stage.** However, with the fine-tuning continues, {{< math >}}$\sigma_{1}${{< /math >}} drops as shown in Fig. 4, leading to a decreasing prediction error {{< math >}}$\delta_{t}${{< /math >}}. This indicates that the corruption stage vanishes, and the fine-tuned DMs gradually move to the state where they only strictly regenerate the training image {{< math >}}$x' \in \mathcal{D}${{< /math >}}. This is a classic case of “overfitting”, where the fine-tuned DMs lose their ability to generate diverse outputs and thus become unusable.

In conclusion, the analysis in this Sec. 3.3 shows how the corruption stage happens when the learned distribution of DMs is highly limited with small {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} and a high standard deviation {{< math >}}$\sigma_{1}${{< /math >}}.

## 4. Applying BNNs to Few-shot Fine-tuning

### 4.1. Motivation

Based on our analysis, the corruption stage primarily arises from a limited learned distribution with a small {{< math >}}$\mathcal{I}_{\theta}${{< /math >}}. Motivated by recent work on BNNs <a href="#ref-3">[3]</a>, <a href="#ref-8">[8]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-23">[23]</a>, which model the parameters {{< math >}}$\theta${{< /math >}} as random variables, we propose to apply BNNs in the few-shot fine-tuning process on DMs as a simple yet effective method to expand {{< math >}}$\mathcal{I}_\theta${{< /math >}}. Intuitively, the modeling of BNNs hinders the DMs to learn the exact distribution of the training dataset {{< math >}}$\mathcal{D}${{< /math >}}. Without BNNs, the model outputs images with high probability, focusing on high-confidence cases. However, DMs trained with BNNs inherently generate some lower-probability images, compelling the model to handle lower-confidence cases, thereby hindering the DMs from learning the exact distribution. Therefore, the DMs are encouraged to learn a larger and more robust distribution to counter the randomness.

Moreover, the sampling randomness during the fine-tuning process can be regarded as an inherent data augmentation, For instance, in a encoder-decoder view structure, BNNs in the encoder introduce perturbations within the encoding space. This acts as encoding-space level augmentation, ultimately influencing final decoded images by enhancing robustness and generalization without diminishing image quality, hence implicitly expanding the corresponding {{< math >}}$\mathcal{I}_\theta${{< /math >}}.

### 4.2. Formulation

**Modeling.** BNNs model the parameters {{< math >}}$\theta${{< /math >}} as random variables. Therefore, the learned distribution of DMs with BNN is {{< math >}}$P(x|\mathcal{D}) = \int P(x|\theta) P(\theta|\mathcal{D}) d\theta${{< /math >}}. Concretely, {{< math >}}$P(x|\theta)${{< /math >}} is the image distribution modeled by the DMs, and {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} is the posterior parameter distribution with a given dataset {{< math >}}$\mathcal{D}${{< /math >}}. As the posterior distribution {{< math >}}$P(\theta|\mathcal{D})${{< /math >}} is intractable, a variational distribution {{< math >}}$Q_W (\theta)${{< /math >}} is applied to approximate it. We model the variational distribution of each parameter {{< math >}}$\theta${{< /math >}} as a Gaussian distribution: {{< math >}}$\theta \sim \mathcal{N}(\mu_{\theta}, \sigma_{\theta}^2)${{< /math >}}, where {{< math >}}$W = \left\{\mu_{\theta}, \sigma_{\theta}\right\}${{< /math >}} are trainable parameters. Considering the fine-tuning process, we initialize the expectation term {{< math >}}$\mu_{\theta}${{< /math >}} from the corresponding parameter of the pretrained DMs, denoted as {{< math >}}$\theta_0${{< /math >}}. Following previous work <a href="#ref-3">[3]</a>, we apply the re-parameterization trick to obtain the gradients of the parameters, as detailed in Appendix Sec. A3.

**Table 1**: Performance of fine-tuning with BNNs under object-driven and subject-driven generation. The averages are reported here, and the standard deviations among 5 different seeds are provided in Table A6.

| Method (Object-Driven Generation: DreamBooth Dataset) | Clip-T ↑ | Dino ↑ | Clip-I ↑ | Lpips ↑ | Clip-IQA ↑ | Method (Subject-Driven Generation: CelebA Dataset) | Clip-T ↑ | Dino ↑ | Clip-I ↑ | Lpips ↑ | Clip-IQA ↑ |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamBooth | 0.246 | 0.614 | 0.771 | 0.611 | 0.875 | DreamBooth | 0.186 | 0.642 | 0.723 | 0.511 | 0.789 |
| DreamBooth w/ BNNs | **0.256** | **0.633** | **0.785** | **0.640** | **0.893** | DreamBooth w/ BNNs | **0.205** | **0.696** | **0.757** | **0.515** | **0.811** |
| LoRA | 0.252 | 0.542 | 0.722 | 0.650 | 0.864 | LoRA | 0.216 | 0.602 | 0.656 | 0.644 | 0.804 |
| LoRA w/ BNNs | **0.261** | **0.618** | **0.769** | **0.678** | **0.890** | LoRA w/ BNNs | **0.227** | **0.604** | **0.663** | **0.688** | **0.824** |
| OFT | 0.233 | 0.649 | 0.786 | 0.629 | 0.861 | OFT | 0.164 | 0.675 | 0.728 | 0.549 | 0.784 |
| OFT w/ BNNs | **0.242** | **0.661** | **0.791** | **0.646** | **0.884** | OFT w/ BNNs | **0.185** | **0.696** | **0.743** | **0.570** | **0.798** |

**Training.** During the fine-tuning process, the DMs are trained by minimizing the Kullback–Leibler (KL) divergence {{< math >}}$\mathrm{KL}(Q_W(\theta)|| P(\theta|\mathcal{D}))${{< /math >}}, which is equivalent to minimizing

{{< math >}}
$$
\begin{aligned}
    \mathcal{L} &= - \int Q_W(\theta)\log \frac{P(\theta, \mathcal{D})}{Q_W(\theta)}  d\theta    \\ 
        &=  \mathbb{E}_{\theta\sim Q_W(\theta)} \underbrace{ -\log P(\mathcal{D}|\theta)}_{\mathcal{L}_{DM}}+\underbrace{\mathrm{KL}(Q_W(\theta) ||P(\theta))}_{\mathcal{L}_r}.
\end{aligned}
\tag{3}
$$
{{< /math >}}

Following previous work <a href="#ref-33">[33]</a>, the above loss {{< math >}}$\mathcal{L}${{< /math >}} can be divided into two terms. In DMs, the first term can be seen as the modeled probability for the training dataset {{< math >}}$\mathcal{D}${{< /math >}}, and is equivalent to the expectation of the diffusion loss {{< math >}}$\mathcal{L}_{DM}${{< /math >}} shown in Sec. 2.1 on the parameters {{< math >}}$\theta${{< /math >}}. The second term can be seen as a regularization restricting the discrepancy between the variational distribution {{< math >}}$Q_W(\theta)${{< /math >}} and the prior distribution {{< math >}}$P(\theta)${{< /math >}}. We name it as the regularization loss {{< math >}}$\mathcal{L}_r${{< /math >}}. In few-shot fine-tuning, we regard the pretrained DMs naturally represent the prior information, so we set the prior distribution {{< math >}}$P(\theta)${{< /math >}} from the pretrained DMs, i.e., {{< math >}}$P(\theta) = \mathcal{N}(\theta_0, \sigma^2)${{< /math >}}, where {{< math >}}$\sigma${{< /math >}} is a hyperparameter which represents the parameter randomness.

In practice, we formulate our learning target as a linear combination of {{< math >}}$\mathcal{L}_{DM}${{< /math >}} and {{< math >}}$\mathcal{L}_r${{< /math >}} with a hyperparameter {{< math >}}$\lambda${{< /math >}}, i.e.,

{{< math >}}
$$
W^* = \arg \min\limits_{W} \mathbb{E}_{\theta\sim Q_W(\theta)} \mathcal{L}_{DM} + \lambda \mathcal{L}_r. \tag{4}
$$
{{< /math >}}

The training process is summarized in Appendix Alg. 1. Empirically, we find that using only {{< math >}}$\mathbb{E}_{\theta\sim Q_W(\theta)} \mathcal{L}_{DM}${{< /math >}}, i.e., setting {{< math >}}$\lambda${{< /math >}} as 0, is enough to improve few-shot fine-tuning. Nevertheless, we can reach a further trade-off between the generation diversity and image fidelity by adjusting {{< math >}}$\lambda${{< /math >}}.

**Inference.** During the inference, we explicitly replace each parameter {{< math >}}$\theta${{< /math >}} with its mean value {{< math >}}$\mu_\theta${{< /math >}} and perform inference just as DMs without BNNs. It guarantees that we do not introduce any additional costs compared to fine-tuned DMs without BNNs when deployed in production.

Motivated by previous approaches on BNN modules <a href="#ref-8">[8]</a>, <a href="#ref-15">[15]</a>, <a href="#ref-23">[23]</a>, we only model a subset of parameters as random variables in practice, which reduces the computational costs. Fine-tuning DMs with BNNs is compatible with existing few-shot fine-tuning methods, including DreamBooth <a href="#ref-22">[22]</a>, LoRA <a href="#ref-11">[11]</a>, and OFT <a href="#ref-18">[18]</a>. More details are presented in Appendix Sec. A4.

## 5. Experiments

{{< figgrid caption="**Figure 6**: Comparison of few-shot fine-tuning methods with and without BNNs across subject-driven and object-driven scenarios. We show both best-case and average-case generated images measured by Clip-I, Dino and Clip-IQA. See Appendix Sec. A9 and A12 for the selection criterion and more visualized results." >}}
bfm/visualization_0.jpg | 99
{{< /figgrid >}}


We apply BNNs to different few-shot fine-tuning methods across different tasks.
For object-driven generation, where the fine-tuned DMs synthesize images with the details of given objects, we use all the 30 classes from DreamBooth <a href="#ref-22">[22]</a> dataset, each containing 4-6 images.
For subject-driven generation, where the fine-tuned DMs synthesize images with the identities of given people, we follow previous research <a href="#ref-27">[27]</a>, randomly selecting 30 classes of images from CelebA-HQ <a href="#ref-16">[16]</a>, each containing 5 images.
Most training settings follow previous approaches <a href="#ref-11">[11]</a>, <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>, <a href="#ref-27">[27]</a>. All experiments are conducted with 5 different seeds by default and we report the average performance here.
We use Stable Diffusion v1.5 (Note: <https://huggingface.co/runwayml/stable-diffusion-v1-5>) (SD v1.5) as the default model for fine-tuning.

As for the BNNs, we set the default initialized standard deviation {{< math >}}$\sigma_{\theta}${{< /math >}} and prior standard deviation {{< math >}}$\sigma${{< /math >}} as 0.01.
The {{< math >}}$\lambda${{< /math >}} is set as 0 by default.
We show more details in Appendix Sec. A8.1.

Following previous approaches <a href="#ref-11">[11]</a>, <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>, for each class, we fine-tune one DM and generate 100 images with various prompts.
These generated images are used to measure the performance of different few-shot fine-tuning methods.
In specific, we use Clip-T <a href="#ref-19">[19]</a> to measure the text prompt fidelity, Clip-I <a href="#ref-9">[9]</a> and Dino <a href="#ref-5">[5]</a> to assess image fidelity, and Lpips <a href="#ref-34">[34]</a> to evaluate generation diversity. Additionally, we apply Clip-IQA <a href="#ref-28">[28]</a> to measure no-reference image quality.
We show more details about these metrics in Appendix Sec. A7.

### 5.1. Quantitative and Visualized Comparisons

We apply BNNs on different few-shot fine-tuning methods under both object-driven and subject-driven generation tasks.
As shown in Tab. 1 and Fig. 6, BNNs bring considerable improvements on all few-shot fine-tuning methods across text prompt fidelity (Clip-T) and image fidelity (Dino and Clip-I).
These improvements arise from the expanded learned distribution contributed by BNNs, which makes the DMs more capable to generate reasonable images about the learned objects/subjects based on different prompts. BNNs also largely enhance the no-reference image quality (Clip-IQA).
It is mainly because BNNs largely reduce the corruption phenomenon, which also partially improve the image fidelity (Dino and Clip-I) as the corrupted images are semantically distant to training images.
Additionally, we observe BNNs boost the generation diversity (Lpips).
We believe it naturally comes from the randomness introduced by BNNs.

**Table 2**: Performance under different DMs. All DMs are fine-tuned with DreamBooth on DreamBooth dataset.

|  |  | Clip-T ↑ | Dino ↑ | Clip-I ↑ | Lpips ↑ | Clip-IQA ↑ |
| --- | --- | --- | --- | --- | --- | --- |
| SD v1.5 | w/o BNNs | 0.246 | 0.614 | 0.771 | 0.611 | 0.875 |
| SD v1.5 | w/ BNNs | **0.256** | **0.633** | **0.785** | **0.640** | **0.893** |
| SD v1.4 | w/o BNNs | 0.248 | 0.594 | 0.762 | 0.618 | 0.872 |
| SD v1.4 | w/ BNNs | **0.249** | **0.620** | **0.777** | **0.656** | **0.895** |
| SD v2.0 | w/o BNNs | 0.240 | 0.563 | 0.739 | 0.604 | 0.875 |
| SD v2.0 | w/ BNNs | **0.248** | **0.610** | **0.764** | **0.649** | **0.925** |

### 5.2. Generalization

In this section, we further demonstrate BNNs can be applied to broader scenarios with notable performance improvement, including different DMs, varying training steps and a different number of training images.
By default, we experiment on DreamBooth with BNNs for fine-tuning.

**Different DMs.** Following previous work <a href="#ref-32">[32]</a>, we experiment on different DMs. Concretely, besides default SD v1.5, we also experiment on SD v1.4 and v2.0 <a href="#ref-20">[20]</a>. We provide training details in Appendix Sec. A8.2. Tab. 2 indicates that applying BNNs on different DMs of SD consistently improves few-shot fine-tuning across multiple metrics.

{{< figgrid caption="**Figure 7**: Comparison of performance with and without BNNs on DreamBooth with different training iterations per image." >}}
bfm/generalization_on_training_steps_dino.png | 49 | **(a)** Dino
bfm/generalization_on_training_steps_clipiqa.png | 49 | **(b)** Clip-IQA
{{< /figgrid >}}


**Training Iterations.** Fig. 7 shows that our method consistently improves the image quality (Clip-IQA). It also improves the image fidelity (Dino) when the training steps are larger than {{< math >}}$100\times \rm{Num}${{< /math >}}, where {{< math >}}$\rm{Num}${{< /math >}} represents the number of images utilized during fine-tuning. With fewer iterations, the model suffers from underfitting. In this case, BNNs may make the underfitting problem further severe as BNNs encourage the model to learn a larger distribution. This results in the slightly decreased image fidelity (Dino) observed in {{< math >}}$100\times \rm{Num}${{< /math >}}.

{{< figgrid caption="**Figure 8**: Comparison of the performance with different number of training images." >}}
bfm/generalization_on_training_images_dino.png | 49 | **(a)** Dino
bfm/generalization_on_training_images_clipiqa.png | 49 | **(b)** Clip-IQA
{{< /figgrid >}}


{{< figgrid caption="**Figure 9**: Ablation study on different λ and initialized σ<sub>θ</sub>." >}}
bfm/ablation_on_sigma_all.png | 49 | **(a)** Different Initialized σ<sub>θ</sub>
bfm/ablation_on_lambda_all.png | 49 | **(b)** Different λ
{{< /figgrid >}}


**Table 3**: User study results of fine-tuned DMs with and without BNNs across various measurements under both best-case and average-case scenarios. The table depicts the percentage of users favoring generated images from fine-tuned DMs with and without BNNs.

| Best-case Generation: Method | Best-case Generation: Subject Fidelity | Best-case Generation: Text Alignment | Best-case Generation: Image Quality | Average-case Generation: Method | Average-case Generation: Subject Fidelity | Average-case Generation: Text alignment | Average-case Generation: Image Quality |
| --- | --- | --- | --- | --- | --- | --- | --- |
| DreamBooth | 34.4% | 32.3% | 30.2% | DreamBooth | **50.5%** | 35.6% | 28.7% |
| DreamBooth w/ BNNs | **65.6%** | **67.7%** | **69.8%** | DreamBooth w/ BNNs | 49.5% | **64.4%** | **71.3%** |
| LoRA | 48.0% | 34.7% | 31.6% | LoRA | 27.5% | 26.5% | 24.5% |
| LoRA w/ BNNs | **52.0%** | **65.3%** | **68.4%** | LoRA w/ BNNs | **72.5%** | **73.5%** | **75.5%** |
| OFT | 30.6% | 34.7% | 40.8% | OFT | 41.1% | 26.8% | 39.3% |
| OFT w/ BNNs | **69.4%** | **65.3%** | **59.2%** | OFT w/ BNNs | **58.9%** | **73.2%** | **60.7%** |

**Table 4**: Comparison of performance when BNNs are applied to different layers in DMs. All experiments are conducted on DreamBooth dataset fine-tuning with DreamBooth. ‘N.A.’ refers to no BNN applied. ‘CA’ refers to BNN applied to cross-attention modules. ‘UB’ refers to only apply BNNs on the up block. We report the GPU memory costs and average time costs during fine-tuning for each class on one A100 GPU.

| Layer | Clip-T | Dino | Clip-I | Lpips | Clip-IQA | Memory | Time(s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| N.A. | 0.246 | 0.614 | 0.771 | 0.611 | 0.875 | 26.6G | 933 |
| Linear | 0.256 | 0.633 | 0.785 | 0.640 | 0.893 | 31.2G | 1107 |
| Linear (UB) | 0.259 | 0.614 | 0.776 | 0.646 | 0.890 | 29.0G | 1040 |
| Linear+CA | **0.272** | 0.611 | 0.760 | **0.672** | **0.897** | 32.5G | 1157 |
| LN+GN | 0.254 | **0.650** | **0.792** | 0.643 | 0.889 | 26.6G | 1088 |

**Numbers of Training Images**. We also experiment under different numbers of training images. Concretely, we use the CelebA-HQ <a href="#ref-16">[16]</a> dataset as it contains enough images per class. We randomly select five classes from the CelebA-HQ dataset and conduct experiments with different numbers of training images. We fix the training iterations to {{< math >}}$250\times\rm{Num}${{< /math >}}.

Fig. 8 illustrates that our method consistently improves image fidelity and quality under different numbers of training images. Such improvement is more obvious with more training images. This is primarily because the {{< math >}}$250\times\rm{Num}${{< /math >}} generally results in more severe corruption problems when the training image number {{< math >}}$\rm{Num}${{< /math >}} is larger, hence BNNs bring in much improvement by expanding the learning distribution and relieving corruptions.

### 5.3. Ablation Study

**Scale of Initialized {{< math >}}$\sigma_{\theta}${{< /math >}}.** The initialized standard deviation {{< math >}}$\sigma_{\theta}${{< /math >}} determines the extent of randomness during fine-tuning. We experiment with applying BNNs under varying initialized {{< math >}}$\sigma_{\theta}${{< /math >}}. Experimental results in Fig. 9a indicate that both the image fidelity (Dino) and quality (Clip-IQA) have been improved with a moderate initialized {{< math >}}$\sigma_{\theta}${{< /math >}}. However, when the initialized {{< math >}}$\sigma_{\theta}${{< /math >}} is too large, the DMs collapse and the performance rapidly decreases. This indicates the DMs are almost randomly updated because of the too large randomness introduced in this scenario.

**Trade-off Between Diversity and Fidelity with Adjusted {{< math >}}$\lambda${{< /math >}}.** Eq. (4) indicates {{< math >}}$\lambda${{< /math >}} controls the trade-off between the generation diversity and image fidelity.

As shown in Fig. 9b, an increasing {{< math >}}$\lambda${{< /math >}} leads to improving generation diversity (Lpips), albeit at the expense of image fidelity (Dino).

**Where to Apply BNNs.** As mentioned in Sec. 4.2, we could model only a subset of parameters as random variables, i.e., applying BNNs on a part of layers in DMs. By default, we apply BNNs to all linear layers in the U-Net <a href="#ref-21">[21]</a> except the ones in the cross-attention modules and explore how different choices influence the performance and the training costs.

As Tab. 4 shows, the DM can achieve relatively good performance with only the upblock of the U-Net applied with BNNs, reducing the ratio of the parameters modified to {{< math >}}$\sim 13.8\%${{< /math >}}. We can further decrease the training costs by only applying BNNs on the normalization layers, i.e., Layer Normalization (LN) <a href="#ref-2">[2]</a> and Group Normalization (GN) <a href="#ref-30">[30]</a> layers. This decreases the ratio of the parameters modified to {{< math >}}$\sim 0.02\%${{< /math >}} with relatively strong performance.

Additionally, when BNNs are applied to cross-attention modules, there is a significant increase in text prompt fidelity (Clip-T) at the expenses of image fidelity (Dino and Clip-I). Intuitively, this is because the input image aligns with only a limited set of prompts, and the applying of BNNs in cross-attention modules produces a further broader distribution matching more prompts.

### 5.4. User Study

**User Study Setting.** We conduct user study to comprehensively present the superiority of applying BNNs on few-shot fine-tuning DMs. Concretely, we follow previous approaches <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a> and conduct a structured human evaluation for generated images, involving 101 participants. For this purpose, we utilize the DreamBooth dataset and the CelebA-HQ dataset. Subsequently, we generate four images per subject or object using a random prompt selected from 25 prompts across five models trained with different seeds. For comprehensive comparisons, we select both the best and average cases from the generated images (see Appendix Sec. A9 for details).

Each participant is requested to compare image pairs generated by models that have been fine-tuned with and without BNNs, using three baseline models: DreamBooth, LoRA, and OFT. For each task, there are three binary-selection questions:

- **Subject fidelity:** Which of the given two images contains a subject or object that is most similar to the following reference image (one from the training dataset)?
- **Text alignment:** Which of the given two images best matches the text description provided below (the prompt used to generate the images)?
- **Image quality:** Which of the given two images exhibits the higher image quality?

**Results.** The results are presented in Tab. 3, which shows the percentage of participants favoring each method (with and without BNNs) based on the criteria described above. It is evident that the methods with BNNs are preferred in almost all scenarios for both best-case and average-case generations. This preference is particularly significant in terms of text alignment and overall image quality.

## 6. Conclusion

In this paper, we focus on few-shot fine-tuning in DMs, and reveal an unusual “corruption stage” where image fidelity first improves, then deteriorates due to noisy patterns, before recovering. With theoretical modeling, we attribute this phenomenon to the constrained learned distribution inherent in few-shot fine-tuning. By applying BNNs to broaden the learned distribution, we mitigate the corruption. Experiment results across various fine-tuning methods and datasets underscore the versatility of our approach.

## Acknowledgements

This work was supported by the National Key R&D Program of China (2022YFB4402102), and the Shanghai Key Laboratory of Scalable Computing and Systems. (Corresponding author: Tao Song)

## A1. Drivation of Eq. (2)

We first restate our assumptions formally. For convenience, we use equal signs instead of approximate signs in our assumptions and derivations.

- The joint distribution of {{< math >}}$x_0${{< /math >}} and {{< math >}}$x_t${{< /math >}} is modeled by the DM as a multivariate Gaussian distribution

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

where {{< math >}}$c${{< /math >}} represents the unknown covariance between {{< math >}}$x_0${{< /math >}} and {{< math >}}$x_t${{< /math >}}.

- The conditional probability of {{< math >}}$x_t${{< /math >}} given {{< math >}}$x_0 = x'${{< /math >}} is

{{< math >}}
$$
P_\theta(x_t \mid x_0=x') = \mathcal{N}(\sqrt{\alpha_t} x', (1 - \alpha_t)).
\tag{6}
$$
{{< /math >}}

Denote the inverse matrix of {{< math >}}$\boldsymbol{\Sigma}${{< /math >}} as

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

where {{< math >}}$|\boldsymbol{\Sigma}| = \sigma_1^2 (\alpha_t \sigma_1^2 + (1 - \alpha_t)) - c^2${{< /math >}} is the determinant of {{< math >}}$\boldsymbol{\Sigma}${{< /math >}}. According to the property of a joint Gaussian distribution, the conditional distribution {{< math >}}$P(x_t|x_0)${{< /math >}} can be represented as

{{< math >}}
$$
P(x_t|x_0) = \mathcal{N}(\frac{\lambda_{22} \sqrt{\alpha_t x'} + \lambda_{12}x_0 - \lambda_{12}x' }{\lambda_{22}}, \frac{1}{\lambda_{22}}).
\tag{8}
$$
{{< /math >}}

According to Eq. (6),

{{< math >}}
$$
\frac{1}{\lambda_{22}} = (1-\alpha_t),
\tag{9}
$$
{{< /math >}}

which means {{< math >}}$c = \pm \sqrt{\alpha_t} \sigma_1^2${{< /math >}}. Hence we know the joint distribution is

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

Repeatedly, according to the property of a joint Gaussian distribution, we have

{{< math >}}
$$
P(x_0 | x_t) = \mathcal{N}(x' \pm \frac{ \sqrt{\alpha_t} \sigma_1^2 (y - \sqrt{\alpha_t} x')}{\alpha_t \sigma_1^2 + (1 - \alpha_t)}, \frac{1 - \alpha_t}{\alpha_t \sigma_1^2 + 1 - \alpha_t}).
\tag{11}
$$
{{< /math >}}

In practice, the positive sign is more reasonable as it indicates the deviations of the predicted image {{< math >}}$x_0${{< /math >}} and the input noisy image {{< math >}}$x_t${{< /math >}} are aligned.

Therefore,

{{< math >}}
$$
P(x_0 | x_t) = \mathcal{N}(x' + \frac{ \sqrt{\alpha_t} \sigma_1^2 (y - \sqrt{\alpha_t} x')}{\alpha_t \sigma_1^2 + (1 - \alpha_t)}, \frac{1 - \alpha_t}{\alpha_t \sigma_1^2 + 1 - \alpha_t}).
\tag{12}
$$
{{< /math >}}

## A2. Proof for the Error term Approaches Zero during Fine-tuning

In this section, we provide a direct proof for the error term approaching zero when the diffusion loss approaches zero.

**Proof.** As previous work points out <a href="#ref-10">[10]</a>, the probability of a diffusion model on a given distribution is lower bounded by its ELBO:

{{< math >}}
$$
-\log P_{\theta}(x_{0}) \geq \mathbf{E}_{q}(x_{1:T} |x_0) \log \frac{ P_{\theta}(x_{0:T})}{Q(x_{1:T} |x_0)},
\tag{13}
$$
{{< /math >}}

and is simplified to diffusion loss:

{{< math >}}
$$
L_{DM}(x_0):=\mathbf{E}_{t, \epsilon\in N(0,1)} \Vert\varepsilon_{\theta} (\sqrt{\alpha_{t}}x_0 + \sqrt{1-\alpha_{t}}\epsilon, t) -\epsilon\Vert^{2}.
\tag{14}
$$
{{< /math >}}

The main target loss of fine-tuning methods is to directly minimize the diffusion loss for data within the fine-tuning data distribution. This can be interpreted as minimizing the KL divergence {{< math >}}$\mathbf{E}_{Q}(x'_{1:T} | x'_0) \log \frac{P_{\theta}(x'_{0:T})}{Q(x'_{1:T} | x'_0)}${{< /math >}}, where {{< math >}}$Q(x'_0)${{< /math >}} represents the fine-tuning data distribution.

Under optimal conditions where {{< math >}}$L_{DM}(x'_0) \to 0${{< /math >}}, we have {{< math >}}$P_{\theta}(x'_0) \to 1${{< /math >}} and {{< math >}}$P_{\theta}(x'_{t+1} | x'_{0}) \to Q(x'_{t+1} | x'_{0})${{< /math >}}. This implies the identity of the learned {{< math >}}$P${{< /math >}} and fine-tuned data distribution {{< math >}}$Q${{< /math >}}, hence we have

{{< math >}}
$$
\arg\max P_\theta(x_0|x_t) = \arg\max Q(x_0|x_t) = x'.
\tag{15}
$$
{{< /math >}}

It corresponds to {{< math >}}$\sigma_{1}=0${{< /math >}} in Eq. (1), leading to {{< math >}}$\delta_{t}=0${{< /math >}}.

It verifies the correctness of our modeling from another side, hence further supports our theoretical analysis.

## A3. Details of Applying BNNs

The training process of applying BNNs in fine-tuning is summarized in Alg. 1.
To obtain the gradients of the variational parameters, i.e., gradients of {{< math >}}$W = \{\mu_\theta, \sigma_\theta\}${{< /math >}}, we apply the commonly used reparameterization trick in BNNs.

Concretely, we first sample a unit Gaussian variable {{< math >}}$\varepsilon_\theta${{< /math >}} for each {{< math >}}$\theta${{< /math >}}, and then perform {{< math >}}$\theta = \mu_\theta + \sigma_\theta \times \varepsilon_\theta${{< /math >}} to obtain a posterior sample of {{< math >}}$\theta${{< /math >}}.
Hence, the gradients can be calculated by

{{< math >}}
$$
\begin{aligned}
\frac{\partial}{\partial W} \mathcal{L} &=
    \frac{\partial}{\partial W} \left[\mathbb{E}_{Q_W(\theta)}\mathcal{L}_{DM} + \mathcal{L}_r\right] \\ &= \mathbb{E}_{\varepsilon_\theta \sim \mathcal{N}(0,I)}\left[ \frac{\partial \mathcal{L}_{DM}}{\partial \theta} \frac{\partial \theta}{\partial W} + \frac{\partial \mathcal{L}_r}{\partial W}\right].
\end{aligned}
\tag{16}
$$
{{< /math >}}

We refer to the Proposition 1 in previous work <a href="#ref-3">[3]</a> for the detailed derivation.

**Algorithm 1**: Fine-tuning DMs with BNNs

```
Input:  Initialized variational parameters W = {μ_θ, σ_θ}, prior distributions
        P(θ) = N(θ_0, σ²), fine-tuning dataset D, number of fine-tuning
        iterations N, hyperparameter λ
Output: Fine-tuned variational parameters W = {μ_θ, σ_θ}

for i = 0 to N-1 do
    Sample ε_θ ~ N(0, I).
    Compute θ = μ_θ + ε_θ ∘ σ_θ.
    Sample x ∈ D, t ~ U(1, 1000), noise ε_t ~ N(0,1)
    Compute L_DM = ||ε_t - ε_θ(x_t, t)||².
    Compute L_r = KL(P(θ) || N(μ_θ, σ_θ²)).
    Compute L = L_DM + λ L_r
    Backward L and update μ_θ, σ_θ.
end for
```

## A4. Applying BNNs on Different Few-shot Fine-tuning Methods

**Applying BNNs on DreamBooth.** DreamBooth is a full-parameter fine-tuning method, and is one of the mainstream fine-tuning methods <a href="#ref-22">[22]</a>. Therefore, all parameters in DreamBooth can be modeled as the BNNs parameters.

**Applying BNNs on LoRA.** LoRA <a href="#ref-11">[11]</a> is a classic lightweight yet effective method for few-shot fine-tuning. In LoRA layers, the weight matrix {{< math >}}$\mathbf{W} \in \mathbb{R}^{d \times k}${{< /math >}} is modeled as a sum of fixed weight from the pretrained model and a trainable low-rank decomposition: {{< math >}}$\mathbf{W} = \mathbf{W}_0 + \mathbf{B} \mathbf{A}${{< /math >}}, where {{< math >}}$\mathbf{W}_0 \in \mathbb{R}^{d \times k}, \mathbf{B} \in \mathbb{R}^{d \times r}, \mathbf{A} \in \mathbb{R}^{r \times k}${{< /math >}} with rank {{< math >}}$r${{< /math >}} <a href="#ref-11">[11]</a>. In our implementation, we only convert the up matrix {{< math >}}$\mathbf{A}${{< /math >}} into random variables, and the down matrix {{< math >}}$\mathbf{B}${{< /math >}} is still kept as usual trainable parameters to make {{< math >}}$P(\mathbf{W})${{< /math >}} a Gaussian distribution. This design also reduces additional computational costs during training while keeps its effectiveness.

**Applying BNNs on OFT.** OFT is a few-shot fine-tuning method where the weights are tuned only by orthogonal transformations <a href="#ref-18">[18]</a>. In OFT layers, the weight matrix {{< math >}}$\mathbf{W} \in \mathbb{R}^{d \times k}${{< /math >}} is modeled as {{< math >}}$ \mathbf{W} = \mathbf{R} \mathbf{W}_0 ${{< /math >}}, where {{< math >}}$\mathbf{R}${{< /math >}} is guaranteed to be an orthogonal matrix by {{< math >}}$    \mathbf{R} = (\boldsymbol{I}+ 0.5 (\mathbf{Q} - \mathbf{Q}^T))(I-0.5 (\mathbf{Q} - \mathbf{Q}^T))^{-1}${{< /math >}}, and {{< math >}}$\mathbf{Q}${{< /math >}} is the trainable parameter in the original OFT method. To guarantee the orthogonality is not destroyed by random sampling in BNNs, we only convert the trainable parameter {{< math >}}$\mathbf{Q}${{< /math >}} into random variables.

Therefore, after the above transformation, {{< math >}}$\mathbf{R}${{< /math >}} is kept as an orthogonal matrix, and {{< math >}}$\mathbf{W}${{< /math >}} is kept as an orthogonalization transformation of the original pretrained {{< math >}}$\mathbf{W}_0${{< /math >}}.

**Table A5**: Prompts used for evaluation. [V] indicates the special token and [object] indicates the type of the object.

| Prompts for object-driven generation | Prompts for subject-driven generation |
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

## A5. Details of the Validation Experiments

**Proof for {{< math >}}$x_{t}=0${{< /math >}} is within the {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} of the pretrained DMs.** We provide two proofs that {{< math >}}$x_{t}=0${{< /math >}} is within the {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} of the pretrained DMs. The SD v1.5 is used as the pretrained DM.

- As shown in Fig. A10a, we use the prompt “a simple, solid gray image with no textures or variations” to generate images, and we observe the pretrained DM is capable of generating such images free of noise.
- We use the Img2Img Pipeline (Note: <https://github.com/huggingface/diffusers/blob/main/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py>) provided by diffusers with no prompt provided, i.e., unconditional generation. Then we set the input image as a blank one and the img2img strength as 0.1, which means input noisy image {{< math >}}$x_{100}=\alpha_{100}\varepsilon${{< /math >}}, where {{< math >}}$\varepsilon\in \mathcal{N}(0,1)${{< /math >}}. As shown in Fig. A10b. we can also observe the denoised result is completely free of noise.

Both results support our argument that {{< math >}}$x_{t}=0${{< /math >}} is naturally within {{< math >}}$\mathcal{I}_{\theta}${{< /math >}} of the pretrained DMs.

{{< figgrid caption="**Figure A10**: Proof for x<sub>t</sub> = 0 is within the ℐ<sub>θ</sub> of the pretrained DM." >}}
bfm/show_zero_2.jpg | 48 | **(a)** Generated Images based on a given prompt.
bfm/show_zero_1.png | 48 | **(b)** Img2img result for pure-color image as input.
{{< /figgrid >}}


We fine-tune an SD v1.5 with DreamBooth without PPL loss <a href="#ref-22">[22]</a>. The learning rate is fixed to {{< math >}}$5\times10^{-6}${{< /math >}} and only the U-Net <a href="#ref-21">[21]</a> is fine-tuned. We use the backpack class in DreamBooth dataset as an example, and use prompt “a [V] backpack” to train where the “[V]” is the special token. We set {{< math >}}$x_{100}=0${{< /math >}} and show the one-step denoised result using the Img2Img Pipeline provided by diffusers. During denoising, for both pretrained and fine-tuned DMs, the prompt is fixed to “a [V] backpack”. More results are shown in Fig. A11.

{{< figgrid caption="**Figure A11**: More results for generated images when input x<sub>t</sub> = 0 under different DMs and different inference steps." >}}
bfm/full_zero.jpg | 100
{{< /figgrid >}}


## A6. Additional Support for Our Modeling

To further support our modeling in Sec. 3.2, we present more results with {{< math >}}$x_t = kx'${{< /math >}} for different {{< math >}}$k${{< /math >}}s.
When giving the training prompt, according to our modeling in Eq. (2), a fine-tuned DM should predict the original image {{< math >}}$\hat{x_0} = (1 + \frac{\sqrt{\alpha_t} \sigma_1^2}{\alpha_t \sigma_1^2 + (1 - \alpha_t)} (k - \sqrt{\alpha_t}))x'${{< /math >}}, which is a scaling of {{< math >}}$x'${{< /math >}}.

In comparison, the generated result for the pretrained model should not exhibit a similar correlation with the given training sample {{< math >}}$kx'${{< /math >}}, as {{< math >}}$kx'${{< /math >}} is not part of the pretrained dataset or its training distribution {{< math >}}$\mathcal{I}_\theta${{< /math >}}.

Experimental results shown in Fig. A12 support our analysis, hence further confirming the rationality of our modeling.

{{< figgrid caption="**Figure A12**: Denoised images from pretrained and fine-tuned DMs using x<sub>t</sub> = kx′ for different ks." >}}
bfm/merged_with_labels.png | 100
{{< /figgrid >}}


## A7. Metrics

We use the following metrics to robustly measure different aspects of the fine-tuned DMs:

**Text prompt fidelity:** Following previous papers <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>, we use the average similarity between the clip <a href="#ref-19">[19]</a> embeddings of the text prompt and generated images, denoted as Clip-T.

**Image fidelity:** Following previous papers <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>, we compute the average similarity between the clip <a href="#ref-19">[19]</a> and dino <a href="#ref-5">[5]</a> embeddings of the generated images and training images, denoted as Clip-I <a href="#ref-9">[9]</a> and Dino.

**Generation Diversity:** We compute the average Lpips <a href="#ref-34">[34]</a> distance between the generated images of the fine-tuned DMs, following previous papers <a href="#ref-18">[18]</a>, <a href="#ref-22">[22]</a>.

**Image quality:** We find that corruption largely decreases the visual quality of the images, making them unusable. However, full-reference images quality measurement cannot fully represent this kind of degradation in image quality. Therefore, we add a no-reference image metric for measurements. We use Clip-IQA <a href="#ref-28">[28]</a> for measurements, which is one of the SOTA no-reference image quality measurements.

## A8. Settings of Fine-tuning

**Table A6**: Standard deviation of the results shown in Tab. 1.

| Method (Object-Driven Generation: DreamBooth Dataset) | Clip-T | Dino | Clip-I | Lpips | Clip-IQA | Method (Subject-Driven Generation: CelebA Dataset) | Clip-T | Dino | Clip-I | Lpips | Clip-IQA |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DreamBooth | 0.0017 | 0.0057 | 0.0037 | 0.0051 | 0.0028 | DreamBooth | 0.0065 | 0.0179 | 0.0115 | 0.0087 | 0.0078 |
| DreamBooth w/ BNNs | 0.0016 | 0.0056 | 0.0035 | 0.0054 | 0.0023 | DreamBooth w/ BNNs | 0.0036 | 0.0102 | 0.0090 | 0.0191 | 0.0095 |
| LoRA | 0.0018 | 0.0087 | 0.0045 | 0.0048 | 0.0084 | LoRA | 0.0034 | 0.0085 | 0.0059 | 0.0060 | 0.0068 |
| LoRA w/ BNNs | 0.0032 | 0.0104 | 0.0064 | 0.0072 | 0.0021 | LoRA w/ BNNs | 0.0025 | 0.0049 | 0.0163 | 0.0078 | 0.0065 |
| OFT | 0.0030 | 0.0063 | 0.0041 | 0.0081 | 0.0045 | OFT | 0.0026 | 0.0194 | 0.0141 | 0.0130 | 0.0115 |
| OFT w/ BNNs | 0.0010 | 0.0028 | 0.0026 | 0.0062 | 0.0068 | OFT w/ BNNs | 0.0024 | 0.0078 | 0.0066 | 0.0061 | 0.0094 |

We experiment on applying BNNs on different fine-tuning methods with one A100 GPU. All experiments are conducted under all 30 classes with 5 different seeds by default and we report the average performance. The standard deviation of our main result in Tab. 1 is shown in Tab. A6.

### A8.1. Few-shot Fine-tuning Hyper-parameters

The details of the parameters in the few-shot fine-tuning methods on our default model, i.e., SD v1.5, are presented below. We use {{< math >}}$\rm{Num}${{< /math >}} to represent the number of images utilized for training.

**Dreambooth**: We use the training script provided by Diffusers (Note: <https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth.py>). Only the U-Net is fine-tuned during the training process. By default, the number of training steps is set to {{< math >}}$200\times\rm{Num}${{< /math >}} on DreamBooth dataset and {{< math >}}$250\times\rm{Num}${{< /math >}} on CelebA, with a learning rate of {{< math >}}$5 \times 10^{-6}${{< /math >}}. The batch size is set to 1, and the number of class images used for computing the prior loss is {{< math >}}$200\times\rm{Num}${{< /math >}} by default. The prior loss weight remains fixed at 1.0. For the DreamBooth dataset, the training instance prompt is “a photo of a [V] {class prompt}”, where{class prompt} refers to the type of the image (such as dog, cat and so on). For the CelebA dataset, the training instance prompt is “a photo of a [V] person”.

**LoRA**: We use the training script provided by Diffusers (Note: <https://github.com/huggingface/diffusers/blob/main/examples/dreambooth/train_dreambooth_lora.py>). All default parameters remain consistent with the case in Dreambooth (No Prior), with the exception of the learning rate and training steps, which are adjusted to {{< math >}}$1 \times 10^{-4}${{< /math >}} and to {{< math >}}$400\times\rm{Num}${{< /math >}}, respectively.

**OFT**: We use the training script provided by the authors (Note: <https://github.com/Zeju1997/oft>). All default parameters remain consistent with the case in Dreambooth (No Prior), with the exception of the learning rate, which is adjusted to {{< math >}}$1 \times 10^{-4}${{< /math >}}.

For all experiments involving the applying of BNNs, we maintain the default settings for the number of learning steps, learning rate, training prompts, and other hyper-parameters.

For evaluation purposes, each training checkpoint generates four images per prompt, resulting in a total of 100 images from 25 different prompts. The prompts used for evaluation are displayed in Tab. A5, encompassing a broad set to thoroughly assess the variety and quality of the DMs.

### A8.2. Training Setting for DMs with Different Architectures

In this section, we provide training settings for applying BNNs on different DMs under DreamBooth. The learning rate is fixed to {{< math >}}$5 \times 10^{-6}${{< /math >}} in all cases. The numbers of training iterations are set as {{< math >}}$200 \times \mathrm{Num}${{< /math >}} and {{< math >}}$400 \times \mathrm{Num}${{< /math >}} for SD v1.4 and SD v2.0, respectively.

For BNNs applying on SD v1.4, we set the hyperparameter {{< math >}}$\lambda=0.1${{< /math >}}. All other hyperparameters are set as default.

## A9. Best-case vs. Average-case Generation

In practical scenarios, users typically generate multiple images and manually choose the most suitable one for use. Therefore, comparing the best cases essentially evaluates the quality of the chosen images, while the comparison of average cases assesses the difficulty of selecting an adequate image. Conversely, worst-case scenarios may hold less practical relevance, as users may regenerate images until achieving a satisfactory result, effectively bypassing such cases.

Therefore, we place emphasis on presenting best-case and average-case generation in this paper. Specifically, we first filter the generated images by selecting the top 90% using CLIP-T and Dino to ensure alignment with the prompt and the learned concept. Subsequently, we employ CLIP-IQA to identify both the top-quality and average-quality images. This approach provides a more comprehensive evaluation of the models' performance.

Nonetheless, we also include some comparisons of the worst-case generation results before and after applying BNNs illustrated in Fig. A13 for comprehensiveness. Concretely, we present the images with the lowest image qualities evaluated using CLIP-IQA.
Our experiments span 10 random classes in the DreamBooth Dataset, with DreamBooth as the baseline fine-tuning method.
The results show that DMs fine-tuned without BNNs usually present the corruption while BNNs successfully mitigate it, leading to better generation quality.

{{< figgrid caption="**Figure A13**: Visualization of worst-case generations measured by CLIP-IQA." >}}
bfm/stitched_image.jpg | 100
{{< /figgrid >}}


## A10. Limitations and Future Work

Even though few-shot fine-tuning DMs with BNNs applied has shown promising improvements, this paper also introduces a few interesting open problems.

Firstly, the extra randomness may make fine-tuning slower. This may lower the generation quality when the DMs are under-fitting.
In addition, the ability of learning extremely detailed patterns in the image may be reduced when the number of fine-tuning iterations is insufficient.
Future work could focus on these problems.

## A11. Broader Impact

This paper focuses on advancing few-shot fine-tuning techniques in DMs to provide more effective tools for creating personalized images in various contexts.
Previous few-shot fine-tuning methods have faced corruption phenomenon, as mentioned in this paper. Our approach, utilizing BNNs, addresses this phenomenon and provides generated images with higher quality.

However, there is potential for misuse, as malicious entities could exploit these technologies to deceive or misinform. Such challenges underscore the critical need for continuous exploration in this field. The development and ethical application of personalized generative models are not only paramount but also ripe for future research.

## A12. More Visualizations

We show more visualized results in Fig. A14 and Fig. A15.

{{< figgrid caption="**Figure A14**: More visualizations on subject-driven and object-driven scenarios." >}}
bfm/compress_visualization_1.jpg | 78
bfm/compress_visualization_2.jpg | 78
{{< /figgrid >}}


{{< figgrid caption="**Figure A15**: More visualizations on subject-driven and object-driven scenarios." >}}
bfm/compress_visualization_3.jpg | 80
{{< /figgrid >}}


## References

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

## About this page

This is a web transcription of the paper's full text, produced with **Claude Opus 5 + ultracode**. The prose follows the original word for word; equations, tables, figures and numbers are reproduced as published. **It is provided for reference only, with no guarantee of accuracy — the original PDF is authoritative.** Where this page and the PDF disagree, the PDF is correct.

- Original PDF: [https://arxiv.org/pdf/2405.19931](https://arxiv.org/pdf/2405.19931)
- Paper page: [Exploring Diffusion Models' Corruption Stage in Few-Shot Fine-tuning and Mitigating with Bayesian Neural Networks]({{< relref "/publication/bfm" >}})
- Chinese translation: [探究扩散模型少样本微调中的崩坏阶段并用贝叶斯神经网络加以缓解]({{< relref "/publication/bfm-cn" >}})
