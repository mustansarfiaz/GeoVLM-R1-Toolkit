# GeoVLM-R1 <img src="images/logo.png" height="40">: Reinforcement Fine-Tuning for Improved Remote Sensing Reasoning
<p align="center">
    <img src="https://i.imgur.com/waxVImv.png" alt="Oryx Video-ChatGPT">
</p>

#### [Mustansar Fiaz](https://sites.google.com/view/mustansarfiaz/home), [Hiyam Debary](https://www.linkedin.com/in/hiyam-debary/), Paolo Fraccaro, [Danda Paudel](https://insait.ai/dr-danda-paudel/), [Luc Van Gool](https://insait.ai/prof-luc-van-gool/), [Fahad Khan](https://sites.google.com/view/fahadkhans/home), and [Salman Khan](https://salman-h-khan.github.io/)


#### **IBM Research, INSAIT, ETH Zürich, Mohamed bin Zayed University of AI, Linköping University, Australian National University**

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://mustansarfiaz.github.io/GeoVLM-R1/)
[![paper](https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg)](https://www.arxiv.org/abs/2509.25026)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Model-FF9900)](https://huggingface.co/mustansarfiaz/GeoVLM-R1)


---

## 📢 Latest Updates
- **2026**: We will open-source the code, model, dataset, and evaluation scripts, which are coming soon. 
- **2026**: Our model checkpoints will be released on HuggingFace.
- **Sep-30-2025**: GeoVLM-R1 paper is released [arxiv link](https://www.arxiv.org/abs/2509.25026). 🔥🔥
- **Sep-30-2025**: GeoVLM-R1 2025 project is live. 🔥🔥
---

## <img src="images/logo.png" height="40">Overview

Earth Observation (EO) tasks introduce unique challenges, spanning referred object detection, image/region captioning, change detection, grounding, and temporal analysis, that demand task-aware reasoning. We propose a novel post-training framework that incorporates task-aware rewards to enable effective adaptation of reasoning-based RL models to diverse EO tasks. This training strategy enhances reasoning capabilities for remote-sensing images, stabilizes optimization, and improves robustness. Extensive experiments across multiple EO benchmarks show consistent performance gains over state-of-the-art generic and specialized vision–language models

---

## Install

1. Clone this repository and navigate to GeoVLM-R1-Toolkit folder
```bash
git clone https://github.com/mustansarfiaz/GeoVLM-R1-Toolkit.git
cd GeoVLM-R1-Toolkit
```

2. Install Package
```Shell
conda create -n GeoVLM-R1-Toolkit python=3.10 -y
conda activate GeoVLM-R1-Toolkit
pip install --upgrade pip  # enable PEP 660 support
pip install -e .
```

3. Install additional packages for training cases
```
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt 
pip install qwen-vl-utils
pip install ninja
pip install flash-attn --no-build-isolation
pip install modelscope, math_verify, openai, json_repair
pip install --upgrade typing_extensions
pip install openai
pip install json_repair
pip install trl==0.17.0
pip install transformers==4.49.0
pip install math_verify
pip install Babel
```

---
## 🏆 Contributions

- **GeoVLM-R1: A specialized VLM for high-resolution remote sensing image Reasoning.** We propose GeoVLM-R1, a reinforcement learning framework that encourages VLM to enhance its reasoning capabilities with flexibility, scalability, and ease of experimentation in mind for diverse EO tasks.
 using their corresponding datasets. This results in a total of 318k instruction pairs for RS domain.
- **Reward Mechanism.** We have a sophisticated reward mechanism, enabling effective RL in EO reasoning contexts. To generate structurally coherent and semantically accurate reasoning outputs, we introduce format and task-aware accuracy rewards to better guide reasoning optimization.  

- **Evaluation Benchmark.** Our experimental results demonstrate the effectiveness of GeoVLM-R1 on multiple challenging EO tasks. Experimental results on 28 downstream benchmarks show that our method performs well compared to existing VLMs and achieves better performance, demonstrating its merits.

---
## 👁️💬 GeoVLM-R1: RL Training Paradigm

Illustration of the overall proposed training paradigm for GeoVLM-R1. The model is first initialized via supervised fine-tuning using diverse earth observation tasks. It is then successively optimized using GRPO-based reinforcement learning (RL) for each task. The GeoVLM-R1 processes queries and outputs a structured format that comprises an interpretable reasoning trace (<think> ... </think>) and a final prediction (<answer> ...</answer>). 

<p align="center">
  <img src="images/Fig1.png" alt="GeoVLM-R1: RL Training Paradigm">
</p>

---

## 🛰️ GeoVLM-R1: RL Policy Update Mechanism

Overall pipeline of GeoVLM-R1 policy update mechanism (left). During fine-tuning, the GRPO module generates multiple candidate responses. These responses are evaluated, and each is assigned a distinct reward equipped with our reward mechanism. In particular, our reward mechanism comprises (i) a format reward to enforce structural compliance and (ii) a task-aware accuracy reward to ensure accuracy compliance. We present a few examples showcasing GeoVLM-R1 using a unique task-aware accuracy reward function, resulting in better performance (right).

<p align="center">
  <img src="images/Fig2.png" alt="GeoVLM-R1: RL Policy Update Mechanism">
</p>

---

## 🔍 State-of-the-art Comparison across EO Tasks

Comparison of recent generic and specialized VLMs over diverse EO tasks. GeoVLM-R1 shows favorable improvements across classification, detection, and captioning tasks.

<p align="center">
  <img src="images/sota_comparison.png" alt="SOTA Comparison">
</p>


## 📊 Image Classification Task

GeoVLM-R1 illustrates a consistent improvement among zero-shot (ZS), multi-label BigEarthNet, 
and temporal classification datasets compared to other existing VLMs.

| Model          | AID (ZS) | UCMerced (ZS) | WHU-19 (ZS) | BigEarthNet | xBD Set 1 (Temporal) | FMoW (Temporal) |
|----------------|----------|---------------|-------------|-------------|-----------------------|-----------------|
| GPT-4o         | 74.73    | 88.76         | 91.14       | 49.00       | 67.95                 | 21.43           |
| InternVL-8B    | 60.40    | 58.23         | 79.30       | 19.73       | 51.44                 | 21.04           |
| Qwen2.5-VL-3B  | 58.27    | 60.86         | 78.21       | 24.75       | 51.44                 | 34.36           |
| GeoChat        | 72.03    | 84.43         | 80.09       | 20.35       | 53.32                 | 59.20           |
| EarthDial      | **88.76**| 92.42         | 96.21       | 73.03       | 96.37                 | 70.03           |
| **GeoVLM-R1**  | 88.46    | **97.81**     | **97.91**   | **80.91**   | **98.93**             | **76.93**       |

## 📊 Referred Object Detection, Region-Captioning, Grounding Description Tasks


<div style="overflow-x:auto;">
<table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 13px;">
  <caption style="caption-side: bottom; text-align: justify; color: gray; font-size: 14px;">
    GeoVLM-R1 illustrates a consistent performance gain across referred object detection, 
    region-captioning, and grounding description tasks.
  </caption>
  <thead>
    <tr>
      <th rowspan="3" style="border: 1px solid #ddd; padding: 6px;">Model</th>
      <th colspan="10" style="border: 1px solid #ddd; padding: 6px;">Referred Object Detection Task</th>
      <th colspan="6" style="border: 1px solid #ddd; padding: 6px;">Region-Captioning Task</th>
      <th colspan="5" style="border: 1px solid #ddd; padding: 6px;">Grounding Task</th>
    </tr>
    <tr>
      <th colspan="5" style="border: 1px solid #ddd; padding: 6px;">GeoChat-Instruct</th>
      <th colspan="5" style="border: 1px solid #ddd; padding: 6px;">NWPU VHR-10 (Zero-Shot)</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">GeoChat-Instruct</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">NWPU VHR-10 (Zero-Shot)</th>
      <th colspan="5" style="border: 1px solid #ddd; padding: 6px;">NWPU VHR-10 (Zero-Shot)</th>
    </tr>
    <tr>
      <th style="border: 1px solid #ddd; padding: 6px;">Small</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Med.</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Large</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Single</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Mult.</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Small</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Med.</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Large</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Single</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Mult.</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">@0.5</th>
      <th style="border: 1px solid #ddd; padding: 6px;">@0.25</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;">GPT-4o</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9.41</td>
      <td style="border: 1px solid #ddd; padding: 6px;">7.6</td>
      <td style="border: 1px solid #ddd; padding: 6px;">8.02</td>
      <td style="border: 1px solid #ddd; padding: 6px;">17.68</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.81</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9.63</td>
      <td style="border: 1px solid #ddd; padding: 6px;">0.7</td>
      <td style="border: 1px solid #ddd; padding: 6px;">6.1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">14.72</td>
      <td style="border: 1px solid #ddd; padding: 6px;">10.82</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9.41</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;">InternVL2-4B</td>
      <td style="border: 1px solid #ddd; padding: 6px;">6.3</td>
      <td style="border: 1px solid #ddd; padding: 6px;">24.37</td>
      <td style="border: 1px solid #ddd; padding: 6px;">37.38</td>
      <td style="border: 1px solid #ddd; padding: 6px;">24.96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.72</td>
      <td style="border: 1px solid #ddd; padding: 6px;">7.1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">12.68</td>
      <td style="border: 1px solid #ddd; padding: 6px;">25.48</td>
      <td style="border: 1px solid #ddd; padding: 6px;">22.96</td>
      <td style="border: 1px solid #ddd; padding: 6px;">8.1</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">10.6</td>
      <td style="border: 1px solid #ddd; padding: 6px;">29.87</td>
      <td style="border: 1px solid #ddd; padding: 6px;">30.67</td>
      <td style="border: 1px solid #ddd; padding: 6px;">29.09</td>
      <td style="border: 1px solid #ddd; padding: 6px;">21.92</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;">InternVL2-8B</td>
      <td style="border: 1px solid #ddd; padding: 6px;">7.20</td>
      <td style="border: 1px solid #ddd; padding: 6px;">23.76</td>
      <td style="border: 1px solid #ddd; padding: 6px;">31.99</td>
      <td style="border: 1px solid #ddd; padding: 6px;">25.77</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9.30</td>
      <td style="border: 1px solid #ddd; padding: 6px;">4.26</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.85</td>
      <td style="border: 1px solid #ddd; padding: 6px;">20.72</td>
      <td style="border: 1px solid #ddd; padding: 6px;">21.66</td>
      <td style="border: 1px solid #ddd; padding: 6px;">5.86</td>
      <td style="border: 1px solid #ddd; padding: 6px;">10.58</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9.06</td>
      <td style="border: 1px solid #ddd; padding: 6px;">8.5</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.88</td>
      <td style="border: 1px solid #ddd; padding: 6px;">9.63</td>
      <td style="border: 1px solid #ddd; padding: 6px;">7.7</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      <td style="border: 1px solid #ddd; padding: 6px;">-</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;">GeoChat</td>
      <td style="border: 1px solid #ddd; padding: 6px;">2.9</td>
      <td style="border: 1px solid #ddd; padding: 6px;">13.6</td>
      <td style="border: 1px solid #ddd; padding: 6px;">21.7</td>
      <td style="border: 1px solid #ddd; padding: 6px;">16</td>
      <td style="border: 1px solid #ddd; padding: 6px;">4.3</td>
      <td style="border: 1px solid #ddd; padding: 6px;">2.5</td>
      <td style="border: 1px solid #ddd; padding: 6px;">3.2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">14.7</td>
      <td style="border: 1px solid #ddd; padding: 6px;">13.23</td>
      <td style="border: 1px solid #ddd; padding: 6px;">1.9</td>
      <td style="border: 1px solid #ddd; padding: 6px;">72.77</td>
      <td style="border: 1px solid #ddd; padding: 6px;">72.74</td>
      <td style="border: 1px solid #ddd; padding: 6px;">61.9</td>
      <td style="border: 1px solid #ddd; padding: 6px;">62.02</td>
      <td style="border: 1px solid #ddd; padding: 6px;">62.02</td>
      <td style="border: 1px solid #ddd; padding: 6px;">53.31</td>
      <td style="border: 1px solid #ddd; padding: 6px;">2.2</td>
      <td style="border: 1px solid #ddd; padding: 6px;">15.27</td>
      <td style="border: 1px solid #ddd; padding: 6px;">21.46</td>
      <td style="border: 1px solid #ddd; padding: 6px;">20.74</td>
      <td style="border: 1px solid #ddd; padding: 6px;">21.38</td>
    </tr>
	    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;">EarthDial</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.43</td>
      <td style="border: 1px solid #ddd; padding: 6px;">31.76</td>
      <td style="border: 1px solid #ddd; padding: 6px;">39.07</td>
      <td style="border: 1px solid #ddd; padding: 6px;">34.29</td>
      <td style="border: 1px solid #ddd; padding: 6px;">13.41</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.66</td>
      <td style="border: 1px solid #ddd; padding: 6px;">14.21</td>
      <td style="border: 1px solid #ddd; padding: 6px;">23.12</td>
      <td style="border: 1px solid #ddd; padding: 6px;">25.37</td>
      <td style="border: 1px solid #ddd; padding: 6px;">8.9</td>
      <td style="border: 1px solid #ddd; padding: 6px;">73.38</td>
      <td style="border: 1px solid #ddd; padding: 6px;">73.34</td>
      <td style="border: 1px solid #ddd; padding: 6px;">62.72</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>72.14</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>72.14</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>60.01</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;">17.07</td>
      <td style="border: 1px solid #ddd; padding: 6px;">41.00</td>
      <td style="border: 1px solid #ddd; padding: 6px;">27.05</td>
      <td style="border: 1px solid #ddd; padding: 6px;">26.35</td>
      <td style="border: 1px solid #ddd; padding: 6px;">23.12</td>
    </tr>
    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>GeoVLM-R1</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>36.02</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>54.72</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>55.03</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>57.1</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>35.04</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>34.44</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>48.76</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>64.91</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>55.97</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>41.45</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>75.92</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>75.9</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>66.43</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;">72.10</td>
      <td style="border: 1px solid #ddd; padding: 6px;">72.10</td>
      <td style="border: 1px solid #ddd; padding: 6px;">55.49</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>38.74</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>61.45</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>31.31</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>30.08</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>26.10</b></td>
    </tr>
  </tbody>
</table>
</div>

## 📊 Change Detection (CD) and Image Captioning (IC) Tasks

<table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 13px;">
  <caption style="caption-side: bottom; text-align: justify; color: gray; font-size: 14px;">
    Comparison of GeoVLM-R1 over change detection (CD) and image captioning (IC) datasets.  
    Results indicate better capabilities of our method to generate captions compared to existing VLMs 
    for both temporal CD and image-captioning datasets. ZS means zero-shot evaluation.
  </caption>
  <thead>
    <tr>
      <th rowspan="2" style="border: 1px solid #ddd; padding: 6px;">Model</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">CD Dubai-CC</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">CD LEVIR-MCI</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">CD MUDS</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">CD SYSU (ZS)</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">IC NWPU-Captions</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">IC RSCID-Captions</th>
      <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">IC RSITMD-Captions (ZS)</th>
    </tr>
    <tr>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
      <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="border: 1px solid #ddd; padding: 6px;">GPT-4o</td>
      <td style="border: 1px solid #ddd; padding: 6px;">8.81</td>
      <td style="border: 1px solid #ddd; padding: 6px;">7.45</td>
      <td style="border: 1px solid #ddd; padding: 6px;">18.68</td>
      <td style="border: 1px solid #ddd; padding: 6px;">10.33</td>
      <td style="border: 1px solid #ddd; padding: 6px;">8.4</td>
      <td style="border: 1px solid #ddd; padding: 6px;">22.05</td>
      <td style="border: 1px solid #ddd; padding: 6px;">14.18</td>
      <td style="border: 1px solid #ddd; padding: 6px;">11.02</td>
      <td style="border: 1px solid #ddd; padding: 6px;">20.92</td>
      <td style="border: 1px solid #ddd; padding: 6px;">16.48</td>
      <td style="border: 1px solid #ddd; padding: 6px;">12.32</td>
      <td style="border: 1px solid #ddd; padding: 6px;"><b>17.49</b></td>
      <td style="border: 1px solid #ddd; padding: 6px;">19.43</td>
      <td style="border: 1px solid #ddd; padding: 6px;">14.86</td>
      <td style="border: 1px solid #ddd; padding: 6px;">28.16</td>
      <td style="border: 1px solid #ddd; padding: 6px;">20.53</td>
      <td style="border: 1px solid #ddd; padding: 6px;">15.59</td>
      <td style="border: 1px solid #ddd; padding: 6px;">26.03</td>
      <td style="border: 1px solid #ddd; padding: 6px;">18.31</td>
      <td style="border: 1px solid #ddd; padding: 6px;">14.22</td>
      <td style="border: 1px solid #ddd; padding: 6px;">24.83</td>
    </tr>
	 <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">InternVL2-4B</td>
        <td style="border: 1px solid #ddd; padding: 6px;">7.31</td>
        <td style="border: 1px solid #ddd; padding: 6px;">6.38</td>
        <td style="border: 1px solid #ddd; padding: 6px;">21.12</td>
        <td style="border: 1px solid #ddd; padding: 6px;">8.88</td>
        <td style="border: 1px solid #ddd; padding: 6px;">7.43</td>
        <td style="border: 1px solid #ddd; padding: 6px;">22.14</td>
        <td style="border: 1px solid #ddd; padding: 6px;">10.25</td>
        <td style="border: 1px solid #ddd; padding: 6px;">7.90</td>
        <td style="border: 1px solid #ddd; padding: 6px;">17.73</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.27</td>
        <td style="border: 1px solid #ddd; padding: 6px;">9.98</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.36</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">InternVL2-8B</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
		<td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">20.69</td>
        <td style="border: 1px solid #ddd; padding: 6px;">15.64</td>
        <td style="border: 1px solid #ddd; padding: 6px;">30.18</td>
        <td style="border: 1px solid #ddd; padding: 6px;">21.59</td>
        <td style="border: 1px solid #ddd; padding: 6px;">16.13</td>
        <td style="border: 1px solid #ddd; padding: 6px;">28.17</td>
        <td style="border: 1px solid #ddd; padding: 6px;">18.91</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.65</td>
        <td style="border: 1px solid #ddd; padding: 6px;">26.02</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">Qwen2.5-VL-3B</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.41</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.62</td>
        <td style="border: 1px solid #ddd; padding: 6px;">27.59</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.27</td>
        <td style="border: 1px solid #ddd; padding: 6px;">10.11</td>
        <td style="border: 1px solid #ddd; padding: 6px;">26.11</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.13</td>
        <td style="border: 1px solid #ddd; padding: 6px;">9.30</td>
        <td style="border: 1px solid #ddd; padding: 6px;">18.22</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.61</td>
        <td style="border: 1px solid #ddd; padding: 6px;">10.34</td>
        <td style="border: 1px solid #ddd; padding: 6px;">16.06</td>
        <td style="border: 1px solid #ddd; padding: 6px;">18.82</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.72</td>
        <td style="border: 1px solid #ddd; padding: 6px;">26.79</td>
        <td style="border: 1px solid #ddd; padding: 6px;">21.37</td>
        <td style="border: 1px solid #ddd; padding: 6px;">16.42</td>
        <td style="border: 1px solid #ddd; padding: 6px;">26.53</td>
        <td style="border: 1px solid #ddd; padding: 6px;">18.79</td>
        <td style="border: 1px solid #ddd; padding: 6px;">15.02</td>
        <td style="border: 1px solid #ddd; padding: 6px;">25.05</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">GeoChat</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.21</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.19</td>
        <td style="border: 1px solid #ddd; padding: 6px;">28.91</td>
        <td style="border: 1px solid #ddd; padding: 6px;">17.15</td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>35.42</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.35</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.28</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.23</td>
        <td style="border: 1px solid #ddd; padding: 6px;">15.98</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.45</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.02</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.96</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.86</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.54</td>
        <td style="border: 1px solid #ddd; padding: 6px;">15.21</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.48</td>
        <td style="border: 1px solid #ddd; padding: 6px;">11.59</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.39</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.41</td>
        <td style="border: 1px solid #ddd; padding: 6px;">11.50</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.33</td>
      </tr>
	<tr>
	  <td style="border: 1px solid #ddd; padding: 6px;">EarthDial</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">31.94</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">30.66</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">55.83</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">33.78</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">30.47</td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>74.80</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;">28.16</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">24.03</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">33.56</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">18.03</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">17.42</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">14.98</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">45.84</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">39.96</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">80.61</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">33.77</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">27.61</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">56.18</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">26.74</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">21.72</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">34.06</td>
	</tr>
	<tr>
	  <td style="border: 1px solid #ddd; padding: 6px;">GeoVLM-R1</td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>36.60</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>34.15</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>61.22</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>37.85</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;">34.02</td>
	  <td style="border: 1px solid #ddd; padding: 6px;">73.56</td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>34.07</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>27.65</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>45.94</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>19.64</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>18.46</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;">15.45</td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>46.94</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>40.96</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>82.00</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>34.64</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>28.63</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>56.54</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>30.62</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>25.39</b></td>
	  <td style="border: 1px solid #ddd; padding: 6px;"><b>39.07</b></td>
	</tr>
   </tbody>
</table>

## 📊 Temporal Damage Assessment Tasks

<table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 13px;">
  <caption style="caption-side: bottom; text-align: justify; color: gray; font-size: 14px;">
    GeoVLM-R1 comparison for various tasks on the xBD dataset for eight diverse tasks, such as temporal image captioning, region classification, image classification, object detection, and referred object detection. Our method exhibits substantial progress across the tasks. In particular, our approach shows a notable performance gain over object detection and referred object detection tasks, compared to other VLMs.
  </caption>
  <thead>
	<tr>
        <th rowspan="2" style="border: 1px solid #ddd; padding: 6px;">Model</th>
        <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">Image Captioning</th>
        <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">Region Classification</th>
        <th colspan="3" style="border: 1px solid #ddd; padding: 6px;">Image Classification</th>
        <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">Object Detection</th>
        <th colspan="2" style="border: 1px solid #ddd; padding: 6px;">Referred Object Detection</th>
      </tr>
      <tr>
        <th style="border: 1px solid #ddd; padding: 6px;">Rouge1</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Rouge-L</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Meteor</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Test Set-1</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Test Set-2</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Test Set-1</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Test Set-2</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Test Set-3</th>
        <th style="border: 1px solid #ddd; padding: 6px;">mAP@0.5</th>
        <th style="border: 1px solid #ddd; padding: 6px;">mAP@0.25</th>
        <th style="border: 1px solid #ddd; padding: 6px;">mAP@0.5</th>
        <th style="border: 1px solid #ddd; padding: 6px;">mAP@0.25</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">GPT-4o</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.21</td>
        <td style="border: 1px solid #ddd; padding: 6px;">10.35</td>
        <td style="border: 1px solid #ddd; padding: 6px;">19.52</td>
        <td style="border: 1px solid #ddd; padding: 6px;">51.68</td>
        <td style="border: 1px solid #ddd; padding: 6px;">71.62</td>
        <td style="border: 1px solid #ddd; padding: 6px;">67.95</td>
        <td style="border: 1px solid #ddd; padding: 6px;">75.45</td>
        <td style="border: 1px solid #ddd; padding: 6px;">70.41</td>
        <td style="border: 1px solid #ddd; padding: 6px;">0.2</td>
        <td style="border: 1px solid #ddd; padding: 6px;">2.15</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">InternVL2-8B</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.89</td>
        <td style="border: 1px solid #ddd; padding: 6px;">10.37</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.92</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.39</td>
        <td style="border: 1px solid #ddd; padding: 6px;">58.33</td>
        <td style="border: 1px solid #ddd; padding: 6px;">51.44</td>
        <td style="border: 1px solid #ddd; padding: 6px;">61.52</td>
        <td style="border: 1px solid #ddd; padding: 6px;">51.12</td>
        <td style="border: 1px solid #ddd; padding: 6px;">0.6</td>
        <td style="border: 1px solid #ddd; padding: 6px;">1.07</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">0.7</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">Qwen2.5-VL-3B</td>
        <td style="border: 1px solid #ddd; padding: 6px;">11.98</td>
        <td style="border: 1px solid #ddd; padding: 6px;">8.12</td>
        <td style="border: 1px solid #ddd; padding: 6px;">19.94</td>
        <td style="border: 1px solid #ddd; padding: 6px;">71.19</td>
        <td style="border: 1px solid #ddd; padding: 6px;">59.69</td>
        <td style="border: 1px solid #ddd; padding: 6px;">51.44</td>
        <td style="border: 1px solid #ddd; padding: 6px;">56.16</td>
        <td style="border: 1px solid #ddd; padding: 6px;">41.26</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
        <td style="border: 1px solid #ddd; padding: 6px;">-</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">GeoChat</td>
        <td style="border: 1px solid #ddd; padding: 6px;">14.18</td>
        <td style="border: 1px solid #ddd; padding: 6px;">10.67</td>
        <td style="border: 1px solid #ddd; padding: 6px;">12.20</td>
        <td style="border: 1px solid #ddd; padding: 6px;">25.30</td>
        <td style="border: 1px solid #ddd; padding: 6px;">57.65</td>
        <td style="border: 1px solid #ddd; padding: 6px;">53.32</td>
        <td style="border: 1px solid #ddd; padding: 6px;">52.19</td>
        <td style="border: 1px solid #ddd; padding: 6px;">49.51</td>
        <td style="border: 1px solid #ddd; padding: 6px;">1.15</td>
        <td style="border: 1px solid #ddd; padding: 6px;">7.2</td>
        <td style="border: 1px solid #ddd; padding: 6px;">0.2</td>
        <td style="border: 1px solid #ddd; padding: 6px;">3.09</td>
      </tr>
      <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">EarthDial</td>
        <td style="border: 1px solid #ddd; padding: 6px;">87.26</td>
        <td style="border: 1px solid #ddd; padding: 6px;">87.26</td>
        <td style="border: 1px solid #ddd; padding: 6px;">88.53</td>
        <td style="border: 1px solid #ddd; padding: 6px;">53.70</td>
        <td style="border: 1px solid #ddd; padding: 6px;">83.09</td>
        <td style="border: 1px solid #ddd; padding: 6px;">96.37</td>
        <td style="border: 1px solid #ddd; padding: 6px;">82.85</td>
        <td style="border: 1px solid #ddd; padding: 6px;">54.01</td>
        <td style="border: 1px solid #ddd; padding: 6px;">7.6</td>
        <td style="border: 1px solid #ddd; padding: 6px;">21.11</td>
        <td style="border: 1px solid #ddd; padding: 6px;">5.1</td>
        <td style="border: 1px solid #ddd; padding: 6px;">13.09</td>
      </tr>
      <tr style="font-weight: bold;">
        <td style="border: 1px solid #ddd; padding: 6px;">GeoVLM-R1</td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>92.26</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>92.26</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>93.37</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>81.36</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>83.55</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>98.93</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>86.39</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;">68.60</td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>38.15</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>48.13</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>24.52</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>34.52</b></td>
      </tr>
   </tbody>
</table>

## 📊 Visual Question Answer Task

<table style="width: 100%; border-collapse: collapse; text-align: center; font-size: 13px;">
  <caption style="caption-side: bottom; text-align: justify; color: gray; font-size: 14px;">
    GeoVLM-R1 performs better compared to existing VLMs for Comp and R/U categories over RSVQA-LRBEN (left) and obtains a better average score for RSVQA-HRBEN (right). Comp: Comparison, R/U: Rural/Urban.
  </caption>
  <thead>
	<tr>
        <th rowspan="2" style="border:1px solid #ddd; padding:6px;">Model</th>
        <th colspan="4" style="border:1px solid #ddd; padding:6px;">RSVQA-LRBEN</th>
        <th rowspan="2" style="border:1px solid #ddd; padding:6px;">Model</th>
        <th colspan="3" style="border:1px solid #ddd; padding:6px;">RSVQA-HRBEN (zero-shot)</th>
      </tr>
      <tr>
        <th style="border:1px solid #ddd; padding:6px;">Presence</th>
        <th style="border:1px solid #ddd; padding:6px;">Comp</th>
        <th style="border:1px solid #ddd; padding:6px;">R/U</th>
        <th style="border:1px solid #ddd; padding:6px;">Avg.</th>
        <th style="border:1px solid #ddd; padding:6px;">Presence</th>
        <th style="border:1px solid #ddd; padding:6px;">Comp</th>
        <th style="border:1px solid #ddd; padding:6px;">Avg.</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">MiniGPTv2</td>
        <td style="border:1px solid #ddd; padding:6px;">55.16</td>
        <td style="border:1px solid #ddd; padding:6px;">55.22</td>
        <td style="border:1px solid #ddd; padding:6px;">39.00</td>
        <td style="border:1px solid #ddd; padding:6px;">54.96</td>
        <td style="border:1px solid #ddd; padding:6px;">MiniGPTv2</td>
        <td style="border:1px solid #ddd; padding:6px;">40.79</td>
        <td style="border:1px solid #ddd; padding:6px;">50.91</td>
        <td style="border:1px solid #ddd; padding:6px;">46.46</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">Qwen2-VL</td>
        <td style="border:1px solid #ddd; padding:6px;">38.57</td>
        <td style="border:1px solid #ddd; padding:6px;">67.59</td>
        <td style="border:1px solid #ddd; padding:6px;">61.00</td>
        <td style="border:1px solid #ddd; padding:6px;">55.35</td>
        <td style="border:1px solid #ddd; padding:6px;">Qwen2-VL</td>
        <td style="border:1px solid #ddd; padding:6px;">66.44</td>
        <td style="border:1px solid #ddd; padding:6px;">60.41</td>
        <td style="border:1px solid #ddd; padding:6px;">63.06</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">InternVL2-8B</td>
        <td style="border:1px solid #ddd; padding:6px;">58.54</td>
        <td style="border:1px solid #ddd; padding:6px;">72.28</td>
        <td style="border:1px solid #ddd; padding:6px;">71.00</td>
        <td style="border:1px solid #ddd; padding:6px;">66.51</td>
        <td style="border:1px solid #ddd; padding:6px;">InternVL2-8B</td>
        <td style="border:1px solid #ddd; padding:6px;">67.35</td>
        <td style="border:1px solid #ddd; padding:6px;">76.91</td>
        <td style="border:1px solid #ddd; padding:6px;">72.70</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">Qwen2.5-VL-3B</td>
        <td style="border:1px solid #ddd; padding:6px;">59.59</td>
        <td style="border:1px solid #ddd; padding:6px;">75.04</td>
        <td style="border:1px solid #ddd; padding:6px;">63.00</td>
        <td style="border:1px solid #ddd; padding:6px;">68.40</td>
        <td style="border:1px solid #ddd; padding:6px;">Qwen2.5-VL-3B</td>
        <td style="border:1px solid #ddd; padding:6px;">59.89</td>
        <td style="border:1px solid #ddd; padding:6px;">72.26</td>
        <td style="border:1px solid #ddd; padding:6px;">66.81</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">GeoChat</td>
        <td style="border:1px solid #ddd; padding:6px;">91.09</td>
        <td style="border:1px solid #ddd; padding:6px;">90.33</td>
        <td style="border:1px solid #ddd; padding:6px;">94.00</td>
        <td style="border:1px solid #ddd; padding:6px;">90.70</td>
        <td style="border:1px solid #ddd; padding:6px;">GeoChat</td>
        <td style="border:1px solid #ddd; padding:6px;">58.45</td>
        <td style="border:1px solid #ddd; padding:6px;"><b>83.19</b></td>
        <td style="border:1px solid #ddd; padding:6px;">72.30</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">LHRS-Bot</td>
        <td style="border:1px solid #ddd; padding:6px;">88.51</td>
        <td style="border:1px solid #ddd; padding:6px;">90.00</td>
        <td style="border:1px solid #ddd; padding:6px;">89.07</td>
        <td style="border:1px solid #ddd; padding:6px;">89.19</td>
        <td style="border:1px solid #ddd; padding:6px;">EarthGPT</td>
        <td style="border:1px solid #ddd; padding:6px;">62.77</td>
        <td style="border:1px solid #ddd; padding:6px;">79.53</td>
        <td style="border:1px solid #ddd; padding:6px;">72.06</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">TeoChat</td>
        <td style="border:1px solid #ddd; padding:6px;">91.70</td>
        <td style="border:1px solid #ddd; padding:6px;">92.70</td>
        <td style="border:1px solid #ddd; padding:6px;">94.00</td>
        <td style="border:1px solid #ddd; padding:6px;">92.29</td>
        <td style="border:1px solid #ddd; padding:6px;">TeoChat</td>
        <td style="border:1px solid #ddd; padding:6px;">67.50</td>
        <td style="border:1px solid #ddd; padding:6px;">81.10</td>
        <td style="border:1px solid #ddd; padding:6px;">75.04</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px;">EarthDial</td>
        <td style="border:1px solid #ddd; padding:6px;"><b>92.58</b></td>
        <td style="border:1px solid #ddd; padding:6px;">92.75</td>
        <td style="border:1px solid #ddd; padding:6px;">94.00</td>
        <td style="border:1px solid #ddd; padding:6px;"><b>92.70</b></td>
        <td style="border:1px solid #ddd; padding:6px;">EarthDial</td>
        <td style="border:1px solid #ddd; padding:6px;">58.89</td>
        <td style="border:1px solid #ddd; padding:6px;">83.11</td>
        <td style="border:1px solid #ddd; padding:6px;">72.45</td>
      </tr>
      <tr>
        <td style="border:1px solid #ddd; padding:6px; font-weight:bold;">GeoVLM-R1</td>
        <td style="border:1px solid #ddd; padding:6px;">91.81</td>
        <td style="border:1px solid #ddd; padding:6px;"><b>93.20</b></td>
        <td style="border:1px solid #ddd; padding:6px;"><b>96</b></td>
        <td style="border:1px solid #ddd; padding:6px;">92.66</td>
        <td style="border:1px solid #ddd; padding:6px; font-weight:bold;">GeoVLM-R1</td>
        <td style="border:1px solid #ddd; padding:6px;"><b>66.38</b></td>
        <td style="border:1px solid #ddd; padding:6px;">82.26</td>
        <td style="border:1px solid #ddd; padding:6px;"><b>75.27</b></td>
      </tr>
</tbody>
</table>



---

## 📜 Citation
```bibtex
  @article{fiaz2025geovlmr1,
          title={GeoVLM-R1: Reinforcement Fine-Tuning for Improved Remote Sensing Reasoning}, 
          author={Mustansar Fiaz, Hiyam Debary, Paolo Fraccaro, Danda Paudel, Luc Van Gool, Fahad Shahbaz Khan, Salman Khan},
          journal={ArXiv},
          year={2025},
          url={https://arxiv.org/pdf/2509.25026}
        } 
```
## 🙏 Acknowledgement
We are thankful to [EarthDail](https://github.com/hiyamdebary/EarthDial), [Qwen2-VL Series](https://github.com/2U1/Qwen2-VL-Finetune), and [VLM-R1](https://github.com/om-ai-lab/VLM-R1) for releasing their models and code as open-source contributions.

---
[<img src="images/IBM_logo.png" width="80" height="80">](https://ibm.com/)
[<img src="images/insait.png" width="100" height="100">](https://insait.ai/)
[<img src="images/eth.png" width="200" height="100">](https://ethz.ch/en.html)
[<img src="images/mbzuai_logo.png" width="150" height="100">](https://mbzuai.ac.ae)
[<img src="images/linkoping.png" width="120" height="100">](https://liu.se/en)
[<img src="images/anu.png" width="120" height="100">](https://www.anu.edu.au/)

