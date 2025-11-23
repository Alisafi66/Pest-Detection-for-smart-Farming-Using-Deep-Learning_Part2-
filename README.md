### Project: Pest Detection for Smart Farming Using Deep Learning, Part 2

[![Project Part](https://img.shields.io/badge/Project-Part_2-orange)]()
[![Model](https://img.shields.io/badge/Model-AIMv2-black)](https://github.com/apple/ml-aim)
[![Sponsor](https://img.shields.io/badge/Sponsor-MDEC-purple)]()

## 📖 Project Context
This repository represents **Phase 2** of the *Pest Detection for Smart Farming using Deep Learning* project.

While **Phase 1 (Grounding DINO)** focused on high-precision pest identification, Phase 2 explores **AIMv2 (Autoregressive Image Models)** to enhance the robotic system's safety and generalized visual understanding. Given that autonomous farming robots operate in dynamic environments, distinguishing between a "target pest" and a "human" is critical for operational safety.

**Key Outcome:**
Although AIMv2 achieved lower raw pest-detection accuracy compared to Grounding DINO, it demonstrated **11% higher accuracy in Human Detection**. This makes it the superior module for the robot's navigation and obstacle avoidance systems.

## 🧠 About the Model: AIMv2
We utilized **AIMv2**, a family of vision models pre-trained with a multimodal autoregressive objective. Unlike standard object detectors, AIMv2 is designed to scale effectively and exhibits strong zero-shot capabilities.

* **Architecture:** We leveraged the **AIMv2-3B** checkpoint, known for achieving high recognition performance on ImageNet using a frozen trunk.
* **Capabilities:** AIMv2 outperforms OAI CLIP and SigLIP on the majority of multimodal understanding benchmarks.

## ⚙️ Methodology
To ensure a fair comparison with Phase 1, we applied the same **stochastic testing pipeline**:

1.  **Dataset:** A curated custom dataset of **>100,000 agricultural images** (pests, crops, foliage, and humans).
2.  **Sampling:** A randomized selection algorithm that pulls **500 images** per evaluation cycle to validate performance efficiently.
3.  **Objective:** Zero-shot classification and bounding box regression comparing "Pest" vs. "Human" classes.

## 📊 Comparative Results: AIMv2 vs. Grounding DINO

| Metric | Grounding DINO (Phase 1) | AIMv2 (Phase 2) | Engineering Analysis |
| :--- | :--- | :--- | :--- |
| **Pest Detection** | **74% Accuracy** | 62% Accuracy | Grounding DINO is better suited for the pesticide spraying module. |
| **Human Detection** | Baseline | **+11% vs Baseline** | **AIMv2 is significantly safer** for collision avoidance and safety protocols. |
| **Role in System** | Precision Targeting | Safety & Navigation | |

## 🚀 Installation & Usage

To replicate this implementation, you must install the official Apple AIM package.

### 1. Prerequisites
Ensure you have PyTorch installed.

### 2. Install AIMv2
Following the official documentation:
```bash
# Install the AIMv2 package
pip install 'git+[https://github.com/apple/ml-aim.git#subdirectory=aim-v2](https://github.com/apple/ml-aim.git#subdirectory=aim-v2)'
