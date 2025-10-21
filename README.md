<div align="center">
    <img src="https://github.com/SnowNation101/NYX/blob/main/assets/Nyx.webp" alt="Nyx Logo" style="width: 35%;" />
</div>

<h1 align="center"> 🌓 Nyx: Unified Multimodal Retriever for Universal RAG </h1>

<div align="center"> 
  <a href="https://arxiv.org/abs/2510.17354" target="_blank">
    <img alt="Arxiv Paper" src="https://img.shields.io/badge/arXiv-Paper-b5212f.svg?logo=arxiv">
  </a>
  
  <a href="https://opensource.org/license/MIT" target="_blank">
    <img alt="LICENSE" src="https://img.shields.io/github/license/SnowNation101/Nyx?color=lightgreen&label=⚖️%20LICENSE">
  </a>

  <a href="https://huggingface.co/collections/SnowNation/nyx-685a63158e4825919b6dd09a" target="_blank">
    <img alt="Hugging Face Collection" src="https://img.shields.io/badge/%20Hugging%20Face-Collection-yellow.svg?logo=huggingface">
  </a>

  <a href="https://github.com/SnowNation101/Nyx" target="_self">
    <img alt="GitHub Repo stars" src="https://img.shields.io/github/stars/SnowNation101/Nyx?style=flat&logo=github&color=lightblue">
  </a>
</div>


---

<div align="center">

**Authors:**  
Chenghao Zhang · Guanting Dong · Xinyu Yang · Zhicheng Dou  

</div>


This repository contains the official implementation of our paper *"Towards Mixed-Modal Retrieval for Universal Retrieval-Augmented Generation"*.


## Introduction
We propose **Nyx**, a unified mixed-modal retriever tailored for URAG scenarios, and construct **NyxQA**, a large-scale mixed-modal QA dataset. Our framework includes:  
- A four-stage automated pipeline for generating realistic multimodal QA pairs.  
- A two-stage training framework combining pre-training on NyxQA and supervised fine-tuning with VLM feedback.  
- Strong performance on both text-only RAG benchmarks and vision-language URAG tasks.  

## Preparation

We recommend using **Conda** for package management.

```bash
conda create -n nyx python=3.11
conda activate nyx
pip install -r requirements.txt
```

Our implementation uses `torch==2.4.0`, `faiss-cpu==1.8.0`, and `transformers==4.52.2`. Please note that `faiss-cpu` and `transformers` might have `numpy` version conflicts. We prefer keeping `numpy` at version `1.26.4` (the version compatible with `faiss-cpu`), so you may need to uninstall any newer `numpy` versions.

Suggested installation order: PyTorch → faiss-cpu → transformers → accelerate → deepspeed


## Acknowledgements

The core implementation of this project is built upon [VLM2Vec](https://github.com/TIGER-AI-Lab/VLM2Vec). We extend our sincere gratitude to the original authors for their foundational work.

We also want to acknowledge and thank the developers of these essential tools that made our work possible:
- [vLLM](https://github.com/vllm-project/vllm) for efficient LLM inferencing
- [FlashAttention](https://github.com/Dao-AILab/flash-attention) for optimized attention computation
- [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) for distributed training acceleration

Our work stands on the shoulders of these remarkable open-source projects and the generous research community.

We also want to note that the logo at the top of this README is adapted from the character **Nyx** in the game *Hades* by Supergiant Games.
