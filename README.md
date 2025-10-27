# 🚀Efficient-VLAs-Survey
> This is a curated list of "A Survey on Efficient Vision-Language Action Models" research.

To the best of our knowledge, this work presents the first comprehensive survey specifically dedicated to the realm of **Efficient VLAs** that covers the entire "data-model-training" process. We will continue to UPDATE this repository to provide you with the latest cutting-edge developments, so stay tuned!😘 We hope that our work will bring some inspiration to you.😉

❗Pre-print version to be released soon.

## Overview
![TOC](assets/TOC.png)
Fig. 1: **The Organization of Our Survey.** We systematically categorize efficient VLAs into three core pillars: (1) **Efficient Model Design**, encompassing efficient architectures and model compression techniques; (2) **Efficient Training**, covering efficient pre-training and post-training strategies; and (3) **Efficient Data Collection**, including efficient data collection and augmentation methods. The framework also reviews VLA foundations, key applications, challenges, and future directions, establishing the groundwork for advancing scalable embodied intelligence.

## Efficient VLAs

### Efficient Model Design

#### Efficient Architectures
![Efficient_Architectures](assets/Efficient_Architectures.png)
Fig. 2: Key strategies for **Efficient Architectures** in VLAs. We illustrate six primary approaches: (a) **Efficient Attention**, mitigating the O(n^2) complexity of standard self-attention; (b) **Transformer Alternatives**, such as Mamba; (c) **Efficient Action Decoding**, advancing from autoregressive generation to parallel and generative methods; (d) **Lightweight Components**, adopting smaller model backbones; (e) **Mixture-of-Experts**, employing sparse activation via input routing; and (f) **Hierarchical Systems**, which decouple high-level VLM planning from low-level VLA execution.

##### Efficient Attention
| Year | Venue | Paper | Website | Code |
|------|-------|-------|---------|------|
| 2024 | ICRA | [SARA-RT: Scaling up Robotics Transformers with Self-Adaptive Robust Attention](https://arxiv.org/abs/2312.01990) | - | - |
| 2025 | arXiv | [Long-VLA: Unleashing Long-Horizon Capability of Vision Language Action Model for Robot Manipulation](https://arxiv.org/abs/2508.19958) | [🌐](https://long-vla.github.io/) | - |
| 2025 | arXiv | [RetoVLA: Reusing Register Tokens for Spatial Reasoning in Vision-Language-Action Models](https://arxiv.org/abs/2509.21243) | [🌐](https://www.youtube.com/watch?v=2CseBR-snZg&feature=youtu.be) | - |
| 2025 | arXiv | [KV-Efficient VLA: A Method of Speed up Vision Language Model with RNN-Gated Chunked KV Cache](https://arxiv.org/abs/2509.21354) | - | - |
| 2025 | arXiv | [dVLA: Diffusion Vision-Language-Action Model with Multimodal Chain-of-Thought](https://arxiv.org/abs/2509.25681) | - | - |
##### Transformer Alternatives
| Year | Venue | Paper | Website | Code |
|------|-------|-------|---------|------|
| 2024 | NeurIPS | [RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation](https://arxiv.org/abs/2406.04339) | [🌐](https://sites.google.com/view/robomamba-web) | [💻](https://github.com/lmzpai/roboMamba) |
| 2025 | arXiv | [FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation](https://arxiv.org/abs/2506.16201) | - | - |

##### Efficient Action Decoding
| Year | Venue | Paper | Website | Code |
|------|-------|-------|---------|------|
| 2025 | RA-L | [TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation](https://arxiv.org/abs/2406.04339) | [🌐](https://tiny-vla.github.io/) | [💻](https://github.com/liyaxuanliyaxuan/TinyVLA) |
| 2025 | arXiv | [Accelerating vision-language-action model integrated with action chunking via parallel decoding](https://arxiv.org/abs/2503.02310) | - | - |
| 2025 | arXiv | [Fine-Tuning Vision-Language-Action Models: Optimizing Speed and Success](https://arxiv.org/abs/2502.19645) | [🌐](https://openvla-oft.github.io/) | [💻](https://github.com/moojink/openvla-oft) |
| 2025 | arXiv | [HybridVLA: Collaborative Diffusion and Autoregression in a Unified Vision-Language-Action Model](https://arxiv.org/abs/2503.10631) | [🌐](https://hybrid-vla.github.io/) | [💻](https://github.com/PKU-HMI-Lab/Hybrid-VLA) |
| 2025 | arXiv | [FreqPolicy: Efficient Flow-based Visuomotor Policy via Frequency Consistency](https://arxiv.org/abs/2506.08822) | - | - |
| 2025 | arXiv | [CEED-VLA: Consistency Vision-Language-Action Model with Early-Exit Decoding](https://arxiv.org/abs/2506.13725) | [🌐](https://irpn-eai.github.io/CEED-VLA/) | [💻](https://github.com/OpenHelix-Team/CEED-VLA) |
| 2025 | arXiv | [FlowRAM: Grounding Flow Matching Policy with Region-Aware Mamba Framework for Robotic Manipulation](https://arxiv.org/abs/2506.16201) | - | - |
| 2025 | arXiv | [MinD: Learning A Dual-System World Model for Real-Time Planning and Implicit Risk Analysis](https://arxiv.org/abs/2506.18897) | [🌐](https://manipulate-in-dream.github.io/) | [💻](https://github.com/manipulate-in-dream/MinD) |
| 2025 | arXiv | [VQ-VLA: Improving Vision-Language-Action Models via Scaling Vector-Quantized Action Tokenizers](https://arxiv.org/abs/2507.01016) | [🌐](https://xiaoxiao0406.github.io/vqvla.github.io/) | [💻](https://github.com/xiaoxiao0406/VQ-VLA) |
| 2025 | arXiv | [Spec-VLA: Speculative Decoding for Vision-Language-Action Models with Relaxed Acceptance](https://arxiv.org/abs/2507.22424) | - | - |
| 2025 | arXiv | [Leveraging OS-Level Primitives for Robotic Action Management](https://arxiv.org/abs/2508.10259) | - | - |
| 2025 | arXiv | [NinA: Normalizing Flows in Action. Training VLA Models with Normalizing Flows](https://arxiv.org/abs/2508.16845) | - | - |
| 2025 | arXiv | [Discrete Diffusion VLA: Bringing Discrete Diffusion to Action Decoding in Vision-Language-Action Policies](https://arxiv.org/abs/2508.20072) | - | - |
##### Lightweight Component
| Year | Venue | Paper | Website | Code |
|------|-------|-------|---------|------|
| 2024 | NeurIPS | [RoboMamba: Efficient Vision-Language-Action Model for Robotic Reasoning and Manipulation](https://arxiv.org/abs/2406.04339) | [🌐](https://sites.google.com/view/robomamba-web) | [💻](https://github.com/lmzpai/roboMamba) |
| 2025 | RA-L | [TinyVLA: Towards Fast, Data-Efficient Vision-Language-Action Models for Robotic Manipulation](https://arxiv.org/abs/2406.04339) | [🌐](https://tiny-vla.github.io/) | [💻](https://github.com/liyaxuanliyaxuan/TinyVLA) |
| 2024 | arXiv | [CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision](https://arxiv.org/abs/2411.00508) | [🌐](https://clip-rt.github.io/) | [💻](https://github.com/clip-rt/clip-rt) |
| 2025 | arXiv | [Scalable, Training-Free Visual Language Robotics: a modular multi-model framework for consumer-grade GPUs](https://arxiv.org/abs/2502.01071) | [🌐](https://scalable-visual-language-robotics.github.io/) | [💻](https://github.com/bastien-muraccioli/svlr) |
| 2025 | arXiv | [NORA: A Small Open-Sourced Generalist Vision Language Action Model for Embodied Tasks](https://arxiv.org/abs/2504.19854) | [🌐](https://declare-lab.github.io/nora) | [💻](https://github.com/declare-lab/nora) |
| 2025 | arXiv | [SmolVLA: A vision-language-action model for affordable and efficient robotics](https://arxiv.org/abs/2506.01844) | [🌐](https://huggingface.co/blog/smolvla) | [💻](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla) |
| 2025 | arXiv | [SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration](https://arxiv.org/abs/2506.12723) | - | - |
| 2025 | arXiv | [EdgeVLA: Efficient Vision-Language-Action Models](https://arxiv.org/abs/2507.14049) | - | - |
| 2025 | arXiv | [MiniVLA: A Better VLA with a Smaller Footprint](https://ai.stanford.edu/blog/minivla/) | [🌐](https://ai.stanford.edu/blog/minivla/) | - |

##### Mixture-of-Experts
| Year | Venue | Paper | Website | Code |
|------|-------|-------|---------|------|
| 2024 | IROS | [GeRM: A Generalist Robotic Model with Mixture-of-experts for Quadruped Robot](https://arxiv.org/abs/2403.13358) | [🌐](https://songwxuan.github.io/GeRM/) | [💻](https://github.com/Songwxuan/GeRM) |
| 2025 | arXiv | [FedVLA: Federated Vision-Language-Action Learning with Dual Gating Mixture-of-Experts for Robotic Manipulation](https://arxiv.org/abs/2508.02190) | - | - |
| 2025 | arXiv | [Learning to See and Act: Task-Aware View Planning for Robotic Manipulation](https://arxiv.org/abs/2508.05186) | [🌐](https://hcplab-sysu.github.io/TAVP/) | [💻](https://github.com/HCPLab-SYSU/TAVP) |

##### Hierarchical Systems
Year,Venue,Paper,Website,Code
2024,CoRL,HiRT: Enhancing Robotic Control with Hierarchical Robot Transformers,-,-
2024,arXiv,"Towards Synergistic, Generalized, and Efficient Dual-System for Robotic Manipulation",🌐,-
2024,arXiv,A Dual Process VLA: Efficient Robotic Manipulation Leveraging VLM,-,-
2025,arXiv,HAMSTER: Hierarchical Action Models For Open-World Robot Manipulation,🌐,💻
#### Model Compression
![Model_Compression](assets/Model_Compression.png)
Fig. 3: Key strategies for **Model Compression** in VLAs. We illustrate three primary approaches: (a) **Layer Pruning**, which removes redundant layers to reduce model depth and computational cost; (b) **Quantization**, which reduces the numerical precision of model parameters to decrease memory footprint and accelerate inference; and (c) **Token Optimization**, which minimizes the number of processed tokens via token compression (merging tokens), token pruning (dropping non-essential tokens), and token caching (reusing static tokens).

##### Layer Pruning
- FAST: Efficient Action Tokenization for Vision-Language-Action Models. <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2501.09747)]

##### Quantization
- SP-VLA: A Joint Model Scheduling and Token Pruning Approach for VLA Model Acceleration. <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2506.12723)]

##### Token Optimization
- VLA-Cache: Towards Efficient Vision-Language-Action Model via Adaptive Token Caching in Robotic Manipulation. <ins>arXiv, 2025</ins> [[Paper](https://arxiv.org/abs/2502.02175)]

### Efficient Training
![Efficient_Training](assets/Efficient_Training.png)
Fig. 4: Key strategies for **Efficient Training** in VLAs, divided into two main stages. (a) **Efficient Pre-Training** migrates general-purpose VLMs into the embodied domain to create an initial, action-aware policy, encompassing **Data-Efficient Pre-training**, **Efficient Action Representation**, and **Other Pre-training Strategies**. (b) **Efficient Post-Training** subsequently specializes this policy for specific tasks, leveraging **Supervised Fine-tuning** and **RL-Based Methods**.

#### Efficient Pre-Training

##### Data-Efficient Pre-training

##### Efficient Action Representation

##### Other Pre-training Strategies

#### Efficient Post-Training

##### Supervised Fine-tuning

##### RL-Based Method

#### Efficient Data Collection
![Efficient_Data_Collection](assets/Efficient_Data_Collection.png)
Fig. 5: Taxonomy of **Efficient Data Collection** strategies in VLAs. This figure illustrates the primary approaches, encompassing human-in-the-loop, simulated, reusability-oriented, self-driven, and augmentative techniques for scalable acquisition of high-quality robotic datasets while minimizing resource overhead.

##### Human-in-the-Loop Data Collection
| Year | Venue | Paper                                                                                                                                              | Website                                  | Code                                      |
| ---- | ----- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------- | ----------------------------------------- |
| 2024 | RSS   | [CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision](https://arxiv.org/abs/2411.00508)                      | [🌐](https://clip-rt.github.io/)         | [💻](https://github.com/clip-rt/clip-rt)  |
| 2025 | arXiv | [GCENT: Genie Centurion — Accelerating Scalable Real-World Robot Training with Human Rewind-and-Refine Guidance](https://arxiv.org/abs/2505.18793) | [🌐](https://genie-centurion.github.io/) | [💻](https://huggingface.co/agibot-world) |
##### Simulation Data Collection
| Year | Venue | Paper                                                                                                                                                               | Website                                                       | Code                                                |
| ---- | ----- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| 2024 | IROS  | [GeRM: A Generalist Robotic Model with Mixture-of-Experts for Quadruped Robot](https://arxiv.org/abs/2403.13358)                                                    | [🌐](https://songwxuan.github.io/GeRM/)                       | [💻](https://github.com/Songwxuan/GeRM)             |
| 2025 | arXiv | [GraspVLA: A Grasping Foundation Model Pre-trained on Billion-scale Synthetic Action Data](https://arxiv.org/abs/2505.03233)                                        | [🌐](https://pku-epic.github.io/GraspVLA-web/)                | [💻](https://github.com/PKU-EPIC/GraspVLA)          |
| 2025 | arXiv | [cVLA: Towards Efficient Camera-Space VLAs](https://arxiv.org/abs/2507.02190)                                                                                       | -                                                             | -                                                   |
| 2025 | arXiv | [RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation](https://arxiv.org/abs/2506.18088) | [🌐](https://robotwin-platform.github.io/)                    | [💻](https://github.com/robotwin-Platform/RoboTwin) |
| 2025 | arXiv | [ReBot: Scaling Robot Learning with Real-to-Sim-to-Real Robotic Video Synthesis](https://arxiv.org/abs/2503.14526)                                                  | [🌐](https://yuffish.github.io/rebot/)                        | [💻](https://github.com/yuffish/rebot)              |
| 2025 | arXiv | [Real2Render2Real: Scaling Robot Data Without Dynamics Simulation or Robot Hardware](https://arxiv.org/abs/2505.09601)                                              | [🌐](https://real2render2real.com/)                           | [💻](https://github.com/uynitsuj/real2render2real)  |
| 2025 | arXiv | [RealMirror: A Comprehensive, Open-Source Vision-Language-Action Platform for Embodied AI](https://arxiv.org/abs/2509.14687)                                        | [🌐](https://terminators2025.github.io/RealMirror.github.io/) | -                                                   |

##### Internet-Scale and Cross-Domain Data Utilization
| Year | Venue | Paper                                                                                                               | Website                                                            | Code                                                                                |
| ---- | ----- | ------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| 2025 | arXiv | [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)   | [🌐](https://huggingface.co/blog/smolvla)                          | [💻](https://github.com/huggingface/lerobot/tree/main/src/lerobot/policies/smolvla) |
| 2025 | arXiv | [EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos](https://arxiv.org/abs/2507.12440)     | [🌐](https://rchalyang.github.io/EgoVLA/)                          | -                                                                                   |
| 2025 | arXiv | [RynnVLA-001: Using Human Demonstrations to Improve Robot Manipulation](https://arxiv.org/abs/2509.15212)           | [🌐](https://huggingface.co/blog/Alibaba-DAMO-Academy/rynnvla-001) | [💻](https://github.com/alibaba-damo-academy/RynnVLA-001)                           |
| 2025 | arXiv | [EgoScaler: Developing Vision-Language-Action Model from Egocentric Videos](https://arxiv.org/abs/2509.21986)       | -                                                                  | -                                                                                   |
| 2025 | arXiv | [Being-H0: Vision-Language-Action Pretraining from Large-Scale Human Videos](https://arxiv.org/abs/2507.15597)      | [🌐](https://beingbeyond.github.io/Being-H0/)                      | [💻](https://github.com/BeingBeyond/Being-H0)                                       |
| 2025 | arXiv | [MimicDreamer: Aligning Human and Robot Demonstrations for Scalable VLA Training](https://arxiv.org/abs/2509.22199) | -                                                                  | -                                                                                   |
| 2025 | arXiv | [EMMA: Generalizing Real-World Robot Manipulation via Generative Visual Transfer](https://arxiv.org/abs/2509.22407) | [🌐](https://emma-gigaai.github.io/)                               | -                                                                                   |
| 2025 | arXiv | [Humanoid-VLA: Towards Universal Humanoid Control with Visual Integration](https://arxiv.org/abs/2502.14795)        | -                                                                  | -                                                                                   |

##### Self-Exploration Data Collection

| Year | Venue | Paper                                                                                                                                     | Website                          | Code                                            |
| ---- | ----- | ----------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------- | ----------------------------------------------- |
| 2025 | arXiv | [AnyPos: Automated Task-Agnostic Actions for Bimanual Manipulation](https://arxiv.org/abs/2507.12768)                                     | -                                | -                                               |
| 2025 | arXiv | [SimpleVLA-RL: Scaling VLA Training via Reinforcement Learning](https://arxiv.org/abs/2509.09674)                                         | -                                | [💻](https://github.com/PRIME-RL/SimpleVLA-RL)  |
| 2025 | arXiv | [Beyond Human Demonstrations: Diffusion-Based Reinforcement Learning to Generate Data for VLA Training](https://arxiv.org/abs/2509.19752) | -                                | -                                               |
| 2025 | arXiv | [World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training](https://arxiv.org/abs/2509.24948)                      | -                                | -                                               |
| 2025 | arXiv | [VLA-RFT: Vision-Language-Action Reinforcement Fine-Tuning with Verified Rewards in World Simulators](https://arxiv.org/abs/2510.00406)   | [🌐](https://vla-rft.github.io/) | [💻](https://github.com/OpenHelix-Team/VLA-RFT) |

##### Data Augmentation

| Year | Venue | Paper                                                                                                                         | Website                                                       | Code                                                |
| ---- | ----- | ----------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------- | --------------------------------------------------- |
| 2025 | arXiv | [LLaRA: Supercharging Robot Learning Data for Vision-Language Policy](https://arxiv.org/abs/2406.20095)                       | -                                                             | [💻](https://github.com/LostXine/LLaRA)             |
| 2025 | arXiv | [Vision-Language-Action Instruction Tuning: From Understanding to Manipulation](https://arxiv.org/abs/2507.17520)             | [🌐](https://yangs03.github.io/InstructVLA_Home/)             | [💻](https://github.com/InternRobotics/InstructVLA) |
| 2025 | arXiv | [RoboChemist: Long-Horizon and Safety-Compliant Robotic Chemical Experimentation](https://arxiv.org/abs/2509.08820)           | [🌐](https://zzongzheng0918.github.io/RoboChemist.github.io/) | -                                                   |
| 2024 | arXiv | [CLIP-RT: Learning Language-Conditioned Robotic Policies from Natural Language Supervision](https://arxiv.org/abs/2411.00508) | [🌐](https://clip-rt.github.io/)                              | [💻](https://github.com/clip-rt/clip-rt)            |
| 2025 | arXiv | [ERMV: Editing 4D Robotic Multi-view Images to Enhance Embodied Agents](https://arxiv.org/abs/2507.17462)                     | -                                                             | [💻](https://github.com/IRMVLab/ERMV)               |

## Contact Us

For any questions or suggestions, please feel free to contact us at:

Email: yuzhaoshu@gmail.com
