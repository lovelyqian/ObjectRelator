# **ObjectRelator: Enabling Cross-View Object Relation Understanding Across Ego-Centric and Exo-Centric Perspectives (ICCV 2025 Highlight)**

> #### Yuqian Fu, Runze Wang, Bin Ren, Guolei Sun, Biao Gong, Yanwei Fu, Danda Pani Paudel, Xuanjing Huang, Luc Van Gool
>

[Paper](https://arxiv.org/abs/2411.19083) 🌟
[Project Page](http://yuqianfu.com/ObjectRelator/) 🚀


## Updates
- [08/2025] Data, models, codes, and training/testing scripts are released. 🔧
- [07/2025] [Project website](http://yuqianfu.com/ObjectRelator/) is released. 📖
- [06/2025] Our paper is accepted by ICCV 25 (Highlight Paper). 🎉
- [06/2025] We were awarded 2nd place in the Correspondences track of the 2025 EgoVis Ego-Exo4D Challenge. [Technical report](https://arxiv.org/pdf/2506.05856?) 🏅

### Features

* 🔥**Ego-Exo Object Correspondence Task:** We conduct an early exploration of this challenging task, analyzing its unique difficulties, constructing several baselines, and proposing a new method.

* 🔥**ObjectRelator Framework:** We introduce ObjectRelator, a cross-view object segmentation method combining MCFuse and XObjAlign. MCFuse first introduces the text modality into this task and improves localization using multimodal cues for the same object(s), while XObjAlign boosts performance under appearance variations with an object-level consistency constraint.

* 🔥**New Testbed** & **SOTA Results:** Alongside Ego-Exo4D, we present HANDAL-X as an additional benchmark. Our proposed ObjectRelator achieves state-of-the-art (SOTA) results on both datasets.

  ![](assets/teaser.png)
  
  ![](assets/compressed-compressed-demo.gif)

  More video demos can be found: http://yuqianfu.com/ObjectRelator/. 


## Installation

See [Installation instructions.](docs/INSTALL.md)

## Data

See [Prepare Datasets for ObjectRelator.](docs/DATASET.md)

## Model Zoo & Quick Start

See [Quick Start With ObjectRelator.](docs/ModelZoo_QuickStart.md)

## Train & Evaluation

See [Train & Evaluation.](docs/Train_Evaluation.md)

## Comparative Results

<img width="1342" height="500" alt="image" src="https://github.com/user-attachments/assets/b3698002-0aed-4bb0-8b53-2ca4314812b8" />

Results on val set, main results from our [ICCV25 paper](https://arxiv.org/pdf/2411.19083).

<img width="1385" height="282" alt="image" src="https://github.com/user-attachments/assets/6ba16b7b-6e57-4c89-99d2-95a72c1d77ac" />

Results on test set, same as [EgoExo4D Correspondence Challenge](https://eval.ai/web/challenges/challenge-page/2288/) and our [technical report](https://arxiv.org/pdf/2506.05856?).


## Citation

If you think this work is useful for your research, please use the following BibTeX entry.

```
@article{fu2024objectrelator,
  title={Objectrelator: Enabling cross-view object relation understanding in ego-centric and exo-centric videos},
  author={Fu, Yuqian and Wang, Runze and Fu, Yanwei and Paudel, Danda Pani and Huang, Xuanjing and Van Gool, Luc},
  journal={ICCV},
  year={2025}
}

@article{fu2025cross,
  title={Cross-View Multi-Modal Segmentation@ Ego-Exo4D Challenges 2025},
  author={Fu, Yuqian and Wang, Runze and Fu, Yanwei and Paudel, Danda Pani and Van Gool, Luc},
  journal={arXiv preprint arXiv:2506.05856},
  year={2025}
}
```

## Acknowledgement

Thanks for awesome works: [PSALM](https://github.com/zamling/PSALM/blob/main/) , [LLaVA](https://github.com/haotian-liu/LLaVA) and [Ego-Exo4D](https://ego-exo4d-data.org). Code is based on these works.
