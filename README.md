<div align="center">

<h1>Relation Rectification in Diffusion Model</h1>

<div>
Yinwei Wu&emsp;Xingyi Yang&emsp;Xinchao Wang<sup>&dagger;</sup>
</div>
<div>
    LV-Lab, National University of Singapore&emsp;
    <sup>&dagger;</sup>corresponding author 
</div>


<div>
   <strong>CVPR 2024</strong>
</div>

<div>
<a target="_blank" href="https://arxiv.org/abs/2403.20249">
  <img src="https://img.shields.io/badge/arXiv-2403.20249-b31b1b.svg" alt="arXiv Paper"/>
</a>

<a target="_blank" href="https://wuyinwei-hah.github.io/rrnet.github.io/">
  <img src="https://img.shields.io/badge/Project-Page-blue" alt="arXiv Paper"/>
</a>
<!-- [![Website](https://img.shields.io/website-up-down-green-red/http/shields.io.svg)](https://shields.io/) -->
<a target="_blank" href="https://cvpr.thecvf.com/virtual/2024/poster/30070">
  <img src="https://img.shields.io/badge/CVPR-Poster-green" alt="CVPR Poster"/>
</a>
</div>
</div>

---
## Overview
![overall_structure](./assets/structure.jpg)

---

## Installation
1.
```bash
git clone https://github.com/WUyinwei-hah/RRNet.git


cd RRNet

pip install -r requirements.txt

```

2.

Install the dgl from https://dgl.ai/pages/start.html

---
## Data preparation

There are some examples in the ./Dataset, which you can try directly.

You can also download the whole RR dataset from https://drive.google.com/file/d/1rKmeR1k4S6Im8YMq-FSBGK11nNwQzhwM/view?usp=sharing

---

## Training
```python
python3 train.py --data_folder ./Dataset/contain/ --device 0 
```
The generation result and the model checkpoints will be placed at ./generation_result.

---

## Generation
```python
python3 generate.py --data_folder ./Dataset/contain/ --save_folder ./generation_result  --device 0 
```
---

## Citation
```
@misc{wu2024relation,
      title={Relation Rectification in Diffusion Model}, 
      author={Yinwei Wu and Xingyi Yang and Xinchao Wang},
      year={2024},
      eprint={2403.20249},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
