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
<!-- <a target="_blank" href="https://arxiv.org/abs/2312.17142">
  <img src="https://img.shields.io/badge/arXiv-2312.17142-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fjiawei-ren%2Fdreamgaussian4d&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a> -->
</div>
</div>

---
## Overview
![overall_structure](https://wuyinwei-hah.github.io/rrnet.github.io/static/images/cover.png)

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

## Acknowledgement

---

## Citation


