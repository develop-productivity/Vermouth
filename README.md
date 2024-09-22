# Vermouth: Bridging Generative and Discriminative Models for Unified Visual Perception with Diffusion Priors | IJCAI2024
This is office PyTorch implementation for Vermouth that was accepted by IJCAI2024.

Vermouth is a simple yet effective framework to migrate the diffusion model to non-generated tasks, which comprising a pre-trained Stable Diffusion (SD) model containing rich generative priors, a pre-trained Stable Diffusion (SD) model containing rich generative priors, a unified head (U-head) capable of integrating hierarchical representations, and an adapted expert providing discriminative priors

![alt text](https://s2.loli.net/2024/09/22/TExn2ZBQJU7KbNW.png)


## Results


### ZS-SBIR
| Model | Sketchy | TU_Berlin | QuickDraw|
| - | - | - | - | 
| MAE-L | 39.23 | 41.99 | 11.71   |
|BeiTv3-G | 54.54 | 50.93 | 13.67 |
| Swinv2-L | 43.39 | 45.51 | 12.08 | 
| DINO-B | 38.51 | 25.49 &|10.15 | 
| Vermouth | 56.8 | 52.83 | 15.11 |
### OV Semantic Segmentation
| Model | ADE-150 | PC-59 | VOC20 | ADE-847 | PC-459|
| - | - | - | - | - | - | 
| MAE-L | 17.5 | 53.27 | 93.51 | 3.42 | 8.82 |
| ConvNeXt-L | 18.65 | 53.42 | 94.62 | 3.53 | 9.53 |
| Swinv2-L | 18.8| 53.37 | 94.76 | 3.8 | 9.42 |
| DINO-B | 17.13 | 47.84 | 92.44 | 3.16 | 7.75|
| Vermouth | 19.0 | 52.88 | 92.87 | 3.7 | 9.0|

## How to use


## TODO
- [x] release core code
- [ ] clean the project code 
- [ ] release model pre-train weight

## cite
If you find our work useful in your research, please consider citing:
> @article{dong2024bridging,
  title={Bridging Generative and Discriminative Models for Unified Visual Perception with Diffusion Priors},
  author={Dong, Shiyin and Zhu, Mingrui and Cheng, Kun and Wang, Nannan and Gao, Xinbo},
  journal={arXiv preprint arXiv:2401.16459},
  year={2024}
}