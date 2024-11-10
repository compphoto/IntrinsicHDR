
---

<div align="center">    
 
# Intrinsic Single-Image HDR Reconstruction  

[![Project](http://img.shields.io/badge/project-intrinsicHDR-cc9933.svg)](https://yaksoy.github.io/intrinsicHDR/)
[![Video](http://img.shields.io/badge/video-YouTube-4b44ce.svg)](https://www.youtube.com/watch?v=EiyH52BcKkw)
[![Paper](http://img.shields.io/badge/paper-ECCV2024-B31B1B.svg)](https://arxiv.org/abs/2409.13803)
[![Supplementary](http://img.shields.io/badge/suppl.-intrinsicHDR-B31B1B.svg)](https://yaksoy.github.io/papers/ECCV24-IntrinsicHDR-supp.pdf)
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/IntrinsicHDR/blob/main/notebooks/intrinsicHDR.ipynb)  


</div>
 


## Description   


The low dynamic range (LDR) of common cameras fails to capture the rich contrast in natural scenes, resulting in loss of color and details in saturated pixels. Reconstructing the high dynamic range (HDR) of luminance present in the scene from single LDR photographs is an important task with many applications in computational photography and realistic display of images. The HDR reconstruction task aims to infer the lost details using the context present in the scene, requiring neural networks to understand high-level geometric and illumination cues. This makes it challenging for data-driven algorithms to generate accurate and high-resolution results. 

![teaser](./figures/representative.jpg)

In this work, we introduce a physically-inspired remodeling of the HDR reconstruction problem in the intrinsic domain. The intrinsic model allows us to train separate networks to extend the dynamic range in the shading domain and to recover lost color details in the albedo domain. We show that dividing the problem into two simpler sub-tasks improves performance in a wide variety of photographs.   

![pipeline](./figures/pipeline.jpg)

Try out our pipeline on your own images in Colab! [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/compphoto/IntrinsicHDR/blob/main/notebooks/intrinsicHDR.ipynb)

## How to run   
First, install dependencies. The code was tested with Python 3.9. It is recommended to start with a fresh environment:
```bash
# create empty env
python3 -m venv intrHDR_env

# activate env
source intrHDR_env/bin/activate
```
Next, clone this repository and install the requirements. Make sure that pip is up-to-date (python3 -m pip install --upgrade pip):

```bash
# clone project   
git clone https://github.com/compphoto/IntrinsicHDR

# install project   
cd IntrinsicHDR
pip install .
 ```   

The pipeline expects input images to be linear. 
To dequantize and linearize images, run:

 ```bash
# download pretrained weights 'model.ckpt.*' and put them into "./baselines/SingleHDR/checkpoints"
wget https://github.com/compphoto/IntrinsicHDR/releases/download/v1.0/model.ckpt.data-00000-of-00001
wget https://github.com/compphoto/IntrinsicHDR/releases/download/v1.0/model.ckpt.index 
wget https://github.com/compphoto/IntrinsicHDR/releases/download/v1.0/model.ckpt.meta 

# create checkpoint directory
mkdir ./baselines/SingleHDR/checkpoints

# move weights to checkpoint directory.
mv model.ckpt* ./baselines/SingleHDR/checkpoints/.

# run linearization, e.g.  
python3 dequantize_and_linearize.py --test_imgs /path/to/input/imgs --output_path /path/to/results --root .
```
 Now, run our HDR reconstruction pipeline. The results will be saved as EXR files in --output_path:  
 ```bash
# run module, e.g.  
python3 inference.py --test_imgs /path/to/input/imgs --output_path /path/to/results --use_exr
```



### Citation
This implementation is provided for academic use only. Please cite our paper if you use this code or any of the models.   
```
@INPROCEEDINGS{dilleIntrinsicHDR,
author={Sebastian Dille and Chris Careaga and Ya\u{g}{\i}z Aksoy},
title={Intrinsic Single-Image HDR Reconstruction},
booktitle={Proc. ECCV},
year={2024},
} 
```   

### Credits
"./baselines/SingleHDR/" is adapted from [SingleHDR](https://github.com/alex04072000/SingleHDR) for their dequantization and linearization network.

".intrinsic_decomposition" is adapted from [IntrinsicImageDecomposition](https://github.com/compphoto/Intrinsic) for the decomposition network.

".src/midas/" is adapted from [MiDaS](https://github.com/intel-isl/MiDaS/tree/v2) for their EfficientNet implementation.
