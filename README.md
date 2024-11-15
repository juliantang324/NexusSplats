<p align="center">
<h1 align="center">NexusSplats: Efficient 3D Gaussian Splatting in the Wild</h1>
  <p align="center">
    <a>Yuzhou Tang</a>
    ·
    <a>Dejun Xu</a>
    ·
    <a>Yongjie Hou</a>
    ·
    <a>Zhenzhong Wang</a>
    ·
    <a>Min Jinag</a>
  </p>

[//]: # (  <h3 align="center"><a href="https://arxiv.org/pdf/2407.08447">📄 Paper</a> | <a href="https://wild-gaussians.github.io/">🌐 Project Page</a></h3>)
  <div align="center"></div>
<br/>
<p align="center">

[//]: # (  <img width="51%" alt="WildGaussians model appearance" src=".assets/cover-trevi.webp" />)
[//]: # (  <img width="43%" alt="WildGaussians remove occluders" src=".assets/cover-onthego.webp" />)
</p>
<p align="justify">
we propose a nexus kernel-driven approach, called {\em NexusSplats}, for efficient and finer 3D scene reconstruction under complex lighting and occlusion conditions.
Experimental results demonstrate that {\em NexusSplats} achieves state-of-the-art rendering quality and reduces reconstruction time in different scenes by up to 70.4\% compared to the current best method in quality.
</p>
<br>

[//]: # (> <b>:dizzy:	NEWS: WildGaussians is now integrated into <a href="https://nerfbaselines.github.io">NerfBaselines</a>. Checkout the results here: https://nerfbaselines.github.io/phototourism</b>)


## Installation
Clone the repository and create a `python == 3.11` Anaconda environment with CUDA toolkit 11.8 installed using
```bash
git clone git@github.com:juliantang324/NexusSplats.git
cd NexusSplats

conda create -y -n ns python=3.11
conda activate ns
conda install -y --override-channels -c nvidia/label/cuda-11.8.0 cuda-toolkit
conda env config vars set NERFBASELINES_BACKEND=python
pip install --upgrade pip
pip install -r requirements.txt
pip install nerfbaselines>=1.2.0
pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install -e .
```

## Checkpoints, predictions, and data
<ul>

</ul>

In order to train/evaluate on the NeRF On-the-go dataset, please download the undistorted version
from the following link:
[https://huggingface.co/datasets/jkulhanek/nerfonthego-undistorted/tree/main](https://huggingface.co/datasets/jkulhanek/nerfonthego-undistorted/tree/main)

## Interactive viewer
To start the viewer and explore the trained models, run one of the following:
```bash
# Photo Tourism
ns viewer --checkpoint  --data external://phototourism/trevi-fountain
ns viewer --checkpoint  --data external://phototourism/sacre-coeur
ns viewer --checkpoint  --data external://phototourism/brandenburg-gate
```

## Training
To start the training on the Photo Tourism dataset, run one of following commands:
```bash
# Photo Tourism
ns train --method nexus-splats --data external://phototourism/trevi-fountain
ns train --method nexus-splats --data external://phototourism/sacre-coeur
ns train --method nexus-splats --data external://phototourism/brandenburg-gate
```

The training will also generate output artifacts containing the **test set predictions**, **checkpoint**, and **tensorboard logs**.

## Rendering videos
To render a video on a trajectory (e.g., generated from the interactive viewer), run:
```bash
ns render --checkpoint {checkpoint} --trajectory {trajectory file}
```

## Concurrent works
There are several concurrent works that also aim to extend 3DGS to handle in-the-wild data:
<ul>
<li><a href="https://arxiv.org/pdf/2407.08447">WildGaussians: 3D Gaussian Splatting in the Wild</a></li>
<li><a href="https://arxiv.org/pdf/2406.10373v1">Wild-GS: Real-Time Novel View Synthesis from Unconstrained Photo Collections</a></li>
<li><a href="https://arxiv.org/pdf/2403.15704">Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections</a></li>
<li><a href="https://arxiv.org/pdf/2406.20055">SpotlessSplats: Ignoring Distractors in 3D Gaussian Splatting</a></li>
<li><a href="https://arxiv.org/pdf/2403.10427">SWAG: Splatting in the Wild images with Appearance-conditioned Gaussians</a></li>
<li><a href="https://arxiv.org/pdf/2406.02407">WE-GS: An In-the-wild Efficient 3D Gaussian Representation for Unconstrained Photo Collections</a></li>
</ul>

## Acknowledgements

[//]: # (The renderer is built on [3DGS]&#40;https://github.com/graphdeco-inria/gaussian-splatting&#41; and [Mip-Splatting]&#40;https://niujinshuchong.github.io/mip-splatting/&#41;.)

[//]: # (Please follow the license of 3DGS and Mip-Splatting. We thank all the authors for their great work and released code.)

## Citation
If you find our code or paper useful, please cite:
```bibtex

```
