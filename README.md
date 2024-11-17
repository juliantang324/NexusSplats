<p align="center">
<h1 align="center">NexusSplats: Efficient 3D Gaussian Splatting in the Wild</h1>
  <p align="center">
    Yuzhou Tang
    ·
    Dejun Xu
    ·
    Yongjie Hou
    ·
    Zhenzhong Wang
    ·
    Min Jiang
  </p>
  <p align="center">School of Informatics, Xiamen University</p>

[//]: # (  <h3 align="center"><a href="https://arxiv.org/pdf/2407.08447">📄 Paper</a> | <a href="https://wild-gaussians.github.io/">🌐 Project Page</a></h3>)
<br/>
<p align="center">
  <img width="100%" alt="NexusSplats model appearance" src="assets/ours.webp" />
</p>

<p align="justify">
we propose a nexus kernel-driven approach, called NexusSplats, for efficient and finer 3D scene reconstruction under complex lighting and occlusion conditions.
Experimental results demonstrate that NexusSplats achieves state-of-the-art rendering quality and reduces reconstruction time in different scenes by up to 70.4% compared to the current best method in quality.
</p>
<br>

[//]: # (> <b>:dizzy:	NEWS: WildGaussians is now integrated into <a href="https://nerfbaselines.github.io">NerfBaselines</a>. Checkout the results here: https://nerfbaselines.github.io/phototourism</b>)

<img width="100%" alt="overview of NexusSplats" src="assets/overview.png" />

## Installation
Clone the repository and create a `python == 3.11` Anaconda environment with CUDA toolkit 11.8 installed using
```bash
git clone git@github.com:juliantang324/NexusSplats.git
cd NexusSplats

conda create -y -n ns python=3.11
conda activate ns
conda env config vars set NERFBASELINES_BACKEND=python
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
pip install -r requirements.txt
pip install nerfbaselines>=1.2.0
pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install -e .
```

## Interactive viewer
To start the viewer and explore the trained models, run one of the following:
```bash
# Photo Tourism
ns viewer --checkpoint https://github.com/juliantang324/NexusSplats/releases/tag/v1.0.0/phototourism.zip/trevi-fountain/checkpoint --data external://phototourism/trevi-fountain
ns viewer --checkpoint https://github.com/juliantang324/NexusSplats/releases/tag/v1.0.0/phototourism.zip/sacre-coeur/checkpoint --data external://phototourism/sacre-coeur
ns viewer --checkpoint https://github.com/juliantang324/NexusSplats/releases/tag/v1.0.0/phototourism.zip/brandenburg-gate/checkpoint --data external://phototourism/brandenburg-gate
```

## Training
To start the training on the Photo Tourism dataset, run one of following commands:
```bash
# Photo Tourism
ns train --data external://phototourism/trevi-fountain
ns train --data external://phototourism/sacre-coeur
ns train --data external://phototourism/brandenburg-gate
```

The training will also generate output artifacts containing the **test set predictions**, **checkpoint**, and **tensorboard logs**.

## Rendering videos
To render a video on a trajectory (e.g., generated from the interactive viewer), run:
```bash
ns render-trajectory --checkpoint {checkpoint} --trajectory {trajectory file}
```

## Concurrent works
There are several concurrent works that also aim to extend 3DGS to handle in-the-wild scenarios:
<ul>
<li><a href="https://arxiv.org/pdf/2407.08447">WildGaussians: 3D Gaussian Splatting in the Wild</a></li>
<li><a href="https://arxiv.org/pdf/2403.15704">Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections</a></li>
</ul>

## Acknowledgements
We sincerely appreciate the authors of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [NerfBaselines](https://github.com/nerfbaselines/nerfbaselines) for their great work and released code.
Please follow their licenses when using our code.

[//]: # (## Citation)

[//]: # (If you find our code or paper useful, please cite:)
```bibtex

```
