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
    Min Jiang*
  </p>
  <p align="center">Xiamen University</p>

  <div align="center">
    <a href="https://arxiv.org/pdf/2411.14514v4"><img src='https://img.shields.io/badge/arXiv-2411.14514-b31b1b'></a>
    <a href="https://nexus-splats.github.io/"><img src='https://img.shields.io/badge/Project-Page-blue'></a>
  </div>
<br/>
<p align="center">
  <img width="100%" alt="NexusSplats model appearance" src="assets/ours.webp"/>
</p>

<br>

## News

**[2024.11.25]** 🎈 We have released the codebase for NexusSplats.

## TODO List
- [x] Codebase release
- [ ] Pretrained models
- [ ] Merge with the latest version of NerfBaselines

## Abstract
we propose a nexus kernel-driven approach, called NexusSplats, for efficient and finer 3D scene reconstruction under complex lighting and occlusion conditions.
Experimental results demonstrate that NexusSplats achieves state-of-the-art rendering quality and reduces reconstruction time in different scenes by up to 70.4% compared to the current best method in quality.
<br>

<img width="100%" alt="overview of NexusSplats" src="assets/overview.png" />
<em>Left:</em> From the reference image, we extract light embedding and transient embedding to capture
global lighting and occlusion conditions. <em>Middle:</em> Our nexus kernels enable hierarchical management
of Gaussian primitives, allowing efficient local adaptations to different lighting and occlusion conditions
via the light decoupling module and the uncertainty splatting module. <em>Right:</em> Through tile
rasterization, we project raw colors, mapped colors, and uncertainties onto 2D visible planes. A boundary
penalty finally refines the filtering mask in handling occlusions.

## Installation
Clone the repository and create a `python == 3.11` Anaconda environment with CUDA toolkit 11.8.
Our code is implemented based on [NerfBaselines](https://github.com/nerfbaselines/nerfbaselines).
Install the dependencies and the codebase:
```bash
git clone git@github.com:juliantang324/NexusSplats.git
cd NexusSplats

conda create -y -n ns python=3.11
conda activate ns
conda env config vars set NERFBASELINES_BACKEND=python
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install --upgrade pip
pip install -r requirements.txt
pip install nerfbaselines==1.2.5
pip install -e ./submodules/diff-gaussian-rasterization ./submodules/simple-knn
pip install -e .
```

## Training
To start the training on the Photo Tourism dataset, run one of following commands:
```bash
ns train --data external://phototourism/trevi-fountain
ns train --data external://phototourism/sacre-coeur
ns train --data external://phototourism/brandenburg-gate
```

## Evaluation
To evaluate the trained model on the Photo Tourism dataset, run the following commands:
```bash
# render predictions
ns render --checkpoint {checkpoint} --data external://phototourism/trevi-fountain --output {output_path}
ns render --checkpoint {checkpoint} --data external://phototourism/sacre-coeur --output {output_path}
ns render --checkpoint {checkpoint} --data external://phototourism/brandenburg-gate --output {output_path}

# evaluate predictions
ns evaluate {path/to/predictions} --output results.json
```

## Concurrent works
There are several concurrent works that also aim to extend 3DGS to handle in-the-wild scenarios:
<ul>
<li><a href="https://arxiv.org/pdf/2407.08447">WildGaussians: 3D Gaussian Splatting in the Wild</a></li>
<li><a href="https://arxiv.org/pdf/2403.15704">Gaussian in the Wild: 3D Gaussian Splatting for Unconstrained Image Collections</a></li>
</ul>

## Acknowledgements
We sincerely appreciate the authors of [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [NerfBaselines](https://github.com/nerfbaselines/nerfbaselines) 
for their great work and released code.
Please follow their licenses when using our code.


## Citation

If you find our code or paper useful, please star this repository and cite:
```bibtex
@article{tang2024nexussplats,
    title={NexusSplats: Efficient 3D Gaussian Splatting in the Wild},
    author={Tang, Yuzhou and Xu, Dejun and Hou, Yongjie and Wang, Zhenzhong and Jiang, Min},
    journal={arXiv preprint arXiv:2411.14514},
    year={2024}
}
```
