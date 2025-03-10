<p align="center">
<h1 align="center">NexusSplats: Efficient 3D Gaussian Splatting in the Wild</h1>
  <p align="center">
    <a href="https://juliantang324.github.io/">Yuzhou Tang</a>
    ·
    <a href="https://orcid.org/0000-0001-6272-0941">Dejun Xu</a>
    ·
    <a href="https://github.com/Maximilian1794">Yongjie Hou</a>
    ·
    <a href="https://zhenzhongxmu.github.io/">Zhenzhong Wang</a>
    ·
    <a href="https://minjiang.xmu.edu.cn/">Min Jiang</a>*
  </p>
  <p align="center">Xiamen University</p>

  <div align="center">
    <a href="https://arxiv.org/pdf/2411.14514"><img src='https://img.shields.io/badge/arXiv-2411.14514-b31b1b'></a>
    <a href="https://nexus-splats.github.io/"><img src='https://img.shields.io/badge/Project-Page-blue'></a>
  </div>
<br/>
<p align="center">
  <img width="100%" alt="NexusSplats model appearance" src="assets/ours.webp"/>
</p>

<br>

## News
**[2025.2.20]** 😊 We have supported custom datasets with COLMAP format in NexusSplats.

**[2025.1.16]**  🤗 We have released the pretrained models for NexusSplats in [#release](https://github.com/juliantang324/NexusSplats/releases/download/v1.0.0/phototourism.zip).

**[2024.11.25]** 🎈 We have released the codebase for NexusSplats.

## TODO List
- [x] Codebase release
- [x] Pretrained models
- [ ] Merge with the latest version of NerfBaselines

## Abstract
Photorealistic 3D reconstruction of unstructured real-world scenes remains challenging due to complex illumination variations and transient occlusions. 
Existing methods based on Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) struggle with inefficient light decoupling and structure-agnostic occlusion handling. 
To address these limitations, we propose <strong>NexusSplats</strong>, an approach tailored for efficient and high-fidelity 3D scene reconstruction under complex lighting and occlusion conditions. 
In particular, NexusSplats leverages a <strong>hierarchical light decoupling</strong> strategy that performs centralized appearance learning, efficiently and effectively decoupling varying lighting conditions. 
Furthermore, a <strong>structure-aware occlusion handling</strong> mechanism is developed, establishing a nexus between 3D and 2D structures for fine-grained occlusion handling. 
Experimental results demonstrate that NexusSplats achieves state-of-the-art rendering quality and reduces the number of total parameters by 65.4%, leading to 2.7× faster reconstruction.
<br>

<img width="100%" alt="overview of NexusSplats" src="assets/overview.png" />
<strong>Overview of NexusSplats.</strong> Our framework operates in three stages:
<em>First</em>, the <strong>Hierarchical Gaussian Management</strong> organizes 3D Gaussians into dynamic <em>nexus kernels</em>, which generate Gaussian attributes and perform <strong>Centralized Appearance Learning</strong> and <strong>Uncertainty Propagation</strong>.
<em>Second</em>, a raw image, a mapped image, and an uncertainty mask are rendered though tile rasterization.
<em>Third</em>, the <strong>Boundary-Aware Refinement</strong> corrects misclassified scene boundaries.
The system optimizes via a combination of color loss and uncertainty loss.

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

## Interactive Viewer
To start the viewer and explore the trained models, run one of the following commands:
```bash
ns viewer --checkpoint {checkpoint} --data external://phototourism/trevi-fountain
ns viewer --checkpoint {checkpoint} --data external://phototourism/sacre-coeur
ns viewer --checkpoint {checkpoint} --data external://phototourism/brandenburg-gate
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
