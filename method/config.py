from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING, Optional


if TYPE_CHECKING:
    UncertaintyMode = Literal["disabled", "l2reg", "l1reg", "dino", "dino+mssim"]
else:
    UncertaintyMode = str


@dataclass
class Config:
    debug: bool = False

    num_sky_gaussians: int = 0

    iterations: int = 30_000
    position_lr_delay_mult: float = 0.01
    position_lr_max_steps: int = 30_000
    percent_dense: float = 0.01
    lambda_dssim: float = 0.2
    densify_grad_threshold: float = 0.0002

    appearance_n_fourier_freqs: int = 4
    embedding_lr: float = 0.005
    embedding_regularization: float = 0.0

    appearance_enabled: bool = True
    appearance_embedding_dim: int = 32
    appearance_embedding_lr: float = 0.001
    appearance_mlp_lr: float = 0.0005
    appearance_embedding_regularization: float = 0.0
    appearance_embedding_optim_lr: float = 0.1
    appearance_embedding_optim_iters: int = 128
    appearance_optim_type: str = "dssim+l1"
    """Either 'mse', 'dssim+l1'"""
    appearance_separate_tuned_color: bool = True
    appearance_dropout: float = 0.2
    appearance_init_fourier: bool = True

    # Uncertainty model
    uncertainty_enabled: bool = True
    uncertainty_embedding_dim: int = 32
    uncertainty_embedding_lr: float = 0.001
    transient_embedding_dim: int = 32
    transient_embedding_lr: float = 0.001
    transient_regularization: float = 0.0
    uncertainty_mode: UncertaintyMode = "dino"
    uncertainty_backbone: str = "dinov2_vits14_reg"
    uncertainty_regularizer_weight: float = 0.5
    uncertainty_lr: float = 0.0005
    uncertainty_scale_grad: bool = False
    uncertainty_center_mult: bool = False
    uncertainty_warmup_iters: int = 0
    uncertainty_warmup_start: int = 2000
    use_boundary_penalty: bool = True

    # Nexus Kernels
    feat_dim: int = 32
    n_offsets: int = 10
    voxel_size: float = 0.001  # if voxel_size<=0, using 1nn dist
    update_depth: int = 3
    update_init_factor: int = 16
    update_hierachy_factor: int = 4

    use_feat_bank: bool = False
    ratio: int = 1  # sampling the input point cloud

    add_opacity_dist: bool = False
    add_cov_dist: bool = False
    add_color_dist: bool = False

    position_lr_init: float = 0.0
    position_lr_final: float = 0.0

    offset_lr_init: float = 0.01
    offset_lr_final: float = 0.0001
    offset_lr_delay_mult: float = 0.01
    offset_lr_max_steps: int = 30_000

    feature_lr: float = 0.0075
    opacity_lr: float = 0.02
    scaling_lr: float = 0.007
    rotation_lr: float = 0.002

    mlp_opacity_lr_init: float = 0.002
    mlp_opacity_lr_final: float = 0.00002
    mlp_opacity_lr_delay_mult: float = 0.01
    mlp_opacity_lr_max_steps: int = 30_000

    mlp_cov_lr_init: float = 0.004
    mlp_cov_lr_final: float = 0.004
    mlp_cov_lr_delay_mult: float = 0.01
    mlp_cov_lr_max_steps: int = 30_000

    mlp_color_lr_init: float = 0.008
    mlp_color_lr_final: float = 0.00005
    mlp_color_lr_delay_mult: float = 0.01
    mlp_color_lr_max_steps: int = 30_000

    mlp_featurebank_lr_init: float = 0.01
    mlp_featurebank_lr_final: float = 0.00001
    mlp_featurebank_lr_delay_mult: float = 0.01
    mlp_featurebank_lr_max_steps: int = 30_000

    # for kernel densification
    start_stat: int = 500
    update_from: int = 1500
    update_interval: int = 100
    update_until: int = 15_000

    min_opacity: float = 0.005
    success_threshold: float = 0.8
