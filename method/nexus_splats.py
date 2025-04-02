# Copyright (c) 2024, XMU.
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#
#     * Neither the name of XMU nor the names of its contributors may be used
#       to endorse or promote products derived from this software without
#       specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# Author: Yuzhou Tang (juliantang-at-stu-dot-xmu-dot-edu-dot-cn)

import hashlib
import struct
from typing import Optional, cast, Any, TypeVar, Dict

from torch_scatter import scatter_max
from einops import repeat

import logging
import itertools
import random
from tqdm import tqdm
from functools import reduce
from operator import mul
from torch.nn import functional as F
from torch import Tensor
from torch import nn
import math
from pathlib import Path
import os
import numpy as np
from random import randint

import torch
from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement

from simple_knn._C import distCUDA2  # type: ignore
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer  # type: ignore
from . import dinov2
from .config import Config
from .types import (
    Method,
    MethodInfo,
    RenderOutput,
    ModelInfo,
    camera_model_to_int,
    Dataset,
    Cameras,
    GenericCameras,
    OptimizeEmbeddingOutput,
)

T = TypeVar("T")


def convert_image_dtype(image: np.ndarray, dtype) -> np.ndarray:
    if image.dtype == dtype:
        return image
    if image.dtype != np.uint8 and dtype != np.uint8:
        return image.astype(dtype)
    if image.dtype == np.uint8 and dtype != np.uint8:
        return image.astype(dtype) / 255.0
    if image.dtype != np.uint8 and dtype == np.uint8:
        return np.clip(image * 255.0, 0, 255).astype(np.uint8)
    raise ValueError(f"cannot convert image from {image.dtype} to {dtype}")


def get_torch_checkpoint_sha(checkpoint_data):
    sha = hashlib.sha256()

    def update(d):
        if type(d).__name__ == "Tensor" or type(d).__name__ == "Parameter":
            sha.update(d.cpu().numpy().tobytes())
        elif isinstance(d, dict):
            items = sorted(d.items(), key=lambda x: x[0])
            for k, v in items:
                update(k)
                update(v)
        elif isinstance(d, (list, tuple)):
            for v in d:
                update(v)
        elif isinstance(d, (int, float)):
            sha.update(struct.pack("f", d))
        elif isinstance(d, str):
            sha.update(d.encode("utf8"))
        elif d is None:
            sha.update("(None)".encode("utf8"))
        else:
            raise ValueError(f"Unsupported type {type(d)}")

    update(checkpoint_data)
    return sha.hexdigest()


def assert_not_none(value: Optional[T]) -> T:
    assert value is not None
    return value


def camera_project(cameras: GenericCameras[Tensor], xyz: Tensor) -> Tensor:
    eps = torch.finfo(xyz.dtype).eps  # type: ignore
    assert xyz.shape[-1] == 3

    # World -> Camera
    origins = cameras.poses[..., :3, 3]
    rotation = cameras.poses[..., :3, :3]
    # Rotation and translation
    uvw = xyz - origins
    uvw = (rotation * uvw[..., :, None]).sum(-2)

    # Camera -> Camera distorted
    uv = torch.where(uvw[..., 2:] > eps, uvw[..., :2] / uvw[..., 2:], torch.zeros_like(uvw[..., :2]))

    # We assume pinhole camera model in 3DGS anyway
    # uv = _distort(cameras.camera_models, cameras.distortion_parameters, uv, xnp=xnp)

    x, y = torch.moveaxis(uv, -1, 0)

    # Transform to image coordinates
    # Camera distorted -> Image
    fx, fy, cx, cy = torch.moveaxis(cameras.intrinsics, -1, 0)
    x = fx * x + cx
    y = fy * y + cy
    return torch.stack((x, y), -1)


def safe_state():
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))


def scale_grads(values, scale):
    grad_values = values * scale
    rest_values = values.detach() * (1 - scale)
    return grad_values + rest_values


def ssim_down(x, y, max_size=None):
    osize = x.shape[2:]
    if max_size is not None:
        scale_factor = max(max_size / x.shape[-2], max_size / x.shape[-1])
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')
    out = ssim(x, y, size_average=False).unsqueeze(1)
    if max_size is not None:
        out = F.interpolate(out, size=osize, mode='bilinear', align_corners=False)
    return out.squeeze(1)


def _ssim_parts(img1, img2, window_size=11):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    sigma1 = torch.sqrt(sigma1_sq.clamp_min(0))
    sigma2 = torch.sqrt(sigma2_sq.clamp_min(0))

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2
    C3 = C2 / 2

    luminance = (2 * mu1_mu2 + C1) / (mu1_sq + mu2_sq + C1)
    contrast = (2 * sigma1 * sigma2 + C2) / (sigma1_sq + sigma2_sq + C2)
    structure = (sigma12 + C3) / (sigma1 * sigma2 + C3)
    return luminance, contrast, structure


def msssim(x, y, max_size=None, min_size=200):
    raw_orig_size = x.shape[-2:]
    if max_size is not None:
        scale_factor = min(1, max(max_size / x.shape[-2], max_size / x.shape[-1]))
        x = F.interpolate(x, scale_factor=scale_factor, mode='area')
        y = F.interpolate(y, scale_factor=scale_factor, mode='area')

    ssim_maps = list(_ssim_parts(x, y))
    orig_size = x.shape[-2:]
    while x.shape[-2] > min_size and x.shape[-1] > min_size:
        x = F.avg_pool2d(x, 2)
        y = F.avg_pool2d(y, 2)
        ssim_maps.extend(tuple(F.interpolate(x, size=orig_size, mode='bilinear') for x in _ssim_parts(x, y)[1:]))
    out = torch.stack(ssim_maps, -1).prod(-1)
    if max_size is not None:
        out = F.interpolate(out, size=raw_orig_size, mode='bilinear')
    return out.mean(1)


def dino_downsample(x, max_size=None):
    if max_size is None:
        return x
    h, w = x.shape[2:]
    if max_size < h or max_size < w:
        scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
        nh = int(h * scale_factor)
        nw = int(w * scale_factor)
        nh = ((nh + 13) // 14) * 14
        nw = ((nw + 13) // 14) * 14
        x = F.interpolate(x, size=(nh, nw), mode='bilinear')
    return x


def get_expon_lr_func(lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param max_steps: int, the number of steps during optimization.
    :return: HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))


def getWorld2View2(R, t, translate, scale):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)


def getProjectionMatrixFromOpenCV(w, h, fx, fy, cx, cy, znear, zfar):
    z_sign = 1.0
    P = torch.zeros((4, 4))
    P[0, 0] = 2.0 * fx / w
    P[1, 1] = 2.0 * fy / h
    # P[0, 2] = (w - 2.0 * cx) / w
    P[0, 2] = (2.0 * cx - w) / w
    P[1, 2] = (2.0 * cy - h) / h
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -(zfar * znear) / (zfar - znear)
    return P


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


# SSIM
def ssim(img1, img2, window_size=11, size_average=True):
    sigma = 1.5
    channel = img1.size(-3)
    # Create window
    gauss = torch.Tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    window = window.to(img1.device).type_as(img1)

    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(-3)


def get_uniform_points_on_sphere_fibonacci(num_points, *, dtype=None, xnp=torch):
    # https://arxiv.org/pdf/0912.4540.pdf
    # Golden angle in radians
    if dtype is None:
        dtype = xnp.float32
    phi = math.pi * (3. - math.sqrt(5.))
    N = (num_points - 1) / 2
    i = xnp.linspace(-N, N, num_points, dtype=dtype)
    lat = xnp.arcsin(2.0 * i / (2 * N + 1))
    lon = phi * i

    # Spherical to cartesian
    x = xnp.cos(lon) * xnp.cos(lat)
    y = xnp.sin(lon) * xnp.cos(lat)
    z = xnp.sin(lat)
    return xnp.stack([x, y, z], -1)


@torch.no_grad()
def get_sky_points(num_points, points3D: Tensor, cameras: GenericCameras[Tensor]):
    xnp = torch
    points = get_uniform_points_on_sphere_fibonacci(num_points, xnp=xnp)
    points = points.to(points3D.device)
    mean = points3D.mean(0)[None]
    sky_distance = xnp.quantile(xnp.linalg.norm(points3D - mean, 2, -1), 0.97) * 10
    points = points * sky_distance
    points = points + mean
    gmask = torch.zeros((points.shape[0],), dtype=xnp.bool, device=points.device)
    for cam in tqdm(cameras, desc="Generating skybox"):
        uv = camera_project(cam, points[xnp.logical_not(gmask)])
        mask = xnp.logical_not(xnp.isnan(uv).any(-1))
        # Only top 2/3 of the image
        assert cam.image_sizes is not None
        mask = xnp.logical_and(mask, uv[..., -1] < 2 / 3 * cam.image_sizes[..., 1])
        gmask[xnp.logical_not(gmask)] = xnp.logical_or(gmask[xnp.logical_not(gmask)], mask)
    return points[gmask], sky_distance / 2


def add_fourier_features(features: torch.Tensor, scale=(0.0, 1.0), num_frequencies=3):
    features = (features - scale[0]) / (scale[1] - scale[0])
    freqs = 2 ** torch.linspace(0, num_frequencies - 1, num_frequencies, dtype=features.dtype, device=features.device)
    offsets = torch.tensor([0, 0.5 * math.pi], dtype=features.dtype, device=features.device)
    sin_cos_features = torch.sin(
        (2 * math.pi * (freqs[..., None, :] * features[..., None]).view(*freqs.shape[:-1], -1)).unsqueeze(-1).add(
            offsets)).view(*features.shape[:-1], -1)
    return torch.cat((features, sin_cos_features), -1)


def srgb_to_linear(img):
    limit = 0.04045

    # NOTE: torch.where is not differentiable, so we use the following
    # return torch.where(img > limit, torch.pow((img + 0.055) / 1.055, 2.4), img / 12.92)

    mask = img > limit
    out = img / 12.92
    out[mask] = torch.pow((img[mask] + 0.055) / 1.055, 2.4)
    return out


def linear_to_srgb(img):
    limit = 0.0031308

    # NOTE: torch.where is not differentiable, so we use the following
    # return torch.where(img > limit, 1.055 * (img ** (1.0 / 2.4)) - 0.055, 12.92 * img)

    mask = img > limit
    out = 12.92 * img
    out[mask] = 1.055 * torch.pow(img[mask], 1.0 / 2.4) - 0.055
    return out


def get_cameras_extent(cameras: Cameras):
    c2w = cameras.poses
    cam_centers = c2w[:, :3, 3:4]
    cam_centers = np.hstack(list(cam_centers))
    avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
    center = avg_cam_center
    dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
    diagonal = np.max(dist)
    # center = center.flatten()
    radius = diagonal * 1.1
    # translate = -center
    return radius


def _get_fourier_features(xyz, num_features=3):
    xyz = torch.from_numpy(xyz).to(dtype=torch.float32)
    xyz = xyz - xyz.mean(dim=0, keepdim=True)
    xyz = xyz / torch.quantile(xyz.abs(), 0.97, dim=0) * 0.5 + 0.5
    freqs = torch.repeat_interleave(
        2 ** torch.linspace(0, num_features - 1, num_features, dtype=xyz.dtype, device=xyz.device), 2)
    offsets = torch.tensor([0, 0.5 * math.pi] * num_features, dtype=xyz.dtype, device=xyz.device)
    feat = xyz[..., None] * freqs[None, None] * 2 * math.pi + offsets[None, None]
    feat = torch.sin(feat).view(-1, reduce(mul, feat.shape[1:]))
    return feat


class LightDecoupling(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.dist_dim = 1 if self.config.add_color_dist else 0
        feat_in = 3 * config.n_offsets
        self.mlp = nn.Sequential(
            nn.Linear(
                config.light_embedding_dim + feat_in + 3 + self.dist_dim + 6 * self.config.appearance_n_fourier_freqs,
                256),
            nn.ReLU(),
            nn.Dropout(config.appearance_dropout),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Dropout(config.appearance_dropout),
            nn.Linear(256, feat_in),
            nn.Sigmoid()
        )

    def forward(self, gembedding, aembedding, color, local_view):
        inp = torch.cat((color, gembedding, aembedding, local_view), dim=-1)
        return self.mlp(inp)


def create_gaussian_mask(height, width, center_x=0.5, center_y=0, sigma_x=0.5, sigma_y=0.8):
    # Create a grid of (x, y) coordinates
    x = torch.linspace(-1, 1, height)
    y = torch.linspace(-1, 1, width)
    x_grid, y_grid = torch.meshgrid(x, y)

    # Calculate the Gaussian mask with the center shifted down
    gaussian_mask = torch.exp(-((x_grid - center_x)**2 / (2 * sigma_x**2) + (y_grid - center_y)**2 / (2 * sigma_y**2)))

    return gaussian_mask


class UncertaintyModel(nn.Module):
    img_norm_mean: Tensor
    img_norm_std: Tensor
    gaussian_mask: Optional[Tensor]

    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        self.backbone = getattr(dinov2, config.uncertainty_backbone)(pretrained=True)
        self.patch_size = self.backbone.patch_size
        feat_in = config.n_offsets
        self.mlp = nn.Sequential(
            nn.Linear(config.transient_embedding_dim + config.uncertainty_embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, feat_in),
        )

        if config.use_boundary_penalty:
            self.register_buffer("gaussian_mask", create_gaussian_mask(256, 256))
        else:
            self.gaussian_mask = None

        img_norm_mean = torch.tensor([123.675, 116.28, 103.53], dtype=torch.float32)
        img_norm_std = torch.tensor([58.395, 57.12, 57.375], dtype=torch.float32)
        self.register_buffer("img_norm_mean", img_norm_mean / 255.)
        self.register_buffer("img_norm_std", img_norm_std / 255.)

        self._images_cache = {}

        # Freeze dinov2 backbone
        for p in self.backbone.parameters():
            p.requires_grad = False

    def forward(self, gembedding, aembedding):
        inp = torch.cat((gembedding, aembedding), dim=-1)
        return self.mlp(inp)

    def _get_pad(self, size):
        new_size = math.ceil(size / self.patch_size) * self.patch_size
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    def _get_dino_cached(self, x, cache_entry=None):
        if cache_entry is None or (cache_entry, x.shape) not in self._images_cache:
            with torch.no_grad():
                x = self.backbone.get_intermediate_layers(x, n=[self.backbone.num_heads - 1], reshape=True)[-1]
            if cache_entry is not None:
                self._images_cache[(cache_entry, x.shape)] = x.detach().cpu()
        else:
            x = self._images_cache[(cache_entry, x.shape)].to(x.device)
        return x

    def _compute_cosine_similarity(self, x, y, _x_cache=None, _y_cache=None, max_size=None):
        # Normalize data
        h, w = x.shape[2:]
        if max_size is not None and (max_size < h or max_size < w):
            assert max_size % 14 == 0, "max_size must be divisible by 14"
            scale_factor = min(max_size / x.shape[-2], max_size / x.shape[-1])
            nh = int(h * scale_factor)
            nw = int(w * scale_factor)
            nh = ((nh + 13) // 14) * 14
            nw = ((nw + 13) // 14) * 14
            x = F.interpolate(x, size=(nh, nw), mode='bilinear')
            y = F.interpolate(y, size=(nh, nw), mode='bilinear')

        x = (x - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        y = (y - self.img_norm_mean[None, :, None, None]) / self.img_norm_std[None, :, None, None]
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        x = F.pad(x, pads)
        padded_shape = x.shape
        y = F.pad(y, pads)

        with torch.no_grad():
            x = self._get_dino_cached(x, _x_cache)
            y = self._get_dino_cached(y, _y_cache)

        cosine = F.cosine_similarity(x, y, dim=1).unsqueeze(1)
        cosine: Tensor = F.interpolate(cosine, size=padded_shape[2:], mode="bilinear", align_corners=False)

        # Remove padding
        cosine = cosine[:, :, pads[2]:h + pads[2], pads[0]:w + pads[0]]
        if max_size is not None and (max_size < h or max_size < w):
            cosine = F.interpolate(cosine, size=(h, w), mode='bilinear', align_corners=False)
        return cosine.squeeze(1)

    def get_loss(self, gt, prediction, uncertainty, _cache_entry=None):
        uncertainty = F.softplus(uncertainty)
        log_uncertainty = torch.log(uncertainty + 1e-6)
        loss_mult = 1 / (2 * uncertainty.pow(2))
        if self.config.uncertainty_mode == "dino":
            gt_down = dino_downsample(gt.unsqueeze(0), max_size=350)
            prediction_down = dino_downsample(prediction.unsqueeze(0), max_size=350)
            dino_cosine = self._compute_cosine_similarity(gt_down, prediction_down, _x_cache=_cache_entry).detach()
            dino_part = (1 - dino_cosine.sub(0.5).div(0.5)).clip(0.0, 1)
            uncertainty_loss = dino_part * dino_downsample(loss_mult.unsqueeze(0).unsqueeze(0), max_size=350)
        elif self.config.uncertainty_mode == "ssim":
            _msssim = msssim(gt, prediction, max_size=400, min_size=80).unsqueeze(1)
            _msssim = ssim(gt, prediction)
            uncertainty_loss = (1 - _msssim.detach()) * loss_mult

        else:
            raise ValueError(f"Invalid uncertainty_mode: {self.config.uncertainty_mode}")

        beta = log_uncertainty.mean()
        loss = uncertainty_loss.mean() + self.config.uncertainty_regularizer_weight * beta

        height, width = gt.shape[-2:]
        loss_mult = loss_mult.detach()
        if self.config.use_boundary_penalty and self.gaussian_mask is not None:
            gaussian_mask = F.interpolate(self.gaussian_mask.unsqueeze(0).unsqueeze(0), size=(height, width), mode="bilinear")
            loss_mult = loss_mult / gaussian_mask.squeeze(0).squeeze(0)

        metrics = {
            "uncertainty_loss": uncertainty_loss.mean().item(),
            "beta": beta.item(),
        }
        return loss, metrics, loss_mult.clamp_max(3)


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


class GaussianModel(nn.Module):
    kernel: nn.Parameter
    offset: nn.Parameter
    kernel_feat: nn.Parameter
    opacity_accum: Tensor
    scaling: nn.Parameter
    rotation: nn.Parameter
    opacity: nn.Parameter
    offset_gradient_accum: Tensor
    offset_denom: Tensor
    kernel_denom: Tensor
    light_embeddings: Optional[nn.Parameter]
    appearance_embeddings: Optional[nn.Parameter]
    appearance_mlp: Optional[LightDecoupling]

    # Setup functions
    @staticmethod
    def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
        L = build_scaling_rotation(scaling_modifier * scaling, rotation)
        actual_covariance = L @ L.transpose(1, 2)
        symm = strip_symmetric(actual_covariance)
        return symm

    scaling_activation = staticmethod(torch.exp)
    scaling_inverse_activation = staticmethod(torch.log)
    covariance_activation = build_covariance_from_scaling_rotation
    opacity_activation = staticmethod(torch.sigmoid)
    inverse_opacity_activation = staticmethod(torch.special.logit)
    rotation_activation = staticmethod(F.normalize)

    def __init__(self, config: Config, training_setup: bool = True):
        super().__init__()
        self.optimizer = None
        self.percent_dense = config.percent_dense
        self.spatial_lr_scale = 0
        self.voxel_size = config.voxel_size
        self.n_offsets = config.n_offsets
        self._optimizer_state = None
        self.config = config

        self.register_parameter("kernel", cast(nn.Parameter, nn.Parameter(torch.empty(0, 3, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("offset", cast(nn.Parameter, nn.Parameter(torch.empty(0, self.n_offsets, 3, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("kernel_feat", cast(nn.Parameter, nn.Parameter(torch.empty(0, config.feat_dim, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("scaling", cast(nn.Parameter, nn.Parameter(torch.empty(0, 6, dtype=torch.float32, requires_grad=True))))
        self.register_parameter("rotation", cast(nn.Parameter, nn.Parameter(torch.empty(0, 4, dtype=torch.float32, requires_grad=False))))
        self.register_parameter("opacity", cast(nn.Parameter, nn.Parameter(torch.empty(0, 1, dtype=torch.float32, requires_grad=False))))

        self.register_buffer("opacity_accum", torch.empty(0, 1, dtype=torch.float32))
        self.register_buffer("offset_gradient_accum", torch.empty(0, 1, dtype=torch.float32))
        self.register_buffer("offset_denom", torch.empty(0, 1, dtype=torch.float32))
        self.register_buffer("kernel_denom", torch.empty(0, 1, dtype=torch.float32))

        self._dynamically_sized_props = [
            "kernel", "offset", "kernel_feat", "scaling", "rotation", "opacity",
            "opacity_accum", "offset_gradient_accum", "offset_denom", "kernel_denom",
            "appearance_embeddings", "uncertainty_embeddings",
        ]

        if self.config.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                nn.Linear(3 + 1, self.config.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.config.feat_dim, 3),
                nn.Softmax(dim=1)
            ).cuda()

        self.opacity_dist_dim = 1 if self.config.add_opacity_dist else 0
        self.mlp_opacity = nn.Sequential(
            nn.Linear(self.config.feat_dim + 3 + self.opacity_dist_dim, self.config.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.config.feat_dim, self.n_offsets),
            nn.Tanh()
        ).cuda()

        self.cov_dist_dim = 1 if self.config.add_cov_dist else 0
        self.mlp_cov = nn.Sequential(
            nn.Linear(self.config.feat_dim + 3 + self.cov_dist_dim, self.config.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.config.feat_dim, 7 * self.n_offsets),
        ).cuda()

        self.color_dist_dim = 1 if self.config.add_color_dist else 0
        self.mlp_color = nn.Sequential(
            nn.Linear(self.config.feat_dim + 3 + self.color_dist_dim, self.config.feat_dim),
            nn.ReLU(True),
            nn.Linear(self.config.feat_dim, 3 * self.n_offsets),
            nn.Sigmoid()
        ).cuda()

        if self.config.appearance_enabled:
            self.register_parameter("light_embeddings", cast(nn.Parameter, nn.Parameter(
                torch.empty(0, config.light_embedding_dim, dtype=torch.float32, requires_grad=True))))
            self.register_parameter("appearance_embeddings", cast(nn.Parameter, nn.Parameter(torch.empty(
                0, 6 * self.config.appearance_n_fourier_freqs, dtype=torch.float32, requires_grad=True))))
            self.appearance_mlp = LightDecoupling(config)
        else:
            self.light_embeddings = None
            self.appearance_embeddings = None
            self.appearance_mlp = None

        if self.config.uncertainty_enabled:
            self.register_parameter("transient_embeddings", cast(nn.Parameter, nn.Parameter(
                torch.empty(0, config.transient_embedding_dim, dtype=torch.float32, requires_grad=True)
            )))
            self.register_parameter("uncertainty_embeddings", cast(nn.Parameter, nn.Parameter(
                torch.empty(0, config.uncertainty_embedding_dim, dtype=torch.float32, requires_grad=True)))
            )
            self.uncertainty_model = UncertaintyModel(self.config)
        else:
            self.transient_embeddings = None
            self.uncertainty_embeddings = None
            self.uncertainty_model = None

        if training_setup:
            self.train()
            self._setup_optimizers()
        else:
            self.eval()

    def voxelize_sample(self, data=None, voxel_size=0.01):
        np.random.shuffle(data)
        data = np.unique(np.round(data / voxel_size), axis=0) * voxel_size
        return data

    def initialize_from_points(self, xyz, spatial_lr_scale: float):
        self.spatial_lr_scale = spatial_lr_scale
        points = xyz[::self.config.ratio]

        if self.voxel_size <= 0:
            init_points = torch.tensor(points).float().cuda()
            init_dist = distCUDA2(init_points).float().cuda()
            median_dist, _ = torch.kthvalue(init_dist, int(init_dist.shape[0] * 0.5))
            self.voxel_size = median_dist.item()
            del init_dist
            del init_points
            torch.cuda.empty_cache()

        logging.info(f'Initial voxel_size: {self.voxel_size}')

        points = self.voxelize_sample(points, voxel_size=self.voxel_size)
        fused_point_cloud = torch.tensor(np.asarray(points)).float().cuda()
        offsets = torch.zeros((fused_point_cloud.shape[0], self.n_offsets, 3)).float().cuda()
        kernels_feat = torch.zeros((fused_point_cloud.shape[0], self.config.feat_dim)).float().cuda()

        logging.info(f"Number of points at initialisation: {fused_point_cloud.shape[0]}")

        dist2 = torch.clamp_min(distCUDA2(fused_point_cloud).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 6)

        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = 0.1 * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        opacities = torch.special.logit(opacities)

        self._resize_parameters(fused_point_cloud.shape[0])
        self.kernel.data.copy_(fused_point_cloud)
        self.offset.data.copy_(offsets)
        self.kernel_feat.data.copy_(kernels_feat)
        self.scaling.data.copy_(scales)
        self.rotation.data.copy_(rots)
        self.opacity.data.copy_(opacities)
        if self.appearance_embeddings is not None:
            embeddings = _get_fourier_features(points, num_features=self.config.appearance_n_fourier_freqs)
            embeddings.add_(torch.randn_like(embeddings) * 0.0001)
            if not self.config.appearance_init_fourier:
                embeddings.normal_(0, 0.01)
            self.appearance_embeddings.data.copy_(embeddings)

    def _setup_optimizers(self):
        config = self.config

        l = [
            {'params': [self.kernel], 'lr': config.position_lr_init * self.spatial_lr_scale, "name": "kernel"},
            {'params': [self.offset], 'lr': config.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self.kernel_feat], 'lr': config.feature_lr, "name": "kernel_feat"},
            {'params': [self.opacity], 'lr': config.opacity_lr, "name": "opacity"},
            {'params': [self.scaling], 'lr': config.scaling_lr, "name": "scaling"},
            {'params': [self.rotation], 'lr': config.rotation_lr, "name": "rotation"},
            {'params': self.mlp_opacity.parameters(), 'lr': config.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': config.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': config.mlp_color_lr_init, "name": "mlp_color"},
        ]

        if config.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': config.mlp_featurebank_lr_init,
                      "name": "mlp_featurebank"})

        if self.light_embeddings is not None:
            l.append({'params': [self.light_embeddings], 'lr': config.light_embedding_lr,
                      "name": "light_embeddings", "weight_decay": config.light_embedding_regularization})

        if self.transient_embeddings is not None:
            l.append({'params': [self.transient_embeddings], 'lr': config.transient_embedding_lr,
                      "name": "transient_embeddings", "weight_decay": config.transient_regularization})

        if self.uncertainty_embeddings is not None:
            l.append({'params': [self.uncertainty_embeddings], 'lr': config.uncertainty_embedding_lr,
                      "name": "uncertainty_embeddings"})

        if self.appearance_embeddings is not None:
            l.append({'params': [self.appearance_embeddings], 'lr': config.appearance_embedding_lr,
                      "name": "appearance_embeddings"})

        if self.appearance_mlp is not None:
            l.append({'params': list(self.appearance_mlp.parameters()), 'lr': config.appearance_mlp_lr,
                      "name": "appearance_mlp"})

        if self.uncertainty_model is not None:
            l.append({'params': list(self.uncertainty_model.parameters()), 'lr': config.uncertainty_lr,
                      "name": "uncertainty_model"})

        self.optimizer = torch.optim.Adam(l, lr=1.0, eps=1e-15)

        self.kernel_scheduler_args = get_expon_lr_func(lr_init=config.position_lr_init * self.spatial_lr_scale,
                                                       lr_final=config.position_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=config.position_lr_delay_mult,
                                                       max_steps=config.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=config.offset_lr_init * self.spatial_lr_scale,
                                                       lr_final=config.offset_lr_final * self.spatial_lr_scale,
                                                       lr_delay_mult=config.offset_lr_delay_mult,
                                                       max_steps=config.offset_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=config.mlp_opacity_lr_init,
                                                            lr_final=config.mlp_opacity_lr_final,
                                                            lr_delay_mult=config.mlp_opacity_lr_delay_mult,
                                                            max_steps=config.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=config.mlp_cov_lr_init,
                                                        lr_final=config.mlp_cov_lr_final,
                                                        lr_delay_mult=config.mlp_cov_lr_delay_mult,
                                                        max_steps=config.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=config.mlp_color_lr_init,
                                                          lr_final=config.mlp_color_lr_final,
                                                          lr_delay_mult=config.mlp_color_lr_delay_mult,
                                                          max_steps=config.mlp_color_lr_max_steps)
        if config.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=config.mlp_featurebank_lr_init,
                                                                    lr_final=config.mlp_featurebank_lr_final,
                                                                    lr_delay_mult=config.mlp_featurebank_lr_delay_mult,
                                                                    max_steps=config.mlp_featurebank_lr_max_steps)

    def set_num_training_images(self, num_images):
        if self.light_embeddings is not None:
            self._resize_parameter("light_embeddings", (num_images, self.light_embeddings.shape[1]))
            self.light_embeddings.data.normal_(0, 0.01)
        if self.transient_embeddings is not None:
            self._resize_parameter("transient_embeddings", (num_images, self.transient_embeddings.shape[1]))
            self.transient_embeddings.data.normal_(0, 0.01)

    @property
    def get_rotation(self):
        return self.rotation_activation(self.rotation)

    @property
    def get_scaling(self):
        return 1.0 * self.scaling_activation(self.scaling)

    @property
    def get_opacity(self):
        return self.opacity_activation(self.opacity)

    def get_covariance(self, scaling_modifier=1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self.rotation)

    def _resize_parameter(self, name, shape):
        tensor = getattr(self, name)
        new_tensor = torch.zeros(shape, device=tensor.device, dtype=tensor.dtype)
        new_tensor[:tensor.shape[0]] = tensor
        if isinstance(tensor, nn.Parameter):
            new_param = nn.Parameter(new_tensor.requires_grad_(True))
            if self.optimizer is not None:
                for group in self.optimizer.param_groups:
                    if group["name"] == name:
                        stored_state = self.optimizer.state.get(group['params'][0], None)
                        if stored_state is not None:
                            stored_state["exp_avg"] = torch.zeros_like(new_tensor)
                            stored_state["exp_avg_sq"] = torch.zeros_like(new_tensor)
                            del self.optimizer.state[group['params'][0]]
                            self.optimizer.state[new_param] = stored_state  # type: ignore
                        group["params"][0] = new_param
                        break
                else:
                    raise ValueError(f"Parameter {name} not found in optimizer")
            setattr(self, name, new_param)
        else:
            self.register_buffer(name, new_tensor)

    def _resize_parameters(self, num_points):
        for name in self._dynamically_sized_props:
            if getattr(self, name, None) is None:
                continue
            if name in ("offset_gradient_accum", "offset_denom"):
                self._resize_parameter(name, (num_points * self.n_offsets, *getattr(self, name).shape[1:]))
            else:
                self._resize_parameter(name, (num_points, *getattr(self, name).shape[1:]))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # Resize all buffers to match the new state_dict
        self._resize_parameters(state_dict["kernel"].shape[0])
        if self.light_embeddings is not None:
            self._resize_parameter("light_embeddings", state_dict["light_embeddings"].shape)
        if self.transient_embeddings is not None:
            self._resize_parameter("transient_embeddings", state_dict["transient_embeddings"].shape)
        optimizer = state_dict.pop("optimizer")
        if strict and optimizer is None:
            missing_keys.append("optimizer")
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(optimizer)
        else:
            self._optimizer_state = optimizer

    def state_dict(self, *args, destination: Any = None, prefix='', keep_vars=False):
        state = super().state_dict(destination=destination, prefix=prefix, keep_vars=keep_vars)
        if self.optimizer is None:
            state["optimizer"] = self._optimizer_state
        else:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def get_light_embedding(self, train_image_id=None):
        if self.light_embeddings is None:
            return None
        if train_image_id is not None:
            return self.light_embeddings[train_image_id]
        return torch.zeros_like(self.light_embeddings[0])

    def get_transient_embedding(self, train_image_id=None):
        if self.transient_embeddings is None:
            return None
        if train_image_id is not None:
            return self.transient_embeddings[train_image_id]
        return torch.zeros_like(self.transient_embeddings[0])

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        assert self.optimizer is not None, "Not set up for training"
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "kernel":
                lr = self.kernel_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.config.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr

    def save_ply(self, path):
        def construct_list_of_attributes():
            l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
            for i in range(self.offset.shape[1] * self.offset.shape[2]):
                l.append('f_offset_{}'.format(i))
            for i in range(self.kernel_feat.shape[1]):
                l.append('f_kernel_feat_{}'.format(i))
            l.append('opacity')
            for i in range(self.scaling.shape[1]):
                l.append('scale_{}'.format(i))
            for i in range(self.rotation.shape[1]):
                l.append('rot_{}'.format(i))
            return l

        os.makedirs(os.path.dirname(path), exist_ok=True)

        kernel = self.kernel.detach().cpu().numpy()
        normals = np.zeros_like(kernel)
        kernel_feat = self.kernel_feat.detach().cpu().numpy()
        offset = self.offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self.opacity.detach().cpu().numpy()
        scale = self.scaling.detach().cpu().numpy()
        rotation = self.rotation.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in construct_list_of_attributes()]

        elements = np.empty(kernel.shape[0], dtype=dtype_full)
        attributes = np.concatenate((kernel, normals, offset, kernel_feat, opacities, scale, rotation), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors

    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self._dynamically_sized_props:
                continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(
                    torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask,
                        kernel_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[kernel_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # update kernel visiting statis
        self.kernel_denom[kernel_visible_mask] += 1

        # update neural gaussian statis
        kernel_visible_mask = kernel_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[kernel_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter, :2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_kernel_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] not in self._dynamically_sized_props:
                continue
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]
                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:, 3:]
                    temp[temp > 0.05] = 0.05
                    group["params"][0][:, 3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

    def prune_kernel(self, mask):
        valid_points_mask = ~mask
        optimizable_tensors = self._prune_kernel_optimizer(valid_points_mask)

        self.kernel = optimizable_tensors["kernel"]
        self.offset = optimizable_tensors["offset"]
        self.kernel_feat = optimizable_tensors["kernel_feat"]
        self.opacity = optimizable_tensors["opacity"]
        self.scaling = optimizable_tensors["scaling"]
        self.rotation = optimizable_tensors["rotation"]
        if self.config.appearance_enabled:
            self.appearance_embeddings = optimizable_tensors["appearance_embeddings"]
        if self.config.uncertainty_enabled:
            self.uncertainty_embeddings = optimizable_tensors["uncertainty_embeddings"]

    def kernel_growing(self, grads, threshold, offset_mask):
        init_length = self.kernel.shape[0] * self.n_offsets
        for i in range(self.config.update_depth):
            # update threshold
            cur_threshold = threshold * ((self.config.update_hierachy_factor // 2) ** i)
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold)
            candidate_mask = torch.logical_and(candidate_mask, offset_mask)

            # random pick
            rand_mask = torch.rand_like(candidate_mask.float()) > (0.5 ** (i + 1))
            rand_mask = rand_mask.cuda()
            candidate_mask = torch.logical_and(candidate_mask, rand_mask)

            length_inc = self.kernel.shape[0] * self.n_offsets - init_length
            if length_inc == 0:
                if i > 0:
                    continue
            else:
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            all_xyz = self.kernel.unsqueeze(dim=1) + self.offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            size_factor = self.config.update_init_factor // (self.config.update_hierachy_factor ** i)
            cur_size = self.voxel_size * size_factor

            grid_coords = torch.round(self.kernel / cur_size).int()

            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round(selected_xyz / cur_size).int()

            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True,
                                                                        dim=0)

            # split data for reducing peak memory calling
            use_chunk = True
            if use_chunk:
                chunk_size = 4096
                max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
                remove_duplicates_list = []
                for i in range(max_iters):
                    cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) ==
                                             grid_coords[i * chunk_size:(i + 1) * chunk_size, :]).all(-1).any(-1).view(-1)
                    remove_duplicates_list.append(cur_remove_duplicates)

                remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
            else:
                remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)

            remove_duplicates = ~remove_duplicates
            candidate_kernel = selected_grid_coords_unique[remove_duplicates] * cur_size

            if candidate_kernel.shape[0] > 0:
                new_scaling = torch.ones_like(candidate_kernel).repeat([1, 2]).float().cuda() * cur_size  # *0.05
                new_scaling = torch.log(new_scaling)
                new_rotation = torch.zeros([candidate_kernel.shape[0], 4], device=candidate_kernel.device).float()
                new_rotation[:, 0] = 1.0
                new_opacities = torch.special.logit(0.1 * torch.ones((candidate_kernel.shape[0], 1), dtype=torch.float, device="cuda"))
                new_feat = self.kernel_feat.unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).view([-1, self.config.feat_dim])[candidate_mask]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_offsets = torch.zeros_like(candidate_kernel).unsqueeze(dim=1).repeat([1, self.n_offsets, 1]).float().cuda()

                d = {
                    "kernel": candidate_kernel,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "kernel_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                }

                temp_kernel_denom = torch.cat([self.kernel_denom, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.kernel_denom
                self.kernel_denom = temp_kernel_denom

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self.kernel = optimizable_tensors["kernel"]
                self.scaling = optimizable_tensors["scaling"]
                self.rotation = optimizable_tensors["rotation"]
                self.kernel_feat = optimizable_tensors["kernel_feat"]
                self.offset = optimizable_tensors["offset"]
                self.opacity = optimizable_tensors["opacity"]
                if self.config.appearance_enabled:
                    self.appearance_embeddings = optimizable_tensors["appearance_embeddings"]
                if self.config.uncertainty_enabled:
                    self.uncertainty_embeddings = optimizable_tensors["uncertainty_embeddings"]

    def adjust_kernel(self, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, min_opacity=0.005):
        # # adding kernels
        grads = self.offset_gradient_accum / self.offset_denom  # [N*k, 1]
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)

        self.kernel_growing(grads_norm, grad_threshold, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_denom = torch.zeros([self.kernel.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32, device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_denom], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.kernel.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32, device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune kernels
        prune_mask = (self.opacity_accum < min_opacity * self.kernel_denom).squeeze(dim=1)
        kernels_mask = (self.kernel_denom > check_interval * success_threshold).squeeze(dim=1)  # [N, 1]
        prune_mask = torch.logical_and(prune_mask, kernels_mask)  # [N]

        # update offset_denom
        offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
        offset_denom = offset_denom.view([-1, 1])
        del self.offset_denom
        self.offset_denom = offset_denom

        offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
        offset_gradient_accum = offset_gradient_accum.view([-1, 1])
        del self.offset_gradient_accum
        self.offset_gradient_accum = offset_gradient_accum

        # update opacity accum
        if kernels_mask.sum() > 0:
            self.opacity_accum[kernels_mask] = torch.zeros([kernels_mask.sum(), 1], device='cuda').float()
            self.kernel_denom[kernels_mask] = torch.zeros([kernels_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_kernel_denom = self.kernel_denom[~prune_mask]
        del self.kernel_denom
        self.kernel_denom = temp_kernel_denom

        if prune_mask.shape[0] > 0:
            self.prune_kernel(prune_mask)

    def _generate_neural_gaussians(self, camera_center, visible_mask, light_embedding=None, transient_embedding=None):
        # view frustum filtering for acceleration
        feat = self.kernel_feat[visible_mask]
        kernel = self.kernel[visible_mask]
        grid_offsets = self.offset[visible_mask]
        grid_scaling = self.get_scaling[visible_mask]

        # get view properties for kernel
        ob_view = kernel - camera_center
        # dist
        ob_dist = ob_view.norm(dim=1, keepdim=True)
        # view
        ob_view = ob_view / ob_dist

        # view-adaptive feature
        if self.config.use_feat_bank:
            cat_view = torch.cat([ob_view, ob_dist], dim=1)
            bank_weight = self.mlp_feature_bank(cat_view).unsqueeze(dim=1)  # [n, 1, 3]

            # multi-resolution feat
            feat = feat.unsqueeze(dim=-1)
            feat = feat[:, ::4, :1].repeat([1, 4, 1]) * bank_weight[:, :, :1] + \
                   feat[:, ::2, :1].repeat([1, 2, 1]) * bank_weight[:, :, 1:2] + \
                   feat[:, ::1, :1] * bank_weight[:, :, 2:]
            feat = feat.squeeze(dim=-1)  # [n, c]

        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)  # [N, c+3+1]
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)  # [N, c+3]

        # get offset's opacity
        if self.config.add_opacity_dist:
            neural_opacity = self.mlp_opacity(cat_local_view)  # [N, k]
        else:
            neural_opacity = self.mlp_opacity(cat_local_view_wodist)

        # opacity mask generation
        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0)
        mask = mask.view(-1)

        # select opacity
        opacity = neural_opacity[mask]

        # get offset's color
        if self.config.add_color_dist:
            color = self.mlp_color(cat_local_view)
        else:
            color = self.mlp_color(cat_local_view_wodist)

        if self.config.appearance_enabled:
            embedding_expanded = assert_not_none(light_embedding)[None].repeat(len(kernel), 1)
            assert self.appearance_mlp is not None
            assert self.appearance_embeddings is not None
            local_view = torch.cat([ob_view, ob_dist], dim=1) if self.config.add_color_dist else ob_view
            color_toned = self.appearance_mlp(self.appearance_embeddings[visible_mask], embedding_expanded, color, local_view)
            color_toned = color_toned.reshape([kernel.shape[0] * self.n_offsets, 3])[mask]
        else:
            color_toned = None

        color = color.reshape([kernel.shape[0] * self.n_offsets, 3])  # [mask]

        if self.config.uncertainty_enabled and transient_embedding is not None:
            embedding_expanded = transient_embedding[None].repeat(len(kernel), 1)
            assert self.uncertainty_model is not None
            assert self.uncertainty_embeddings is not None
            uncertainty = self.uncertainty_model(self.uncertainty_embeddings[visible_mask], embedding_expanded)
            uncertainty = uncertainty.reshape([kernel.shape[0] * self.n_offsets, 1])[mask]
        else:
            uncertainty = None

        # get offset's cov
        if self.config.add_cov_dist:
            scale_rot = self.mlp_cov(cat_local_view)
        else:
            scale_rot = self.mlp_cov(cat_local_view_wodist)
        scale_rot = scale_rot.reshape([kernel.shape[0] * self.n_offsets, 7])  # [mask]

        # offsets
        offsets = grid_offsets.view([-1, 3])  # [mask]

        # combine for parallel masking
        concatenated = torch.cat([grid_scaling, kernel], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=self.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_kernel, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        # post-process cov
        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])  # * (1+torch.sigmoid(repeat_dist))
        rot = self.rotation_activation(scale_rot[:, 3:7])

        # post-process offsets to get centers for gaussians
        offsets = offsets * scaling_repeat[:, :3]
        xyz = repeat_kernel + offsets

        return xyz, color, color_toned, opacity, scaling, rot, neural_opacity, mask, uncertainty

    def render_internal(self,
                        viewpoint_camera: Cameras,
                        config: Config,
                        *,
                        scaling_modifier=1.0,
                        light_embedding: Optional[torch.Tensor],
                        transient_embedding: Optional[torch.Tensor] = None,
                        prefilter_voxel: bool = True,
                        retain_grad: bool = False,
                        return_raw: bool = True,
                        render_depth: bool = False):
        """
        Render the scene.
        """
        device = self.kernel.device
        assert len(viewpoint_camera.poses.shape) == 2, "Expected a single camera"
        assert viewpoint_camera.image_sizes is not None, "Expected image sizes to be set"
        pose = np.copy(viewpoint_camera.poses)
        pose = np.concatenate([pose, np.array([[0, 0, 0, 1]], dtype=pose.dtype)], axis=0)
        pose = np.linalg.inv(pose)
        R = pose[:3, :3]
        T = pose[:3, 3]
        R = np.transpose(R)
        width, height = viewpoint_camera.image_sizes
        fx, fy, cx, cy = viewpoint_camera.intrinsics

        zfar = 100.0
        znear = 0.01
        trans = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        scale = 1.0

        world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).to(device=device)
        projection_matrix = (getProjectionMatrixFromOpenCV(width, height, fx, fy, cx, cy, znear, zfar).transpose(0, 1).to(device=device))
        full_proj_transform = (world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))).squeeze(0)
        camera_center = world_view_transform.inverse()[3, :3]

        # Set up rasterization configuration
        FoVx = focal2fov(float(fx), float(width))
        FoVy = focal2fov(float(fy), float(height))
        tanfovx = math.tan(FoVx * 0.5)
        tanfovy = math.tan(FoVy * 0.5)

        settings = {
            "image_height": int(height),
            "image_width": int(width),
            "tanfovx": tanfovx,
            "tanfovy": tanfovy,
            "bg": torch.zeros((3,), dtype=torch.float32, device="cuda"),
            "scale_modifier": scaling_modifier,
            "viewmatrix": world_view_transform,
            "projmatrix": full_proj_transform,
            "campos": camera_center,
            "debug": config.debug,
        }

        if prefilter_voxel:
            visible_mask = self._prefilter_voxel(settings)
        else:
            visible_mask = torch.ones(self.kernel.shape[0], dtype=torch.bool, device=device)

        xyz, color, color_toned, opacity, scaling, rot, neural_opacity, mask, uncertainty = self._generate_neural_gaussians(
            camera_center, visible_mask, light_embedding, transient_embedding)

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(xyz, dtype=self.kernel.dtype, requires_grad=True, device=device) + 0
        if retain_grad:
            try:
                screenspace_points.retain_grad()
            except Exception:
                pass

        raster_settings = GaussianRasterizationSettings(
            image_height=int(height),
            image_width=int(width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=torch.zeros((3,), dtype=torch.float32, device="cuda"),
            scale_modifier=scaling_modifier,
            viewmatrix=world_view_transform,
            projmatrix=full_proj_transform,
            sh_degree=1,
            campos=camera_center,
            prefiltered=False,
            debug=config.debug,
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # Rasterize visible Gaussians to image, obtain their radii (on screen).
        rendered_image = None
        raw_rendered_image = None
        radii = None

        if not self.config.appearance_enabled or (self.config.appearance_separate_tuned_color and return_raw):
            raw_rendered_image, radii = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=color,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None)
            rendered_image = raw_rendered_image

        if self.config.appearance_enabled:
            rendered_image, _radii = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                shs=None,
                colors_precomp=color_toned,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                cov3D_precomp=None)

            radii = _radii if radii is None else radii
            raw_rendered_image = rendered_image if not self.config.appearance_separate_tuned_color else raw_rendered_image

        visibility_filter = assert_not_none(radii) > 0
        out = {"render": rendered_image,
               "viewspace_points": screenspace_points,
               "visibility_filter": visibility_filter,
               "radii": assert_not_none(radii)}

        if self.training:
            out["selection_mask"] = mask
            out["neural_opacity"] = neural_opacity
            out["scaling"] = scaling
            out["voxel_visible_mask"] = visible_mask

        if return_raw:
            out["raw_render"] = raw_rendered_image

        if self.config.uncertainty_enabled:
            if uncertainty is not None:
                uncertainty = uncertainty.repeat(1, 3)
                out["uncertainty"] = rasterizer(
                    means3D=xyz,
                    means2D=screenspace_points,
                    colors_precomp=uncertainty,
                    opacities=opacity,
                    scales=scaling,
                    rotations=rot,
                    shs=None,
                    cov3D_precomp=None
                )[0][0]
            else:
                out["uncertainty"] = None

        if render_depth:
            dist = torch.norm(xyz - camera_center[None], dim=-1).unsqueeze(-1).repeat(1, 3)
            out["depth"] = rasterizer(
                means3D=xyz,
                means2D=screenspace_points,
                colors_precomp=dist,
                opacities=opacity,
                scales=scaling,
                rotations=rot,
                shs=None,
                cov3D_precomp=None)[0][0]
        return out

    def _prefilter_voxel(self, settings: Dict[str, Any]):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """
        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(self.kernel, dtype=self.kernel.dtype, requires_grad=True,
                                              device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

        # Set up rasterization configuration
        raster_settings = GaussianRasterizationSettings(
            image_height=int(settings["image_height"]),
            image_width=int(settings["image_width"]),
            tanfovx=settings["tanfovx"],
            tanfovy=settings["tanfovy"],
            bg=settings["bg"],
            scale_modifier=settings["scale_modifier"],
            viewmatrix=settings["viewmatrix"],
            projmatrix=settings["projmatrix"],
            sh_degree=1,
            campos=settings["campos"],
            prefiltered=False,
            debug=settings["debug"],
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = self.get_scaling
        rotations = self.get_rotation
        radii_pure = rasterizer.visible_filter(means3D=self.kernel, scales=scales[:, :3],
                                               rotations=rotations, cov3D_precomp=None)
        return radii_pure > 0


class NexusSplats(Method):
    config_overrides: Optional[dict] = None

    def __init__(self, *, checkpoint: Optional[Path] = None,
                 train_dataset: Optional[Dataset] = None, config_overrides: Optional[dict] = None):
        self.optimizer = None
        self.checkpoint = checkpoint
        self.step = 0

        # Setup parameters
        load_state_dict = None
        self.config: Config = OmegaConf.structured(Config)
        self._loaded_step = None
        if checkpoint is not None:
            if not os.path.exists(checkpoint):
                raise RuntimeError(f"Model directory {checkpoint} does not exist")
            logging.info(f"Loading config file {os.path.join(checkpoint, 'config.yaml')}")
            self.config = cast(Config, OmegaConf.merge(self.config, OmegaConf.load(os.path.join(checkpoint, "config.yaml"))))
            self._loaded_step = self.step = sorted(
                int(x[x.find("-") + 1: x.find(".")]) for x in os.listdir(str(self.checkpoint)) if
                x.startswith("chkpnt-"))[-1]
            state_dict_name = f"chkpnt-{self._loaded_step}.pth"
            load_state_dict = torch.load(os.path.join(checkpoint, state_dict_name))
        else:
            if config_overrides is not None:
                if "config" in config_overrides:
                    config_overrides = dict(config_overrides)
                    config_file = config_overrides.pop("config")
                else:
                    config_file = "default.yml"
                logging.info(f"Loading config file {config_file}")
                config_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "configs", config_file)
                self.config = cast(Config, OmegaConf.merge(self.config, OmegaConf.load(config_file)))
                oc_config_overrides = OmegaConf.from_dotlist([f"{k}={v}" for k, v in config_overrides.items()])
                self.config = cast(Config, OmegaConf.merge(self.config, oc_config_overrides))

        self._viewpoint_stack = []

        self.train_cameras = None
        self.cameras_extent = None

        # Used for saving
        self._json_cameras = None

        # Initialize system state (RNG)
        safe_state()

        device = torch.device("cuda")
        bg_color = [0, 0, 0]
        self.background = torch.tensor(bg_color, dtype=torch.float32, device=device)
        self.model = GaussianModel(self.config, train_dataset is not None).to(device)
        self._sky_distance = None
        if train_dataset is not None:
            self._setup_train(train_dataset, load_state_dict)
        elif load_state_dict is not None:
            self.model.load_state_dict(load_state_dict, strict=False)

    def _setup_train(self, train_dataset: Dataset, load_state_dict):
        points3D_xyz = train_dataset["points3D_xyz"]
        assert points3D_xyz is not None, "Train points3D_xyz are required for training"
        if self.checkpoint is None and self.config.num_sky_gaussians:
            th_cameras = train_dataset["cameras"].apply(lambda x, _: torch.from_numpy(x).cuda())
            skybox, self._sky_distance = get_sky_points(self.config.num_sky_gaussians, torch.from_numpy(points3D_xyz).cuda(), th_cameras)
            skybox = skybox.cpu().numpy()
            logging.info(f"Adding skybox with {skybox.shape[0]} points")
            train_dataset = train_dataset.copy()
            train_dataset["points3D_xyz"] = np.concatenate((points3D_xyz, skybox))

        self.cameras_extent = get_cameras_extent(train_dataset["cameras"])
        self.train_cameras = train_dataset["cameras"]
        self.train_images = [
            torch.from_numpy(np.moveaxis(convert_image_dtype(img, np.float32), -1, 0)) for img in
            train_dataset["images"]
        ]
        self.train_sampling_masks = None
        if train_dataset["sampling_masks"] is not None:
            self.train_sampling_masks = [
                torch.from_numpy(convert_image_dtype(img, np.float32)[None]) for img in train_dataset["sampling_masks"]
            ]
        # Clear memory
        train_dataset["images"] = None  # type: ignore
        train_dataset["sampling_masks"] = None  # type: ignore

        # Setup model
        if self.checkpoint is None:
            xyz = train_dataset["points3D_xyz"]
            self.model.initialize_from_points(xyz, self.cameras_extent)
            self.model.set_num_training_images(len(train_dataset["cameras"]))
        else:
            self.model.load_state_dict(load_state_dict, strict=False)

        self._viewpoint_stack = []

    @classmethod
    def get_method_info(cls) -> MethodInfo:
        return MethodInfo(
            method_id="nexus-splats",  # Will be filled by the registry
            required_features=frozenset(("color", "points3D_xyz")),
            supported_camera_models=frozenset(("pinhole",)),
        )

    def get_info(self) -> ModelInfo:
        return ModelInfo(
            **self.get_method_info(),
            num_iterations=self.config.iterations,
            loaded_step=self._loaded_step,
        )

    def optimize_embedding(self, dataset: Dataset, *, embedding: Optional[np.ndarray] = None) -> OptimizeEmbeddingOutput:
        device = self.model.kernel.device
        camera = dataset["cameras"].item()
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"

        self.model.eval()
        i = 0
        losses, psnrs, mses = [], [], []

        embedding = (
            torch.from_numpy(embedding).to(device) if embedding is not None else self.model.get_light_embedding(None)
        )
        if self.config.appearance_enabled:
            light_embedding_param = torch.nn.Parameter(assert_not_none(embedding).requires_grad_(True))
            optimizer = torch.optim.Adam([light_embedding_param], lr=self.config.light_embedding_optim_lr)

            gt_image = torch.tensor(convert_image_dtype(dataset["images"][i], np.float32), dtype=torch.float32,
                                    device=device).permute(2, 0, 1)
            gt_mask = torch.tensor(convert_image_dtype(dataset["sampling_masks"][i], np.float32), dtype=torch.float32,
                                   device=device)[..., None].permute(2, 0, 1) if dataset["sampling_masks"] is not None else None

            with torch.enable_grad():
                app_optim_type = self.config.appearance_optim_type
                for _ in range(self.config.light_embedding_optim_iters):
                    optimizer.zero_grad()
                    render_pkg = self.model.render_internal(camera, config=self.config,
                                                            light_embedding=light_embedding_param,
                                                            transient_embedding=None)
                    image = render_pkg["render"]
                    if gt_mask is not None:
                        image = scale_grads(image, gt_mask.float())

                    mse = F.mse_loss(image, gt_image)

                    if app_optim_type == "mse":
                        loss = mse
                    elif app_optim_type == "dssim+l1":
                        Ll1 = F.l1_loss(image, gt_image)
                        ssim_value = ssim(image, gt_image, size_average=True)
                        loss = (
                                (1.0 - self.config.lambda_dssim) * Ll1 +
                                self.config.lambda_dssim * (1.0 - ssim_value)
                        )
                    else:
                        raise ValueError(f"Unknown appearance optimization type {app_optim_type}")
                    loss.backward()
                    optimizer.step()

                    losses.append(loss.detach().cpu().item())
                    mses.append(mse.detach().cpu().item())
                    psnrs.append(20 * math.log10(1.0) - 10 * torch.log10(mse).detach().cpu().item())

            if self.model.optimizer is not None:
                self.model.optimizer.zero_grad()
            embedding_np = light_embedding_param.detach().cpu().numpy()
            return {
                "embedding": embedding_np,
                "metrics": {
                    "psnr": psnrs,
                    "mse": mses,
                    "loss": losses,
                }
            }
        else:
            raise NotImplementedError("Trying to optimize embedding with appearance_enabled=False")

    def render(self, camera: Cameras, options=None, **kwargs) -> RenderOutput:
        del kwargs
        camera = camera.item()
        device = self.model.kernel.device
        assert camera.camera_models == camera_model_to_int("pinhole"), "Only pinhole cameras supported"
        render_depth = False
        if options is not None and "depth" in options.get("outputs", ()):
            render_depth = True

        self.model.eval()
        with torch.no_grad():
            _np_embedding = (options or {}).get("embedding", None)
            embedding = (
                torch.from_numpy(_np_embedding)
                if _np_embedding is not None else self.model.get_light_embedding(None)
            )
            del _np_embedding
            embedding = embedding.to(device) if embedding is not None else None
            out = self.model.render_internal(camera, config=self.config, light_embedding=embedding, render_depth=render_depth)
            image = out["render"]
            image = torch.clamp(image, 0.0, 1.0).nan_to_num_(0.0)
            color = image.detach().permute(1, 2, 0).cpu().numpy()

            ret_out: RenderOutput = {
                "color": color,
            }
            if out.get("depth") is not None:
                ret_out["depth"] = out["depth"].detach().cpu().numpy()
            return ret_out

    def _get_viewpoint_stack(self, step: int):
        assert self.train_cameras is not None, "Method not initialized"
        generator = torch.Generator()
        generator.manual_seed(step // 300)
        num_images = 30
        indices = torch.multinomial(
            torch.ones(len(self.train_cameras), dtype=torch.float32),
            num_images,
            generator=generator,
        )
        return [self.train_cameras[i] for i in indices]

    def train_iteration(self, step):
        assert self.train_cameras is not None, "Method not initialized"
        assert self.model.optimizer is not None, "Method not initialized"
        assert self.train_cameras.image_sizes is not None, "Image sizes are required for training"

        self.step = step
        iteration = step + 1  # Gaussian Splatting is 1-indexed
        del step

        self.model.train()
        self.model.update_learning_rate(iteration)
        device = self.model.kernel.device

        # Pick a random Camera
        if not self._viewpoint_stack:
            self._viewpoint_stack = list(range(len(self.train_cameras)))
        camera_id = self._viewpoint_stack.pop(randint(0, len(self._viewpoint_stack) - 1))
        viewpoint_cam = self.train_cameras[camera_id]
        assert viewpoint_cam.image_sizes is not None, "Image sizes are required for training"
        image_width, image_height = viewpoint_cam.image_sizes

        # Render
        # NOTE: random background color is not supported

        light_embedding = self.model.get_light_embedding(train_image_id=camera_id)
        transient_embedding = self.model.get_transient_embedding(train_image_id=camera_id)
        retain_grad = iteration < self.config.update_until
        render_pkg = self.model.render_internal(viewpoint_cam, config=self.config, light_embedding=light_embedding,
                                                transient_embedding=transient_embedding, retain_grad=retain_grad)
        image_toned, image, scaling, opacity = render_pkg["render"], render_pkg["raw_render"], render_pkg["scaling"], render_pkg["neural_opacity"]
        viewspace_point_tensor, visibility_filter = render_pkg["viewspace_points"], render_pkg["visibility_filter"]
        offset_selection_mask, voxel_visible_mask = render_pkg["selection_mask"], render_pkg["voxel_visible_mask"]

        # Apply exposure modelling
        assert image.shape == (3, image_height, image_width), f"image.shape={image.shape}"

        # Loss
        gt_image = self.train_images[camera_id].to(device)
        sampling_mask = self.train_sampling_masks[camera_id].to(device) if self.train_sampling_masks is not None else None

        # Apply mask
        if sampling_mask is not None:
            image = scale_grads(image, sampling_mask)
            image_toned = scale_grads(image_toned, sampling_mask)

        metrics = {}

        if self.config.uncertainty_enabled:
            uncertainty = render_pkg["uncertainty"]
            uncertainty_loss, metrics, mask = self.model.uncertainty_model.get_loss(gt_image,
                                                                                    image_toned,
                                                                                    uncertainty,
                                                                                    _cache_entry=('train', camera_id))
            mask = (mask > 1).to(dtype=mask.dtype)
            # mask = mask.clamp_max(1)

            if iteration < self.config.uncertainty_warmup_start:
                mask = 1
            elif iteration < self.config.uncertainty_warmup_start + self.config.uncertainty_warmup_iters:
                p = (iteration - self.config.uncertainty_warmup_start) / self.config.uncertainty_warmup_iters
                mask = 1 + p * (mask - 1)
            if self.config.uncertainty_center_mult:
                mask = mask.sub(mask.mean() - 1).clamp(0, 2)
            if self.config.uncertainty_scale_grad:
                image = scale_grads(image, mask)
                image_toned = scale_grads(image_toned, mask)
                mask = 1
        else:
            mask = 1.0
            uncertainty_loss = 0.0

        Ll1 = F.l1_loss(image_toned, gt_image, reduction='none')
        ssim_value = 1.0 - ssim(image, gt_image, size_average=False)
        scaling_reg = scaling.prod(dim=1).mean()

        loss = (
                (1.0 - self.config.lambda_dssim) * (Ll1 * mask).mean() +
                self.config.lambda_dssim * (ssim_value * mask).mean() +
                uncertainty_loss + scaling_reg * 0.01
        )
        loss.backward()

        with torch.no_grad():
            mse = (image_toned - gt_image).pow_(2)
            psnr_value = 20 * math.log10(1.) - 10 * torch.log10(mse.mean())
            metrics.update({
                "l1_loss": Ll1.detach().mean().cpu().item(),
                "ssim": ssim_value.detach().mean().cpu().item(),
                "mse": mse.detach().mean().cpu().item(),
                "loss": loss.detach().cpu().item(),
                "psnr": psnr_value.detach().cpu().item(),
                "num_kernels": len(self.model.kernel),
            })

            def _reduce_masked(tensor, mask):
                return ((tensor * mask).sum() / mask.sum()).detach().cpu().item()

            if sampling_mask is not None:
                mask_percentage = sampling_mask.detach().mean().cpu().item()
                metrics["mask_percentage"] = mask_percentage
                metrics["ssim_masked"] = _reduce_masked(ssim_value, sampling_mask)
                metrics["mse_masked"] = masked_mse = _reduce_masked(mse, sampling_mask)
                masked_psnr_value = 20 * math.log10(1.) - 10 * math.log10(masked_mse)
                metrics["psnr_masked"] = masked_psnr_value
                metrics["l1_loss_masked"] = _reduce_masked(Ll1, sampling_mask)

            # densification
            if self.config.update_until > iteration > self.config.start_stat:
                # add statis
                self.model.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask)

                # densification
                if iteration > self.config.update_from and iteration % self.config.update_interval == 0:
                    self.model.adjust_kernel(check_interval=self.config.update_interval,
                                             success_threshold=self.config.success_threshold,
                                             grad_threshold=self.config.densify_grad_threshold,
                                             min_opacity=self.config.min_opacity)

            elif iteration == self.config.update_until:
                del self.model.opacity_accum
                del self.model.offset_gradient_accum
                del self.model.offset_denom
                del self.model.kernel_denom

            stats = torch.cuda.memory_stats()
            allocated_memory = stats["allocated_bytes.all.current"]
            reserved_memory = stats["reserved_bytes.all.current"]
            allocated = (allocated_memory + reserved_memory) / (1024 ** 3)
            if allocated > 10 and iteration >= self.config.update_until:
                torch.cuda.empty_cache()
                logging.info(f"Memory usage: {allocated:.2f} GB. Clearing cache...")

            # Optimizer step
            if iteration < self.config.iterations:
                self.model.optimizer.step()
                self.model.optimizer.zero_grad(set_to_none=True)

        self.step = self.step + 1
        self.model.eval()
        return metrics

    def get_train_embedding(self, index: int) -> Optional[np.ndarray]:
        embed = self.model.get_light_embedding(index)
        if embed is not None:
            return embed.detach().cpu().numpy()
        return embed

    def save(self, path):
        self.model.save_ply(os.path.join(path, "point_cloud.ply"))
        ckpt = self.model.state_dict()
        ckpt_path = str(path) + f"/chkpnt-{self.step}.pth"
        torch.save(ckpt, ckpt_path)
        OmegaConf.save(self.config, os.path.join(path, "config.yaml"))

        # Note, since the torch checkpoint does not have deterministic SHA, we compute the SHA here.
        sha = get_torch_checkpoint_sha(ckpt)
        with open(ckpt_path + ".sha256", "w", encoding="utf8") as f:
            f.write(sha)
