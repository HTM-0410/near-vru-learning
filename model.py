"""METEOR-like Vehicle/VRU detection + metric-depth trong MỘT file.

Mục tiêu
--------
Nhận một ảnh RGB, phát hiện hai nhóm đối tượng ``Vehicle`` và ``VRU``, dự đoán
camera-Z depth theo mét, rồi suy ra khoảng cách từ camera/ego đến từng object.
VRU sau đó được chia thành ``critical``, ``near``, ``far`` hoặc
``unknown_depth`` để phục vụ logic an toàn phía sau.

Luồng tensor
------------

    RGB [B,3,H,W], giá trị [0,1]
      -> ImageNet normalization
      -> ResNet-34 backbone
      -> top-down FPN 160 channels: P2/P3/P4 @ stride 4/8/16
          |-> P2 depth decoder: [B,64,H/4,W/4]
          |      -> softmax 64 bins -> expected metric depth [B,H/4,W/4]
          `-> CenterNet detector 3 scale
                 s4  : object < 40 px
                 s8  : object 40-120 px
                 s16 : object >= 120 px
                 heatmap channels: 0=Vehicle, 1=VRU
      -> decode box + class-wise NMS
      -> lấy depth bền vững trong lower-central region của bbox
      -> pinhole unprojection: pixel + camera-Z -> camera XYZ
      -> optional camera->ego transform -> ego ground distance

Phạm vi
--------
Đây là phần image-space lấy cấu hình chính từ METEOR. File KHÔNG triển khai toàn
bộ multi-camera depth-weighted IPM, BEV segmentation hoặc planner. ResNet-34 có
thể khởi tạo bằng ImageNet, nhưng depth head và detection head hai lớp phải được
train bằng dữ liệu phù hợp trước khi inference thật.
"""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import pickle
import sys
import tempfile
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms
from torch.utils.data import DataLoader, Dataset


# =============================================================================
# BLOCK 1 — HẰNG SỐ VÀ QUY ƯỚC MODEL
# =============================================================================

# Taxonomy rút gọn. Tất cả pedestrian/cyclist/motorcyclist phải được convert về
# class 1 ở bước chuẩn bị label; ô tô/bus/truck... convert về class 0.
CLASS_NAMES = ("vehicle", "vru")

# Detection pyramid. Một pixel trên từng feature map lần lượt tương ứng
# 4/8/16 pixel trên ảnh đầu vào.
DET_STRIDES = (4, 8, 16)

# Quy tắc gán GT theo cạnh dài nhất của bbox, giống detection pyramid METEOR.
DET_SIZE_SPLITS = (40.0, 120.0)

# Depth classification: centre(i) = 1.0 + i*1.25 m, i=0..63.
# Bin cuối có tâm 79.75 m.
DEPTH_BINS = 64
DEPTH_MIN_M = 1.0
DEPTH_STEP_M = 1.25


# =============================================================================
# BLOCK 2 — KHỐI CONV DÙNG CHUNG
# =============================================================================

class ConvBlock(nn.Sequential):
    """Hai lớp Conv-BN-ReLU giữ nguyên kích thước không gian."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


# =============================================================================
# BLOCK 3 — DATA CONTRACT: CALIBRATION, DETECTION, DEPTH VÀ OUTPUT
# =============================================================================

@dataclass(frozen=True)
class CameraCalibration:
    """Camera intrinsics plus optional 4x4 camera-to-ego transform.

    Camera axes là [x sang phải, y hướng xuống, z hướng trước].

    ``ego_from_camera`` biến điểm homogeneous ``[x_cam,y_cam,z_cam,1]`` sang
    ego frame. Nếu không truyền extrinsic, code chỉ trả Euclidean range tính từ
    tâm camera; đó chưa phải khoảng cách ego đã hiệu chuẩn.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    ego_from_camera: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.fx <= 0 or self.fy <= 0:
            raise ValueError("fx and fy must be positive")
        if self.ego_from_camera is not None:
            transform = np.asarray(self.ego_from_camera, dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError("ego_from_camera must have shape [4,4]")
            object.__setattr__(self, "ego_from_camera", transform)


@dataclass(frozen=True)
class Detection:
    """Một bbox đã decode về hệ pixel của ảnh gốc."""

    class_id: int
    class_name: str
    score: float
    box_xyxy: tuple[float, float, float, float]


@dataclass(frozen=True)
class ObjectDepth:
    """Depth và vị trí 3D đã suy ra cho một detection.

    ``camera_z_m`` là depth dọc optical axis, khác với Euclidean range.
    ``depth_mad_m`` là median absolute deviation để quan sát độ phân tán/noise.
    """

    camera_z_m: float
    camera_range_m: float
    ego_distance_m: float
    camera_xyz_m: tuple[float, float, float]
    ego_xyz_m: tuple[float, float, float] | None
    valid_pixels: int
    valid_fraction: float
    depth_mad_m: float


@dataclass(frozen=True)
class PerceivedObject:
    """Detection sau khi ghép depth và phân loại khoảng cách an toàn."""

    detection: Detection
    depth: ObjectDepth | None
    distance_band: str
    requires_attention: bool


@dataclass
class ModelOutput:
    """Raw differentiable outputs dùng cho cả training và decoding."""

    depth_logits: torch.Tensor
    depth_m: torch.Tensor
    heatmaps: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    regressions: tuple[torch.Tensor, torch.Tensor, torch.Tensor]


# =============================================================================
# BLOCK 4 — KIẾN TRÚC MODEL: RESNET-34 + FPN + DEPTH + DETECTION
# =============================================================================

class MeteorLikeVRUDepthModel(nn.Module):
    """Shared ResNet-34/FPN model for depth and Vehicle/VRU detection."""

    def __init__(self, pretrained_backbone: bool = True):
        super().__init__()
        weights = (
            torchvision.models.ResNet34_Weights.IMAGENET1K_V1
            if pretrained_backbone
            else None
        )
        backbone = torchvision.models.resnet34(weights=weights)
        self.stem = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool
        )
        self.layer1 = backbone.layer1  # 64 channels, stride 4
        self.layer2 = backbone.layer2  # 128, stride 8
        self.layer3 = backbone.layer3  # 256, stride 16
        self.layer4 = backbone.layer4  # 512, stride 32

        # ---- FPN ảnh -------------------------------------------------------
        # Top-down FPN C5 -> C2. Khác với cộng trực tiếp mọi stage tại s4,
        # mỗi mức pyramid giữ đúng độ phân giải riêng rồi mới được smooth.
        # Detection nhỏ dùng P2, trung bình dùng P3, lớn dùng P4; vì vậy head
        # không phải tự downsample lại một feature đã bị trộn ở duy nhất s4.
        self.lat1 = nn.Conv2d(64, 160, 1)
        self.lat2 = nn.Conv2d(128, 160, 1)
        self.lat3 = nn.Conv2d(256, 160, 1)
        self.lat4 = nn.Conv2d(512, 160, 1)
        self.fpn_smooth2 = ConvBlock(160, 160)
        self.fpn_smooth3 = ConvBlock(160, 160)
        self.fpn_smooth4 = ConvBlock(160, 160)

        # ---- Depth head ----------------------------------------------------
        # Decoder convolutional tại stride 4. Output không phải một số depth
        # trực tiếp mà là phân phối xác suất trên 64 metric-depth bins.
        self.depth_head = nn.Sequential(
            ConvBlock(160, 256),
            ConvBlock(256, 256),
            ConvBlock(256, 192),
            ConvBlock(192, 128),
            nn.Conv2d(128, DEPTH_BINS, 1),
        )

        # ---- Detection head -----------------------------------------------
        # Pyramid CenterNet ba scale. Mỗi scale trả:
        #   heatmap [B,2,h,w] = logit tâm Vehicle/VRU
        #   reg     [B,4,h,w] = off_y, off_x, log(w/stride), log(h/stride)
        self.det_s4 = ConvBlock(160, 128)
        self.det_s8 = ConvBlock(160, 192)
        self.det_s16 = ConvBlock(160, 256)
        self.heatmap_heads = nn.ModuleList(
            [nn.Conv2d(128, 2, 1), nn.Conv2d(192, 2, 1), nn.Conv2d(256, 2, 1)]
        )
        self.regression_heads = nn.ModuleList(
            [nn.Conv2d(128, 4, 1), nn.Conv2d(192, 4, 1), nn.Conv2d(256, 4, 1)]
        )
        for head in self.heatmap_heads:
            nn.init.constant_(head.bias, -2.19)  # initial p ~= 0.10
        for head in self.regression_heads:
            # Initial offset sigmoid(0)=0.5 và size exp(0)=1 cell. Khởi tạo
            # hữu hạn này tránh bbox cực lớn trước khi regression hội tụ.
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

        # Các buffer này đi theo device của model nhưng không phải parameter.
        bins = DEPTH_MIN_M + torch.arange(DEPTH_BINS) * DEPTH_STEP_M
        self.register_buffer("depth_centres_m", bins, persistent=False)
        self.register_buffer(
            "image_mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None],
            persistent=False,
        )
        self.register_buffer(
            "image_std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None],
            persistent=False,
        )

    def _image_pyramid(
        self, images: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Trả P2/P3/P4 top-down tương ứng stride 4/8/16."""
        x0 = self.stem(images)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        p5 = self.lat4(x4)
        p4 = self.lat3(x3) + F.interpolate(
            p5, size=x3.shape[-2:], mode="bilinear", align_corners=False
        )
        p3 = self.lat2(x2) + F.interpolate(
            p4, size=x2.shape[-2:], mode="bilinear", align_corners=False
        )
        p2 = self.lat1(x1) + F.interpolate(
            p3, size=x1.shape[-2:], mode="bilinear", align_corners=False
        )
        return (
            self.fpn_smooth2(p2),
            self.fpn_smooth3(p3),
            self.fpn_smooth4(p4),
        )

    def forward(self, images: torch.Tensor) -> ModelOutput:
        """Forward differentiable dùng cho cả train và inference.

        Args:
            images: float RGB trong [0,1], shape [B,3,H,W]. H/W nên chia hết
                cho 32 để ba scale và backbone căn chỉnh đẹp nhất.

        Returns:
            ModelOutput chứa depth logits/depth expectation và ba cặp
            heatmap/regression. Hàm này chưa decode bbox và chưa threshold.
        """
        if images.ndim != 4 or images.shape[1] != 3:
            raise ValueError("images must have shape [B,3,H,W]")
        normalized = (images - self.image_mean) / self.image_std
        feature_s4, feature_s8, feature_s16 = self._image_pyramid(normalized)

        # Depth expectation E[z] = sum_i P(bin_i)*centre_i. Giữ depth_logits
        # riêng vì training cần cross-entropy trên bin, còn inference dùng E[z].
        depth_logits = self.depth_head(feature_s4)
        depth_probability = depth_logits.float().softmax(1)
        depth_m = (
            depth_probability * self.depth_centres_m[None, :, None, None]
        ).sum(1)

        # Mỗi detection head nhận trực tiếp đúng mức top-down FPN của nó.
        det4 = self.det_s4(feature_s4)
        det8 = self.det_s8(feature_s8)
        det16 = self.det_s16(feature_s16)
        features = (det4, det8, det16)
        heatmaps = tuple(head(feature) for head, feature in zip(self.heatmap_heads, features))
        regressions = tuple(
            head(feature) for head, feature in zip(self.regression_heads, features)
        )
        return ModelOutput(depth_logits, depth_m, heatmaps, regressions)

    # =========================================================================
    # BLOCK 5 — TARGET VÀ LOSS CHO TRAINING
    # =========================================================================

    @staticmethod
    def _gaussian_targets(
        boxes: torch.Tensor,
        counts: torch.Tensor,
        feature_hw: tuple[int, int],
        stride: int,
        size_low: float,
        size_high: float,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rasterize bbox thành CenterNet targets cho đúng một pyramid scale.

        Args:
            boxes: [B,K,5] = class,cx,cy,width,height theo pixel ảnh input.
            counts: số phần tử hợp lệ trong K của từng sample.
            feature_hw: kích thước feature map đang được supervise.
            stride: 4, 8 hoặc 16.
            size_low/size_high: chỉ bbox thuộc dải size này mới vào scale.

        Returns:
            heatmap: Gaussian centre target [B,2,h,w].
            regression: off_y/off_x/log_w/log_h [B,4,h,w].
            mask: 1 duy nhất tại cell tâm được regression supervise.
        """
        batch, max_boxes = boxes.shape[:2]
        height, width = feature_hw
        heatmap = torch.zeros(batch, 2, height, width, device=device)
        regression = torch.zeros(batch, 4, height, width, device=device)
        mask = torch.zeros(batch, 1, height, width, device=device)
        rows = torch.arange(height, device=device, dtype=torch.float32)
        cols = torch.arange(width, device=device, dtype=torch.float32)
        for batch_index in range(batch):
            for box_index in range(min(int(counts[batch_index]), max_boxes)):
                cls, cx, cy, box_w, box_h = boxes[batch_index, box_index].tolist()
                size = max(box_w, box_h)
                if not size_low <= size < size_high or cls not in (0.0, 1.0):
                    continue
                row, col = cy / stride, cx / stride
                row_i, col_i = int(row), int(col)
                if not 0 <= row_i < height or not 0 <= col_i < width:
                    continue
                # Gaussian shoulder giúp các cell sát tâm không bị coi như
                # background chắc chắn. Nó là label encoding, không phải
                # class/distance loss weight.
                radius = max(1.0, 0.35 * size / stride / 2)
                gaussian = torch.exp(
                    -(
                        ((rows - row) ** 2).view(-1, 1)
                        + ((cols - col) ** 2).view(1, -1)
                    )
                    / (2 * radius**2)
                )
                channel = int(cls)
                heatmap[batch_index, channel] = torch.maximum(
                    heatmap[batch_index, channel], gaussian
                )
                # Ensure a fractional centre always creates one focal positive.
                heatmap[batch_index, channel, row_i, col_i] = 1.0
                regression[batch_index, :, row_i, col_i] = torch.tensor(
                    [
                        row - row_i,
                        col - col_i,
                        math.log(max(box_w / stride, 0.25)),
                        math.log(max(box_h / stride, 0.25)),
                    ],
                    device=device,
                )
                mask[batch_index, 0, row_i, col_i] = 1.0
        return heatmap, regression, mask

    @staticmethod
    def _ohem_focal_loss(
        logits: torch.Tensor,
        target: torch.Tensor,
        min_negatives: int = 256,
        negative_ratio: int = 10,
    ) -> torch.Tensor:
        """Focal loss với Online Hard-Negative Mining.

        Mọi positive đều có hệ số 1. Background dễ không được cộng hết; mỗi
        ảnh chỉ giữ ``max(min_negatives, negative_ratio*n_positive)`` negative
        có loss lớn nhất. Đây là cách xử lý foreground/background imbalance mà
        không gán VRU weight hoặc distance weight thủ công.
        """
        probability = logits.float().sigmoid().clamp(1e-4, 1 - 1e-4)
        positive = (target > 0.99).float()
        positive_loss = -positive * (1 - probability).pow(2) * probability.log()
        negative_loss = -(
            (1 - positive)
            * (1 - target).pow(4)
            * probability.pow(2)
            * (1 - probability).log()
        )
        total = probability.new_zeros(())
        for batch_index in range(probability.shape[0]):
            positive_count = int(positive[batch_index].sum().item())
            hard_count = min(
                negative_loss[batch_index].numel(),
                max(min_negatives, negative_ratio * positive_count),
            )
            hard_negative = negative_loss[batch_index].flatten().topk(hard_count).values.sum()
            total = total + (
                positive_loss[batch_index].sum() + hard_negative
            ) / max(positive_count, 1)
        return total / max(probability.shape[0], 1)

    def detection_loss(
        self,
        output: ModelOutput,
        boxes: torch.Tensor,
        counts: torch.Tensor,
    ) -> torch.Tensor:
        """Loss trung bình của ba detection scales.

        Bbox nhỏ/vừa/lớn chỉ được gán vào một scale, tránh cùng một GT bị đếm
        ba lần. Positive Vehicle và VRU có cùng weight; trọng tâm học VRU đến
        từ dữ liệu/kiến trúc và hard-example mining, không phải multiplier.
        """
        lows = (0.0, *DET_SIZE_SPLITS)
        highs = (*DET_SIZE_SPLITS, float("inf"))
        total = output.depth_logits.new_zeros(())
        for heatmap, regression, stride, low, high in zip(
            output.heatmaps,
            output.regressions,
            DET_STRIDES,
            lows,
            highs,
        ):
            hm_target, reg_target, reg_mask = self._gaussian_targets(
                boxes,
                counts,
                heatmap.shape[-2:],
                stride,
                low,
                high,
                heatmap.device,
            )
            heatmap_loss = self._ohem_focal_loss(heatmap, hm_target)
            # Offset đi qua sigmoid ở cả train và decode, thay vì clamp tại
            # inference. Smooth-L1 ít nhạy với outlier log-size hơn L1 thuần.
            regression_prediction = torch.cat(
                [regression[:, :2].float().sigmoid(), regression[:, 2:].float()],
                dim=1,
            )
            regression_loss = (
                F.smooth_l1_loss(
                    regression_prediction,
                    reg_target,
                    reduction="none",
                    beta=0.1,
                )
                * reg_mask
            ).sum() / reg_mask.sum().clamp(min=1) / 4
            giou_loss = self._regression_giou_loss(
                regression_prediction, reg_target, reg_mask, stride
            )
            total = total + heatmap_loss + regression_loss + giou_loss
        return total / len(DET_STRIDES)

    @staticmethod
    def _regression_giou_loss(
        prediction: torch.Tensor,
        target: torch.Tensor,
        mask: torch.Tensor,
        stride: int,
    ) -> torch.Tensor:
        """GIoU tại positive center cells để tối ưu trực tiếp hình học bbox.

        Smooth-L1 trên offset/log-size giữ parameter ổn định, nhưng một sai số
        nhỏ trong log-size vẫn có thể làm IoU của VRU nhỏ tụt mạnh. GIoU bổ
        sung gradient theo mép box thật, đồng đều cho Vehicle và VRU.
        """
        batch, _, height, width = prediction.shape
        rows = torch.arange(
            height, device=prediction.device, dtype=prediction.dtype
        ).view(1, 1, height, 1)
        cols = torch.arange(
            width, device=prediction.device, dtype=prediction.dtype
        ).view(1, 1, 1, width)

        def decode_map(regression: torch.Tensor) -> tuple[torch.Tensor, ...]:
            center_y = (rows + regression[:, 0:1]) * stride
            center_x = (cols + regression[:, 1:2]) * stride
            box_w = regression[:, 2:3].clamp(-4.0, 7.0).exp() * stride
            box_h = regression[:, 3:4].clamp(-4.0, 7.0).exp() * stride
            return (
                center_x - box_w / 2,
                center_y - box_h / 2,
                center_x + box_w / 2,
                center_y + box_h / 2,
            )

        px1, py1, px2, py2 = decode_map(prediction)
        tx1, ty1, tx2, ty2 = decode_map(target)
        ix1, iy1 = torch.maximum(px1, tx1), torch.maximum(py1, ty1)
        ix2, iy2 = torch.minimum(px2, tx2), torch.minimum(py2, ty2)
        intersection = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
        pred_area = (px2 - px1).clamp(min=0) * (py2 - py1).clamp(min=0)
        target_area = (tx2 - tx1).clamp(min=0) * (ty2 - ty1).clamp(min=0)
        union = pred_area + target_area - intersection
        iou = intersection / union.clamp(min=1e-6)
        cx1, cy1 = torch.minimum(px1, tx1), torch.minimum(py1, ty1)
        cx2, cy2 = torch.maximum(px2, tx2), torch.maximum(py2, ty2)
        enclosing = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
        giou = iou - (enclosing - union) / enclosing.clamp(min=1e-6)
        positive = mask.expand(batch, 1, height, width)
        return ((1.0 - giou) * positive).sum() / positive.sum().clamp(min=1)

    def depth_loss(self, output: ModelOutput, depth_gt_m: torch.Tensor) -> torch.Tensor:
        """Depth loss = 64-bin cross entropy + 0.1*metric L1.

        ``depth_gt_m`` là camera-Z theo mét, không phải inverse/relative depth.
        Pixel ngoài [1.0,79.75] m hoặc không finite bị ignore. CE học phân phối
        bin; L1 giữ expected depth đúng đơn vị mét và hạn chế scale drift.
        """
        if depth_gt_m.ndim == 4:
            depth_gt_m = depth_gt_m[:, 0]
        if depth_gt_m.shape[-2:] != output.depth_logits.shape[-2:]:
            depth_gt_m = F.interpolate(
                depth_gt_m[:, None].float(),
                output.depth_logits.shape[-2:],
                mode="nearest",
            )[:, 0]
        continuous_bin = (depth_gt_m - DEPTH_MIN_M) / DEPTH_STEP_M
        target_bin = continuous_bin.round().long()
        valid = (
            torch.isfinite(depth_gt_m)
            & (depth_gt_m >= DEPTH_MIN_M)
            & (target_bin >= 0)
            & (target_bin < DEPTH_BINS)
        )
        ce_target = target_bin.clamp(0, DEPTH_BINS - 1)
        ce_target[~valid] = -1
        if valid.any():
            cross_entropy = F.cross_entropy(
                output.depth_logits, ce_target, ignore_index=-1
            )
            metric_l1 = (output.depth_m - depth_gt_m).abs()[valid].mean()
        else:
            cross_entropy = output.depth_logits.sum() * 0.0
            metric_l1 = output.depth_m.sum() * 0.0
        return cross_entropy + 0.1 * metric_l1

    # =========================================================================
    # BLOCK 6 — DECODE HEATMAP THÀNH BBOX VÀ NMS
    # =========================================================================

    @staticmethod
    def decode(
        output: ModelOutput,
        input_hw: tuple[int, int],
        score_threshold: float = 0.3,
        topk_per_scale: int = 64,
        nms_iou: float = 0.5,
    ) -> list[list[Detection]]:
        """Decode ba scale và chạy NMS riêng cho Vehicle/VRU.

        Quy trình mỗi scale: sigmoid -> local-maximum 3x3 -> top-k -> threshold
        -> áp regression -> xyxy. Cuối cùng class-wise NMS gộp các bbox trùng
        nhau giữa s4/s8/s16.
        """
        input_h, input_w = input_hw
        batches: list[list[Detection]] = [[] for _ in range(output.depth_m.shape[0])]
        for heatmap, regression, stride in zip(
            output.heatmaps, output.regressions, DET_STRIDES
        ):
            probability = heatmap.sigmoid()
            probability = probability * (
                F.max_pool2d(probability, 3, 1, 1) == probability
            )
            batch, _, height, width = probability.shape
            for batch_index in range(batch):
                scores, indices = probability[batch_index].flatten().topk(
                    min(topk_per_scale, probability[batch_index].numel())
                )
                keep = scores > score_threshold
                for score, index in zip(scores[keep], indices[keep]):
                    class_id = int(index // (height * width))
                    spatial = int(index % (height * width))
                    row, col = spatial // width, spatial % width
                    offset = regression[batch_index, :, row, col].float()
                    # Cùng parameterization với loss: raw logit -> [0,1].
                    offset_y = float(offset[0].sigmoid())
                    offset_x = float(offset[1].sigmoid())
                    center_y = (row + offset_y) * stride
                    center_x = (col + offset_x) * stride
                    box_w = float(offset[2].clamp(-4, 7).exp()) * stride
                    box_h = float(offset[3].clamp(-4, 7).exp()) * stride
                    x1 = min(max(0.0, center_x - box_w / 2), float(input_w - 1))
                    y1 = min(max(0.0, center_y - box_h / 2), float(input_h - 1))
                    x2 = min(max(0.0, center_x + box_w / 2), float(input_w - 1))
                    y2 = min(max(0.0, center_y + box_h / 2), float(input_h - 1))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    batches[batch_index].append(
                        Detection(
                            class_id,
                            CLASS_NAMES[class_id],
                            float(score),
                            (x1, y1, x2, y2),
                        )
                    )

        merged: list[list[Detection]] = []
        for detections in batches:
            kept: list[Detection] = []
            for class_id in range(2):
                group = [d for d in detections if d.class_id == class_id]
                if not group:
                    continue
                boxes = torch.tensor([d.box_xyxy for d in group])
                scores = torch.tensor([d.score for d in group])
                for index in nms(boxes, scores, nms_iou).tolist():
                    kept.append(group[index])
            kept.sort(key=lambda item: item.score, reverse=True)
            merged.append(kept)
        return merged


# =============================================================================
# BLOCK 7 — TỪ DEPTH MAP + BBOX ĐẾN KHOẢNG CÁCH 3D
# =============================================================================

def estimate_object_depth(
    detection: Detection,
    depth_z_m: np.ndarray,
    calibration: CameraCalibration,
    min_valid_pixels: int = 8,
) -> ObjectDepth | None:
    """Ước lượng vị trí 3D bền vững của một detection.

    Vì bbox còn chứa background/road/occluder, không lấy depth tại đúng một
    pixel. Code dùng vùng giữa theo chiều ngang và nửa dưới bbox theo chiều dọc,
    loại invalid depth, trim quantile 15-85%, rồi lấy median 3D.

    Pinhole unprojection với camera-Z ``z``:
        x_cam = (u-cx)*z/fx
        y_cam = (v-cy)*z/fy
        z_cam = z

    Nếu có extrinsic, khoảng cách ego là norm của hai trục ground-plane đầu
    tiên trong ego frame. Nếu không có, code dùng Euclidean camera range làm
    xấp xỉ và caller phải biết giới hạn này.
    """
    depth = np.asarray(depth_z_m, dtype=np.float32)
    image_h, image_w = depth.shape
    x1, y1, x2, y2 = detection.box_xyxy
    x1, x2 = sorted((max(0.0, x1), min(float(image_w), x2)))
    y1, y2 = sorted((max(0.0, y1), min(float(image_h), y2)))
    box_w, box_h = x2 - x1, y2 - y1
    sx1, sx2 = int(x1 + 0.25 * box_w), int(math.ceil(x1 + 0.75 * box_w))
    sy1, sy2 = int(y1 + 0.50 * box_h), int(math.ceil(y1 + 0.90 * box_h))
    # Lower-central crop tránh phần đầu/sky phía trên và hạn chế road ở đáy.
    patch = depth[sy1:sy2, sx1:sx2]
    valid = np.isfinite(patch) & (patch >= 0.2) & (patch <= 120.0)
    if int(valid.sum()) < min_valid_pixels:
        return None
    values = patch[valid]
    # Trim hai đuôi phân phối để giảm nhiễu từ background và object che khuất.
    low, high = np.quantile(values, [0.15, 0.85])
    retained = valid & (patch >= low) & (patch <= high)
    if int(retained.sum()) < min_valid_pixels:
        retained = valid

    local_v, local_u = np.nonzero(retained)
    z = patch[retained].astype(np.float64)
    u = local_u.astype(np.float64) + sx1
    v = local_v.astype(np.float64) + sy1
    x = (u - calibration.cx) * z / calibration.fx
    y = (v - calibration.cy) * z / calibration.fy
    camera_xyz = np.median(np.stack([x, y, z], axis=1), axis=0)
    camera_range = float(np.linalg.norm(camera_xyz))

    ego_xyz = None
    ego_distance = camera_range
    if calibration.ego_from_camera is not None:
        camera_h = np.concatenate([camera_xyz, np.ones(1)])
        ego_xyz = (calibration.ego_from_camera @ camera_h)[:3]
        ego_distance = float(np.linalg.norm(ego_xyz[:2]))
    median_z = float(np.median(z))
    return ObjectDepth(
        camera_z_m=median_z,
        camera_range_m=camera_range,
        ego_distance_m=ego_distance,
        camera_xyz_m=tuple(float(value) for value in camera_xyz),
        ego_xyz_m=(
            None if ego_xyz is None else tuple(float(value) for value in ego_xyz)
        ),
        valid_pixels=int(retained.sum()),
        valid_fraction=float(retained.sum() / max(patch.size, 1)),
        depth_mad_m=float(np.median(np.abs(z - median_z))),
    )


# =============================================================================
# BLOCK 8 — PIPELINE INFERENCE TRÊN ẢNH THẬT
# =============================================================================

class InferencePipeline:
    """Tiền xử lý ảnh, forward, decode, depth fusion và safety banding."""

    def __init__(
        self,
        model: MeteorLikeVRUDepthModel,
        calibration: CameraCalibration,
        device: str | None = None,
        input_hw: tuple[int, int] = (432, 768),
        critical_distance_m: float = 10.0,
        near_distance_m: float = 20.0,
    ):
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.model = model.to(self.device).eval()
        self.calibration = calibration
        self.input_hw = input_hw
        self.critical_distance_m = critical_distance_m
        self.near_distance_m = near_distance_m

    def predict(
        self, image: Image.Image, score_threshold: float = 0.3
    ) -> tuple[list[PerceivedObject], np.ndarray]:
        """Chạy một ảnh PIL và trả object đã sắp theo mức ưu tiên an toàn.

        Ảnh được resize về 432x768 giống cấu hình METEOR. Bbox và depth map sau
        đó được scale ngược về ảnh gốc trước khi dùng camera calibration gốc.
        """
        image = image.convert("RGB")
        original_w, original_h = image.size
        input_h, input_w = self.input_hw
        resized = image.resize((input_w, input_h), Image.Resampling.BILINEAR)
        array = np.asarray(resized, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)[None].to(self.device)
        with torch.inference_mode():
            output = self.model(tensor)
        # Bbox decode hiện nằm trong không gian ảnh resize.
        decoded = self.model.decode(
            output, self.input_hw, score_threshold=score_threshold
        )[0]
        # Metric value được bilinear-resample về ảnh gốc; chỉ thay đổi sampling
        # grid, không đổi đơn vị mét của depth.
        depth = F.interpolate(
            output.depth_m[:, None],
            size=(original_h, original_w),
            mode="bilinear",
            align_corners=False,
        )[0, 0].cpu().numpy()

        scale_x, scale_y = original_w / input_w, original_h / input_h
        perceived: list[PerceivedObject] = []
        for detection in decoded:
            x1, y1, x2, y2 = detection.box_xyxy
            detection = Detection(
                detection.class_id,
                detection.class_name,
                detection.score,
                (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y),
            )
            object_depth = estimate_object_depth(
                detection, depth, self.calibration
            )
            # Chỉ VRU được chia safety bands. Vehicle vẫn giữ depth để logging,
            # nhưng không tự động được đánh dấu requires_attention tại đây.
            if detection.class_id == 0:
                band, attention = "vehicle", False
            elif object_depth is None:
                band, attention = "unknown_depth", True
            elif object_depth.ego_distance_m <= self.critical_distance_m:
                band, attention = "critical", True
            elif object_depth.ego_distance_m <= self.near_distance_m:
                band, attention = "near", True
            else:
                band, attention = "far", False
            perceived.append(
                PerceivedObject(detection, object_depth, band, attention)
            )
        priority = {
            "critical": 0,
            "unknown_depth": 1,
            "near": 2,
            "far": 3,
            "vehicle": 4,
        }
        perceived.sort(
            key=lambda item: (
                priority[item.distance_band],
                float("inf") if item.depth is None else item.depth.ego_distance_m,
            )
        )
        return perceived, depth

    @staticmethod
    def annotate(image: Image.Image, objects: list[PerceivedObject]) -> Image.Image:
        """Vẽ bbox, confidence, khoảng cách và safety band lên ảnh."""
        colors = {
            "critical": "#ff2d2d",
            "unknown_depth": "#ff7a00",
            "near": "#ffb000",
            "far": "#21c55d",
            "vehicle": "#2f80ed",
        }
        canvas = image.convert("RGB").copy()
        draw = ImageDraw.Draw(canvas)
        font = ImageFont.load_default()
        for item in objects:
            distance = "depth?" if item.depth is None else f"{item.depth.ego_distance_m:.1f}m"
            label = (
                f"{item.detection.class_name.upper()} {item.detection.score:.2f} "
                f"{distance} {item.distance_band}"
            )
            draw.rectangle(
                item.detection.box_xyxy,
                outline=colors[item.distance_band],
                width=3,
            )
            draw.text(
                item.detection.box_xyxy[:2],
                label,
                fill=colors[item.distance_band],
                font=font,
            )
        return canvas


# =============================================================================
# BLOCK 9 — NAVSIM RAW DATA: INDEX, GT PROJECTION VÀ SPARSE DEPTH
# =============================================================================

# NAVSIM mini dùng semantic name từ annotation 3D. Mapping được khai báo tường
# minh; tuyệt đối không đoán VRU bằng kích thước bbox.
NAVSIM_TAXONOMY = {
    "vehicle": 0,
    "pedestrian": 1,
    "bicycle": 1,
    "motorcycle": 1,
}


def lidar_to_camera(
    points_lidar: np.ndarray,
    sensor2lidar_rotation: np.ndarray,
    sensor2lidar_translation: np.ndarray,
) -> np.ndarray:
    """Đổi điểm LiDAR sang camera bằng nghịch đảo camera→LiDAR.

    NAVSIM lưu ``p_lidar = R * p_camera + t`` nên chiều cần dùng là
    ``p_camera = inv(R) * (p_lidar - t)``. Camera-Z dương là phía trước camera.
    """
    points = np.asarray(points_lidar, dtype=np.float32)
    rotation = np.asarray(sensor2lidar_rotation, dtype=np.float32)
    translation = np.asarray(sensor2lidar_translation, dtype=np.float32)
    return (np.linalg.inv(rotation) @ (points - translation).T).T.astype(np.float32)


def box3d_corners_lidar(box: Iterable[float]) -> np.ndarray:
    """Tạo tám đỉnh box NAVSIM [x,y,z,length,width,height,heading]."""
    x, y, z, length, width, height, heading = map(float, box)
    signs = np.array(
        [
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1],
        ],
        dtype=np.float32,
    )
    local = signs * np.array([length, width, height], dtype=np.float32) / 2.0
    cosine, sine = math.cos(heading), math.sin(heading)
    rotation = np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return (rotation @ local.T).T + np.array([x, y, z], dtype=np.float32)


def project_box3d_to_image(
    box: Iterable[float],
    sensor2lidar_rotation: np.ndarray,
    sensor2lidar_translation: np.ndarray,
    intrinsic: np.ndarray,
    image_size: tuple[int, int],
    min_area_px: float = 4.0,
) -> tuple[float, float, float, float] | None:
    """Chiếu box 3D LiDAR thành bbox 2D đã clip trong ảnh camera trước."""
    camera_points = lidar_to_camera(
        box3d_corners_lidar(box),
        sensor2lidar_rotation,
        sensor2lidar_translation,
    )
    visible = camera_points[:, 2] > 1e-3
    if not np.any(visible):
        return None
    points = camera_points[visible]
    projected = (np.asarray(intrinsic, dtype=np.float32) @ points.T).T
    pixels = projected[:, :2] / projected[:, 2:3].clip(min=1e-3)
    width, height = image_size
    x1, y1 = pixels.min(axis=0)
    x2, y2 = pixels.max(axis=0)
    x1, x2 = float(np.clip(x1, 0, width - 1)), float(np.clip(x2, 0, width - 1))
    y1, y2 = float(np.clip(y1, 0, height - 1)), float(np.clip(y2, 0, height - 1))
    if x2 <= x1 or y2 <= y1 or (x2 - x1) * (y2 - y1) < min_area_px:
        return None
    return x1, y1, x2, y2


def build_navsim_index(
    dataset_root: str | Path,
    output_path: str | Path,
    max_samples: int | None = None,
) -> dict[str, Any]:
    """Quét NAVSIM mini hiện có và tạo index portable bằng relative paths.

    File pickle chỉ được đọc từ dataset NAVSIM cục bộ mà người dùng đã cung
    cấp. Index chỉ lưu metadata/calibration/bbox; pixel và point cloud vẫn được
    lazy-load khi Dataset thực sự cần sample.
    """
    root = Path(dataset_root).resolve()
    logs_root = root / "navsim_logs" / "mini"
    sensors_root = root / "sensor_blobs" / "mini"
    if not logs_root.is_dir() or not sensors_root.is_dir():
        raise FileNotFoundError(
            f"NAVSIM mini requires {logs_root} and {sensors_root}"
        )

    records: list[dict[str, Any]] = []
    class_counts: Counter[str] = Counter()
    skipped_missing = 0
    for log_path in sorted(logs_root.glob("*.pkl")):
        # NAVSIM logs are trusted local dataset artifacts, not arbitrary input.
        with log_path.open("rb") as stream:
            frames = pickle.load(stream)
        for frame in frames:
            camera_meta = frame["cams"]["CAM_F0"]
            camera_path = sensors_root / camera_meta["data_path"]
            lidar_path = sensors_root / frame["lidar_path"]
            if not camera_path.is_file() or not lidar_path.is_file():
                skipped_missing += 1
                continue
            with Image.open(camera_path) as image:
                image_size = image.size

            rotation = np.asarray(camera_meta["sensor2lidar_rotation"], dtype=np.float32)
            translation = np.asarray(camera_meta["sensor2lidar_translation"], dtype=np.float32)
            intrinsic = np.asarray(camera_meta["cam_intrinsic"], dtype=np.float32)
            objects: list[dict[str, Any]] = []
            for box, raw_name in zip(
                frame["anns"]["gt_boxes"], frame["anns"]["gt_names"]
            ):
                name = str(raw_name)
                class_id = NAVSIM_TAXONOMY.get(name)
                if class_id is None:
                    continue
                bbox = project_box3d_to_image(
                    box, rotation, translation, intrinsic, image_size
                )
                if bbox is None:
                    continue
                box_values = np.asarray(box, dtype=np.float32)
                camera_center = lidar_to_camera(
                    box_values[None, :3], rotation, translation
                )[0]
                objects.append(
                    {
                        "box_xyxy": list(bbox),
                        "class_id": class_id,
                        "original_name": name,
                        # Dùng để audit near-VRU và chia scenario sampler;
                        # không được nhân vào loss như một distance weight.
                        "ego_distance_m": float(np.linalg.norm(box_values[:2])),
                        "camera_z_m": float(camera_center[2]),
                    }
                )
                class_counts[CLASS_NAMES[class_id]] += 1

            records.append(
                {
                    "token": str(frame["token"]),
                    "log_name": str(frame["log_name"]),
                    "camera_rel": str(camera_path.relative_to(root)),
                    "lidar_rel": str(lidar_path.relative_to(root)),
                    "image_size": list(image_size),
                    "intrinsic": intrinsic.tolist(),
                    "sensor2lidar_rotation": rotation.tolist(),
                    "sensor2lidar_translation": translation.tolist(),
                    "objects": objects,
                }
            )
            if max_samples is not None and len(records) >= max_samples:
                break
        if max_samples is not None and len(records) >= max_samples:
            break

    payload = {
        "format_version": 3,
        "dataset_root_hint": str(root),
        "camera": "CAM_F0",
        "samples": records,
        "summary": {
            "samples": len(records),
            "vehicle_boxes": class_counts["vehicle"],
            "vru_boxes": class_counts["vru"],
            "skipped_missing_sensor_frames": skipped_missing,
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_pcd_xyz(path: str | Path) -> np.ndarray:
    """Đọc XYZ từ PCD binary/ascii của NAVSIM mà không cần nuPlan runtime."""
    path = Path(path)
    with path.open("rb") as stream:
        header: dict[str, list[str]] = {}
        while True:
            line = stream.readline()
            if not line:
                raise ValueError(f"PCD header ended before DATA: {path}")
            decoded = line.decode("ascii").strip()
            if not decoded or decoded.startswith("#"):
                continue
            parts = decoded.split()
            header[parts[0].upper()] = parts[1:]
            if parts[0].upper() == "DATA":
                break

        fields = header["FIELDS"]
        sizes = list(map(int, header["SIZE"]))
        types = header["TYPE"]
        counts = list(map(int, header.get("COUNT", ["1"] * len(fields))))
        point_count = int(header["POINTS"][0])
        if header["DATA"][0].lower() == "ascii":
            values = np.loadtxt(stream, dtype=np.float32, max_rows=point_count)
            return values[:, [fields.index(axis) for axis in ("x", "y", "z")]]
        if header["DATA"][0].lower() != "binary":
            raise ValueError(f"Unsupported PCD DATA kind in {path}")

        codes = {
            ("F", 4): "<f4", ("F", 8): "<f8", ("U", 1): "u1",
            ("U", 2): "<u2", ("U", 4): "<u4", ("I", 1): "i1",
            ("I", 2): "<i2", ("I", 4): "<i4",
        }
        dtype_fields = []
        for field, size, value_type, count in zip(fields, sizes, types, counts):
            base = codes.get((value_type.upper(), size))
            if base is None:
                raise ValueError(f"Unsupported PCD field {value_type}{size}")
            dtype_fields.append(
                (field, base) if count == 1 else (field, base, (count,))
            )
        records = np.frombuffer(
            stream.read(), dtype=np.dtype(dtype_fields), count=point_count
        )
        return np.column_stack(
            [records["x"], records["y"], records["z"]]
        ).astype(np.float32)


def sparse_camera_depth(
    points_lidar: np.ndarray,
    rotation: np.ndarray,
    translation: np.ndarray,
    intrinsic: np.ndarray,
    source_size: tuple[int, int],
    output_size: tuple[int, int],
) -> np.ndarray:
    """Project LiDAR thành sparse camera-Z target trên feature stride 4."""
    camera_points = lidar_to_camera(points_lidar, rotation, translation)
    source_w, source_h = source_size
    output_w, output_h = output_size
    scaled_k = np.asarray(intrinsic, dtype=np.float32).copy()
    scaled_k[0, :] *= output_w / source_w
    scaled_k[1, :] *= output_h / source_h
    z = camera_points[:, 2]
    projected = (scaled_k @ camera_points.T).T
    safe_z = projected[:, 2].clip(min=1e-3)
    u = np.floor(projected[:, 0] / safe_z).astype(np.int64)
    v = np.floor(projected[:, 1] / safe_z).astype(np.int64)
    valid = (
        (z >= DEPTH_MIN_M)
        & (z <= float(DEPTH_MIN_M + (DEPTH_BINS - 1) * DEPTH_STEP_M))
        & (u >= 0) & (u < output_w)
        & (v >= 0) & (v < output_h)
    )
    depth = np.full((output_h, output_w), np.inf, dtype=np.float32)
    flat = v[valid] * output_w + u[valid]
    np.minimum.at(depth.reshape(-1), flat, z[valid])
    depth[~np.isfinite(depth)] = 0.0
    return depth


class NavsimFrontDataset(Dataset):
    """Lazy CAM_F0 + LiDAR dataset dùng trực tiếp index portable ở trên."""

    def __init__(
        self,
        records: list[dict[str, Any]],
        dataset_root: str | Path,
        image_hw: tuple[int, int],
        include_depth: bool = True,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.root = Path(dataset_root)
        self.image_hw = image_hw  # (height, width)
        self.include_depth = include_depth
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        camera_path = self.root / record["camera_rel"]
        with Image.open(camera_path) as source:
            source = source.convert("RGB")
            source_w, source_h = source.size
            target_h, target_w = self.image_hw
            resized = source.resize((target_w, target_h), Image.Resampling.BILINEAR)
            image = np.asarray(resized, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image.copy()).permute(2, 0, 1)

        scale_x, scale_y = target_w / source_w, target_h / source_h
        encoded: list[list[float]] = []
        for obj in record["objects"]:
            x1, y1, x2, y2 = obj["box_xyxy"]
            x1, x2 = x1 * scale_x, x2 * scale_x
            y1, y2 = y1 * scale_y, y2 * scale_y
            encoded.append(
                [
                    float(obj["class_id"]),
                    (x1 + x2) / 2,
                    (y1 + y2) / 2,
                    x2 - x1,
                    y2 - y1,
                ]
            )
        boxes = torch.tensor(encoded, dtype=torch.float32).reshape(-1, 5)

        result: dict[str, Any] = {
            "image": image_tensor,
            "boxes": boxes,
            "record": record,
        }
        if self.include_depth:
            points = load_pcd_xyz(self.root / record["lidar_rel"])
            depth = sparse_camera_depth(
                points,
                np.asarray(record["sensor2lidar_rotation"], dtype=np.float32),
                np.asarray(record["sensor2lidar_translation"], dtype=np.float32),
                np.asarray(record["intrinsic"], dtype=np.float32),
                tuple(record["image_size"]),
                (target_w // 4, target_h // 4),
            )
            result["depth"] = torch.from_numpy(depth)
        if self.augment:
            # Geometry-preserving augmentation: image, bbox và sparse depth
            # luôn flip cùng nhau. Photometric jitter chỉ tác động RGB.
            if bool(torch.rand(()) < 0.5):
                result["image"] = result["image"].flip(-1)
                if len(result["boxes"]):
                    result["boxes"][:, 1] = target_w - result["boxes"][:, 1]
                if "depth" in result:
                    result["depth"] = result["depth"].flip(-1)
            brightness = 0.85 + 0.30 * float(torch.rand(()))
            contrast = 0.85 + 0.30 * float(torch.rand(()))
            mean = result["image"].mean(dim=(-2, -1), keepdim=True)
            result["image"] = (
                (result["image"] - mean) * contrast + mean
            ).mul(brightness).clamp(0.0, 1.0)
        return result


def navsim_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable-length bbox lists; stack ảnh và sparse depth tensors."""
    counts = torch.tensor([len(item["boxes"]) for item in batch], dtype=torch.long)
    max_boxes = max(1, int(counts.max()))
    boxes = torch.zeros(len(batch), max_boxes, 5, dtype=torch.float32)
    for index, item in enumerate(batch):
        boxes[index, : len(item["boxes"])] = item["boxes"]
    result = {
        "images": torch.stack([item["image"] for item in batch]),
        "boxes": boxes,
        "counts": counts,
        "records": [item["record"] for item in batch],
    }
    if "depth" in batch[0]:
        result["depth"] = torch.stack([item["depth"] for item in batch])
    return result


def draw_navsim_ground_truth(
    image: Image.Image, record: dict[str, Any]
) -> Image.Image:
    """Vẽ projected GT để kiểm tra geometry trước khi tin kết quả train."""
    canvas = image.convert("RGB").copy()
    draw = ImageDraw.Draw(canvas)
    colors = {0: "#2f80ed", 1: "#ffb000"}
    for obj in record["objects"]:
        class_id = int(obj["class_id"])
        draw.rectangle(obj["box_xyxy"], outline=colors[class_id], width=3)
        draw.text(
            tuple(obj["box_xyxy"][:2]),
            f"GT {CLASS_NAMES[class_id]} ({obj['original_name']})",
            fill=colors[class_id],
        )
    return canvas


def evaluate_detection_iou(
    objects: list[PerceivedObject],
    ground_truth: list[dict[str, Any]],
    iou_threshold: float = 0.5,
    include_matches: bool = False,
) -> dict[str, Any]:
    """Greedy same-class IoU matching for detection and object-depth metrics.

    ``include_matches`` chỉ dùng nội bộ khi đánh giá depth tại các bbox TP;
    cặp object/GT không được ghi trực tiếp vào JSON vì chứa dataclass.
    """

    def iou(a: Iterable[float], b: Iterable[float]) -> float:
        ax1, ay1, ax2, ay2 = map(float, a)
        bx1, by1, bx2, by2 = map(float, b)
        ix1, iy1, ix2, iy2 = max(ax1, bx1), max(ay1, by1), min(ax2, bx2), min(ay2, by2)
        intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
        area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
        area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
        return intersection / max(area_a + area_b - intersection, 1e-9)

    summary: dict[str, Any] = {}
    matched_pairs: dict[str, list[tuple[PerceivedObject, dict[str, Any]]]] = {}
    for class_id, class_name in enumerate(CLASS_NAMES):
        predictions = sorted(
            [item for item in objects if item.detection.class_id == class_id],
            key=lambda item: item.detection.score,
            reverse=True,
        )
        targets = [item for item in ground_truth if int(item["class_id"]) == class_id]
        matched: set[int] = set()
        pairs: list[tuple[PerceivedObject, dict[str, Any]]] = []
        for prediction in predictions:
            candidates = [
                (iou(prediction.detection.box_xyxy, target["box_xyxy"]), index)
                for index, target in enumerate(targets)
                if index not in matched
            ]
            best_iou, best_index = max(candidates, default=(0.0, -1))
            if best_iou >= iou_threshold:
                matched.add(best_index)
                pairs.append((prediction, targets[best_index]))
        true_positive = len(pairs)
        false_positive = len(predictions) - true_positive
        false_negative = len(targets) - true_positive
        summary[class_name] = {
            "tp": true_positive,
            "fp": false_positive,
            "fn": false_negative,
            "precision": true_positive / max(true_positive + false_positive, 1),
            "recall": true_positive / max(true_positive + false_negative, 1),
        }
        if include_matches:
            matched_pairs[class_name] = pairs
    result = {"iou_threshold": iou_threshold, "per_class": summary}
    if include_matches:
        result["matched_pairs"] = matched_pairs
    return result


def sample_records_across_logs(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Lấy mẫu xác định nhưng trải đều giữa các recording.

    Cắt ``records[:limit]`` dễ lấy toàn bộ frame từ log đầu tiên vì index được
    tạo theo thứ tự file metadata. Round-robin sau khi shuffle trong từng log
    giúp mini-run nhìn thấy nhiều bối cảnh hơn mà vẫn tái lập được bằng seed.
    """
    if limit <= 0:
        raise ValueError("sample limit must be positive")
    groups: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        groups.setdefault(record["log_name"], []).append(record)
    rng = np.random.default_rng(seed)
    for group in groups.values():
        rng.shuffle(group)

    selected: list[dict[str, Any]] = []
    log_names = sorted(groups)
    row = 0
    target = min(limit, len(records))
    while len(selected) < target:
        added = False
        for log_name in log_names:
            group = groups[log_name]
            if row < len(group):
                selected.append(group[row])
                added = True
                if len(selected) == target:
                    break
        if not added:
            break
        row += 1
    return selected


def sample_training_records(
    records: list[dict[str, Any]], limit: int, seed: int
) -> list[dict[str, Any]]:
    """Scenario-balanced sampling mà không đổi trọng số loss.

    50% budget ưu tiên frame có VRU <=30 m, 30% frame có VRU xa hơn và 20%
    context còn lại. Mỗi bucket vẫn round-robin theo log và không duplicate
    frame. Đây là data curriculum minh bạch, không phải distance/class weight.
    """
    near_vru: list[dict[str, Any]] = []
    other_vru: list[dict[str, Any]] = []
    context: list[dict[str, Any]] = []
    for record in records:
        vrus = [obj for obj in record["objects"] if int(obj["class_id"]) == 1]
        if any(float(obj.get("ego_distance_m", float("inf"))) <= 30.0 for obj in vrus):
            near_vru.append(record)
        elif vrus:
            other_vru.append(record)
        else:
            context.append(record)

    target = min(limit, len(records))
    quotas = (round(0.50 * target), round(0.30 * target))
    selected: list[dict[str, Any]] = []
    selected_tokens: set[str] = set()
    for bucket, quota, bucket_seed in (
        (near_vru, quotas[0], seed + 11),
        (other_vru, quotas[1], seed + 23),
        (context, target - sum(quotas), seed + 37),
    ):
        if not bucket or quota <= 0:
            continue
        for record in sample_records_across_logs(bucket, quota, bucket_seed):
            if record["token"] not in selected_tokens:
                selected.append(record)
                selected_tokens.add(record["token"])

    if len(selected) < target:
        remaining = [r for r in records if r["token"] not in selected_tokens]
        selected.extend(
            sample_records_across_logs(remaining, target - len(selected), seed + 53)
        )
    return selected[:target]


def run_navsim_smoke(args: argparse.Namespace) -> None:
    """Build index → smoke train → held-out inference → checkpoint/artifacts.

    Đây là kiểm tra pipeline, không phải benchmark accuracy. Split được thực
    hiện theo ``log_name`` để sample của recording validation không lọt vào
    train. Trên CPU, backbone mặc định bị freeze để vòng smoke hoàn thành nhanh.
    """
    dataset_root = Path(args.navsim_root).resolve()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    index_path = Path(args.navsim_index)
    if args.rebuild_navsim_index or not index_path.exists():
        payload = build_navsim_index(
            dataset_root, index_path, max_samples=args.max_index_samples
        )
    else:
        payload = json.loads(index_path.read_text(encoding="utf-8"))
    records = payload["samples"]
    if len(records) < 2:
        raise RuntimeError("NAVSIM index needs at least two valid samples")

    logs = sorted({record["log_name"] for record in records})
    validation_log = logs[-1]
    train_records = [r for r in records if r["log_name"] != validation_log]
    validation_records = [r for r in records if r["log_name"] == validation_log]
    if not train_records or not validation_records:
        split = max(1, len(records) - 1)
        train_records, validation_records = records[:split], records[split:]
    train_records = (
        sample_records_across_logs(train_records, args.train_samples, args.seed)
        if args.uniform_training_sampling
        else sample_training_records(train_records, args.train_samples, args.seed)
    )
    validation_records = sample_records_across_logs(
        validation_records, args.validation_samples, args.seed + 1
    )
    training_scenarios: Counter[str] = Counter()
    for record in train_records:
        vrus = [obj for obj in record["objects"] if int(obj["class_id"]) == 1]
        if any(float(obj.get("ego_distance_m", float("inf"))) <= 30.0 for obj in vrus):
            training_scenarios["near_vru"] += 1
        elif vrus:
            training_scenarios["far_vru"] += 1
        else:
            training_scenarios["context"] += 1

    image_hw = (args.image_height, args.image_width)
    dataset = NavsimFrontDataset(
        train_records,
        dataset_root,
        image_hw,
        include_depth=True,
        augment=not args.no_train_augmentation,
    )
    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    loader_kwargs: dict[str, Any] = {
        "batch_size": args.batch_size,
        "shuffle": True,
        "num_workers": args.num_workers,
        "pin_memory": device.type == "cuda",
        "collate_fn": navsim_collate,
        "generator": torch.Generator().manual_seed(args.seed),
    }
    if args.num_workers > 0:
        loader_kwargs.update(persistent_workers=True, prefetch_factor=2)
    loader = DataLoader(dataset, **loader_kwargs)
    model = MeteorLikeVRUDepthModel(
        pretrained_backbone=not args.no_pretrained_backbone
    ).to(device)
    backbone_modules = (
        model.stem, model.layer1, model.layer2, model.layer3, model.layer4
    )
    frozen_backbone_modules: list[nn.Module] = []
    if not args.train_backbone:
        for module in backbone_modules:
            for parameter in module.parameters():
                parameter.requires_grad_(False)
        stages_to_unfreeze = max(0, min(args.unfreeze_backbone_stages, 4))
        for module in backbone_modules[-stages_to_unfreeze:] if stages_to_unfreeze else ():
            for parameter in module.parameters():
                parameter.requires_grad_(True)
        frozen_backbone_modules = [
            module
            for module in backbone_modules
            if not any(parameter.requires_grad for parameter in module.parameters())
        ]
    step = 0
    last_losses: dict[str, float] = {}
    checkpoint_image_hw = image_hw
    if args.init_checkpoint and args.resume_smoke_checkpoint:
        raise ValueError(
            "--init-checkpoint and --resume-smoke-checkpoint are mutually exclusive"
        )
    if args.init_checkpoint:
        initial_payload = torch.load(
            args.init_checkpoint, map_location=device, weights_only=True
        )
        model.load_state_dict(initial_payload["model"])
        step = int(initial_payload.get("step", 0))
        print(
            f"initialized from checkpoint={args.init_checkpoint} step={step}; "
            "continuing training with a fresh optimizer",
            flush=True,
        )
    if args.resume_smoke_checkpoint:
        resume_payload = torch.load(
            args.resume_smoke_checkpoint, map_location=device, weights_only=True
        )
        model.load_state_dict(resume_payload["model"])
        checkpoint_image_hw = tuple(
            resume_payload.get("config", {}).get("image_hw", image_hw)
        )
        if checkpoint_image_hw != image_hw:
            print(
                "warning: evaluation resolution differs from checkpoint training "
                f"resolution: eval={image_hw} trained={checkpoint_image_hw}",
                flush=True,
            )
        last_losses = resume_payload.get("last_losses", {})
        step = int(
            resume_payload.get(
                "step",
                args.epochs * math.ceil(len(train_records) / args.batch_size),
            )
        )
        print(
            f"resumed checkpoint={args.resume_smoke_checkpoint} step={step}; "
            "skipping training and running held-out evaluation",
            flush=True,
        )
    else:
        fine_tune_start_step = step
        fine_tune_steps = max(args.epochs * len(loader), 1)
        backbone_parameter_ids = {
            id(parameter)
            for module in backbone_modules
            for parameter in module.parameters()
        }
        backbone_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) in backbone_parameter_ids
        ]
        task_parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad and id(parameter) not in backbone_parameter_ids
        ]
        parameter_groups: list[dict[str, Any]] = [
            {"params": task_parameters, "lr": args.learning_rate}
        ]
        if backbone_parameters:
            parameter_groups.append(
                {
                    "params": backbone_parameters,
                    "lr": args.learning_rate * args.backbone_lr_scale,
                }
            )
        optimizer = torch.optim.AdamW(parameter_groups, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=fine_tune_steps,
            eta_min=args.learning_rate * 0.05,
        )
        model.train()
        # requires_grad=False không tự khóa running mean/variance của BatchNorm.
        # Giữ các stage frozen ở eval để batch nhỏ không phá pretrained stats.
        for module in frozen_backbone_modules:
            module.eval()
        for epoch in range(args.epochs):
            for batch in loader:
                non_blocking = device.type == "cuda"
                images = batch["images"].to(device, non_blocking=non_blocking)
                boxes = batch["boxes"].to(device, non_blocking=non_blocking)
                counts = batch["counts"].to(device, non_blocking=non_blocking)
                depth_gt = batch["depth"].to(device, non_blocking=non_blocking)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.bfloat16,
                    enabled=args.amp and device.type == "cuda",
                ):
                    output = model(images)
                    detection = model.detection_loss(output, boxes, counts)
                    depth = model.depth_loss(output, depth_gt)
                    total = detection + args.depth_weight * depth
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    args.gradient_clip,
                )
                optimizer.step()
                scheduler.step()
                step += 1
                last_losses = {
                    "total": float(total.detach()),
                    "detection": float(detection.detach()),
                    "depth": float(depth.detach()),
                    "learning_rate": float(optimizer.param_groups[0]["lr"]),
                }
                fine_tune_step = step - fine_tune_start_step
                if (
                    fine_tune_step == 1
                    or fine_tune_step % args.log_every == 0
                    or fine_tune_step == fine_tune_steps
                ):
                    print(
                        f"epoch={epoch + 1} step={step} "
                        + " ".join(
                            f"{key}={value:.4f}"
                            for key, value in last_losses.items()
                        ),
                        flush=True,
                    )

    output_dir = Path(args.smoke_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = (
        Path(args.resume_smoke_checkpoint)
        if args.resume_smoke_checkpoint
        else output_dir / "navsim_smoke.pt"
    )
    if not args.resume_smoke_checkpoint:
        torch.save(
            {
                "model": model.state_dict(),
                "step": step,
                "config": {
                    "image_hw": image_hw,
                    "depth_weight": args.depth_weight,
                    "train_samples": len(train_records),
                    "validation_log": validation_log,
                    "unfreeze_backbone_stages": args.unfreeze_backbone_stages,
                    "scenario_balanced_sampling": not args.uniform_training_sampling,
                    "training_scenarios": dict(training_scenarios),
                    "init_checkpoint": args.init_checkpoint,
                    "device": str(device),
                    "amp": bool(args.amp and device.type == "cuda"),
                    "num_workers": args.num_workers,
                },
                "last_losses": last_losses,
            },
            checkpoint_path,
        )

    # Aggregate metric trên nhiều frame của held-out recording. Depth MAE được
    # weighted theo số pixel LiDAR hợp lệ, còn detection cộng TP/FP/FN trước
    # khi tính precision/recall để frame trống không làm lệch trung bình.
    model.eval()
    aggregate_counts = {
        class_name: {"tp": 0, "fp": 0, "fn": 0}
        for class_name in CLASS_NAMES
    }
    depth_absolute_error = 0.0
    depth_valid_pixels = 0
    near_vru_true_positive = 0
    near_vru_false_negative = 0
    matched_object_depth_errors: dict[str, list[float]] = {
        "vehicle": [],
        "vru": [],
        "near_vru_30m": [],
    }
    matched_without_depth = {key: 0 for key in matched_object_depth_errors}
    validation_rows: list[dict[str, Any]] = []
    first_visual: tuple[
        dict[str, Any], Image.Image, InferencePipeline, list[PerceivedObject]
    ] | None = None
    validation_dataset = NavsimFrontDataset(
        validation_records, dataset_root, image_hw, include_depth=True
    )
    for validation_index, validation_record in enumerate(validation_records):
        camera_path = dataset_root / validation_record["camera_rel"]
        image = Image.open(camera_path).convert("RGB")
        intrinsic = np.asarray(validation_record["intrinsic"], dtype=np.float64)
        pipeline = InferencePipeline(
            model,
            CameraCalibration(
                float(intrinsic[0, 0]),
                float(intrinsic[1, 1]),
                float(intrinsic[0, 2]),
                float(intrinsic[1, 2]),
            ),
            device=str(device),
            input_hw=image_hw,
        )
        objects, _ = pipeline.predict(image, args.smoke_score_threshold)
        sample_metrics = evaluate_detection_iou(
            objects,
            validation_record["objects"],
            iou_threshold=0.5,
            include_matches=True,
        )
        matched_pairs = sample_metrics.pop("matched_pairs")
        for class_name, pairs in matched_pairs.items():
            for prediction, target in pairs:
                groups = [class_name]
                if (
                    class_name == CLASS_NAMES[1]
                    and float(target.get("ego_distance_m", float("inf"))) <= 30.0
                ):
                    groups.append("near_vru_30m")
                target_z = float(target.get("camera_z_m", float("nan")))
                if prediction.depth is None or not math.isfinite(target_z):
                    for group in groups:
                        matched_without_depth[group] += 1
                    continue
                error = abs(prediction.depth.camera_z_m - target_z)
                for group in groups:
                    matched_object_depth_errors[group].append(error)
        near_vru_targets = [
            obj
            for obj in validation_record["objects"]
            if int(obj["class_id"]) == 1
            and float(obj.get("ego_distance_m", float("inf"))) <= 30.0
        ]
        near_vru_metrics = evaluate_detection_iou(
            objects, near_vru_targets, iou_threshold=0.5
        )["per_class"][CLASS_NAMES[1]]
        near_vru_true_positive += int(near_vru_metrics["tp"])
        near_vru_false_negative += int(near_vru_metrics["fn"])
        for class_name, counts_for_class in sample_metrics["per_class"].items():
            for key in ("tp", "fp", "fn"):
                aggregate_counts[class_name][key] += int(counts_for_class[key])

        validation_item = validation_dataset[validation_index]
        with torch.inference_mode():
            validation_output = model(validation_item["image"][None].to(device))
        sparse_gt = validation_item["depth"].to(device)
        sparse_valid = sparse_gt > 0
        valid_count = int(sparse_valid.sum())
        sample_depth_mae = None
        if valid_count:
            absolute_error = (
                validation_output.depth_m[0] - sparse_gt
            ).abs()[sparse_valid]
            depth_absolute_error += float(absolute_error.sum())
            depth_valid_pixels += valid_count
            sample_depth_mae = float(absolute_error.mean())
        validation_rows.append(
            {
                "token": validation_record["token"],
                "sparse_depth_mae_m": sample_depth_mae,
                "sparse_depth_valid_pixels": valid_count,
                "detection_metrics": sample_metrics,
                "prediction_count": len(objects),
                "ground_truth_count": len(validation_record["objects"]),
                "near_vru_ground_truth_count": len(near_vru_targets),
            }
        )
        if first_visual is None:
            first_visual = (validation_record, image, pipeline, objects)

    detection_summary: dict[str, dict[str, float | int]] = {}
    for class_name, counts_for_class in aggregate_counts.items():
        tp = counts_for_class["tp"]
        fp = counts_for_class["fp"]
        fn = counts_for_class["fn"]
        detection_summary[class_name] = {
            **counts_for_class,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(tp + fn, 1),
        }
    detection_metrics = {
        "iou_threshold": 0.5,
        "evaluated_samples": len(validation_records),
        "per_class": detection_summary,
    }
    sparse_depth_mae = (
        depth_absolute_error / depth_valid_pixels if depth_valid_pixels else None
    )
    object_depth_metrics = {
        group: {
            "matched_with_depth": len(errors),
            "matched_without_depth": matched_without_depth[group],
            "camera_z_mae_m": float(np.mean(errors)) if errors else None,
            "camera_z_median_abs_error_m": (
                float(np.median(errors)) if errors else None
            ),
        }
        for group, errors in matched_object_depth_errors.items()
    }
    assert first_visual is not None
    validation_record, image, pipeline, objects = first_visual
    prediction_path = output_dir / "prediction.jpg"
    gt_path = output_dir / "projected_gt.jpg"
    pipeline.annotate(image, objects).save(prediction_path)
    draw_navsim_ground_truth(image, validation_record).save(gt_path)
    result_path = output_dir / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": "pipeline_smoke_only_not_accuracy_evidence",
                "token": validation_record["token"],
                "validation_log": validation_log,
                "train_samples": len(train_records),
                "train_logs": sorted({r["log_name"] for r in train_records}),
                "training_scenarios": dict(training_scenarios),
                "validation_samples": len(validation_records),
                "image_hw": image_hw,
                "checkpoint_training_image_hw": checkpoint_image_hw,
                "evaluation_resolution_matches_checkpoint": (
                    image_hw == checkpoint_image_hw
                ),
                "device": str(device),
                "checkpoint": str(checkpoint_path),
                "steps": step,
                "last_losses": last_losses,
                "score_threshold": args.smoke_score_threshold,
                "sparse_depth_mae_m": sparse_depth_mae,
                "sparse_depth_valid_pixels": depth_valid_pixels,
                "matched_object_depth_metrics": object_depth_metrics,
                "detection_metrics": detection_metrics,
                "near_vru_30m": {
                    "tp": near_vru_true_positive,
                    "fn": near_vru_false_negative,
                    "recall": near_vru_true_positive
                    / max(near_vru_true_positive + near_vru_false_negative, 1),
                },
                "validation_details": validation_rows,
                "ground_truth_objects": validation_record["objects"],
                "predictions": [
                    {
                        "detection": asdict(item.detection),
                        "depth": None if item.depth is None else asdict(item.depth),
                        "distance_band": item.distance_band,
                        "requires_attention": item.requires_attention,
                    }
                    for item in objects
                ],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"NAVSIM smoke complete: index={index_path} checkpoint={checkpoint_path} "
        f"prediction={prediction_path} result={result_path}",
        flush=True,
    )


def run_navsim_e2e(args: argparse.Namespace) -> None:
    """Chạy trọn pipeline NAVSIM bằng một lệnh và chỉ giữ artifact cuối.

    Profile ``full`` tái hiện progressive resizing đã được kiểm chứng:
    144p -> 216p -> 432p. Hai checkpoint trung gian nằm trong temporary
    directory bên trong output và được tự xoá sau khi phase cuối thành công.
    Profile ``quick`` chạy một phase rất nhỏ để kiểm tra installation/I/O.
    """
    dataset_root = Path(args.navsim_root).resolve()
    logs_root = dataset_root / "navsim_logs" / "mini"
    sensors_root = dataset_root / "sensor_blobs" / "mini"
    if not logs_root.is_dir() or not sensors_root.is_dir():
        raise FileNotFoundError(
            "NAVSIM root must contain navsim_logs/mini and sensor_blobs/mini: "
            f"{dataset_root}"
        )
    metadata_files = len(list(logs_root.glob("*.pkl")))
    sensor_scenes = len([path for path in sensors_root.iterdir() if path.is_dir()])
    if metadata_files < 2 or sensor_scenes < 2:
        raise RuntimeError(
            "End-to-end training needs at least two NAVSIM recordings; "
            f"found metadata={metadata_files}, sensor_scenes={sensor_scenes}"
        )

    device = torch.device(
        args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    )
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.benchmark = True
    elif args.e2e_profile == "full":
        print(
            "warning: full profile is running on CPU and will be very slow",
            flush=True,
        )

    if args.e2e_profile == "quick":
        phases = [
            {
                "name": "quick_stage1",
                "train_samples": 4,
                "validation_samples": 2,
                "epochs": 1,
                "batch_size": 2,
                "image_width": 192,
                "image_height": 128,
                "learning_rate": 1e-4,
                "depth_weight": 0.10,
                "unfreeze_backbone_stages": 0,
            },
            {
                "name": "quick_final",
                "train_samples": 8,
                "validation_samples": 4,
                "epochs": 1,
                "batch_size": 2,
                "image_width": 256,
                "image_height": 160,
                "learning_rate": 5e-5,
                "depth_weight": 0.10,
                "unfreeze_backbone_stages": 0,
            },
        ]
    else:
        phases = [
            {
                "name": "stage1_144p",
                "train_samples": 512,
                "validation_samples": 32,
                "epochs": 6,
                "batch_size": 4,
                "image_width": 256,
                "image_height": 144,
                "learning_rate": 3e-4,
                "depth_weight": 0.15,
                "unfreeze_backbone_stages": 2,
            },
            {
                "name": "stage2_216p",
                "train_samples": 768,
                "validation_samples": 32,
                "epochs": 3,
                "batch_size": 4,
                "image_width": 384,
                "image_height": 216,
                "learning_rate": 1.5e-4,
                "depth_weight": 0.10,
                "unfreeze_backbone_stages": 2,
            },
            {
                "name": "final_432p",
                "train_samples": 1024,
                "validation_samples": 128,
                "epochs": 3,
                "batch_size": 4,
                "image_width": 768,
                "image_height": 432,
                "learning_rate": 7.5e-5,
                "depth_weight": 0.10,
                "unfreeze_backbone_stages": 2,
            },
        ]

    output_dir = Path(args.smoke_output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "navsim_index.json"
    print(
        f"NAVSIM E2E profile={args.e2e_profile} device={device} "
        f"recordings={metadata_files} output={output_dir}",
        flush=True,
    )

    previous_checkpoint = ""
    with tempfile.TemporaryDirectory(
        prefix=".progressive-stages-", dir=output_dir
    ) as temporary_root:
        temporary_root_path = Path(temporary_root)
        for phase_index, phase in enumerate(phases):
            is_final = phase_index == len(phases) - 1
            phase_output = (
                output_dir
                if is_final
                else temporary_root_path / str(phase["name"])
            )
            phase_args = argparse.Namespace(
                **{
                    **vars(args),
                    **phase,
                    "navsim_index": str(index_path),
                    "rebuild_navsim_index": phase_index == 0,
                    "max_index_samples": None,
                    "smoke_output_dir": str(phase_output),
                    "smoke_score_threshold": 0.20,
                    "resume_smoke_checkpoint": "",
                    "init_checkpoint": previous_checkpoint,
                    "device": str(device),
                    "amp": device.type == "cuda",
                    "num_workers": 2 if device.type == "cuda" else 0,
                    "backbone_lr_scale": 0.1,
                    "gradient_clip": 5.0,
                    "log_every": 20,
                    "train_backbone": False,
                    "no_train_augmentation": False,
                    "uniform_training_sampling": False,
                    # Chỉ phase đầu cần ImageNet initialization. Phase sau
                    # nạp toàn bộ weights từ checkpoint trước.
                    "no_pretrained_backbone": (
                        args.no_pretrained_backbone if phase_index == 0 else True
                    ),
                }
            )
            print(
                f"E2E phase {phase_index + 1}/{len(phases)}: {phase['name']} "
                f"samples={phase['train_samples']} epochs={phase['epochs']} "
                f"resolution={phase['image_height']}x{phase['image_width']}",
                flush=True,
            )
            run_navsim_smoke(phase_args)
            previous_checkpoint = str(phase_output / "navsim_smoke.pt")

    final_checkpoint = output_dir / "navsim_smoke.pt"
    final_result = output_dir / "result.json"
    if not final_checkpoint.is_file() or not final_result.is_file():
        raise RuntimeError("E2E completed without final checkpoint/result")
    print(
        "NAVSIM E2E complete; intermediate checkpoints removed; "
        f"checkpoint={final_checkpoint} result={final_result}",
        flush=True,
    )


# =============================================================================
# BLOCK 10 — CHECKPOINT I/O
# =============================================================================

def load_checkpoint(model: nn.Module, checkpoint_path: str) -> dict[str, Any]:
    """Nạp checkpoint và fail-fast nếu kiến trúc không khớp.

    Chấp nhận raw state_dict hoặc dict có key ``model``; tự bỏ prefix ``module.``
    do DDP tạo ra. Không cho phép âm thầm bỏ head thiếu/thừa.
    """
    try:
        raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        raw = torch.load(checkpoint_path, map_location="cpu")
    state = raw.get("model", raw)
    state = {key.replace("module.", ""): value for key, value in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise ValueError(
            f"checkpoint is not compatible: missing={missing[:8]}, "
            f"unexpected={unexpected[:8]}"
        )
    return raw if isinstance(raw, dict) and "model" in raw else {}


# =============================================================================
# BLOCK 11 — SELF-TEST KHÔNG CẦN DATA/CHECKPOINT
# =============================================================================

def self_test() -> None:
    """Kiểm tra shape forward và phép tính bbox-depth bằng dữ liệu tổng hợp."""
    torch.manual_seed(0)
    model = MeteorLikeVRUDepthModel(pretrained_backbone=False).eval()
    image = torch.rand(1, 3, 128, 192)
    with torch.inference_mode():
        output = model(image)
    assert output.depth_logits.shape == (1, 64, 32, 48)
    assert output.depth_m.shape == (1, 32, 48)
    assert [tuple(t.shape) for t in output.heatmaps] == [
        (1, 2, 32, 48),
        (1, 2, 16, 24),
        (1, 2, 8, 12),
    ]
    decoded = model.decode(
        output, (128, 192), score_threshold=0.0, topk_per_scale=5
    )[0]
    assert all(
        detection.box_xyxy[2] > detection.box_xyxy[0]
        and detection.box_xyxy[3] > detection.box_xyxy[1]
        for detection in decoded
    )
    train_output = model(image)
    synthetic_boxes = torch.tensor([[[1.0, 72.0, 64.0, 18.0, 28.0]]])
    detection_loss = model.detection_loss(
        train_output, synthetic_boxes, torch.tensor([1])
    )
    assert torch.isfinite(detection_loss)
    detection_loss.backward()
    synthetic_depth = np.full((100, 100), 30.0, dtype=np.float32)
    synthetic_depth[50:82, 30:50] = 8.0
    detection = Detection(1, "vru", 0.9, (20, 10, 60, 90))
    distance = estimate_object_depth(
        detection,
        synthetic_depth,
        CameraCalibration(100, 100, 50, 50),
    )
    assert distance is not None and abs(distance.camera_z_m - 8.0) < 1e-5
    print("self-test passed: METEOR-like backbone + det heads + depth distance")


# =============================================================================
# BLOCK 12 — COMMAND-LINE ENTRYPOINT
# =============================================================================

def main() -> None:
    """CLI cho E2E NAVSIM training, stage nâng cao, self-test và inference."""
    # Windows console có thể mặc định cp1252 và crash chỉ vì workspace/path có
    # tiếng Việt. Reconfigure ngay tại CLI để mọi log E2E luôn in được.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument(
        "--navsim-e2e",
        action="store_true",
        help="raw NAVSIM -> index -> progressive GPU train -> validation artifacts",
    )
    parser.add_argument(
        "--e2e-profile",
        choices=("full", "quick"),
        default="full",
        help="full trains 144p/216p/432p; quick only verifies the complete pipeline",
    )
    parser.add_argument(
        "--navsim-stage",
        "--navsim-smoke",
        dest="navsim_smoke",
        action="store_true",
        help="advanced: run one configurable NAVSIM train/evaluation stage",
    )
    parser.add_argument(
        "--navsim-root", default=r"D:\navsim_workspace\dataset"
    )
    parser.add_argument(
        "--navsim-index", default="artifacts/navsim_index.json"
    )
    parser.add_argument("--rebuild-navsim-index", action="store_true")
    parser.add_argument("--max-index-samples", type=int)
    parser.add_argument("--train-samples", type=int, default=8)
    parser.add_argument("--validation-samples", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--backbone-lr-scale", type=float, default=0.1)
    parser.add_argument("--gradient-clip", type=float, default=5.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--depth-weight", type=float, default=0.3)
    parser.add_argument("--image-width", type=int, default=192)
    parser.add_argument("--image-height", type=int, default=108)
    parser.add_argument("--smoke-score-threshold", type=float, default=0.20)
    parser.add_argument("--smoke-output-dir", default="artifacts/navsim_e2e")
    parser.add_argument("--resume-smoke-checkpoint", default="")
    parser.add_argument(
        "--init-checkpoint",
        default="",
        help="load compatible weights and continue training with a fresh optimizer",
    )
    parser.add_argument("--train-backbone", action="store_true")
    parser.add_argument(
        "--unfreeze-backbone-stages",
        type=int,
        choices=(0, 1, 2, 3, 4),
        default=0,
        help="unfreeze N deepest ResNet stages while keeping earlier BN frozen",
    )
    parser.add_argument("--no-train-augmentation", action="store_true")
    parser.add_argument("--uniform-training-sampling", action="store_true")
    parser.add_argument("--no-pretrained-backbone", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="result.jpg")
    parser.add_argument("--json-output", default="result.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.20)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--ego-from-camera-npy", default="")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    if args.navsim_e2e:
        run_navsim_e2e(args)
        return
    if args.navsim_smoke:
        run_navsim_smoke(args)
        return
    required = (args.image, args.checkpoint, args.fx, args.fy, args.cx, args.cy)
    if any(value is None for value in required):
        parser.error(
            "inference requires --image --checkpoint --fx --fy --cx --cy"
        )

    model = MeteorLikeVRUDepthModel(pretrained_backbone=False)
    checkpoint_payload = load_checkpoint(model, args.checkpoint)
    checkpoint_hw = tuple(
        checkpoint_payload.get("config", {}).get("image_hw", (432, 768))
    )
    transform = (
        None
        if not args.ego_from_camera_npy
        else np.load(args.ego_from_camera_npy)
    )
    pipeline = InferencePipeline(
        model,
        CameraCalibration(args.fx, args.fy, args.cx, args.cy, transform),
        device=args.device,
        input_hw=checkpoint_hw,
    )
    image = Image.open(args.image).convert("RGB")
    objects, _ = pipeline.predict(image, args.score_threshold)
    pipeline.annotate(image, objects).save(args.output)
    records = [
        {
            "detection": asdict(item.detection),
            "depth": None if item.depth is None else asdict(item.depth),
            "distance_band": item.distance_band,
            "requires_attention": item.requires_attention,
        }
        for item in objects
    ]
    Path(args.json_output).write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"objects={len(objects)} image={args.output} json={args.json_output}")


if __name__ == "__main__":
    main()
