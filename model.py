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
      -> FPN 160 channels @ stride 4
          |-> depth decoder: [B,64,H/4,W/4]
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
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision.ops import nms


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
        # Chiếu cả bốn ResNet stage về 160 channel. Các stage s8/s16/s32 được
        # bilinear-upsample về s4 trước khi cộng. Nhờ vậy feature cuối vừa giữ
        # chi tiết của object nhỏ, vừa có receptive field của tầng sâu.
        self.lat1 = nn.Conv2d(64, 160, 1)
        self.lat2 = nn.Conv2d(128, 160, 1)
        self.lat3 = nn.Conv2d(256, 160, 1)
        self.lat4 = nn.Conv2d(512, 160, 1)
        self.fpn_fuse = nn.Sequential(
            nn.Conv2d(160, 160, 3, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.ReLU(inplace=True),
        )

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
        self.det_s8 = nn.Sequential(
            nn.Conv2d(128, 192, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            ConvBlock(192, 192),
        )
        self.det_s16 = nn.Sequential(
            nn.Conv2d(192, 256, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ConvBlock(256, 256),
        )
        self.heatmap_heads = nn.ModuleList(
            [nn.Conv2d(128, 2, 1), nn.Conv2d(192, 2, 1), nn.Conv2d(256, 2, 1)]
        )
        self.regression_heads = nn.ModuleList(
            [nn.Conv2d(128, 4, 1), nn.Conv2d(192, 4, 1), nn.Conv2d(256, 4, 1)]
        )
        for head in self.heatmap_heads:
            nn.init.constant_(head.bias, -2.19)  # initial p ~= 0.10

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

    def _image_features(self, images: torch.Tensor) -> torch.Tensor:
        """Trích xuất và hợp nhất ResNet features thành FPN stride-4.

        Với input 432x768, tensor trả về có shape [B,160,108,192].
        """
        x0 = self.stem(images)
        x1 = self.layer1(x0)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        size = x1.shape[-2:]
        fused = (
            self.lat1(x1)
            + F.interpolate(self.lat2(x2), size=size, mode="bilinear", align_corners=False)
            + F.interpolate(self.lat3(x3), size=size, mode="bilinear", align_corners=False)
            + F.interpolate(self.lat4(x4), size=size, mode="bilinear", align_corners=False)
        )
        return self.fpn_fuse(fused)

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
        feature_s4 = self._image_features(normalized)

        # Depth expectation E[z] = sum_i P(bin_i)*centre_i. Giữ depth_logits
        # riêng vì training cần cross-entropy trên bin, còn inference dùng E[z].
        depth_logits = self.depth_head(feature_s4)
        depth_probability = depth_logits.float().softmax(1)
        depth_m = (
            depth_probability * self.depth_centres_m[None, :, None, None]
        ).sum(1)

        # Tạo pyramid detection tuần tự s4 -> s8 -> s16.
        det4 = self.det_s4(feature_s4)
        det8 = self.det_s8(det4)
        det16 = self.det_s16(det8)
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
        min_negatives: int = 64,
        negative_ratio: int = 3,
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
            regression_loss = (
                (regression.float() - reg_target).abs() * reg_mask
            ).sum() / reg_mask.sum().clamp(min=1) / 4
            total = total + heatmap_loss + regression_loss
        return total / len(DET_STRIDES)

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
                    center_y = (row + float(offset[0])) * stride
                    center_x = (col + float(offset[1])) * stride
                    box_w = float(offset[2].clamp(-4, 7).exp()) * stride
                    box_h = float(offset[3].clamp(-4, 7).exp()) * stride
                    x1 = max(0.0, center_x - box_w / 2)
                    y1 = max(0.0, center_y - box_h / 2)
                    x2 = min(float(input_w), center_x + box_w / 2)
                    y2 = min(float(input_h), center_y + box_h / 2)
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
# BLOCK 9 — CHECKPOINT I/O
# =============================================================================

def load_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
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


# =============================================================================
# BLOCK 10 — SELF-TEST KHÔNG CẦN DATA/CHECKPOINT
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
# BLOCK 11 — COMMAND-LINE ENTRYPOINT
# =============================================================================

def main() -> None:
    """CLI cho self-test hoặc inference một ảnh bằng checkpoint đã train."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--image")
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", default="result.jpg")
    parser.add_argument("--json-output", default="result.json")
    parser.add_argument("--device", default=None)
    parser.add_argument("--score-threshold", type=float, default=0.3)
    parser.add_argument("--fx", type=float)
    parser.add_argument("--fy", type=float)
    parser.add_argument("--cx", type=float)
    parser.add_argument("--cy", type=float)
    parser.add_argument("--ego-from-camera-npy", default="")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        return
    required = (args.image, args.checkpoint, args.fx, args.fy, args.cx, args.cy)
    if any(value is None for value in required):
        parser.error(
            "inference requires --image --checkpoint --fx --fy --cx --cy"
        )

    model = MeteorLikeVRUDepthModel(pretrained_backbone=False)
    load_checkpoint(model, args.checkpoint)
    transform = (
        None
        if not args.ego_from_camera_npy
        else np.load(args.ego_from_camera_npy)
    )
    pipeline = InferencePipeline(
        model,
        CameraCalibration(args.fx, args.fy, args.cx, args.cy, transform),
        device=args.device,
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
