# METEOR-like Vehicle/VRU Detection + Metric Depth

Toàn bộ source nằm trong duy nhất [model.py](./model.py).

## Kiến trúc

```text
RGB image [B,3,H,W]
  → ResNet-34 ImageNet backbone
  → FPN 160 channels @ stride 4
       ├─ depth decoder: 64 bins, 1.0–79.75 m, stride 4
       └─ CenterNet detector:
            ├─ s4:  object nhỏ <40 px
            ├─ s8:  object 40–120 px
            └─ s16: object lớn ≥120 px
                 ├─ channel 0: Vehicle
                 └─ channel 1: VRU
  → decode + class-wise NMS
  → robust depth trong bbox
  → camera XYZ / ego XYZ
  → critical / near / far / unknown_depth
```

Đây là phần image-space được rút từ cấu hình METEOR: ResNet-34, FPN 160,
depth classification 64 bins và detection pyramid s4/s8/s16. File không chứa
toàn bộ multi-camera depth-weighted IPM, BEV segmentation hay planner của
METEOR vì bài toán ở đây chỉ cần Vehicle/VRU detection và khoảng cách.

Loss detection dùng positive đồng đều và online hard-negative mining, không có
`VRU_CW` hoặc multiplier theo khoảng cách. Depth loss gồm bin cross-entropy và
L1 trên expected metric depth.

## Lưu ý bắt buộc

ImageNet chỉ khởi tạo backbone. Detection head hai lớp và depth head chưa có
pretrained weights phù hợp, vì vậy phải train model và tạo checkpoint trước khi
chạy inference thật. Không dùng output random của self-test làm kết quả.

## Cài đặt và self-test

```powershell
Set-Location "D:\AI thực chiến\VinFast\near-vru-learning"
python -m pip install -r requirements.txt
python model.py --self-test
```

## Dữ liệu train

- `images`: float RGB `[B,3,H,W]` trong `[0,1]`.
- `boxes`: `[B,K,5]` gồm `[class,cx,cy,width,height]` theo pixel.
- `class=0`: Vehicle; `class=1`: VRU.
- `counts`: số box hợp lệ của từng ảnh.
- `depth_gt_m`: camera-Z metric depth theo mét.

```python
output = model(images)
loss_det = model.detection_loss(output, boxes, counts)
loss_depth = model.depth_loss(output, depth_gt_m)
loss = loss_det + depth_weight * loss_depth
loss.backward()
```

## Inference bằng checkpoint đã train

```powershell
python model.py `
  --image frame.jpg `
  --checkpoint checkpoint.pt `
  --fx 910 --fy 910 --cx 384 --cy 216 `
  --output result.jpg `
  --json-output result.json
```

Nếu có extrinsic camera→ego 4×4:

```powershell
python model.py ... --ego-from-camera-npy ego_from_camera.npy
```

Không có extrinsic thì khoảng cách chỉ là Euclidean range từ tâm camera, chưa
phải khoảng cách ego đã hiệu chuẩn.
