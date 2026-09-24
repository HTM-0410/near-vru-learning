# Near-VRU Detection + Metric Depth

Repository tối giản gồm một file triển khai chính: `model.py`. Pipeline nhận
NAVSIM/OpenScene raw data, tạo target, train Vehicle/VRU detection cùng metric
depth, đánh giá trên recording held-out và xuất checkpoint/ảnh/JSON kết quả.

## 1. Phạm vi

Model triển khai phần image-space lấy cảm hứng từ METEOR:

```text
CAM_F0 RGB
  → ResNet-34 ImageNet backbone
  → top-down FPN C5→C2, 160 channels
       ├─ P2/s4 depth head: 64 log-spaced bins, 1.0–79.75 m
       ├─ P2/s4 detector: object <40 px
       ├─ P3/s8 detector: object 40–120 px
       └─ P4/s16 detector: object ≥120 px
  → Vehicle/VRU bbox + dense predicted camera-Z
  → robust depth trong bbox
  → camera XYZ / optional ego XYZ
  → critical / near / far
```

Không bao gồm multi-camera IPM, BEV segmentation, tracking hoặc planner đầy
đủ của METEOR.

## 2. Cấu trúc dữ liệu NAVSIM

`--navsim-root` phải trỏ tới thư mục có cấu trúc:

```text
dataset/
├─ navsim_logs/
│  └─ mini/*.pkl
└─ sensor_blobs/
   └─ mini/<recording>/
      ├─ CAM_F0/*.jpg
      └─ MergedPointCloud/*.pcd
```

Code tự thực hiện:

1. Đọc metadata NAVSIM cục bộ.
2. Gom `pedestrian`, `bicycle`, `motorcycle` thành VRU; `vehicle` thành Vehicle.
3. Chiếu annotation box 3D LiDAR thành bbox 2D CAM_F0.
4. Chiếu point cloud thành sparse camera-Z depth target.
5. Split train/validation theo recording, không random frame giữa cùng log.

## 3. Cài đặt

```powershell
Set-Location "D:\AI thực chiến\VinFast\near-vru-learning"
py -3.12 -m pip install -r requirements.txt
py -3.12 model.py --self-test
```

Nếu dùng NVIDIA GPU trên Windows, phải cài PyTorch CUDA thay vì wheel `+cpu`.
Cấu hình đã kiểm chứng trên RTX 5060 Laptop, driver CUDA 13.2:

```powershell
py -3.12 -m pip install --force-reinstall --no-cache-dir --no-deps `
  torch==2.12.1+cu130 torchvision==0.27.1+cu130 `
  --index-url https://download.pytorch.org/whl/cu130

py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 4. Chạy toàn bộ pipeline bằng một lệnh

```powershell
py -3.12 model.py `
  --navsim-e2e `
  --navsim-root "D:\navsim_workspace\dataset" `
  --device cuda `
  --smoke-output-dir "artifacts\navsim_e2e"
```

Profile `full` mặc định tự chạy ba phase progressive resizing:

| Phase | Train frames | Epoch | Resolution | Head LR |
|---|---:|---:|---:|---:|
| Stage 1 | 512 | 6 | 144×256 | 3e-4 |
| Stage 2 | 768 | 3 | 216×384 | 1.5e-4 |
| Final | 1.024 | 3 | 432×768 | 7.5e-5 |

Pipeline tự động:

- Dựng lại portable NAVSIM index từ raw data.
- Lấy mẫu 50% near-VRU≤30 m, 30% far-VRU, 20% context.
- Dùng BF16, pinned memory và 2 workers khi chạy CUDA.
- Fine-tune `layer3/layer4`; backbone LR bằng 10% head LR.
- Dùng AdamW, cosine decay, gradient clipping và augmentation đồng bộ.
- Đánh giá final checkpoint trên 128 frame của recording held-out.
- Xoá checkpoint progressive trung gian sau khi phase cuối thành công.

Output cuối duy nhất:

```text
artifacts/navsim_e2e/
├─ navsim_index.json
├─ navsim_smoke.pt
├─ projected_gt.jpg
├─ prediction.jpg
└─ result.json
```

Kiểm tra nhanh toàn bộ I/O trước khi chạy full:

```powershell
py -3.12 model.py `
  --navsim-e2e `
  --e2e-profile quick `
  --navsim-root "D:\navsim_workspace\dataset" `
  --device cuda `
  --smoke-output-dir "artifacts\navsim_quick"
```

`quick` chạy hai phase nhỏ để kiểm tra cả truyền checkpoint progressive và tự
xoá stage tạm; nó chỉ chứng minh pipeline chạy được, không phải accuracy.

## 5. Inference một ảnh

```powershell
py -3.12 model.py `
  --image frame.jpg `
  --checkpoint "artifacts\navsim_e2e\navsim_smoke.pt" `
  --fx 1545 --fy 1545 --cx 960 --cy 560 `
  --device cuda `
  --score-threshold 0.20 `
  --output result.jpg `
  --json-output result.json
```

Inference tự đọc resolution train từ checkpoint. Nếu có camera→ego extrinsic:

```powershell
py -3.12 model.py ... --ego-from-camera-npy ego_from_camera.npy
```

Không có extrinsic thì `ego_distance_m` chỉ là Euclidean range từ camera, chưa
phải khoảng cách ground-plane tới ego đã hiệu chuẩn.

## 6. Loss và xử lý mất cân bằng

Detection loss:

```text
Gaussian focal loss + online hard-negative mining
+ Smooth-L1 offset/log-size
+ GIoU bbox
```

Mọi positive Vehicle/VRU có cùng trọng số. Code không dùng `VRU_CW` hoặc nhân
loss theo khoảng cách. Near-VRU được tăng hiện diện bằng scenario sampling ở
cấp frame, không sửa gradient của từng object.

Depth loss:

```text
64 geometric/log-bin cross entropy + 0.1 × continuous metric L1
```

Tâm bin khớp METEOR `depth_log_bins`:

```text
center(i) = exp(log(1.0) + i/63 × (log(79.75) - log(1.0)))
```

GT camera-Z được `bucketize` rồi nội suy giữa hai tâm kề nhau trước khi tạo CE
target. Train dùng full expected metric depth; inference dùng expectation trong
modal bin ±2 giống đường deployment METEOR để tránh trộn foreground/background.

Loss tổng:

```text
L = L_detection + depth_weight × L_depth
```

## 7. Lệnh một stage nâng cao

`--navsim-stage` dành cho experiment hoặc đánh giá lại checkpoint. Ví dụ chỉ
evaluation, không train và không nhân đôi file checkpoint:

```powershell
py -3.12 model.py `
  --navsim-stage `
  --navsim-root "D:\navsim_workspace\dataset" `
  --navsim-index "artifacts\navsim_e2e\navsim_index.json" `
  --train-samples 1024 `
  --validation-samples 128 `
  --image-width 768 `
  --image-height 432 `
  --device cuda `
  --resume-smoke-checkpoint "artifacts\navsim_e2e\navsim_smoke.pt" `
  --smoke-output-dir "artifacts\navsim_eval"
```

`--init-checkpoint` khác `--resume-smoke-checkpoint`:

- `--init-checkpoint`: nạp weights, tạo optimizer mới rồi tiếp tục train.
- `--resume-smoke-checkpoint`: bỏ qua train và chỉ evaluation.

## 8. Giới hạn hiện tại

- Validation và threshold tuning vẫn dùng một recording; cần recording test kín.
- Depth supervision là sparse LiDAR, chưa có object-aware depth target.
- Không có temporal tracking/multi-frame fusion.
- Safety band chỉ là camera range nếu thiếu camera→ego extrinsic.
- Không coi loss giảm hoặc một ảnh đẹp là bằng chứng model đủ an toàn.

Checkpoint/dataset/output nằm trong `artifacts/` và không được commit lên Git.
Checkpoint linear-bin cũ không tương thích và bị từ chối khi load; phải train
lại bằng `--navsim-e2e` sau thay đổi log-bin.

## 9. Kết quả log-bin hiện tại

Full E2E log-bin đã chạy 2.112 step trên GPU và đánh giá 128 frame held-out,
score threshold 0.20, IoU ≥0.5:

| Metric | Linear-bin cũ | Log-bin mới |
|---|---:|---:|
| Vehicle TP / FP / FN | 50 / 59 / 21 | **51 / 55 / 20** |
| VRU TP / FP / FN | 57 / 93 / 232 | **59 / 95 / 230** |
| VRU precision / recall | 38,0% / 19,7% | **38,3% / 20,4%** |
| Near-VRU≤30 m TP / FN | 20 / 63 | **23 / 60** |
| Near-VRU recall | 24,1% | **27,7%** |
| VRU object camera-Z MAE | 7,68 m | **6,56 m** |
| Near-VRU object camera-Z MAE | 5,10 m | **4,98 m** |
| Sparse pixel depth MAE | **4,93 m** | 5,29 m |

Log-bin cải thiện mục tiêu object/near-VRU nhưng sparse pixel MAE kém hơn. Đây
là validation trên recording đã dùng để chọn cấu hình, chưa phải test độc lập
và chưa đủ độ chính xác cho quyết định an toàn.
