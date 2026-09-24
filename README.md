# METEOR-like Vehicle/VRU Detection + Metric Depth

Toàn bộ source nằm trong duy nhất [model.py](./model.py).

## Kiến trúc

```text
RGB image [B,3,H,W]
  → ResNet-34 ImageNet backbone
  → top-down FPN 160 channels: C5 → C4 → C3 → C2
       ├─ P2/s4 → depth decoder: 64 bins, 1.0–79.75 m
       └─ CenterNet detector nhận trực tiếp từng mức FPN:
            ├─ P2/s4:  object nhỏ <40 px
            ├─ P3/s8:  object 40–120 px
            └─ P4/s16: object lớn ≥120 px
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

Loss detection dùng positive đồng đều, hard-negative mining, Smooth-L1 và GIoU bbox,
không có `VRU_CW` hoặc multiplier theo khoảng cách. Train sampler chia scenario
near-VRU/far-VRU/context ở cấp frame; nó không nhân class/distance vào loss.
Depth loss gồm bin cross-entropy và L1 trên expected metric depth.

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

Trên Windows có NVIDIA GPU, cần cài PyTorch bản CUDA; bản `+cpu` sẽ không dùng
GPU dù `nvidia-smi` nhìn thấy card. Cấu hình đã kiểm chứng ở máy này là Python
3.12, RTX 5060 Laptop, driver CUDA 13.2 và wheel PyTorch CUDA 13.0:

```powershell
py -3.12 -m pip install --force-reinstall --no-cache-dir --no-deps `
  torch==2.12.1+cu130 torchvision==0.27.1+cu130 `
  --index-url https://download.pytorch.org/whl/cu130
py -3.12 -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Chỉ dùng lệnh `--no-deps` sau khi các dependency trong `requirements.txt` đã
được cài. Với máy/driver khác, chọn wheel tương thích tại trang PyTorch.

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

## Chạy pipeline NAVSIM mini

`model.py` có thể tự xử lý raw NAVSIM mini theo chuỗi:

```text
NAVSIM log + CAM_F0 + LiDAR PCD
  → portable JSON index bằng relative path
  → chiếu box 3D LiDAR thành bbox 2D Vehicle/VRU
  → project LiDAR thành sparse camera-Z depth
  → DataLoader
  → detection + depth smoke training
  → inference trên recording held-out
  → checkpoint + GT image + prediction image + JSON metric
```

Lệnh smoke đã kiểm chứng trên Windows/CPU:

```powershell
python model.py `
  --navsim-smoke `
  --navsim-root "D:\navsim_workspace\dataset" `
  --rebuild-navsim-index `
  --train-samples 512 `
  --validation-samples 32 `
  --epochs 6 `
  --batch-size 4 `
  --image-width 256 `
  --image-height 144 `
  --learning-rate 3e-4 `
  --backbone-lr-scale 0.1 `
  --unfreeze-backbone-stages 2 `
  --depth-weight 0.15 `
  --smoke-score-threshold 0.20 `
  --smoke-output-dir "artifacts\navsim_improved"
```

Output mặc định (có thể đổi bằng `--smoke-output-dir`):

```text
artifacts/navsim_index.json
artifacts/navsim_smoke/navsim_smoke.pt
artifacts/navsim_smoke/projected_gt.jpg
artifacts/navsim_smoke/prediction.jpg
artifacts/navsim_smoke/result.json
```

Train sample được chia near-VRU/far-VRU/context rồi lấy round-robin giữa các
recording. Augmentation flip/photometric được áp dụng đồng bộ ảnh, box và depth.
Toàn bộ validation sample thuộc một recording held-out, không xuất hiện trong train.
`result.json` ghi loss, sparse-depth MAE có trọng số theo pixel LiDAR và
precision/recall IoU 0.5 cộng dồn trên nhiều frame.

### Fine-tune bằng GPU ở độ phân giải cao

Vòng đã chạy từ checkpoint v2: 1.024 frame train, 3 epoch, 768×432, BF16,
2 worker, batch 4. Sau train đã đánh giá lại trên 128 frame held-out.
`--init-checkpoint` nạp trọng số nhưng tạo optimizer mới;
`--resume-smoke-checkpoint` chỉ đánh giá và không train.

```powershell
py -3.12 model.py `
  --navsim-smoke `
  --navsim-root "D:\navsim_workspace\dataset" `
  --navsim-index "artifacts\navsim_index.json" `
  --train-samples 1024 `
  --validation-samples 128 `
  --epochs 3 `
  --batch-size 4 `
  --num-workers 2 `
  --amp `
  --device cuda `
  --image-width 768 `
  --image-height 432 `
  --learning-rate 7.5e-5 `
  --backbone-lr-scale 0.1 `
  --unfreeze-backbone-stages 2 `
  --depth-weight 0.10 `
  --smoke-score-threshold 0.20 `
  --smoke-output-dir "artifacts\navsim_improved_v3" `
  --init-checkpoint "artifacts\navsim_improved_v2\navsim_smoke.pt"
```

Checkpoint được huấn luyện ở độ phân giải nào thì nên đánh giá/inference ở độ
phân giải đó. Đổi resolution chỉ lúc inference có thể làm cả depth và detection
giảm mạnh. Lệnh đánh giá lại checkpoint v3 mà không nhân đôi file `.pt`:

```powershell
py -3.12 model.py `
  --navsim-smoke `
  --navsim-root "D:\navsim_workspace\dataset" `
  --navsim-index "artifacts\navsim_index.json" `
  --train-samples 1024 `
  --validation-samples 128 `
  --epochs 3 `
  --batch-size 4 `
  --device cuda `
  --image-width 768 `
  --image-height 432 `
  --smoke-score-threshold 0.20 `
  --smoke-output-dir "artifacts\navsim_v3_val128" `
  --resume-smoke-checkpoint "artifacts\navsim_improved_v3\navsim_smoke.pt"
```

Nếu train đã xong nhưng bước đánh giá bị ngắt, có thể nạp checkpoint để chỉ chạy
lại held-out evaluation mà không train lại:

```powershell
python model.py `
  --navsim-smoke `
  --navsim-root "D:\navsim_workspace\dataset" `
  --navsim-index "artifacts\navsim_index.json" `
  --train-samples 512 `
  --validation-samples 32 `
  --epochs 6 `
  --batch-size 4 `
  --image-width 256 `
  --image-height 144 `
  --unfreeze-backbone-stages 2 `
  --depth-weight 0.15 `
  --resume-smoke-checkpoint "artifacts\navsim_improved\navsim_smoke.pt"
```

Đây vẫn là pipeline nghiên cứu nhỏ, không phải accuracy benchmark hoàn chỉnh.
Không được xem loss giảm hoặc một ảnh output đẹp là bằng chứng model đã hội tụ.
Threshold đã được xem trên cùng recording validation; để báo cáo độ chính xác
không thiên lệch cần thêm recording test chưa từng dùng để train/tune.

### Kết quả thử nghiệm hiện tại

Trên 128 frame cùng một recording held-out, IoU ≥0.5 và score ≥0.20:

| Checkpoint tại resolution train | Vehicle TP/FP/FN | VRU TP/FP/FN | Near-VRU ≤30 m TP/FN |
| --- | ---: | ---: | ---: |
| v2 384×216 | 46/75/25 | 23/150/266 | 12/71 |
| v3 768×432 | 50/59/21 | 57/93/232 | 20/63 |

VRU recall v3 vẫn chỉ 57/289 (19,7%); near-VRU recall 20/83 (24,1%).
Depth MAE trên sparse LiDAR là 4,74 m ở v2 và 4,93 m ở v3, nhưng hai lần
đo dùng lưới pixel khác nhau nên không được xem như so sánh depth trực tiếp.
Trong 57 VRU detection khớp GT của v3, camera-Z MAE theo object là 7,68 m;
riêng 20 near-VRU khớp là 5,10 m. Đây là lỗi khoảng cách tại object, khác với
pixel-level sparse-depth MAE và hiện vẫn quá cao cho ứng dụng an toàn.
Chưa có recording test độc lập để khẳng định khả năng tổng quát hóa.
