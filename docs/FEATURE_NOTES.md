# Ghi chú: S3D vs CLIP Feature Extraction trong UniVL

## 1. MSRVTT Dataset

| File | Kích thước | Mô tả |
|------|-----------|-------|
| `MSRVTT_data.json` | 23MB | Annotations, captions |
| `msrvtt_videos_features.pickle` | 292MB | S3D features đã extract sẵn |
| `MSRVTT_train.9k.csv` | 87KB | Train split (9k videos) |
| `MSRVTT_JSFUSION_test.csv` | 73KB | Test split |

**Cấu trúc pickle:**
```python
# dict: {video_id: np.array}
pickle['video1436'] = np.array(12, 1024)  # shape (T_seconds, 1024), dtype=float16
```

---

## 2. S3D Feature Extraction

### Pipeline

```
Video gốc (bất kỳ fps: 24/25/30...)
   ↓ ffmpeg resample CỐ ĐỊNH → 16fps
   Video 12s × 16fps = 192 frames
   ↓ chia no-overlap clips, mỗi clip 16 frames
   192 ÷ 16 = 12 clips
   ↓ reshape → tensor (12, 3, 16, 224, 224)
   ↓ S3D forward
   tensor (12, 1024)
   ↓ L2 normalize → lưu pickle float16
```

### Shape tại mỗi bước

| Bước | Shape | Ghi chú |
|------|-------|---------|
| Raw video | `(192, 3, 224, 224)` | T = 12s × 16fps, float32 |
| Sau normalize | `(192, 3, 224, 224)` | ÷255, [0,1] |
| Sau reshape | `(12, 3, 16, 224, 224)` | chia clip 16 frames |
| **S3D input** | **`(12, 3, 16, 224, 224)`** | batch=12 clips |
| **S3D output** | **`(12, 1024)`** | 1 vector/clip |
| Lưu pickle | `(12, 1024)` | float16 |

### Tại sao `shape[0] = số giây`?

```
1 clip = 16 frames liên tiếp @ 16fps = 1 giây
→ shape[0] = số clips = số giây video
```

### S3D học temporal như thế nào?

- **Trong clip (16 frames):** S3D dùng **3D Convolution** — kernel `(k_t, k_h, k_w)` học chuyển động ✅
- **Giữa các clips:** S3D **không học** — 12 vectors là độc lập nhau ❌
- **Giữa các clips:** Do **UniVL Visual Encoder (Transformer)** xử lý ✅

---

## 3. CLIP Feature Extraction

### Pipeline

```
Video 12s, fps gốc = 30fps
   ↓ cv2 đọc, sample_every = round(30/1) = 30
   lấy frame tại: f0, f30, f60,... → 12 frames (1fps)
   ↓ preprocess: Resize → CenterCrop 224×224 → Normalize ImageNet
   tensor (12, 3, 224, 224)
   ↓ CLIP ViT-B/32 encode_image() — từng frame độc lập
   tensor (12, 512)
   ↓ .float().cpu().numpy()
   np.array (12, 512)  float32
```

### Shape tại mỗi bước

| Bước | Shape | Ghi chú |
|------|-------|---------|
| Sample frames | `List[12 PIL Images]` | 1fps, 12 frames/12s |
| Sau preprocess | `(12, 3, 224, 224)` | CenterCrop, Normalize |
| **CLIP input** | **`(12, 3, 224, 224)`** | batch=12 frames ĐỘC LẬP |
| **CLIP output** | **`(12, 512)`** | 1 vector/frame |

### CLIP không học temporal

```
frame_0: f0  → CLIP → vector_0   ┐
frame_1: f1  → CLIP → vector_1   ├── ĐỘC LẬP, không biết thứ tự
frame_2: f2  → CLIP → vector_2   ┘

→ Transformer trong UniVL + positional embedding bù đắp temporal
```

---

## 4. S3D vs CLIP: So sánh trực tiếp

| | S3D | CLIP |
|--|-----|------|
| **Pretrain task** | Action Recognition (Kinetics-400) | Image-Text Matching (400M pairs) |
| **FPS lấy frame** | ffmpeg resample 16fps | cv2 sample 1fps |
| **Frames/video 12s** | 192 frames | 12 frames |
| **Đơn vị xử lý** | clip 16 frames | 1 frame |
| **Shape input** | `(12, 3, 16, 224, 224)` | `(12, 3, 224, 224)` |
| **Shape output** | `(12, 1024)` | `(12, 512)` |
| **1 vector = ?** | tổng hợp chuyển động 1 giây | snapshot 1 frame |
| **Temporal local** | ✅ 3D Conv trong 16 frames | ❌ không có |
| **Temporal global** | ❌ → Transformer bù | ❌ → Transformer bù |
| **Semantic/language alignment** | ❌ yếu | ✅ mạnh |
| **Object/scene recognition** | ❌ yếu | ✅ mạnh |

---

## 5. Tại sao CLIP giúp CIDEr tăng so với S3D?

### MSRVTT captions thiên về appearance

```
"a woman is talking to the camera"    ← appearance
"a dog is running on the grass"       ← object + scene
"a man is playing guitar"             ← object recognition quan trọng
"a news anchor is reporting"          ← scene understanding
```

Phần lớn caption cần **nhận ra object/scene**, không cần phân tích chuyển động phức tạp.

### CLIP có lợi thế gì?

```
S3D:
  Giỏi: phân biệt CHUYỂN ĐỘNG (cách vật thể di chuyển)
  Không học: tên objects cụ thể, scenes, ngôn ngữ

CLIP:
  Giỏi: nhận diện mọi visual concept có trong ngôn ngữ
  Feature space ALIGN với text → caption đúng từ hơn
  Không học: temporal
```

### Phân tích gains/losses

```
CLIP mang lại:
  ✅ Semantic alignment với ngôn ngữ  → caption đúng từ hơn
  ✅ Object/scene recognition tốt    → caption đủ thông tin hơn
  ✅ Pretrain 400M image-text pairs  → feature phong phú hơn

CLIP mất đi:
  ❌ Local temporal (trong 16 frames)
     → nhưng Transformer của UniVL BÙ ĐƯỢC

Trên MSRVTT:
  Gains từ semantic >> Losses từ temporal
  → CIDEr tăng ✓
```

---

## 6. Phân công nhiệm vụ trong UniVL

```
S3D / CLIP:
  → Extract visual features offline
  → Lưu pickle để tái sử dụng

UniVL Visual Encoder (Transformer, 6 layers):
  Input:  (B, 48, 1024) hoặc (B, 48, 512) sau pad
  Output: (B, 48, dim)  — contextualized features
  Học:    temporal dài hạn giữa các clips (self-attention)

UniVL Cross-Encoder:
  Học: quan hệ giữa video features và text features

UniVL Decoder:
  Sinh: caption word-by-word
```
---
