# UniVL: Các Giai Đoạn Huấn Luyện

## Tổng quan kiến trúc 4 khối

```
┌────────────────────────────────────────────────────────────────────┐
│                        UniVL Architecture                          │
│                                                                    │
│  [Text Encoder]     [Video Encoder]                                │
│  BERT-base          Transformer                                    │
│  12 layers          1 layer                                        │
│  hidden=768         hidden=768                                     │
│       │                    │                                       │
│       └──────────┬─────────┘                                       │
│                  ▼                                                  │
│          [Cross Encoder]                                           │
│          Transformer                                               │
│          2 layers, hidden=768                                      │
│          Bidirectional                                             │
│                  │                                                  │
│                  ▼                                                  │
│            [Decoder]                                               │
│            Transformer Decoder                                     │
│            12 layers, hidden=768                                   │
│            Causal (auto-regressive)                                │
└────────────────────────────────────────────────────────────────────┘
```

### Shapes chi tiết

| Module | Input Shape | Output Shape | Ghi chú |
|--------|------------|--------------|---------|
| Text Encoder (BERT) | `(B, L)` token ids | `(B, L, 768)` | L = max_words |
| Video Encoder | `(B, T, 1024)` S3D hoặc `(B, T, 768)` CLIP | `(B, T, 768)` | T = max_frames |
| Cross Encoder | `(B, L+T, 768)` concat | `(B, L+T, 768)` | Bidirectional self-attention |
| Decoder | caption ids `(B, S)` + encoder_outs `(B, L+T, 768)` | `(B, S, 30522)` logits | S = max_words, vocab=30522 |

---

## Pre-training Data

- **Dataset**: HowTo100M
- **Quy mô**: ~1.2 triệu video hướng dẫn
- **Trung bình**: mỗi video dài 6.5 phút, ~110 cặp clip-text
- **Text**: ASR transcript (noisy, không khớp hoàn toàn về thời gian)
- **Video features**: Trích xuất bằng S3D (dim=1024)
- **Hardware**: 8x NVIDIA Tesla V100

---

## Stage 1: Huấn luyện 2 Unimodal Encoders

### Modules được train

```
✅ Text Encoder   (BERT)      ← UPDATE weights
✅ Video Encoder  (Visual)    ← UPDATE weights
❌ Cross Encoder              ← KHÔNG train
❌ Decoder                    ← KHÔNG train
```

### Code tương ứng trong modeling.py

```python
# modeling.py - __init__
self._stage_one = True
self._stage_two = False

# modeling.py - forward()
if self._stage_one:
    sim_matrix = self.get_similarity_logits(
        sequence_output, visual_output,
        attention_mask, video_mask, shaped=True
    )
    sim_loss = self.loss_fct(sim_matrix)
    loss += sim_loss
```

### Mục tiêu duy nhất: Video-Text Joint (MIL-NCE)

```
Text Encoder output:   (B, L, 768) → mean pooling → (B, 768)
Video Encoder output:  (B, T, 768) → mean pooling → (B, 768)
                                ↓
                    MIL-NCE Contrastive Loss
                    (Multiple Instance Learning -
                     Noise Contrastive Estimation)
```

**Tại sao dùng MIL-NCE thay vì NCE thông thường?**

HowTo100M bị **noisy** — một đoạn text có thể khớp với nhiều clip khác nhau trong cùng video (không chỉ clip hiện tại). MIL-NCE cho phép một text match với **nhiều video clips** thay vì chỉ 1, giải quyết vấn đề weak alignment.

### Tại sao cần Stage 1?

```
Nếu train thẳng tất cả 4 khối ngay từ đầu:
  Cross Encoder nhận input từ Text + Video Encoder
  → Hai encoder chưa được align
  → Cross Encoder học trên noisy, unaligned features
  → Training không ổn định

Stage 1 giải quyết:
  → Text và Video Encoder được align trước
  → Stage 2 bắt đầu với nền tảng vững chắc hơn
```

### Thời gian: ~1.5 ngày (8x V100)

---

## Stage 2: Huấn luyện Toàn bộ 4 Modules

### Modules được train

```
✅ Text Encoder   (BERT)      ← CONTINUE training
✅ Video Encoder  (Visual)    ← CONTINUE training
✅ Cross Encoder              ← BẮT ĐẦU train
✅ Decoder                    ← BẮT ĐẦU train
```

### Code tương ứng trong modeling.py

```python
# modeling.py - __init__
self._stage_one = False
self._stage_two = True   # task_config.stage_two = True
```

### 5 Mục tiêu huấn luyện (Pre-training Objectives)

```
L_UniVL = L_joint + L_CMLM + L_CMFM + L_align + L_recon
```

#### 1. Video-Text Joint (MIL-NCE) — tiếp tục từ Stage 1

```python
# Trong forward() - Stage 2
sim_matrix = self.get_similarity_logits(
    sequence_output, visual_output, ..., _pretrain_joint=True
)
sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
loss += sim_loss_joint
```

#### 2. CMLM — Conditioned Masked Language Model

```
Input:  Text với 15% token bị MASK ngẫu nhiên + Video đầy đủ
Target: Predict các token bị mask
Module: Cross Encoder → BertOnlyMLMHead

Shape:  sequence_cross_output (B, L, 768)
        → cls head → (B, L, 30522)
        → CrossEntropyLoss với pairs_token_labels
```

```python
alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
loss += alm_loss
```

#### 3. CMFM — Conditioned Masked Frame Model

```
Input:  Video với 15% frame bị MASK + Text đầy đủ
Target: Identify frame đúng (contrastive giữa các frames)
Module: Cross Encoder → VisualOnlyMLMHead

Shape:  visual_cross_output (B, T, 768)
        → visual cls head → (B, T, 1024)
        → NCE Loss với video_labels_index
```

```python
nce_loss = self._calculate_mfm_loss(
    visual_cross_output, video, video_mask, video_labels_index
)
loss += nce_loss
```

#### 4. Video-Text Alignment

```
Input:  Cặp (Video, Text) matched hoặc mismatched
Target: Phân biệt matched vs mismatched
Module: Cross Encoder → similarity_dense

Shape:  pooled_output (B, 768) → Linear(768, 1) → score
        → CrossEn Loss
```

```python
sim_matrix_text_visual = self.get_similarity_logits(
    sequence_output_alm, visual_output_alm, ...
)
sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
loss += sim_loss_text_visual
```

#### 5. Language Reconstruction (CLM)

```
Input:  Video + Text + Caption prefix (teacher forcing)
Target: Generate toàn bộ caption
Module: Cross Encoder → Decoder

Shape:  cross_output (B, L+T, 768) → Decoder
        input_caption_ids (B, S) → teacher forcing
        → logits (B, S, 30522)
        → CrossEntropyLoss với output_caption_ids
```

```python
decoder_scores, _ = self._get_decoder_score(
    sequence_output_alm, visual_output_alm,
    input_ids, attention_mask, video_mask,
    input_caption_ids, decoder_mask, shaped=True
)
decoder_loss = self.decoder_loss_fct(
    decoder_scores.view(-1, self.bert_config.vocab_size),
    output_caption_ids.view(-1)
)
loss += decoder_loss
```

### Chiến lược EnhancedV

```
Xác suất 15%: input_text bị MASK HOÀN TOÀN
                      ↓
  Model buộc phải dựa 100% vào Video
  để reconstruct caption
                      ↓
  → Ép model học video representation tốt hơn
  → Tránh model phụ thuộc quá nhiều vào text side
```

### Thời gian: ~12 ngày (8x V100)

---

## Fine-tuning (Task-specific)

### Cấu hình

```python
# task_config
do_pretrain = False
task_type   = "caption"
stage_two   = True        # dùng toàn bộ 4 khối
```

### Code trong forward()

```python
# Chỉ tính decoder loss, không có MLM/MFM/Sim losses
if self.task_config.task_type == "caption":
    decoder_scores, _ = self._get_decoder_score(
        sequence_output, visual_output,
        input_ids, attention_mask, video_mask,
        input_caption_ids, decoder_mask, shaped=True
    )
    decoder_loss = self.decoder_loss_fct(
        decoder_scores.view(-1, self.bert_config.vocab_size),
        output_caption_ids.view(-1)
    )
    loss += decoder_loss
```

### Dataset & Kết quả

| Features | Dataset | CIDEr |
|----------|---------|-------|
| S3D (dim=1024) | MSRVTT | 50 |
| CLIP ViT-L/14 (dim=768) | MSRVTT | **58** |

### Khi dùng CLIP ViT-L/14 thay S3D

```
Pretrained checkpoint (S3D-based):
  visual.embeddings.word_embeddings: Linear(1024, 768)  ← RANDOM INIT
  visual.encoder.*:                                     ← LOADED
  bert.*:                                               ← LOADED
  cross.*:                                              ← LOADED
  decoder.*:                                            ← LOADED

Tại sao CLIP cho CIDEr cao hơn?
  S3D pretrain: Kinetics-400, action classification
  → features action-centric, nghèo semantic

  CLIP pretrain: 400M image-text pairs, contrastive
  → features giàu semantic, gần với ngôn ngữ tự nhiên
  → Decoder dễ generate caption phong phú hơn
  → Video Encoder Transformer đã xử lý temporal
     → temporal không còn là lợi thế của S3D
```

---

## Sơ đồ tổng thể

```
HowTo100M (1.2M videos, ASR noisy)
            │
            ▼
┌───────────────────────────────────┐
│         STAGE 1 (~1.5 ngày)       │
│                                   │
│   Text Enc ←── MIL-NCE ──→ Vid Enc│
│            (align spaces)         │
└───────────────────────────────────┘
            │
            ▼ load Stage 1 weights
┌───────────────────────────────────┐
│         STAGE 2 (~12 ngày)        │
│                                   │
│   Text Enc  +  Vid Enc            │
│        ↓                          │
│   Cross Encoder                   │
│        ↓                          │
│   Decoder                         │
│                                   │
│   5 objectives: MIL-NCE + CMLM   │
│                + CMFM + Align     │
│                + Recon            │
└───────────────────────────────────┘
            │
            ▼ load Stage 2 weights
┌───────────────────────────────────┐
│      FINE-TUNING (MSRVTT)         │
│                                   │
│   S3D features  → CIDEr = 50      │
│   CLIP ViT-L/14 → CIDEr = 58      │
└───────────────────────────────────┘
```
