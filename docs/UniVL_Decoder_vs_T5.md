# UniVL Decoder: Phân tích và Đề xuất thay thế bằng T5

## 1. Decoder hiện tại hoạt động như thế nào?

### Kiến trúc Transformer Decoder chuẩn

```
Decoder config:
  hidden_size:        768
  num_hidden_layers:  12       ← 12 transformer layers
  num_decoder_layers: 1        ← số lần stack decoder block
  num_attention_heads: 12
  vocab_size:         30522    ← BERT vocabulary
  max_target_embeddings: 512
```

### Data flow chi tiết

```
Bước 1: Cross Encoder tạo ra context
─────────────────────────────────────
Text Encoder output:  (B, L, 768)   L = max_words  (e.g. 32)
Video Encoder output: (B, T, 768)   T = max_frames (e.g. 28)
           ↓ concat
    (B, L+T, 768) = (B, 60, 768)
           ↓ Cross Encoder (2 transformer layers, bidirectional)
    cross_output: (B, L+T, 768) = (B, 60, 768)   ← context chứa cả video lẫn text

Bước 2: Decoder generate caption (teacher forcing khi train)
─────────────────────────────────────────────────────────────
input_caption_ids: (B, S)  ← "A man is [MASK]..." (shifted right)
           ↓ DecoderEmbeddings
    (B, S, 768)   S = max_words
           ↓
    DecoderLayer (x num_decoder_layers=1):
    ┌──────────────────────────────────────────────────────┐
    │  Self-Attention (causal mask)                        │
    │    Q = K = V = caption embedding (B, S, 768)         │
    │    mask: lower triangular (token i chỉ thấy 0..i-1)  │
    │                    ↓                                 │
    │  Cross-Attention  ← ĐÂY LÀ CHÌA KHÓA                │
    │    Q = caption hidden (B, S, 768)                    │
    │    K = cross_output  (B, L+T, 768)                   │
    │    V = cross_output  (B, L+T, 768)                   │
    │    → Mỗi token caption "nhìn vào" toàn bộ video+text │
    │                    ↓                                 │
    │  FFN (768 → 3072 → 768)                              │
    └──────────────────────────────────────────────────────┘
           ↓ (B, S, 768)
    Linear(768, 30522)
           ↓
    logits: (B, S, 30522)
           ↓
    argmax → predicted token ids
           ↓
    BERT tokenizer decode → "A man is cooking"
```

### Code trong modeling.py

```python
def _get_decoder_score(self, sequence_output, visual_output,
                       input_ids, attention_mask, video_mask,
                       input_caption_ids, decoder_mask, shaped=False):

    # Bước 1: Lấy cross encoder output
    cross_output, pooled_output, concat_mask = self._get_cross_output(
        sequence_output, visual_output, attention_mask, video_mask
    )
    # cross_output: (B, L+T, 768)

    # Bước 2: Decoder forward với cross-attention vào cross_output
    decoder_scores = self.decoder(
        input_caption_ids,           # (B, S) BERT token ids
        encoder_outs=cross_output,   # (B, L+T, 768)  ← KEY INPUT
        answer_mask=decoder_mask,    # (B, S)
        encoder_mask=concat_mask     # (B, L+T)
    )
    # decoder_scores: (B, S, 30522)

    return decoder_scores, ()

# Loss calculation
decoder_loss = self.decoder_loss_fct(
    decoder_scores.view(-1, self.bert_config.vocab_size),  # (B*S, 30522)
    output_caption_ids.view(-1)                            # (B*S,)
)
```

---

## 2. Vấn đề của Decoder hiện tại

### Decoder nhỏ so với khả năng language generation hiện đại

```
UniVL Decoder:
  - Pretrain trên HowTo100M (domain hẹp: video hướng dẫn)
  - Vocabulary: 30522 (BERT wordpiece)
  - Language generation capability: hạn chế
  - Chỉ học từ ~1.2M videos → limited linguistic diversity

Kết quả: CIDEr = 58 (với CLIP ViT-L/14)
         → Còn dư địa để cải thiện phía language generation
```

---

## 3. Tại sao T5 là lựa chọn hợp lý?

### 3.1 T5 là Encoder-Decoder model → có sẵn Cross-Attention

```
GPT-2 / LLaMA architecture:
  Self-Attention → FFN
  ↑ KHÔNG CÓ cross-attention
  → KHÔNG NHẬN ĐƯỢC encoder_outs từ Cross Encoder
  → Không thể dùng trực tiếp

T5 Decoder architecture:
  Self-Attention → Cross-Attention → FFN
                        ↑
                   CÓ SẴN! Nhận encoder hidden states
  → Thay thế trực tiếp cho UniVL Decoder
```

### 3.2 Hidden dimension MATCH hoàn toàn

```
UniVL Cross Encoder output: (B, L+T, 768)
                                   ↑
                                 768!

T5-base hidden_size: 768
                     ↑
                   MATCH!

→ Không cần Projection Layer
→ cross_output đưa thẳng vào T5 Decoder
   như thể đây là T5 Encoder output
```

### 3.3 T5 được pretrain trên lượng data text khổng lồ

```
T5-base pretrain:
  Dataset: C4 (Colossal Clean Crawled Corpus)
  Size: 750GB text
  Task: span corruption (denoising)
  → Language generation capability vượt trội
  → Hiểu ngữ pháp, cấu trúc câu phong phú hơn

UniVL Decoder pretrain:
  Dataset: HowTo100M subtitles
  → Domain hẹp, ít đa dạng ngôn ngữ
```

### 3.4 So sánh T5 vs BART vs GPT-2

| | UniVL Decoder | T5-base | BART-base | GPT-2 |
|--|--------------|---------|-----------|-------|
| **Architecture** | Enc-Dec | Enc-Dec | Enc-Dec | Dec-only |
| **Cross-Attention** | ✅ | ✅ | ✅ | ❌ |
| **hidden_size** | 768 | **768** ✅ | 768 ✅ | 768 ✅ |
| **vocab_size** | 30522 | 32128 | 50265 | 50257 |
| **num_dec_layers** | 1 | **12** | 6 | - |
| **Pretrain data** | HowTo100M | C4 (750GB) | Books+Wiki | WebText |
| **Thay thế trực tiếp** | N/A | ✅ Dễ | ✅ Dễ | ❌ Khó |
| **Projection cần** | N/A | **Không** | Không | Cần |

**→ T5-base là lựa chọn tối ưu**: cùng hidden_size, có cross-attention, pretrain tốt.

---

## 4. Vấn đề vocab khác nhau

### Phạm vi ảnh hưởng

```
BERT tokenizer (WordPiece):
  vocab_size = 30522
  "cooking" → ["cook", "##ing"]
  BOS = [CLS] (id=101), EOS = [SEP] (id=102)

T5 tokenizer (SentencePiece):
  vocab_size = 32128
  "cooking" → ["▁cooking"]
  EOS = </s> (id=1), PAD = <pad> (id=0)
```

### Điều cần thay đổi trong code

```
❌ DecoderEmbedding(30522, 768)     → T5 Embedding(32128, 768)
❌ Linear head: Linear(768, 30522)  → T5 lm_head: Linear(768, 32128)
❌ BertTokenizer trong DataLoader   → T5Tokenizer
❌ output_caption_ids (BERT ids)    → output_caption_ids (T5 ids)
❌ CrossEntropy với vocab=30522     → CrossEntropy với vocab=32128
```

### Điều KHÔNG thay đổi (giữ từ checkpoint CIDEr=58)

```
✅ bert.*          ← Text Encoder, giữ nguyên weights
✅ visual.*        ← Video Encoder, giữ nguyên weights
✅ cross.*         ← Cross Encoder, giữ nguyên weights
✅ normalize_video ← giữ nguyên
✅ similarity_dense← giữ nguyên (dù không dùng cho captioning)
```

---

## 5. Kiến trúc sau khi tích hợp T5 Decoder

```
Video (CLIP ViT-L/14)
  (B, T, 768)
       ↓
  Video Encoder          weights từ checkpoint CIDEr=58
  (B, T, 768)
       ↓
       ↓    concat với Text Encoder output
       ↓
  Cross Encoder          weights từ checkpoint CIDEr=58
  cross_output: (B, L+T, 768)
       ↓
       ↓    không cần projection! (768 = 768)
       ↓
  T5 Decoder             weights từ t5-base pretrained
  ┌──────────────────────────────────┐
  │  Self-Attn (caption tokens)      │
  │          ↓                       │
  │  Cross-Attn ← cross_output       │  ← đây là điểm kết nối
  │             (B, L+T, 768)        │
  │          ↓                       │
  │  FFN                             │
  │          ↓                       │
  │  lm_head: Linear(768, 32128)     │
  └──────────────────────────────────┘
       ↓
  logits: (B, S, 32128)
       ↓
  T5 tokenizer decode
       ↓
  "A man is cooking pasta"
```

---

## 6. Strategy Fine-tuning T5 Decoder

### Option A: Frozen T5 + chỉ train Cross Encoder

```
Frozen:  bert, visual, T5 Decoder
Train:   cross (adapt output cho T5)

Ưu điểm:  Ít params, training nhanh
Nhược:    T5 không adapt với video domain
```

### Option B: LoRA trên T5 Decoder (khuyến nghị)

```python
# Thêm LoRA adapter vào T5 Decoder attention layers
# Chỉ train: LoRA params + Cross Encoder (optional)
# Frozen: bert, visual, T5 base weights

from peft import get_peft_model, LoraConfig, TaskType

lora_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    r=16,               # rank
    lora_alpha=32,
    target_modules=["q", "v"],   # T5 attention projections
    lora_dropout=0.1,
)
t5_decoder = get_peft_model(t5_decoder, lora_config)
# → Chỉ thêm ~0.1% params, T5 tự adapt nhẹ
```

### Option C: Full fine-tune tất cả (tốn GPU nhất)

```
Train toàn bộ: bert + visual + cross + T5 Decoder
→ Tốt nhất lý thuyết nhưng cần nhiều data/compute
→ MSRVTT chỉ ~7K train videos → dễ overfitting
```

---

## 7. Kỳ vọng kết quả

```
Baseline:          CIDEr = 58  (UniVL + CLIP ViT-L/14)

Sau khi thay T5:
  T5 frozen:       CIDEr ≈ ?   (không chắc, T5 chưa adapt)
  T5 + LoRA:       CIDEr ≈ 60-65?  (kỳ vọng)
  T5 full finetune:CIDEr ≈ 62-67?  (kỳ vọng, nhưng overfitting risk)

Lý do kỳ vọng cải thiện:
  T5 language generation tốt hơn Decoder gốc nhiều
  → Caption fluent hơn → CIDEr đo similarity → cao hơn
```

---

## 8. Tóm lại: Vì sao T5 là hợp lý

```
1. CÓ SẴN Cross-Attention
   → Nhận cross_output (B, L+T, 768) trực tiếp
   → Không cần thiết kế lại cơ chế conditioning

2. hidden_size = 768 = MATCH với UniVL
   → Không cần Projection Layer
   → Thay thế "plug-and-play"

3. Language generation mạnh hơn
   → Pretrain trên 750GB text đa dạng
   → Decoder gốc chỉ pretrain trên HowTo100M subtitles

4. Giữ nguyên được phần quan trọng nhất
   → Video Encoder + Cross Encoder từ checkpoint CIDEr=58
   → Encoder side không bị ảnh hưởng
   → Chỉ thay "đầu generate" tốt hơn
```
