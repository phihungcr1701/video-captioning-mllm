
"""
Temporal-aware Instruction-aware Dynamic Q-Former adapter for short-video captioning.

Design goals
------------
- Keep UNIVL-like forward signature:
    forward(concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True)
- Add temporal adaptation on video tokens before the dynamic Q-Former.
- Preserve instruction injection through set_instruction() without changing forward signature.
- Support explicit temporal ids through set_temporal_ids() when upstream can provide frame order.
- Fix a few training issues from the earlier version:
  * avoid creating brand-new Parameters inside forward by using registered Linear/LazyLinear modules
  * use masked attentive pooling instead of q[:, 0]
  * use a temporal-aware query budget instead of only multiplicative query scaling

Conventions
-----------
- concat_type == 0 -> text token
- concat_type == 1 -> video token
- If concat_type is None and assume_all_video_if_no_type=True, all valid tokens are treated as video.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

# --- UNIVL config import (best-effort) ---
try:
    from modules.module_cross import CrossConfig  # type: ignore
except Exception:
    try:
        from .module_cross import CrossConfig  # type: ignore
    except Exception:
        @dataclass
        class CrossConfig:
            vocab_size_or_config_json_file: Union[int, str] = 30522
            hidden_size: int = 768
            num_hidden_layers: int = 6
            num_attention_heads: int = 12
            intermediate_size: int = 3072
            hidden_act: str = "gelu"
            hidden_dropout_prob: float = 0.1
            attention_probs_dropout_prob: float = 0.1
            initializer_range: float = 0.02


def _act(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "swish":
        return lambda x: x * torch.sigmoid(x)
    return lambda x: x


def _to_float_mask(mask: Optional[torch.Tensor], like: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones(like.shape[:2], device=like.device, dtype=like.dtype)
    m = mask.to(device=like.device)
    if m.dtype == torch.bool:
        return m.to(dtype=like.dtype)
    return m.to(dtype=like.dtype)


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean(dim=1)
    m = mask.to(device=x.device, dtype=x.dtype).unsqueeze(-1)
    denom = m.sum(dim=1).clamp_min(1e-6)
    return (x * m).sum(dim=1) / denom


def _masked_softmax(logits: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1) -> torch.Tensor:
    if mask is None:
        return F.softmax(logits, dim=dim)
    m = mask.to(device=logits.device)
    if m.dtype != torch.bool:
        m = m > 0
    fill = torch.finfo(logits.dtype).min
    logits = logits.masked_fill(~m, fill)
    probs = F.softmax(logits, dim=dim)
    probs = probs * m.to(dtype=probs.dtype)
    probs = probs / probs.sum(dim=dim, keepdim=True).clamp_min(1e-6)
    return probs


def _bucket_summaries(x: torch.Tensor, mask: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Return (start, middle, end) masked summaries over valid temporal positions.
    x:    (B, L, H)
    mask: (B, L) float/bool
    """
    B, L, H = x.shape
    if mask is None:
        mask_f = torch.ones(B, L, device=x.device, dtype=x.dtype)
    else:
        mask_f = mask.to(device=x.device, dtype=x.dtype)
    idx = torch.arange(L, device=x.device).unsqueeze(0).expand(B, -1)
    counts = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
    start_bound = torch.floor(counts / 3.0)
    end_bound = torch.floor(2.0 * counts / 3.0)
    start_mask = (idx < start_bound).to(dtype=x.dtype) * mask_f
    middle_mask = ((idx >= start_bound) & (idx < end_bound)).to(dtype=x.dtype) * mask_f
    end_mask = (idx >= end_bound).to(dtype=x.dtype) * mask_f

    # Fallback for very short clips: if a bucket is empty, reuse full valid mask.
    def _safe(bucket_mask: torch.Tensor) -> torch.Tensor:
        empty = bucket_mask.sum(dim=1, keepdim=True) <= 0
        return torch.where(empty, mask_f, bucket_mask)

    start_mask = _safe(start_mask)
    middle_mask = _safe(middle_mask)
    end_mask = _safe(end_mask)
    return _masked_mean(x, start_mask), _masked_mean(x, middle_mask), _masked_mean(x, end_mask)


class _MHA(nn.Module):
    """Compact BERT-style multi-head attention. `mask` is over keys: (B, Nk)."""

    def __init__(self, h: int, heads: int, p_attn: float, p_out: float):
        super().__init__()
        if h % heads != 0:
            raise ValueError("hidden_size % num_heads != 0")
        self.h = h
        self.a = heads
        self.d = h // heads

        self.wq = nn.Linear(h, h)
        self.wk = nn.Linear(h, h)
        self.wv = nn.Linear(h, h)
        self.wo = nn.Linear(h, h)

        self.dp_attn = nn.Dropout(p_attn)
        self.dp_out = nn.Dropout(p_out)

    def _shape(self, x: torch.Tensor) -> torch.Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.a, self.d).transpose(1, 2)  # (B, A, N, D)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: Optional[torch.Tensor],
        topk: int = 0,
    ) -> torch.Tensor:
        qh = self._shape(self.wq(q))
        kh = self._shape(self.wk(k))
        vh = self._shape(self.wv(v))

        scores = (qh @ kh.transpose(-1, -2)) / math.sqrt(self.d)  # (B, A, Nq, Nk)

        if mask is not None:
            m = mask.to(device=scores.device)
            if m.dtype != torch.bool:
                m = m > 0
            m = m.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, Nk)
            scores = scores.masked_fill(~m, torch.finfo(scores.dtype).min)

        if topk and 0 < topk < scores.size(-1):
            vals, idx = torch.topk(scores, topk, dim=-1)
            sparse_scores = scores.new_full(scores.shape, torch.finfo(scores.dtype).min)
            sparse_scores.scatter_(-1, idx, vals)
            scores = sparse_scores

        probs = self.dp_attn(F.softmax(scores, dim=-1))
        ctx = (probs @ vh).transpose(1, 2).contiguous().view(q.size(0), q.size(1), self.h)
        return self.dp_out(self.wo(ctx))


class _TemporalBlock(nn.Module):
    """
    Lightweight temporal block for short-video captioning:
    LN -> temporal self-attn -> depthwise temporal conv -> FFN.
    """

    def __init__(self, cfg: CrossConfig, kernel_size: int = 3):
        super().__init__()
        H = cfg.hidden_size
        hd = cfg.hidden_dropout_prob
        ap = cfg.attention_probs_dropout_prob

        self.ln1 = nn.LayerNorm(H, eps=1e-12)
        self.attn = _MHA(H, cfg.num_attention_heads, ap, hd)

        self.ln2 = nn.LayerNorm(H, eps=1e-12)
        self.dwconv = nn.Conv1d(H, H, kernel_size, padding=kernel_size // 2, groups=H)
        self.pwconv = nn.Conv1d(H, H, kernel_size=1)

        self.ln3 = nn.LayerNorm(H, eps=1e-12)
        self.fc1 = nn.Linear(H, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, H)

        self.drop = nn.Dropout(hd)
        self.act = _act(getattr(cfg, "hidden_act", "gelu"))

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if mask is None:
            mask_f = torch.ones(x.shape[:2], device=x.device, dtype=x.dtype)
        else:
            mask_f = mask.to(device=x.device, dtype=x.dtype)
        xm = mask_f.unsqueeze(-1)

        h = self.ln1(x)
        if film is not None:
            g, b = film
            h = h * (1.0 + g) + b
        x = x + self.attn(h, h, h, mask)
        x = x * xm

        h = self.ln2(x)
        conv = self.dwconv(h.transpose(1, 2))
        conv = self.pwconv(self.act(conv)).transpose(1, 2)
        x = x + self.drop(conv)
        x = x * xm

        h = self.fc2(self.drop(self.act(self.fc1(self.ln3(x)))))
        x = x + self.drop(h)
        return x * xm


class TemporalAdapter(nn.Module):
    """
    Temporal adaptation over video tokens only.

    It mixes:
    - learned temporal positions
    - explicit motion token from first-order differences
    - 2-4 lightweight temporal blocks
    - instruction-conditioned FiLM
    - gated residual blending back to the original video tokens

    The adapter returns temporally-enhanced video tokens plus multi-scale summaries
    that are later used by the query allocator.
    """

    def __init__(
        self,
        config: CrossConfig,
        num_layers: int = 2,
        kernel_size: int = 3,
        max_temporal_positions: int = 256,
        use_motion: bool = True,
    ):
        super().__init__()
        H = config.hidden_size
        self.use_motion = bool(use_motion)
        self.max_temporal_positions = int(max_temporal_positions)

        self.temporal_pos = nn.Embedding(max_temporal_positions, H)
        self.motion_proj = nn.Linear(H, H) if self.use_motion else None
        self.cond_film = nn.Linear(H, 2 * H)

        self.blocks = nn.ModuleList([_TemporalBlock(config, kernel_size=kernel_size) for _ in range(num_layers)])

        self.blend_gate = nn.Sequential(
            nn.Linear(3 * H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

        self.summary_proj = nn.Sequential(
            nn.Linear(6 * H, H),
            nn.GELU(),
            nn.Linear(H, H),
        )

    def forward(
        self,
        video_tokens: torch.Tensor,
        video_mask: torch.Tensor,
        temporal_ids: Optional[torch.Tensor] = None,
        instruction_summary: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, L, H = video_tokens.shape
        dtype = video_tokens.dtype
        device = video_tokens.device
        mask_f = video_mask.to(device=device, dtype=dtype)
        xm = mask_f.unsqueeze(-1)

        if temporal_ids is None:
            temporal_ids = (mask_f.long().cumsum(dim=1) - 1).clamp_min(0)
        temporal_ids = temporal_ids.to(device=device).clamp(min=0, max=self.max_temporal_positions - 1)

        x0 = video_tokens
        x = video_tokens + self.temporal_pos(temporal_ids).to(dtype=dtype)

        if self.use_motion and L > 1:
            dx = x[:, 1:] - x[:, :-1]
            dx = F.pad(dx, (0, 0, 1, 0))
            x = x + self.motion_proj(dx)
        x = x * xm

        film = None
        if instruction_summary is not None:
            gb = self.cond_film(instruction_summary).to(dtype=dtype)
            g, b = gb[:, :H], gb[:, H:]
            film = (g.unsqueeze(1), b.unsqueeze(1))

        for blk in self.blocks:
            x = blk(x, video_mask, film=film)

        blend_input_parts = [x0, x]
        if instruction_summary is None:
            blend_input_parts.append(torch.zeros_like(x0))
        else:
            blend_input_parts.append(instruction_summary.unsqueeze(1).expand(-1, L, -1).to(dtype=dtype))
        gate = torch.sigmoid(self.blend_gate(torch.cat(blend_input_parts, dim=-1))).to(dtype=dtype)
        out = gate * x + (1.0 - gate) * x0
        out = out * xm

        global_summary = _masked_mean(out, video_mask)
        start_summary, mid_summary, end_summary = _bucket_summaries(out, video_mask)
        if L > 1:
            motion_summary = _masked_mean(torch.abs(out[:, 1:] - out[:, :-1]), mask_f[:, 1:] * mask_f[:, :-1])
        else:
            motion_summary = torch.zeros(B, H, device=device, dtype=dtype)

        if instruction_summary is None:
            instr_summary = torch.zeros(B, H, device=device, dtype=dtype)
        else:
            instr_summary = instruction_summary.to(dtype=dtype)

        q_context = self.summary_proj(
            torch.cat(
                [global_summary, start_summary, mid_summary, end_summary, motion_summary, instr_summary],
                dim=-1,
            )
        ).to(dtype=dtype)

        aux = {
            "video_global": global_summary,
            "video_start": start_summary,
            "video_mid": mid_summary,
            "video_end": end_summary,
            "video_motion": motion_summary,
            "q_context": q_context,
        }
        return out, aux


class _DynQFLayer(nn.Module):
    """One dynamic Q-Former layer with optional query mask."""

    def __init__(self, cfg: CrossConfig, use_instr_xattn: bool = True):
        super().__init__()
        H = cfg.hidden_size
        hd = cfg.hidden_dropout_prob
        ap = cfg.attention_probs_dropout_prob

        self.sa = _MHA(H, cfg.num_attention_heads, ap, hd)
        self.ca = _MHA(H, cfg.num_attention_heads, ap, hd)
        self.ia = _MHA(H, cfg.num_attention_heads, ap, hd) if use_instr_xattn else None

        self.ln1 = nn.LayerNorm(H, eps=1e-12)
        self.lnI = nn.LayerNorm(H, eps=1e-12)
        self.ln2 = nn.LayerNorm(H, eps=1e-12)
        self.ln3 = nn.LayerNorm(H, eps=1e-12)

        self.fc1 = nn.Linear(H, cfg.intermediate_size)
        self.fc2 = nn.Linear(cfg.intermediate_size, H)

        self.dp = nn.Dropout(hd)
        self.act = _act(getattr(cfg, "hidden_act", "gelu"))

    def forward(
        self,
        q: torch.Tensor,
        query_mask: Optional[torch.Tensor],
        kv: torch.Tensor,
        kv_mask: Optional[torch.Tensor],
        cond_tokens: Optional[torch.Tensor],
        cond_mask: Optional[torch.Tensor],
        topk_kv: int = 0,
        topk_cond: int = 0,
        film: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if query_mask is None:
            qm = torch.ones(q.shape[:2], device=q.device, dtype=q.dtype)
        else:
            qm = query_mask.to(device=q.device, dtype=q.dtype)
        qx = qm.unsqueeze(-1)

        if film is not None:
            g, b = film
            q = q * (1.0 + g) + b
        q = q * qx

        q = self.ln1(q + self.sa(q, q, q, query_mask))
        q = q * qx

        if self.ia is not None and cond_tokens is not None:
            q = self.lnI(q + self.ia(q, cond_tokens, cond_tokens, cond_mask, topk=topk_cond))
            q = q * qx

        q = self.ln2(q + self.ca(q, kv, kv, kv_mask, topk=topk_kv))
        q = q * qx

        h = self.fc2(self.dp(self.act(self.fc1(q))))
        q = self.ln3(q + self.dp(h))
        return q * qx


class TemporalDynamicQFormerAdapter(nn.Module):
    """
    Temporal-aware, instruction-aware, dynamic Q-Former adapter.

    Backward-compatible forward:
        forward(concat_input, concat_type=None, attention_mask=None, output_all_encoded_layers=True)

    Extra side-channels:
        set_instruction(tensor, mask=None)
        set_temporal_ids(tensor or None)

    Recommended for short-video captioning:
    - temporal adapter focuses on local motion + short-range order
    - query budget is conditioned on instruction + temporal summaries
    - final pooling is masked attentive pooling over active queries
    """

    def __init__(
        self,
        config: CrossConfig,
        input_dim: Optional[int] = None,
        instruction_dim: Optional[int] = None,
        num_query_tokens: int = 32,
        min_query_tokens: int = 8,
        attend_to_text: bool = True,
        attend_to_video: bool = True,
        use_instruction_xattn: bool = True,
        time_buckets: int = 4,
        temporal_layers: int = 2,
        temporal_kernel_size: int = 3,
        max_temporal_positions: int = 256,
        query_topk_kv: int = 0,
        query_topk_cond: int = 0,
        assume_all_video_if_no_type: bool = False,
    ):
        super().__init__()
        self.cfg = config
        self.H = config.hidden_size
        self.K = int(num_query_tokens)
        self.min_query_tokens = int(min_query_tokens)
        self.attend_to_text = bool(attend_to_text)
        self.attend_to_video = bool(attend_to_video)
        self.time_buckets = int(time_buckets)
        self.topk_kv = int(query_topk_kv)
        self.topk_cond = int(query_topk_cond)
        self.assume_all_video_if_no_type = bool(assume_all_video_if_no_type)

        self.q_base = nn.Parameter(torch.zeros(self.K, self.H))
        self.q_time_emb = nn.Embedding(max(1, self.time_buckets), self.H)

        # Registered at init time. If dims are unknown, LazyLinear keeps params registered.
        if input_dim is None:
            self.kv_proj = nn.LazyLinear(self.H)
        elif input_dim == self.H:
            self.kv_proj = nn.Identity()
        else:
            self.kv_proj = nn.Linear(input_dim, self.H)

        if instruction_dim is None:
            self.instr_proj = nn.LazyLinear(self.H)
        elif instruction_dim == self.H:
            self.instr_proj = nn.Identity()
        else:
            self.instr_proj = nn.Linear(instruction_dim, self.H)

        self.temporal_adapter = TemporalAdapter(
            config=config,
            num_layers=temporal_layers,
            kernel_size=temporal_kernel_size,
            max_temporal_positions=max_temporal_positions,
            use_motion=True,
        )

        self.cond_token_proj = nn.Linear(self.H, self.H)
        self.context_fuse = nn.Sequential(
            nn.Linear(2 * self.H, self.H),
            nn.GELU(),
            nn.Linear(self.H, self.H),
        )
        self.context_to_film = nn.Linear(self.H, 2 * self.H)
        self.query_score = nn.Linear(self.H, self.K)
        self.query_len = nn.Linear(self.H, 1)
        self.init_cond_attn = _MHA(self.H, config.num_attention_heads, config.attention_probs_dropout_prob, config.hidden_dropout_prob)

        self.layers = nn.ModuleList(
            [_DynQFLayer(config, use_instr_xattn=use_instruction_xattn) for _ in range(config.num_hidden_layers)]
        )

        self.pool_score = nn.Linear(self.H, 1)
        self.pool_proj = nn.Linear(self.H, self.H)

        self._instr: Optional[torch.Tensor] = None
        self._instr_mask: Optional[torch.Tensor] = None
        self._temporal_ids: Optional[torch.Tensor] = None
        self.last_aux: Dict[str, torch.Tensor] = {}

        self._lazy_init_done: Dict[str, bool] = {"kv_proj": False, "instr_proj": False}
        self._init_weights()

    @property
    def dtype(self):
        """Return dtype of model parameters (for pipeline compatibility with PreTrainedModel)."""
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.float32

    def _init_weights(self) -> None:
        std = float(getattr(self.cfg, "initializer_range", 0.02))

        def init(m: nn.Module):
            if isinstance(m, nn.Linear):
                # Skip LazyLinear before materialization
                if isinstance(m.weight, UninitializedParameter):
                    return
                nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=std)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.apply(init)
        nn.init.normal_(self.q_base, mean=0.0, std=std)

    def _maybe_init_lazy_linear(self, module: nn.Module, name: str) -> None:
        if name in self._lazy_init_done and self._lazy_init_done[name]:
            return
        if not isinstance(module, nn.LazyLinear):
            self._lazy_init_done[name] = True
            return
        if isinstance(module.weight, UninitializedParameter):
            return  # materializes on first actual call
        std = float(getattr(self.cfg, "initializer_range", 0.02))
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
        self._lazy_init_done[name] = True

    def set_instruction(self, instruction: torch.Tensor, instruction_mask: Optional[torch.Tensor] = None) -> None:
        """
        instruction:
            (B, Din) or (B, Lin, Din)
        """
        self._instr = instruction
        self._instr_mask = instruction_mask

    def clear_instruction(self) -> None:
        self._instr = None
        self._instr_mask = None

    def set_temporal_ids(self, temporal_ids: Optional[torch.Tensor]) -> None:
        """
        temporal_ids:
            Either (B, N) aligned with concat_input or None.
            Non-video positions may be any value; they are ignored.
        """
        self._temporal_ids = temporal_ids

    def clear_temporal_ids(self) -> None:
        self._temporal_ids = None

    @torch.no_grad()
    def maybe_init_from_crossmodel(self, cross_model: Any, strict: bool = False) -> None:
        """
        Best-effort initialization from UNIVL CrossModel:
        - copy pooler.dense -> pool_proj
        - copy encoder.layer[i] self-attn + FFN into our self-attn/FFN
        Cross-attn, temporal blocks, and dynamic query heads stay task-specific.
        """
        try:
            if hasattr(cross_model, "pooler") and hasattr(cross_model.pooler, "dense"):
                self.pool_proj.load_state_dict(cross_model.pooler.dense.state_dict(), strict=False)
        except Exception:
            if strict:
                raise

        enc = getattr(cross_model, "encoder", None)
        src = getattr(enc, "layer", None) if enc is not None else None
        if src is None:
            if strict:
                raise ValueError("cross_model.encoder.layer not found")
            return

        for i in range(min(len(self.layers), len(src))):
            s = src[i]
            t = self.layers[i]
            try:
                t.sa.wq.load_state_dict(s.attention.self.query.state_dict())
                t.sa.wk.load_state_dict(s.attention.self.key.state_dict())
                t.sa.wv.load_state_dict(s.attention.self.value.state_dict())
                t.sa.wo.load_state_dict(s.attention.output.dense.state_dict())
                t.ln1.load_state_dict({"weight": s.attention.output.LayerNorm.weight, "bias": s.attention.output.LayerNorm.bias})
                t.fc1.load_state_dict(s.intermediate.dense.state_dict())
                t.fc2.load_state_dict(s.output.dense.state_dict())
                t.ln3.load_state_dict({"weight": s.output.LayerNorm.weight, "bias": s.output.LayerNorm.bias})
            except Exception:
                if strict:
                    raise

    def _project_kv(self, x: torch.Tensor) -> torch.Tensor:
        y = self.kv_proj(x)  # type: ignore[arg-type]
        if isinstance(self.kv_proj, nn.LazyLinear):
            self._maybe_init_lazy_linear(self.kv_proj, "kv_proj")
        return y

    def _project_instruction(self, x: torch.Tensor) -> torch.Tensor:
        y = self.instr_proj(x)  # type: ignore[arg-type]
        if isinstance(self.instr_proj, nn.LazyLinear):
            self._maybe_init_lazy_linear(self.instr_proj, "instr_proj")
        return y

    def _build_instruction(
        self,
        batch_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            instr_tokens: (B, Lin, H) or None
            instr_mask:   (B, Lin) or None
            instr_pool:   (B, H) or None
        """
        if self._instr is None:
            return None, None, None

        ins = self._instr.to(device=device)
        if ins.dim() == 2:
            ins_h = self._project_instruction(ins)
            ins_h = ins_h.to(dtype=dtype)
            mask = torch.ones(batch_size, 1, device=device, dtype=dtype)
            return ins_h.unsqueeze(1), mask, ins_h

        if ins.dim() == 3:
            ins_h = self._project_instruction(ins).to(dtype=dtype)
            if self._instr_mask is None:
                im = torch.ones(batch_size, ins_h.size(1), device=device, dtype=dtype)
            else:
                im = _to_float_mask(self._instr_mask, ins_h)
            pool = _masked_mean(ins_h, im)
            return ins_h, im, pool

        raise ValueError("instruction must be (B, D) or (B, L, D)")

    def _pack_video_tokens(
        self,
        kv: torch.Tensor,
        seq_video_mask: torch.Tensor,
        temporal_ids: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pack variable-length video tokens into a dense (B, Lv_max, H) tensor and
        return indices for scattering back.
        """
        B, N, H = kv.shape
        device = kv.device
        dtype = kv.dtype

        counts = seq_video_mask.long().sum(dim=1)
        max_v = int(counts.max().item()) if B > 0 else 0
        if max_v == 0:
            return (
                kv.new_zeros(B, 0, H),
                kv.new_zeros(B, 0),
                torch.full((B, 0), -1, device=device, dtype=torch.long),
                kv.new_zeros(B, 0, dtype=torch.long),
            )

        packed = kv.new_zeros(B, max_v, H)
        packed_mask = kv.new_zeros(B, max_v)
        scatter_idx = torch.full((B, max_v), -1, device=device, dtype=torch.long)
        packed_temporal_ids = torch.zeros(B, max_v, device=device, dtype=torch.long)

        for b in range(B):
            idx = torch.nonzero(seq_video_mask[b], as_tuple=False).squeeze(-1)
            c = int(idx.numel())
            if c == 0:
                continue
            packed[b, :c] = kv[b, idx]
            packed_mask[b, :c] = 1
            scatter_idx[b, :c] = idx
            if temporal_ids is None:
                packed_temporal_ids[b, :c] = torch.arange(c, device=device, dtype=torch.long)
            else:
                packed_temporal_ids[b, :c] = temporal_ids[b, idx].to(device=device, dtype=torch.long).clamp_min(0)

        return packed, packed_mask.to(dtype=dtype), scatter_idx, packed_temporal_ids

    def _scatter_video_tokens(
        self,
        kv: torch.Tensor,
        adapted_video: torch.Tensor,
        scatter_idx: torch.Tensor,
        packed_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = kv.clone()
        B = kv.size(0)
        for b in range(B):
            valid = packed_mask[b] > 0
            idx = scatter_idx[b, valid]
            if idx.numel() == 0:
                continue
            out[b, idx] = adapted_video[b, valid]
        return out

    def _build_condition_tokens(
        self,
        instr_tokens: Optional[torch.Tensor],
        instr_mask: Optional[torch.Tensor],
        temporal_aux: Optional[Dict[str, torch.Tensor]],
        dtype: torch.dtype,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Build condition tokens used for:
        - initial query synthesis
        - optional query->condition cross-attn inside each Q-Former layer
        Also returns fused context vector for FiLM and query allocation.
        """
        ctx_parts: List[torch.Tensor] = []
        cond_tokens_list: List[torch.Tensor] = []
        cond_masks: List[torch.Tensor] = []

        if temporal_aux is not None:
            temporal_tokens = torch.stack(
                [
                    temporal_aux["video_global"],
                    temporal_aux["video_start"],
                    temporal_aux["video_mid"],
                    temporal_aux["video_end"],
                    temporal_aux["video_motion"],
                ],
                dim=1,
            )
            temporal_tokens = self.cond_token_proj(temporal_tokens).to(dtype=dtype)
            cond_tokens_list.append(temporal_tokens)
            cond_masks.append(torch.ones(temporal_tokens.shape[:2], device=temporal_tokens.device, dtype=dtype))
            ctx_parts.append(temporal_aux["q_context"].to(dtype=dtype))

        instr_pool = None
        if instr_tokens is not None:
            cond_tokens_list.insert(0, instr_tokens.to(dtype=dtype))
            cond_masks.insert(0, instr_mask.to(dtype=dtype) if instr_mask is not None else torch.ones(instr_tokens.shape[:2], device=instr_tokens.device, dtype=dtype))
            instr_pool = _masked_mean(instr_tokens.to(dtype=dtype), instr_mask)
            ctx_parts.append(instr_pool)

        if not cond_tokens_list:
            return None, None, None

        cond_tokens = torch.cat(cond_tokens_list, dim=1)
        cond_mask = torch.cat(cond_masks, dim=1)

        if len(ctx_parts) == 1:
            context = ctx_parts[0]
        else:
            context = self.context_fuse(torch.cat([ctx_parts[0], ctx_parts[1]], dim=-1)).to(dtype=dtype)

        return cond_tokens, cond_mask, context

    def _allocate_queries(self, context: Optional[torch.Tensor], dtype: torch.dtype, device: torch.device, batch_size: int = 1) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            q:          (B, K, H)
            query_mask: (B, K) straight-through mask
            hard_k:     (B,) integer budget
        """
        if context is None:
            # fallback: use all queries
            B = batch_size
            q = self.q_base.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
            query_mask = torch.ones(B, self.K, device=device, dtype=dtype)
            hard_k = torch.full((B,), self.K, device=device, dtype=torch.long)
            return q, query_mask, hard_k

        B = context.size(0)
        q = self.q_base.unsqueeze(0).expand(B, -1, -1).to(device=device, dtype=dtype)
        if self.time_buckets > 0:
            qid = torch.arange(self.K, device=device) % max(1, self.time_buckets)
            q = q + self.q_time_emb(qid).unsqueeze(0).to(dtype=dtype)

        cond_context = context.to(dtype=dtype)
        q = q + self.init_cond_attn(q, cond_context.unsqueeze(1), cond_context.unsqueeze(1), torch.ones(B, 1, device=device, dtype=dtype))

        gb = self.context_to_film(cond_context).to(dtype=dtype)
        g, b = gb[:, :self.H], gb[:, self.H:]
        q = q * (1.0 + g.unsqueeze(1)) + b.unsqueeze(1)

        soft_scores = torch.sigmoid(self.query_score(cond_context)).to(dtype=dtype)  # (B, K)
        raw_k = torch.sigmoid(self.query_len(cond_context)).squeeze(-1)
        hard_k = self.min_query_tokens + torch.round(raw_k * (self.K - self.min_query_tokens)).long()
        hard_k = hard_k.clamp(min=self.min_query_tokens, max=self.K)

        hard_mask = torch.zeros(B, self.K, device=device, dtype=dtype)
        for b_idx in range(B):
            k = int(hard_k[b_idx].item())
            _, idx = torch.topk(soft_scores[b_idx], k=k, dim=-1)
            hard_mask[b_idx, idx] = 1.0

        # Straight-through: forward uses hard selection, backward uses soft gate.
        query_mask = hard_mask + soft_scores - soft_scores.detach()
        q = q * query_mask.unsqueeze(-1)
        return q, query_mask, hard_k

    def get_last_aux(self) -> Dict[str, torch.Tensor]:
        return self.last_aux

    def auxiliary_losses(
        self,
        diversity_weight: float = 1e-2,
        sparsity_weight: float = 1e-3,
        temporal_weight: float = 1e-3,
    ) -> Dict[str, torch.Tensor]:
        """
        Optional regularizers that can be added to the main caption loss.
        Call after a forward().
        """
        aux = self.last_aux
        out: Dict[str, torch.Tensor] = {}

        if "query_states" in aux and diversity_weight > 0:
            q = aux["query_states"]  # (B, K, H)
            q = F.normalize(q, dim=-1)
            sim = q @ q.transpose(-1, -2)
            eye = torch.eye(sim.size(-1), device=sim.device, dtype=sim.dtype).unsqueeze(0)
            out["diversity_loss"] = diversity_weight * ((sim - eye) ** 2).mean()

        if "query_mask_soft" in aux and sparsity_weight > 0:
            out["sparsity_loss"] = sparsity_weight * aux["query_mask_soft"].mean()

        if "temporal_video" in aux and temporal_weight > 0 and aux["temporal_video"].size(1) > 1:
            tv = aux["temporal_video"]
            smooth = (tv[:, 1:] - tv[:, :-1]).pow(2).mean()
            out["temporal_smooth_loss"] = temporal_weight * smooth

        return out

    def forward(
        self,
        concat_input: torch.Tensor,
        concat_type: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        output_all_encoded_layers: bool = True,
    ) -> Tuple[Union[List[torch.Tensor], torch.Tensor], torch.Tensor]:
        if concat_input.dim() != 3:
            raise ValueError("concat_input must be (B, N, D)")

        B, N, _ = concat_input.shape
        device = concat_input.device
        dtype = concat_input.dtype

        kv_mask = _to_float_mask(attention_mask, concat_input)
        if concat_type is None:
            ttype = torch.ones(B, N, device=device, dtype=torch.long) if self.assume_all_video_if_no_type else torch.zeros(B, N, device=device, dtype=torch.long)
        else:
            ttype = concat_type.to(device=device).long()

        if not self.attend_to_text:
            kv_mask = kv_mask * (ttype != 0).to(dtype=dtype)
        if not self.attend_to_video:
            kv_mask = kv_mask * (ttype != 1).to(dtype=dtype)

        kv = self._project_kv(concat_input).to(dtype=dtype)

        instr_tokens, instr_mask, instr_pool = self._build_instruction(B, device, dtype)

        temporal_aux: Optional[Dict[str, torch.Tensor]] = None
        seq_video_mask = (ttype == 1) & (kv_mask > 0)

        if seq_video_mask.any():
            packed_video, packed_video_mask, scatter_idx, packed_temporal_ids = self._pack_video_tokens(
                kv=kv,
                seq_video_mask=seq_video_mask,
                temporal_ids=self._temporal_ids,
            )
            adapted_video, temporal_aux = self.temporal_adapter(
                packed_video,
                packed_video_mask,
                temporal_ids=packed_temporal_ids,
                instruction_summary=instr_pool,
            )
            kv = self._scatter_video_tokens(kv, adapted_video, scatter_idx, packed_video_mask)

        cond_tokens, cond_mask, context = self._build_condition_tokens(instr_tokens, instr_mask, temporal_aux, dtype=dtype)
        if context is None:
            # fallback: use masked global mean over current memory
            context = _masked_mean(kv, kv_mask)

        q, query_mask, hard_k = self._allocate_queries(context, dtype=dtype, device=device, batch_size=B)
        soft_query_mask = torch.sigmoid(self.query_score(context)).to(dtype=dtype)

        outs: List[torch.Tensor] = []
        for layer in self.layers:
            q = layer(
                q=q,
                query_mask=query_mask,
                kv=kv,
                kv_mask=kv_mask,
                cond_tokens=cond_tokens,
                cond_mask=cond_mask,
                topk_kv=self.topk_kv,
                topk_cond=self.topk_cond,
                film=None,
            )
            if output_all_encoded_layers:
                outs.append(q)

        pool_logits = self.pool_score(q).squeeze(-1)
        pool_weights = _masked_softmax(pool_logits, query_mask, dim=-1)
        pooled = torch.tanh(self.pool_proj((pool_weights.unsqueeze(-1) * q).sum(dim=1)))

        self.last_aux = {
            "query_states": q,
            "query_mask": query_mask.detach(),
            "query_mask_soft": soft_query_mask,
            "hard_k": hard_k.detach(),
            "context": context.detach(),
            "memory_states": kv,
            "memory_mask": kv_mask.detach(),
        }
        if temporal_aux is not None:
            self.last_aux["temporal_video_global"] = temporal_aux["video_global"].detach()
            self.last_aux["temporal_video"] = adapted_video.detach()
            self.last_aux["video_motion"] = temporal_aux["video_motion"].detach()

        return (outs if output_all_encoded_layers else q), pooled


class CrossModel(TemporalDynamicQFormerAdapter):
    """Compatibility wrapper so existing code can instantiate CrossModel unchanged."""

    def __init__(self, config: CrossConfig):
        super().__init__(
            config=config,
            input_dim=getattr(config, "hidden_size", None),
            instruction_dim=getattr(config, "hidden_size", None),
            num_query_tokens=getattr(config, "num_query_tokens", 32),
            min_query_tokens=getattr(config, "min_query_tokens", 8),
            attend_to_text=getattr(config, "attend_to_text", True),
            attend_to_video=getattr(config, "attend_to_video", True),
            use_instruction_xattn=getattr(config, "use_instruction_xattn", True),
            time_buckets=getattr(config, "time_buckets", 4),
            temporal_layers=getattr(config, "temporal_layers", 2),
            temporal_kernel_size=getattr(config, "temporal_kernel_size", 3),
            max_temporal_positions=getattr(config, "max_temporal_positions", 256),
            query_topk_kv=getattr(config, "query_topk_kv", 0),
            query_topk_cond=getattr(config, "query_topk_cond", 0),
            assume_all_video_if_no_type=getattr(config, "assume_all_video_if_no_type", False),
        )


def smoke_test() -> None:
    cfg = CrossConfig(
        30522,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        intermediate_size=256,
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
    )
    model = TemporalDynamicQFormerAdapter(
        config=cfg,
        input_dim=192,
        instruction_dim=64,
        num_query_tokens=16,
        min_query_tokens=6,
        temporal_layers=2,
        time_buckets=4,
        assume_all_video_if_no_type=False,
    )

    B, Lt, Lv = 2, 5, 12
    D = 192
    x = torch.randn(B, Lt + Lv, D, requires_grad=True)
    ttype = torch.cat(
        [torch.zeros(B, Lt, dtype=torch.long), torch.ones(B, Lv, dtype=torch.long)],
        dim=1,
    )
    mask = torch.ones(B, Lt + Lv)
    temporal_ids = torch.cat(
        [torch.zeros(B, Lt, dtype=torch.long), torch.arange(Lv).unsqueeze(0).repeat(B, 1)],
        dim=1,
    )
    ins = torch.randn(B, 7, 64)
    ins_mask = torch.ones(B, 7)

    model.set_instruction(ins, ins_mask)
    model.set_temporal_ids(temporal_ids)

    outs, pooled = model(x, ttype, mask, True)
    y = outs[-1]
    assert y.shape == (B, 16, 128), y.shape
    assert pooled.shape == (B, 128), pooled.shape

    loss = y.mean() + pooled.mean()
    reg = model.auxiliary_losses()
    for v in reg.values():
        loss = loss + v
    loss.backward()
    assert x.grad is not None
    assert model.get_last_aux()["hard_k"].shape == (B,)
    print("smoke_test OK")


if __name__ == "__main__":
    smoke_test()
