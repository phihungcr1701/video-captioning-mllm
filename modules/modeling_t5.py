# coding=utf-8
# Adapted from modules/modeling.py
# Changes:
#   - DecoderModel replaced by T5 Decoder from transformers (t5-base)
#   - Q-Former cross encoder (module_cross_qformer)
#   - ITC + ITM losses (BLIP-2 style) added for stage_two caption training

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import math
import numpy as np

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss

from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross_qformer import CrossModel, CrossConfig
from modules.module_decoder import DecoderConfig  # kept for from_pretrained compatibility
from transformers import T5ForConditionalGeneration, T5Config

logger = logging.getLogger(__name__)


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.t5_decoder = None

    @classmethod
    def from_pretrained(cls, pretrained_bert_name, visual_model_name, cross_model_name, decoder_model_name,
                        state_dict=None, cache_dir=None, type_vocab_size=2, *inputs, **kwargs):

        task_config = None
        if "task_config" in kwargs.keys():
            task_config = kwargs["task_config"]
            if not hasattr(task_config, "local_rank"):
                task_config.__dict__["local_rank"] = 0
            elif task_config.local_rank == -1:
                task_config.local_rank = 0

        bert_config, state_dict = BertConfig.get_config(pretrained_bert_name, cache_dir, type_vocab_size, state_dict, task_config=task_config)
        visual_config, _ = VisualConfig.get_config(visual_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        visual_config = update_attr("visual_config", visual_config, "vocab_size", task_config, "video_dim")
        cross_config, _ = CrossConfig.get_config(cross_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)
        # decoder_config kept for interface compatibility; T5 is loaded inside __init__
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)

        assert model.bert is not None
        assert model.visual is not None

        if state_dict is not None:
            # Filter out old decoder keys from the checkpoint — they are incompatible with T5
            state_dict = {k: v for k, v in state_dict.items() if not k.startswith("decoder.")}
            model = cls.init_preweight(model, state_dict, task_config=task_config)

        return model


class NormalizeVideo(nn.Module):
    def __init__(self, task_config):
        super(NormalizeVideo, self).__init__()
        self.visual_norm2d = LayerNorm(task_config.video_dim)

    def forward(self, video):
        video = torch.as_tensor(video).float()
        video = video.view(-1, video.shape[-2], video.shape[-1])
        video = self.visual_norm2d(video)
        return video


def show_log(task_config, info):
    if task_config is None or task_config.local_rank == 0:
        logger.warning(info)


def update_attr(target_name, target_config, target_attr_name, source_config, source_attr_name, default_value=None):
    if hasattr(source_config, source_attr_name):
        if default_value is None or getattr(source_config, source_attr_name) != default_value:
            setattr(target_config, target_attr_name, getattr(source_config, source_attr_name))
            show_log(source_config, "Set {}.{}: {}.".format(target_name,
                                                            target_attr_name, getattr(target_config, target_attr_name)))
    return target_config


def check_attr(target_name, task_config):
    return hasattr(task_config, target_name) and task_config.__dict__[target_name]


class UniVL_T5(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config):
        super(UniVL_T5, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_frames <= visual_config.max_position_embeddings
        assert self.task_config.max_words + self.task_config.max_frames <= cross_config.max_position_embeddings

        self._stage_one = True
        self._stage_two = False

        if check_attr('stage_two', self.task_config):
            self._stage_one = False
            self._stage_two = self.task_config.stage_two
        show_log(task_config, "Stage-One:{}, Stage-Two:{}".format(self._stage_one, self._stage_two))

        self.train_sim_after_cross = False
        if self._stage_one and check_attr('train_sim_after_cross', self.task_config):
            self.train_sim_after_cross = True
            show_log(task_config, "Test retrieval after cross encoder.")

        # Text Encoder ===>
        bert_config = update_attr("bert_config", bert_config, "num_hidden_layers",
                                   self.task_config, "text_num_hidden_layers")
        self.bert = BertModel(bert_config)
        bert_word_embeddings_weight = self.bert.embeddings.word_embeddings.weight
        bert_position_embeddings_weight = self.bert.embeddings.position_embeddings.weight
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # T5 Decoder ===>
                show_log(task_config, "Loading T5-base decoder from HuggingFace...")
                _t5_full = T5ForConditionalGeneration.from_pretrained("t5-base")
                self.t5_decoder = _t5_full.decoder   # T5Stack (decoder)
                self.t5_lm_head = _t5_full.lm_head   # Linear(768, 32128)
                self.t5_vocab_size = _t5_full.config.vocab_size  # 32128
                del _t5_full
                # <=== End of T5 Decoder

                # ====== ITC + ITM Heads (BLIP-2 style) ======
                H = bert_config.hidden_size  # 768
                itc_proj_dim = getattr(task_config, 'itc_proj_dim', 256)

                # ITC: projection heads + learnable logit scale
                self.itc_text_proj = nn.Linear(H, itc_proj_dim)
                self.itc_video_proj = nn.Linear(H, itc_proj_dim)
                init_temp = getattr(task_config, 'itc_init_temp', 0.07)
                self.itc_logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / init_temp))

                # ITM: binary classifier on [pooled_query; text_feat]
                use_richer = getattr(task_config, 'itm_use_richer_fusion', False)
                if use_richer:
                    self.itm_head = nn.Sequential(
                        nn.Linear(H * 4, H),
                        nn.ReLU(),
                        nn.Dropout(0.1),
                        nn.Linear(H, 2)
                    )
                else:
                    self.itm_head = nn.Linear(H * 2, 2)
                self._itm_use_richer = use_richer

                # Config flags (read with safe defaults)
                self._use_itc = getattr(task_config, 'use_itc_loss', False)
                self._use_itm = getattr(task_config, 'use_itm_loss', False)
                self._itc_gather = getattr(task_config, 'itc_gather_distributed', False)

                if self._use_itc:
                    show_log(task_config, "ITC loss ENABLED (proj_dim={}, init_temp={:.3f})".format(
                        itc_proj_dim, init_temp))
                if self._use_itm:
                    show_log(task_config, "ITM loss ENABLED (richer_fusion={})".format(use_richer))
                # ====== End ITC + ITM ======

            if self.task_config.do_pretrain:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)

            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)
            self.decoder_loss_fct = CrossEntropyLoss(ignore_index=-1)

        self.normalize_video = NormalizeVideo(task_config)

        mILNCELoss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair)
        maxMarginRankingLoss = MaxMarginRankingLoss(margin=task_config.margin,
                                                    negative_weighting=task_config.negative_weighting,
                                                    batch_size=task_config.batch_size // task_config.n_gpu,
                                                    n_pair=task_config.n_pair,
                                                    hard_negative_rate=task_config.hard_negative_rate)

        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mILNCELoss
            self._pretrain_sim_loss_fct = mILNCELoss
        else:
            self.loss_fct = CrossEn() if self._stage_two else maxMarginRankingLoss
            self._pretrain_sim_loss_fct = maxMarginRankingLoss

        # Storage for per-step loss metrics (read by trainer for logging)
        self._last_loss_dict = {}

        self.apply(self.init_weights)

    # ====================================================================
    # ITC / ITM helper methods
    # ====================================================================

    def _masked_mean_pool_text(self, sequence_output, attention_mask):
        """Mean-pool text encoder output, excluding CLS (position 0)
        to match existing _mean_pooling_for_similarity behavior."""
        mask = attention_mask.to(dtype=torch.float).unsqueeze(-1)  # (B, T, 1)
        mask[:, 0, :] = 0.  # exclude CLS
        pooled = torch.sum(sequence_output * mask, dim=1) / torch.sum(mask, dim=1, dtype=torch.float).clamp(min=1e-8)
        return pooled  # (B, H)

    def _get_qformer_video_embedding(self, cross_output, pooled_output, encoder_mask):
        """Get video embedding from Q-Former output for ITC.
        Uses pooled_output (attentive pooling from Q-Former) as primary.
        Falls back to masked mean-pool of cross_output if pooled_output is zero."""
        if pooled_output is not None and pooled_output.abs().sum() > 0:
            return pooled_output  # (B, H)
        # Fallback: masked mean-pool of query tokens
        mask = encoder_mask.to(dtype=torch.float).unsqueeze(-1)  # (B, K, 1)
        pooled = torch.sum(cross_output * mask, dim=1) / torch.sum(mask, dim=1, dtype=torch.float).clamp(min=1e-8)
        return pooled  # (B, H)

    def _compute_itc_loss(self, text_feat, video_feat):
        """Bidirectional InfoNCE contrastive loss with learnable temperature.
        text_feat: (B, H) — mean-pooled text encoder output
        video_feat: (B, H) — Q-Former pooled output
        Returns: itc_loss scalar, top1_acc scalar (detached)
        """
        # Project + L2 normalize
        text_proj = F.normalize(self.itc_text_proj(text_feat), dim=-1)   # (B, D)
        video_proj = F.normalize(self.itc_video_proj(video_feat), dim=-1)  # (B, D)

        # Optional DDP all_gather
        if self._itc_gather and torch.distributed.is_initialized() and torch.distributed.get_world_size() > 1:
            text_proj_all = [torch.zeros_like(text_proj) for _ in range(torch.distributed.get_world_size())]
            video_proj_all = [torch.zeros_like(video_proj) for _ in range(torch.distributed.get_world_size())]
            torch.distributed.all_gather(text_proj_all, text_proj)
            torch.distributed.all_gather(video_proj_all, video_proj)
            # Replace local shard with version that has gradients
            rank = torch.distributed.get_rank()
            text_proj_all[rank] = text_proj
            video_proj_all[rank] = video_proj
            text_proj_g = torch.cat(text_proj_all, dim=0)
            video_proj_g = torch.cat(video_proj_all, dim=0)
            # Labels offset by rank
            B_local = text_feat.size(0)
            labels = torch.arange(B_local, device=text_feat.device) + rank * B_local
        else:
            text_proj_g = text_proj
            video_proj_g = video_proj
            labels = torch.arange(text_feat.size(0), device=text_feat.device)

        # Learnable temperature with safety clamp
        logit_scale = self.itc_logit_scale.exp().clamp(max=100.0)

        # Similarity matrices
        sim_t2v = logit_scale * text_proj @ video_proj_g.t()    # (B, B_global)
        sim_v2t = logit_scale * video_proj @ text_proj_g.t()    # (B, B_global)

        loss_t2v = F.cross_entropy(sim_t2v, labels)
        loss_v2t = F.cross_entropy(sim_v2t, labels)
        itc_loss = (loss_t2v + loss_v2t) / 2.0

        # Top-1 retrieval accuracy (detached, for logging)
        with torch.no_grad():
            pred_t2v = sim_t2v.argmax(dim=-1)
            top1_acc = (pred_t2v == labels).float().mean()

        return itc_loss, top1_acc, logit_scale.detach()

    def _sample_hard_negatives(self, text_feat, video_feat, B):
        """Select hard negative indices from ITC similarity.
        Returns neg_text_idx: (B,) indices for negative text-video pairing.
        All operations are detached (no gradient through selection)."""
        with torch.no_grad():
            sim = F.normalize(text_feat, dim=-1) @ F.normalize(video_feat, dim=-1).t()  # (B, B)
            sim.fill_diagonal_(-float('inf'))  # mask positives

            itm_use_hard = getattr(self.task_config, 'itm_use_hard_negative', True) if hasattr(self, 'task_config') else True
            if itm_use_hard and B > 2:
                # For each video (column), pick text with highest similarity
                neg_text_idx = sim.argmax(dim=0)  # (B,)
                # Verify no diagonal selection (safety)
                diag = torch.arange(B, device=sim.device)
                collision = (neg_text_idx == diag)
                if collision.any():
                    # Fallback: for collisions, pick second-best
                    sim_safe = sim.clone()
                    sim_safe[neg_text_idx[collision], diag[collision]] = -float('inf')
                    neg_text_idx[collision] = sim_safe[:, diag[collision]].argmax(dim=0)
            else:
                # Random shuffle fallback (also for batch_size <= 2)
                neg_text_idx = torch.roll(torch.arange(B, device=text_feat.device), shifts=1)

        return neg_text_idx

    def _compute_itm_loss(self, sequence_output, visual_output, attention_mask, video_mask,
                           text_feat, video_feat, cross_output_pos, pooled_output_pos):
        """Image-Text Matching loss with hard negative mining.
        Positive: existing Q-Former output from caption path.
        Negative: re-run Q-Former with mismatched text-video pairs.
        Returns: itm_loss, itm_acc, pos_score_mean, neg_score_mean (all detached except loss)."""
        B = text_feat.size(0)

        # --- Positive ITM features ---
        itm_feat_pos = self._build_itm_features(pooled_output_pos, text_feat)  # (B, feat_dim)

        # --- Negative pairs: select hard negatives ---
        neg_text_idx = self._sample_hard_negatives(text_feat, video_feat, B)

        # Re-run Q-Former with mismatched text + original video
        _, pooled_neg, _ = self._get_cross_output(
            sequence_output[neg_text_idx], visual_output,
            attention_mask[neg_text_idx], video_mask
        )
        text_feat_neg = text_feat[neg_text_idx]
        itm_feat_neg = self._build_itm_features(pooled_neg, text_feat_neg)  # (B, feat_dim)

        # --- Classify ---
        itm_input = torch.cat([itm_feat_pos, itm_feat_neg], dim=0)  # (2B, feat_dim)
        itm_labels = torch.cat([
            torch.ones(B, device=text_feat.device, dtype=torch.long),
            torch.zeros(B, device=text_feat.device, dtype=torch.long)
        ])  # (2B,)

        itm_logits = self.itm_head(itm_input)  # (2B, 2)
        itm_loss = F.cross_entropy(itm_logits, itm_labels)

        # --- Logging metrics (detached) ---
        with torch.no_grad():
            itm_pred = itm_logits.argmax(dim=-1)
            itm_acc = (itm_pred == itm_labels).float().mean()
            pos_scores = F.softmax(itm_logits[:B], dim=-1)[:, 1]  # P(match) for positives
            neg_scores = F.softmax(itm_logits[B:], dim=-1)[:, 1]  # P(match) for negatives
            pos_mean = pos_scores.mean()
            neg_mean = neg_scores.mean()

        return itm_loss, itm_acc, pos_mean, neg_mean

    def _build_itm_features(self, pooled_output, text_feat):
        """Build input features for ITM classifier."""
        if self._itm_use_richer:
            return torch.cat([
                pooled_output, text_feat,
                torch.abs(pooled_output - text_feat),
                pooled_output * text_feat
            ], dim=-1)  # (B, 4H)
        else:
            return torch.cat([pooled_output, text_feat], dim=-1)  # (B, 2H)

    # ====================================================================
    # Core forward
    # ====================================================================

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = self.normalize_video(video)

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        sequence_output, visual_output = self.get_sequence_visual_output(input_ids, token_type_ids, attention_mask,
                                                                         video, video_mask, shaped=True)

        if self.training:
            loss = 0.
            if self._stage_one:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                        video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss += sim_loss

            if self._stage_two:
                # ── Pre-compute text_feat for ITC/ITM (reused across losses) ──
                text_feat = self._masked_mean_pool_text(sequence_output, attention_mask)  # (B, H)

                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_output(pairs_masked_text, token_type_ids,
                                                                                             attention_mask, masked_video, video_mask, shaped=True)

                    cross_output, pooled_output, concat_mask = self._get_cross_output(sequence_output_alm, visual_output_alm, attention_mask, video_mask)
                    expected_concat_len = attention_mask.size(-1) + video_mask.size(-1)
                    if cross_output.size(1) == expected_concat_len:
                        sequence_cross_output, visual_cross_output = torch.split(cross_output, [attention_mask.size(-1), video_mask.size(-1)], dim=1)
                        alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
                        loss += alm_loss
                        nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)
                        loss += nce_loss
                    else:
                        logger.warning("Q-Former cross output shape (%s) != concat length (%d). "
                                       "Skipping MLM/MFM pretrain losses.", cross_output.shape, expected_concat_len)

                    sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                    sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                    loss += sim_loss_joint

                # ── Decoder (caption) loss ──
                decoder_loss = torch.tensor(0.0, device=input_ids.device)
                cross_output_pos = None
                pooled_output_pos = None
                encoder_mask_pos = None

                if (input_caption_ids is not None) and \
                        (self.task_config.do_pretrain
                         or (self.task_config.do_pretrain is False and self.task_config.task_type == "caption")):
                    if self.task_config.do_pretrain:
                        decoder_scores, res_tuples = self._get_decoder_score(sequence_output_alm, visual_output_alm,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                    elif self.task_config.task_type == "caption":
                        decoder_scores, res_tuples = self._get_decoder_score(sequence_output, visual_output,
                                                                             input_ids, attention_mask, video_mask,
                                                                             input_caption_ids, decoder_mask, shaped=True)
                        # Capture cross_output from the decoder path for ITC/ITM reuse
                        # _get_decoder_score already called _get_cross_output internally,
                        # so we re-call here to get pooled_output for ITC/ITM.
                        # This is a second forward through cross encoder for this batch.
                        cross_output_pos, pooled_output_pos, encoder_mask_pos = self._get_cross_output(
                            sequence_output, visual_output, attention_mask, video_mask
                        )
                    else:
                        raise NotImplementedError

                    output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                    decoder_loss = self.decoder_loss_fct(
                        decoder_scores.view(-1, self.t5_vocab_size),
                        output_caption_ids.view(-1)
                    )
                    loss += decoder_loss

                # ── ITC loss ──
                itc_loss = torch.tensor(0.0, device=input_ids.device)
                itc_top1 = torch.tensor(0.0, device=input_ids.device)
                itc_temp = torch.tensor(0.0, device=input_ids.device)

                if self._use_itc:
                    # Video embedding from Q-Former (post cross-encoder)
                    if pooled_output_pos is not None:
                        video_feat = self._get_qformer_video_embedding(
                            cross_output_pos, pooled_output_pos, encoder_mask_pos)
                    else:
                        # Fallback: run cross encoder if not yet computed
                        co, po, em = self._get_cross_output(
                            sequence_output, visual_output, attention_mask, video_mask)
                        video_feat = self._get_qformer_video_embedding(co, po, em)
                        cross_output_pos, pooled_output_pos, encoder_mask_pos = co, po, em

                    itc_loss, itc_top1, itc_temp = self._compute_itc_loss(text_feat, video_feat)

                    current_itc_weight = getattr(self.task_config, 'current_itc_weight',
                                                 getattr(self.task_config, 'itc_weight', 0.05))
                    loss += current_itc_weight * itc_loss

                # ── ITM loss ──
                itm_loss = torch.tensor(0.0, device=input_ids.device)
                itm_acc = torch.tensor(0.0, device=input_ids.device)
                itm_pos_mean = torch.tensor(0.0, device=input_ids.device)
                itm_neg_mean = torch.tensor(0.0, device=input_ids.device)

                if self._use_itm and text_feat.size(0) > 1:
                    if pooled_output_pos is None:
                        co, po, em = self._get_cross_output(
                            sequence_output, visual_output, attention_mask, video_mask)
                        cross_output_pos, pooled_output_pos, encoder_mask_pos = co, po, em

                    video_feat_itm = self._get_qformer_video_embedding(
                        cross_output_pos, pooled_output_pos, encoder_mask_pos)

                    itm_loss, itm_acc, itm_pos_mean, itm_neg_mean = self._compute_itm_loss(
                        sequence_output, visual_output, attention_mask, video_mask,
                        text_feat, video_feat_itm, cross_output_pos, pooled_output_pos
                    )

                    current_itm_weight = getattr(self.task_config, 'current_itm_weight',
                                                 getattr(self.task_config, 'itm_weight', 0.05))
                    loss += current_itm_weight * itm_loss

                # ── Retrieval loss (existing, for pretrain/retrieval tasks) ──
                if self.task_config.do_pretrain or self.task_config.task_type == "retrieval":
                    if self.task_config.do_pretrain:
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output_alm, visual_output_alm,
                                                                            attention_mask, video_mask, shaped=True)
                    elif self.task_config.task_type == "retrieval":
                        sim_matrix_text_visual = self.get_similarity_logits(sequence_output, visual_output,
                                                                            attention_mask, video_mask, shaped=True)
                    else:
                        raise NotImplementedError

                    sim_loss_text_visual = self.loss_fct(sim_matrix_text_visual)
                    loss += sim_loss_text_visual

                # ── Store loss breakdown for trainer logging ──
                self._last_loss_dict = {
                    "loss": loss.detach(),
                    "decoder_loss": decoder_loss.detach(),
                    "itc_loss": itc_loss.detach(),
                    "itm_loss": itm_loss.detach(),
                    "itm_acc": itm_acc.detach() if torch.is_tensor(itm_acc) else itm_acc,
                    "itc_temp": itc_temp.detach() if torch.is_tensor(itc_temp) else itc_temp,
                    "itc_top1": itc_top1.detach() if torch.is_tensor(itc_top1) else itc_top1,
                    "itm_pos_mean": itm_pos_mean.detach() if torch.is_tensor(itm_pos_mean) else itm_pos_mean,
                    "itm_neg_mean": itm_neg_mean.detach() if torch.is_tensor(itm_neg_mean) else itm_neg_mean,
                }

            # Return dict in training mode
            return {"loss": loss, **{k: v for k, v in self._last_loss_dict.items() if k != "loss"}}

        else:
            # ── EVAL mode: return scalar decoder_loss for backward compat ──
            if (self._stage_two and
                input_caption_ids is not None and
                output_caption_ids is not None and
                self.task_config.task_type == "caption"):
                decoder_scores, res_tuples = self._get_decoder_score(sequence_output, visual_output,
                                                                     input_ids, attention_mask, video_mask,
                                                                     input_caption_ids, decoder_mask, shaped=True)
                output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
                decoder_loss = self.decoder_loss_fct(
                    decoder_scores.view(-1, self.t5_vocab_size),
                    output_caption_ids.view(-1)
                )
                return decoder_loss
            else:
                return None

    def _calculate_mlm_loss(self, sequence_output_alm, pairs_token_labels):
        alm_scores = self.cls(sequence_output_alm)
        alm_loss = self.alm_loss_fct(alm_scores.view(-1, self.bert_config.vocab_size), pairs_token_labels.view(-1))
        return alm_loss

    def _calculate_mfm_loss(self, visual_output_alm, video, video_mask, video_labels_index):
        afm_scores = self.cls_visual(visual_output_alm)
        afm_scores_tr = afm_scores.view(-1, afm_scores.shape[-1])

        video_tr = video.permute(2, 0, 1)
        video_tr = video_tr.view(video_tr.shape[0], -1)

        logits_matrix = torch.mm(afm_scores_tr, video_tr)
        video_mask_float = video_mask.to(dtype=torch.float)
        mask_matrix = torch.mm(video_mask_float.view(-1, 1), video_mask_float.view(1, -1))
        masked_logits = logits_matrix + (1. - mask_matrix) * -1e8

        logpt = F.log_softmax(masked_logits, dim=-1)
        logpt = torch.diag(logpt)
        nce_loss = -logpt

        video_labels_index_mask = (video_labels_index != self.ignore_video_index)
        nce_loss = nce_loss.masked_select(video_labels_index_mask.view(-1))
        nce_loss = nce_loss.mean()
        return nce_loss

    def get_sequence_visual_output(self, input_ids, token_type_ids, attention_mask, video, video_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        encoded_layers, _ = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=True)
        sequence_output = encoded_layers[-1]

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]

        return sequence_output, visual_output

    def _get_cross_output(self, sequence_output, visual_output, attention_mask, video_mask):
        concat_features = torch.cat((sequence_output, visual_output), dim=1)
        concat_mask = torch.cat((attention_mask, video_mask), dim=1)
        text_type_ = torch.zeros_like(attention_mask)
        video_type_ = torch.ones_like(video_mask)
        concat_type = torch.cat((text_type_, video_type_), dim=1)

        cross_layers, pooled_output = self.cross(concat_features, concat_type, concat_mask, output_all_encoded_layers=True)
        cross_output = cross_layers[-1]

        # Build encoder_mask that matches cross_output seq dimension.
        if cross_output.size(1) != concat_mask.size(1):
            if hasattr(self.cross, 'get_last_aux'):
                aux = self.cross.get_last_aux()
                encoder_mask = aux.get('query_mask', None)
            else:
                encoder_mask = None
            if encoder_mask is None:
                B, K = cross_output.shape[:2]
                encoder_mask = torch.ones(B, K, device=cross_output.device, dtype=concat_mask.dtype)
        else:
            encoder_mask = concat_mask

        return cross_output, pooled_output, encoder_mask

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum[video_mask_un_sum == 0.] = 1.
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, s_text, h_text = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        attention_mask_splits = torch.split(attention_mask, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]
            attention_mask_row = attention_mask_splits[i]
            sequence_output_l = sequence_output_row.unsqueeze(1).repeat(1, b_visual, 1, 1)
            sequence_output_l = sequence_output_l.view(-1, s_text, h_text)
            attention_mask_l = attention_mask_row.unsqueeze(1).repeat(1, b_visual, 1)
            attention_mask_l = attention_mask_l.view(-1, s_text)

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            cross_output, pooled_output, concat_mask = \
                self._get_cross_output(sequence_output_l, visual_output_r, attention_mask_l, video_mask_r)
            retrieve_logits_row = self.similarity_dense(pooled_output).squeeze(-1).view(step_truth, b_visual)

            retrieve_logits_list.append(retrieve_logits_row)
        retrieve_logits = torch.cat(retrieve_logits_list, dim=0)
        return retrieve_logits

    def get_similarity_logits(self, sequence_output, visual_output, attention_mask, video_mask, shaped=False, _pretrain_joint=False):
        if shaped is False:
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

        if (self._stage_two and _pretrain_joint is False) or self.train_sim_after_cross:
            retrieve_logits = self._cross_similarity(sequence_output, visual_output, attention_mask, video_mask)
        else:
            text_out, video_out = self._mean_pooling_for_similarity(sequence_output, visual_output, attention_mask, video_mask)
            if self.task_config.use_mil is False:
                text_out = F.normalize(text_out, dim=-1)
                video_out = F.normalize(video_out, dim=-1)
            retrieve_logits = torch.matmul(text_out, video_out.t())

        return retrieve_logits

    def _get_decoder_score(self, sequence_output, visual_output, input_ids, attention_mask, video_mask,
                           input_caption_ids, decoder_mask, shaped=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        res_tuples = ()
        cross_output, pooled_output, concat_mask = self._get_cross_output(
            sequence_output, visual_output, attention_mask, video_mask
        )
        t5_output = self.t5_decoder(
            input_ids=input_caption_ids,
            attention_mask=decoder_mask,
            encoder_hidden_states=cross_output,
            encoder_attention_mask=concat_mask,
        )
        decoder_scores = self.t5_lm_head(t5_output.last_hidden_state)

        return decoder_scores, res_tuples

    def decoder_caption(self, sequence_output, visual_output, input_ids, attention_mask, video_mask,
                        input_caption_ids, decoder_mask, shaped=False, get_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores, _ = self._get_decoder_score(sequence_output, visual_output,
                                                    input_ids, attention_mask, video_mask,
                                                    input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)
        return decoder_scores_result
