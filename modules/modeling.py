# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch BERT model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import numpy as np
import itertools

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import T5TokenizerFast


from modules.until_module import PreTrainedModel, LayerNorm, CrossEn, MILNCELoss, MaxMarginRankingLoss
from modules.module_bert import BertModel, BertConfig, BertOnlyMLMHead
from modules.module_visual import VisualModel, VisualConfig, VisualOnlyMLMHead
from modules.module_cross import CrossModel, CrossConfig
from modules.module_decoder import DecoderConfig
from modules.blip2 import Blip2Base
# from modules.modeling_t5 import T5Config, T5ForConditionalGeneration
from transformers import T5Config, T5ForConditionalGeneration


from peft import LoraConfig, TaskType, get_peft_model

logger = logging.getLogger(__name__)


def tokenize(refs, cands, no_op=False):
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer

    tokenizer = PTBTokenizer()

    if no_op:
        refs = {idx: [r for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [c] for idx, c in enumerate(cands)}
    else:
        refs = {idx: [{'caption': r} for r in c_refs] for idx, c_refs in enumerate(refs)}
        cands = {idx: [{'caption': c}] for idx, c in enumerate(cands)}

        refs = tokenizer.tokenize(refs)
        cands = tokenizer.tokenize(cands)

    return refs, cands


class UniVLPreTrainedModel(PreTrainedModel, nn.Module):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs):
        # utilize bert config as base config
        super(UniVLPreTrainedModel, self).__init__(bert_config)
        self.bert_config = bert_config
        self.visual_config = visual_config
        self.cross_config = cross_config
        self.decoder_config = decoder_config

        self.bert = None
        self.visual = None
        self.cross = None
        self.decoder = None

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
        decoder_config, _ = DecoderConfig.get_config(decoder_model_name, cache_dir, type_vocab_size, state_dict=None, task_config=task_config)

        model = cls(bert_config, visual_config, cross_config, decoder_config, *inputs, **kwargs)

        assert model.bert is not None
        assert model.visual is not None

        if state_dict is not None:
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

class UniVL(UniVLPreTrainedModel):
    def __init__(self, bert_config, visual_config, cross_config, decoder_config, task_config):
        super(UniVL, self).__init__(bert_config, visual_config, cross_config, decoder_config)
        self.task_config = task_config
        self.ignore_video_index = -1

        assert self.task_config.max_words <= bert_config.max_position_embeddings
        assert self.task_config.max_words <= decoder_config.max_target_embeddings
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
        # <=== End of Text Encoder

        # Video Encoder ===>
        visual_config = update_attr("visual_config", visual_config, "num_hidden_layers",
                                    self.task_config, "visual_num_hidden_layers")
        self.visual = VisualModel(visual_config)
        self.freeze_vit = getattr(self.task_config, "freeze_vit", False)
        if self.freeze_vit:
            for param in self.visual.parameters():
                param.requires_grad = False
            show_log(task_config, "Freeze vision encoder.")
        visual_word_embeddings_weight = self.visual.embeddings.word_embeddings.weight
        # <=== End of Video Encoder

        if self._stage_one is False or self.train_sim_after_cross:
            # Cross Encoder ===>
            cross_config = update_attr("cross_config", cross_config, "num_hidden_layers",
                                        self.task_config, "cross_num_hidden_layers")
            self.cross = CrossModel(cross_config)
            self.num_query_token = getattr(self.task_config, "num_query_token", 32)
            self.Qformer, self.query_tokens = Blip2Base.init_Qformer(
                self.num_query_token, visual_config.hidden_size
            )
            self.Qformer.cls = None
            self.Qformer.bert.embeddings.word_embeddings = None
            self.Qformer.bert.embeddings.position_embeddings = None
            for layer in self.Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            # <=== End of Cross Encoder

            if self.train_sim_after_cross is False:
                # Decoder ===>
                self.scst = getattr(self.task_config, "scst", False)
                self.beam_size = getattr(self.task_config, "beam_size", 5)
                self.max_txt_len = getattr(self.task_config, "max_txt_len", 32)
                self.prompt = " A video of"

                t5_model_name = getattr(self.task_config, "t5_model", "google/flan-t5-xl")
                self.t5_tokenizer = T5TokenizerFast.from_pretrained(t5_model_name)
                t5_config = T5Config.from_pretrained(t5_model_name)
                t5_config.dense_act_fn = "gelu"
                self.t5_model = T5ForConditionalGeneration.from_pretrained(
                    t5_model_name, config=t5_config,
                )
                for name, param in self.t5_model.named_parameters():
                    param.requires_grad = False
                    param.data = param.data.bfloat16()

                lora = getattr(self.task_config, "lora", False)
                lora_r = getattr(self.task_config, "lora_r", 16)
                lora_alpha = getattr(self.task_config, "lora_alpha", 32)
                lora_dropout = getattr(self.task_config, "lora_dropout", 0.05)
                peft_config = LoraConfig(
                    task_type=TaskType.SEQ_2_SEQ_LM,
                    inference_mode=False,
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    target_modules=['q', 'v']
                )

                if lora:
                    self.t5_model = get_peft_model(self.t5_model, peft_config)
                    self.t5_model.print_trainable_parameters()

                self.t5_proj = nn.Linear(
                    self.Qformer.config.hidden_size, self.t5_model.config.hidden_size
                )
                # <=== End of Decoder

            if self.task_config.do_pretrain:
                self.cls = BertOnlyMLMHead(bert_config, bert_word_embeddings_weight)
                self.cls_visual = VisualOnlyMLMHead(visual_config, visual_word_embeddings_weight)
                self.alm_loss_fct = CrossEntropyLoss(ignore_index=-1)
                
            self.similarity_dense = nn.Linear(bert_config.hidden_size, 1)

        self.normalize_video = NormalizeVideo(task_config)

        mil_nce_loss = MILNCELoss(batch_size=task_config.batch_size // task_config.n_gpu, n_pair=task_config.n_pair, )
        max_margin_ranking_loss = MaxMarginRankingLoss(margin=task_config.margin,
                                   negative_weighting=task_config.negative_weighting,
                                   batch_size=task_config.batch_size // task_config.n_gpu,
                                   n_pair=task_config.n_pair,
                                   hard_negative_rate=task_config.hard_negative_rate, )

        if task_config.use_mil:
            self.loss_fct = CrossEn() if self._stage_two else mil_nce_loss
            self._pretrain_sim_loss_fct = mil_nce_loss
        else:
            self.loss_fct = CrossEn() if self._stage_two else max_margin_ranking_loss
            self._pretrain_sim_loss_fct = max_margin_ranking_loss

        self.apply(self.init_weights)

    def forward(self, input_ids, token_type_ids, attention_mask, video, video_mask=None,
                pairs_masked_text=None, pairs_token_labels=None, masked_video=None, video_labels_index=None,
                input_caption_ids=None, decoder_mask=None, output_caption_ids=None,
                t5_output_caption_ids=None):

        input_ids = input_ids.view(-1, input_ids.shape[-1])
        token_type_ids = token_type_ids.view(-1, token_type_ids.shape[-1])
        attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
        video_mask = video_mask.view(-1, video_mask.shape[-1])
        video = self.normalize_video(video)

        if input_caption_ids is not None:
            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        # Skip text encoder when it's not needed (caption-only fine-tuning)
        _need_text_encoder = (
            self._stage_one
            or (self._stage_two and self.task_config.do_pretrain)
            or (self._stage_two and self.task_config.task_type == "retrieval")
        )

        if _need_text_encoder:
            sequence_output, visual_output = self.get_sequence_visual_output(
                input_ids, token_type_ids, attention_mask, video, video_mask, shaped=True
            )
        else:
            visual_output = self.get_visual_output(video, video_mask, shaped=True)
            sequence_output = None

        if self.training:
            loss = 0.
            if self._stage_one:
                sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask,
                                                        video_mask, shaped=True)
                sim_loss = self.loss_fct(sim_matrix)
                loss += sim_loss

            if self._stage_two:
                if self.task_config.do_pretrain:
                    pairs_masked_text = pairs_masked_text.view(-1, pairs_masked_text.shape[-1])
                    pairs_token_labels = pairs_token_labels.view(-1, pairs_token_labels.shape[-1])

                    masked_video = self.normalize_video(masked_video)
                    video_labels_index = video_labels_index.view(-1, video_labels_index.shape[-1])

                    sequence_output_alm, visual_output_alm = self.get_sequence_visual_output(pairs_masked_text, token_type_ids,
                                                                                             attention_mask, masked_video, video_mask, shaped=True)

                    sequence_cross_output = sequence_output_alm
                    visual_cross_output = visual_output_alm

                    alm_loss = self._calculate_mlm_loss(sequence_cross_output, pairs_token_labels)
                    loss += alm_loss

                    nce_loss = self._calculate_mfm_loss(visual_cross_output, video, video_mask, video_labels_index)
                    loss += nce_loss

                    sim_matrix = self.get_similarity_logits(sequence_output, visual_output, attention_mask, video_mask,
                                                            shaped=True, _pretrain_joint=True)
                    sim_loss_joint = self._pretrain_sim_loss_fct(sim_matrix)
                    loss += sim_loss_joint

                if (input_caption_ids is not None) and \
                        (self.task_config.do_pretrain
                         or (self.task_config.do_pretrain is False and self.task_config.task_type == "caption")):
                    if self.task_config.do_pretrain:
                        decoder_loss = self._get_t5_caption_loss(visual_output_alm,
                                                                 video_mask,
                                                                 output_caption_ids,
                                                                 t5_output_caption_ids)
                    elif self.task_config.task_type == "caption":
                        decoder_loss = self._get_t5_caption_loss(visual_output,
                                                                 video_mask,
                                                                 output_caption_ids,
                                                                 t5_output_caption_ids)
                    else:
                        raise NotImplementedError
                    loss += decoder_loss

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

            return loss
        else:
            # During evaluation, return (loss, visual_output) so callers can
            # reuse visual_output for generation without re-encoding.
            if (self._stage_two and 
                input_caption_ids is not None and 
                output_caption_ids is not None and
                self.task_config.task_type == "caption"):
                decoder_loss = self._get_t5_caption_loss(visual_output,
                                                         video_mask,
                                                         output_caption_ids,
                                                         t5_output_caption_ids)
                return decoder_loss, visual_output
            else:
                return None, visual_output

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

    def get_visual_output(self, video, video_mask, shaped=False):
        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])
            video = self.normalize_video(video)

        visual_layers, _ = self.visual(video, video_mask, output_all_encoded_layers=True)
        visual_output = visual_layers[-1]
        return visual_output

    def _get_cross_output(self, visual_output, video_mask, num_query_token=32):
        # Use BLIP2 Qformer query cross-attention and expose query tokens as encoder outputs.
        b_visual, _, _ = visual_output.size()
        query_len = min(num_query_token, self.query_tokens.size(1))
        qformer_dtype = self.query_tokens.dtype
        query_tokens = self.query_tokens[:, :query_len, :].expand(b_visual, -1, -1).to(
            device=visual_output.device,
            dtype=qformer_dtype,
        )
        visual_for_qformer = visual_output.to(dtype=qformer_dtype)
        image_atts = video_mask.long()
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=visual_for_qformer,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )

        cross_output = query_output.last_hidden_state.to(dtype=visual_output.dtype)
        pooled_output = cross_output[:, 0]

        return cross_output, pooled_output

    def _build_t5_encoder_inputs(self, visual_output, video_mask):
        cross_output, _ = self._get_cross_output(visual_output, video_mask)
        inputs_t5 = self.t5_proj(cross_output)
        atts_t5 = torch.ones(inputs_t5.size()[:-1], dtype=torch.long, device=inputs_t5.device)

        prompt = [self.prompt] * inputs_t5.size(0)
        prompt_tokens = self.t5_tokenizer(
            prompt,
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(inputs_t5.device)

        prompt_embeds = self.t5_model.encoder.embed_tokens(prompt_tokens.input_ids)
        inputs_embeds = torch.cat([inputs_t5, prompt_embeds], dim=1)
        encoder_atts = torch.cat([atts_t5, prompt_tokens.attention_mask], dim=1)
        return inputs_embeds, encoder_atts

    def _compute_xe_caption_loss(self, inputs_embeds, encoder_atts, output_caption_ids):
        pad_token_id = self.t5_tokenizer.pad_token_id
        output_tokens = output_caption_ids.clone()
        output_tokens = output_tokens.masked_fill(output_tokens.lt(0), pad_token_id)
        output_mask = output_tokens.ne(pad_token_id).long()
        targets = output_tokens.masked_fill(output_tokens.eq(pad_token_id), -100)

        outputs = self.t5_model(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            decoder_attention_mask=output_mask,
            return_dict=True,
            labels=targets,
        )
        return outputs.loss

    def _compute_scst_caption_loss(self, inputs_embeds, encoder_atts, output_caption_ids, t5_output_caption_ids=None):
        from pycocoevalcap.cider.cider import Cider

        outputs = self.t5_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=encoder_atts,
            do_sample=False,
            top_p=0.9,
            temperature=1,
            num_beams=self.beam_size,
            max_length=self.max_txt_len,
            repetition_penalty=1.0,
            length_penalty=1.0,
            num_return_sequences=self.beam_size,
            return_dict_in_generate=True,
            output_scores=True,
        )

        transition_scores = self.t5_model.compute_transition_scores(
            outputs.sequences, outputs.scores, outputs.beam_indices, normalize_logits=False
        )
        output_length = torch.sum(transition_scores < 0, dim=1).clamp(min=1)
        sequences_scores = transition_scores.sum(dim=1) / output_length

        batch_size = output_caption_ids.size(0)
        sequences_scores = sequences_scores.view(batch_size, -1)

        caps_gen = self.t5_tokenizer.batch_decode(outputs.sequences, skip_special_tokens=True)
        caps_gen = [text.strip() for text in caps_gen]

        # Use T5-tokenized GT IDs if available (correct vocab), otherwise
        # fall back to output_caption_ids (BERT vocab — legacy/incorrect)
        if t5_output_caption_ids is not None:
            gt_ids = t5_output_caption_ids
        else:
            gt_ids = output_caption_ids
        pad_token_id = self.t5_tokenizer.pad_token_id
        gt_tokens = gt_ids.clone().masked_fill(gt_ids.lt(0), pad_token_id)
        caps_gt = self.t5_tokenizer.batch_decode(gt_tokens, skip_special_tokens=True)
        caps_gt = list(itertools.chain(*([c] * self.beam_size for c in caps_gt)))
        caps_gt = [[c] for c in caps_gt]

        caps_gen, caps_gt = tokenize(caps_gt, caps_gen)
        reward = Cider().compute_score(caps_gt, caps_gen)[1].astype(np.float32)
        reward = torch.from_numpy(reward).to(inputs_embeds.device).view(batch_size, self.beam_size)
        reward_baseline = torch.mean(reward, -1, keepdim=True)

        loss = -(sequences_scores) * (reward - reward_baseline)
        return loss.mean()

    def _get_t5_caption_loss(self, visual_output, video_mask, output_caption_ids, t5_output_caption_ids=None):
        if output_caption_ids is None:
            return torch.tensor(0.0, device=visual_output.device)

        output_caption_ids = output_caption_ids.view(-1, output_caption_ids.shape[-1])
        if t5_output_caption_ids is not None:
            t5_output_caption_ids = t5_output_caption_ids.view(-1, t5_output_caption_ids.shape[-1])
        caption_label_ids = t5_output_caption_ids if t5_output_caption_ids is not None else output_caption_ids

        with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            inputs_embeds, encoder_atts = self._build_t5_encoder_inputs(
                visual_output, video_mask
            )
            if self.training and getattr(self, "scst", False):
                return self._compute_scst_caption_loss(inputs_embeds, encoder_atts, output_caption_ids, t5_output_caption_ids)
            return self._compute_xe_caption_loss(inputs_embeds, encoder_atts, caption_label_ids)

    def generate_caption_ids(self, visual_output, video_mask, num_beams=None, max_length=None):
        if num_beams is None:
            num_beams = max(1, getattr(self, "beam_size", 1))
        if max_length is None:
            max_length = getattr(self, "max_txt_len", 32)

        with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            inputs_embeds, encoder_atts = self._build_t5_encoder_inputs(
                visual_output, video_mask
            )
            outputs = self.t5_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                do_sample=False,
                num_beams=num_beams,
                max_length=max_length,
                repetition_penalty=1.0,
                length_penalty=1.0,
            )

        return outputs

    def generate_caption_text(self, visual_output, video_mask, num_beams=None, max_length=None):
        output_ids = self.generate_caption_ids(
            visual_output, video_mask, num_beams=num_beams, max_length=max_length
        )
        captions = self.t5_tokenizer.batch_decode(output_ids, skip_special_tokens=True)
        return [caption.strip() for caption in captions]

    def _mean_pooling_for_similarity(self, sequence_output, visual_output, attention_mask, video_mask,):
        attention_mask_un = attention_mask.to(dtype=torch.float).unsqueeze(-1)
        attention_mask_un[:, 0, :] = 0.
        sequence_output = sequence_output * attention_mask_un
        text_out = torch.sum(sequence_output, dim=1) / torch.sum(attention_mask_un, dim=1, dtype=torch.float)

        video_mask_un = video_mask.to(dtype=torch.float).unsqueeze(-1)
        visual_output = visual_output * video_mask_un
        video_mask_un_sum = torch.sum(video_mask_un, dim=1, dtype=torch.float)
        video_mask_un_sum = torch.clamp(video_mask_un_sum, min=1.0)
        video_out = torch.sum(visual_output, dim=1) / video_mask_un_sum

        return text_out, video_out

    def _cross_similarity(self, sequence_output, visual_output, attention_mask, video_mask):
        b_text, _, _ = sequence_output.size()
        b_visual, s_visual, h_visual = visual_output.size()

        retrieve_logits_list = []
        step_size = 5

        split_size = [step_size] * (b_text // step_size)
        release_size = b_text - sum(split_size)
        if release_size > 0:
            split_size += [release_size]

        sequence_output_splits = torch.split(sequence_output, split_size, dim=0)
        for i in range(len(split_size)):
            sequence_output_row = sequence_output_splits[i]

            step_truth = sequence_output_row.size(0)
            visual_output_r = visual_output.unsqueeze(0).repeat(step_truth, 1, 1, 1)
            visual_output_r = visual_output_r.view(-1, s_visual, h_visual)
            video_mask_r = video_mask.unsqueeze(0).repeat(step_truth, 1, 1)
            video_mask_r = video_mask_r.view(-1, s_visual)

            _, pooled_output = self._get_cross_output(visual_output_r, video_mask_r)
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

    def _get_decoder_score(self, visual_output, video_mask, input_caption_ids, decoder_mask, shaped=False):

        if shaped is False:
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        with torch.amp.autocast(device_type="cuda",dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            inputs_embeds, encoder_atts = self._build_t5_encoder_inputs(
                visual_output, video_mask
            )

            pad_token_id = self.t5_tokenizer.pad_token_id
            decoder_input_ids = input_caption_ids.clone().masked_fill(input_caption_ids.lt(0), pad_token_id)
            if decoder_mask is not None:
                decoder_att_mask = decoder_mask.long()
            else:
                decoder_att_mask = decoder_input_ids.ne(pad_token_id).long()

            outputs = self.t5_model(
                inputs_embeds=inputs_embeds,
                attention_mask=encoder_atts,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_att_mask,
                return_dict=True,
            )
            decoder_scores = outputs.logits

        return decoder_scores

    def decoder_caption(self, sequence_output, visual_output, input_ids, attention_mask, video_mask, input_caption_ids, decoder_mask,
                        shaped=False, get_logits=False):
        if shaped is False:
            input_ids = input_ids.view(-1, input_ids.shape[-1])
            attention_mask = attention_mask.view(-1, attention_mask.shape[-1])
            video_mask = video_mask.view(-1, video_mask.shape[-1])

            input_caption_ids = input_caption_ids.view(-1, input_caption_ids.shape[-1])
            decoder_mask = decoder_mask.view(-1, decoder_mask.shape[-1])

        decoder_scores = self._get_decoder_score(visual_output,
                             video_mask,
                             input_caption_ids, decoder_mask, shaped=True)

        if get_logits:
            return decoder_scores

        _, decoder_scores_result = torch.max(decoder_scores, -1)

        return decoder_scores_result
