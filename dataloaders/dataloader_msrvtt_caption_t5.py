# Adapted from dataloaders/dataloader_msrvtt_caption.py
# Changes: caption tokenization uses T5Tokenizer instead of BertTokenizer.
# - pairs_text / pairs_mask / pairs_segment / MLM parts → still use bert_tokenizer (Text Encoder is BERT)
# - pairs_input_caption_ids / pairs_output_caption_ids / pairs_decoder_mask → use t5_tokenizer
# The constructor now accepts separate bert_tokenizer and t5_tokenizer.

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import os
from torch.utils.data import Dataset
import numpy as np
import pickle
import pandas as pd
from collections import defaultdict
import json
import random


class MSRVTT_Caption_T5_DataLoader(Dataset):
    """MSRVTT caption dataset with T5 tokenizer for the decoder side."""

    def __init__(
            self,
            csv_path,
            json_path,
            features_path,
            bert_tokenizer,
            t5_tokenizer,
            max_words=30,
            feature_framerate=1.0,
            max_frames=100,
            split_type=""
    ):
        self.csv = pd.read_csv(csv_path)
        self.data = json.load(open(json_path, 'r'))
        self.feature_dict = pickle.load(open(features_path, 'rb'))
        self.feature_framerate = feature_framerate
        self.max_words = max_words
        self.max_frames = max_frames
        self.bert_tokenizer = bert_tokenizer
        self.t5_tokenizer = t5_tokenizer

        self.feature_size = self.feature_dict[self.csv['video_id'].values[0]].shape[-1]

        assert split_type in ["train", "val", "test"]
        # Train: video0 : video6512 (6513)
        # Val: video6513 : video7009 (497)
        # Test: video7010 : video9999 (2990)
        video_ids = [self.data['videos'][idx]['video_id'] for idx in range(len(self.data['videos']))]
        split_dict = {"train": video_ids[:6513], "val": video_ids[6513:6513 + 497], "test": video_ids[6513 + 497:]}
        choiced_video_ids = split_dict[split_type]

        self.sample_len = 0
        self.sentences_dict = {}
        self.video_sentences_dict = defaultdict(list)
        if split_type == "train":
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.sentences_dict[len(self.sentences_dict)] = (itm['video_id'], itm['caption'])
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
        elif split_type in ("val", "test"):
            for itm in self.data['sentences']:
                if itm['video_id'] in choiced_video_ids:
                    self.video_sentences_dict[itm['video_id']].append(itm['caption'])
            for vid in choiced_video_ids:
                self.sentences_dict[len(self.sentences_dict)] = (vid, self.video_sentences_dict[vid][0])
        else:
            raise NotImplementedError

        self.sample_len = len(self.sentences_dict)

    def __len__(self):
        return self.sample_len

    def _get_text(self, video_id, caption=None):
        k = 1
        choice_video_ids = [video_id]

        # ── BERT-side arrays (Text Encoder input) ──────────────────────────
        pairs_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_mask = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_segment = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_masked_text = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_token_labels = np.zeros((k, self.max_words), dtype=np.int64)

        # ── T5-side arrays (Decoder input/output) ──────────────────────────
        pairs_input_caption_ids = np.zeros((k, self.max_words), dtype=np.int64)
        pairs_output_caption_ids = np.full((k, self.max_words), -1, dtype=np.int64)  # -1 = ignore_index
        pairs_decoder_mask = np.zeros((k, self.max_words), dtype=np.int64)

        for i, video_id in enumerate(choice_video_ids):

            # ── BERT text (Text Encoder) ────────────────────────────────────
            words = ["[CLS]"]
            total_length_with_CLS = self.max_words - 1
            words = words + ["[SEP]"]

            # MLM masking (used in pre-training; kept for interface compatibility)
            token_labels = []
            masked_tokens = words.copy()
            for token_id, token in enumerate(masked_tokens):
                if token_id == 0 or token_id == len(masked_tokens) - 1:
                    token_labels.append(-1)
                    continue
                prob = random.random()
                if prob < 0.15:
                    prob /= 0.15
                    if prob < 0.8:
                        masked_tokens[token_id] = "[MASK]"
                    elif prob < 0.9:
                        masked_tokens[token_id] = random.choice(list(self.bert_tokenizer.vocab.items()))[0]
                    try:
                        token_labels.append(self.bert_tokenizer.vocab[token])
                    except KeyError:
                        token_labels.append(self.bert_tokenizer.vocab["[UNK]"])
                else:
                    token_labels.append(-1)

            input_ids = self.bert_tokenizer.convert_tokens_to_ids(words)
            input_mask = [1] * len(input_ids)
            segment_ids = [0] * len(input_ids)
            masked_token_ids = self.bert_tokenizer.convert_tokens_to_ids(masked_tokens)
            while len(input_ids) < self.max_words:
                input_ids.append(0)
                input_mask.append(0)
                segment_ids.append(0)
                masked_token_ids.append(0)
                token_labels.append(-1)

            pairs_text[i] = np.array(input_ids)
            pairs_mask[i] = np.array(input_mask)
            pairs_segment[i] = np.array(segment_ids)
            pairs_masked_text[i] = np.array(masked_token_ids)
            pairs_token_labels[i] = np.array(token_labels)

            # ── T5 caption (Decoder) ────────────────────────────────────────
            # T5Tokenizer.encode() returns token_ids with </s> (=1) appended.
            # Teacher forcing layout:
            #   input_caption  = [<pad>(=0)] + token_ids[:-1]   (decoder start + shifted caption)
            #   output_caption = token_ids                       (caption + </s>)
            if caption is not None:
                raw_caption = caption
            else:
                raw_caption = self._get_single_raw_caption(video_id)

            # Encode; leave room so that EOS fits within max_words
            caption_ids = self.t5_tokenizer.encode(
                raw_caption,
                max_length=self.max_words - 1,  # leave 1 slot for decoder_start at input side
                truncation=True,
                add_special_tokens=True          # appends </s>
            )
            # caption_ids ends with </s> (id=1)

            # Decoder input: start with pad_token_id (=0, T5 decoder_start_token_id)
            input_caption_ids = [self.t5_tokenizer.pad_token_id] + caption_ids[:-1]
            output_caption_ids = caption_ids  # target includes </s>
            decoder_mask_ids = [1] * len(input_caption_ids)

            # Pad to max_words
            pad_len = self.max_words - len(input_caption_ids)
            input_caption_ids = input_caption_ids + [self.t5_tokenizer.pad_token_id] * pad_len
            output_caption_ids = output_caption_ids + [-1] * (self.max_words - len(output_caption_ids))
            decoder_mask_ids = decoder_mask_ids + [0] * pad_len

            pairs_input_caption_ids[i] = np.array(input_caption_ids[:self.max_words])
            pairs_output_caption_ids[i] = np.array(output_caption_ids[:self.max_words])
            pairs_decoder_mask[i] = np.array(decoder_mask_ids[:self.max_words])

        return (pairs_text, pairs_mask, pairs_segment,
                pairs_masked_text, pairs_token_labels,
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids,
                choice_video_ids)

    def _get_single_raw_caption(self, video_id):
        """Return a random raw caption string for video_id (fallback when caption=None)."""
        captions = self.video_sentences_dict.get(video_id, [""])
        return captions[random.randint(0, len(captions) - 1)]

    def _get_video(self, choice_video_ids):
        video_mask = np.zeros((len(choice_video_ids), self.max_frames), dtype=np.int64)
        max_video_length = [0] * len(choice_video_ids)

        video = np.zeros((len(choice_video_ids), self.max_frames, self.feature_size), dtype=np.float32)
        for i, video_id in enumerate(choice_video_ids):
            video_slice = self.feature_dict[video_id]

            if self.max_frames < video_slice.shape[0]:
                video_slice = video_slice[:self.max_frames]

            slice_shape = video_slice.shape
            max_video_length[i] = max_video_length[i] if max_video_length[i] > slice_shape[0] else slice_shape[0]
            if len(video_slice) < 1:
                print("video_id: {}".format(video_id))
            else:
                video[i][:slice_shape[0]] = video_slice

        for i, v_length in enumerate(max_video_length):
            video_mask[i][:v_length] = [1] * v_length

        # Mask Frame Model
        video_labels_index = [[] for _ in range(len(choice_video_ids))]
        masked_video = video.copy()
        for i, video_pair_ in enumerate(masked_video):
            for j, _ in enumerate(video_pair_):
                if j < max_video_length[i]:
                    prob = random.random()
                    if prob < 0.15:
                        masked_video[i][j] = [0.] * video.shape[-1]
                        video_labels_index[i].append(j)
                    else:
                        video_labels_index[i].append(-1)
                else:
                    video_labels_index[i].append(-1)
        video_labels_index = np.array(video_labels_index, dtype=np.int64)

        return video, video_mask, masked_video, video_labels_index

    def __getitem__(self, idx):
        video_id, caption = self.sentences_dict[idx]

        (pairs_text, pairs_mask, pairs_segment,
         pairs_masked_text, pairs_token_labels,
         pairs_input_caption_ids, pairs_decoder_mask,
         pairs_output_caption_ids, choice_video_ids) = self._get_text(video_id, caption)

        video, video_mask, masked_video, video_labels_index = self._get_video(choice_video_ids)

        return (pairs_text, pairs_mask, pairs_segment, video, video_mask,
                pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,
                pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids)
