"""
Adapted from modules/beam.py
Changes: Constants updated for T5 tokenizer.
  T5 uses:
    PAD / decoder_start = <pad> (id=0)
    EOS                 = </s>  (id=1)
  The from_tokenizer classmethod reads these from T5Tokenizer attributes.
"""

import torch


class Constants():
    def __init__(self):
        # T5 defaults
        self.PAD = 0      # <pad>
        self.BOS = 0      # T5 uses pad_token as decoder_start_token
        self.EOS = 1      # </s>

    @classmethod
    def from_tokenizer(cls, tokenizer):
        instance = cls()
        instance.PAD = tokenizer.pad_token_id   # 0
        instance.BOS = tokenizer.pad_token_id   # 0 — T5 decoder_start_token_id
        instance.EOS = tokenizer.eos_token_id   # 1
        return instance


class Beam():
    """Beam search — T5 token id variant."""

    def __init__(self, size, device=False, tokenizer=None):
        if tokenizer is None:
            self.constants = Constants()
        else:
            self.constants = Constants.from_tokenizer(tokenizer)

        self.size = size
        self._done = False

        self.scores = torch.zeros((size,), dtype=torch.float, device=device)
        self.all_scores = []

        self.prev_ks = []

        # First token is BOS (= pad_token_id = 0 for T5)
        self.next_ys = [torch.full((size,), self.constants.BOS, dtype=torch.long, device=device)]

    def get_current_state(self):
        return self.get_tentative_hypothesis()

    def get_current_origin(self):
        return self.prev_ks[-1]

    @property
    def done(self):
        return self._done

    def _block_ngram_repeats(self, word_prob, no_repeat_ngram_size):
        """Set probability of tokens that would create a repeated n-gram to -inf."""
        word_prob = word_prob.clone()
        for beam_idx in range(self.size):
            hyp = self.get_hypothesis(beam_idx)
            hyp_len = len(hyp)
            if hyp_len < no_repeat_ngram_size - 1:
                continue
            # (n-1)-gram suffix of current hypothesis
            ngram_prefix = tuple(hyp[-(no_repeat_ngram_size - 1):])
            # Collect all tokens that ever followed this prefix
            banned = set()
            for i in range(hyp_len - (no_repeat_ngram_size - 1)):
                if tuple(hyp[i:i + no_repeat_ngram_size - 1]) == ngram_prefix:
                    banned.add(hyp[i + no_repeat_ngram_size - 1])
            for token_id in banned:
                word_prob[beam_idx, token_id] = float('-inf')
        return word_prob

    def advance(self, word_prob, word_length=None, no_repeat_ngram_size=3):
        num_words = word_prob.size(1)
        # Block repeated n-grams to prevent "a man in a man in a man" loops
        if len(self.prev_ks) > 0 and no_repeat_ngram_size > 0:
            word_prob = self._block_ngram_repeats(word_prob, no_repeat_ngram_size)
        if len(self.prev_ks) > 0:
            beam_lk = word_prob + self.scores.unsqueeze(1).expand_as(word_prob)
        else:
            beam_lk = word_prob[0]
        flat_beam_lk = beam_lk.view(-1)
        best_scores, best_scores_id = flat_beam_lk.topk(self.size, 0, True, True)
        self.all_scores.append(self.scores)
        self.scores = best_scores
        prev_k = best_scores_id // num_words
        self.prev_ks.append(prev_k)
        self.next_ys.append(best_scores_id - prev_k * num_words)
        # Stop when EOS (</s> = 1) is top of beam
        if self.next_ys[-1][0].item() == self.constants.EOS:
            self._done = True
        return self._done

    def sort_scores(self):
        return torch.sort(self.scores, 0, True)

    def get_the_best_score_and_idx(self):
        scores, ids = self.sort_scores()
        return scores[1], ids[1]

    def get_tentative_hypothesis(self):
        if len(self.next_ys) == 1:
            dec_seq = self.next_ys[0].unsqueeze(1)
        else:
            _, keys = self.sort_scores()
            hyps = [self.get_hypothesis(k) for k in keys]
            hyps = [[self.constants.BOS] + h for h in hyps]
            dec_seq = torch.LongTensor(hyps)
        return dec_seq

    def get_hypothesis(self, k):
        hyp = []
        for j in range(len(self.prev_ks) - 1, -1, -1):
            hyp.append(self.next_ys[j + 1][k])
            k = self.prev_ks[j][k]
        return list(map(lambda x: x.item(), hyp[::-1]))
