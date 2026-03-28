# Adapted from inference/eval_utils.py
# Changes: decode_tokens_to_text uses T5Tokenizer.decode() instead of BERT
#          convert_ids_to_tokens + manual subword joining.

from typing import List, Optional, Any
import os


def decode_tokens_to_text(token_ids: List[int], tokenizer: Any) -> str:
    """Decode a list of T5 token ids to a string, skipping special tokens."""
    # T5Tokenizer.decode handles <pad>, </s>, <unk> etc. transparently
    decode_text = tokenizer.decode(token_ids, skip_special_tokens=True)
    return decode_text.strip()


def save_predictions(
    predictions: List[str],
    references: List[str],
    output_dir: str,
    pred_filename: str = "hyp.txt",
    ref_filename: str = "ref.txt"
) -> tuple:
    os.makedirs(output_dir, exist_ok=True)

    pred_path = os.path.join(output_dir, pred_filename)
    with open(pred_path, "w", encoding='utf-8') as writer:
        for pred_txt in predictions:
            writer.write(pred_txt + "\n")

    ref_path = os.path.join(output_dir, ref_filename)
    with open(ref_path, "w", encoding='utf-8') as writer:
        for ref_txt in references:
            writer.write(ref_txt + "\n")

    return pred_path, ref_path


def save_complete_results(
    predictions: List[str],
    dataset: Any,
    output_dir: str,
    filename: str = "hyp_complete_results.txt"
) -> Optional[str]:
    if not hasattr(dataset, 'iter2video_pairs_dict') or not hasattr(dataset, 'data_dict'):
        return None

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)

    with open(output_path, "w", encoding='utf-8') as writer:
        writer.write("{}\t{}\t{}\n".format("video_id", "start_time", "caption"))
        for idx, pred_txt in enumerate(predictions):
            video_id, sub_id = dataset.iter2video_pairs_dict[idx]
            start_time = dataset.data_dict[video_id]['start'][sub_id]
            writer.write("{}\t{}\t{}\n".format(video_id, start_time, pred_txt))

    return output_path


def log_metrics(logger: Any, metrics: dict) -> None:
    for key, value in metrics.items():
        logger.info("  %s = %s", key, str(value))
