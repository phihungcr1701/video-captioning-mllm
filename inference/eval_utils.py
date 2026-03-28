from typing import List, Optional, Any
import os


def decode_tokens_to_text(token_ids: List[int], tokenizer: Any) -> str:
    decode_text_list = tokenizer.convert_ids_to_tokens(token_ids)
    
    # Remove special tokens
    if "[SEP]" in decode_text_list:
        SEP_index = decode_text_list.index("[SEP]")
        decode_text_list = decode_text_list[:SEP_index]
    if "[PAD]" in decode_text_list:
        PAD_index = decode_text_list.index("[PAD]")
        decode_text_list = decode_text_list[:PAD_index]
    
    # Join tokens and handle subword tokens (##)
    decode_text = ' '.join(decode_text_list)
    decode_text = decode_text.replace(" ##", "").strip("##").strip()
    
    return decode_text


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
    if metrics is None or len(metrics) == 0:
        logger.warning("No metrics to log")
        return
    
    logger.info(
        ">>>  BLEU_1: {:.4f}, BLEU_2: {:.4f}, BLEU_3: {:.4f}, BLEU_4: {:.4f}".format(
            metrics.get("Bleu_1", 0.0),
            metrics.get("Bleu_2", 0.0),
            metrics.get("Bleu_3", 0.0),
            metrics.get("Bleu_4", 0.0)
        )
    )
    logger.info(
        ">>>  METEOR: {:.4f}, ROUGE_L: {:.4f}, CIDEr: {:.4f}".format(
            metrics.get("METEOR", 0.0),
            metrics.get("ROUGE_L", 0.0),
            metrics.get("CIDEr", 0.0)
        )
    )
