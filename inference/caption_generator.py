import torch
from inference.eval_utils import decode_tokens_to_text, save_predictions, save_complete_results, log_metrics
# from tasks.pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


def eval_epoch(
    args,
    model,
    test_dataloader,
    tokenizer,
    device,
    n_gpu,
    logger,
    nlgEvalObj=None,
    test_set=None,
    pred_filename="hyp.txt",
    ref_filename="ref.txt",
    complete_results_filename="hyp_complete_results.txt",
):
    bleu_4 = 0.0

    if hasattr(model, 'module'):
        model = model.module.to(device)

    if model._stage_one:
        return 0.

    all_result_lists = []
    all_caption_lists = []
    total_loss = 0.0
    model.eval()
    for batch in test_dataloader:
        # Last element is sample indices (not needed for eval), rest are tensors
        batch = tuple(t.to(device, non_blocking=True) for t in batch[:-1])

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index, \
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids, \
        pairs_t5_output_caption_ids = batch

        with torch.no_grad():
            # Calculate validation loss
            # forward() returns (loss, visual_output) in eval mode,
            # so we can reuse visual_output for generation without re-encoding.
            loss, visual_output = model(input_ids, segment_ids, input_mask, video, video_mask,
                        pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                        masked_video=masked_video, video_labels_index=video_labels_index,
                        input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                        output_caption_ids=pairs_output_caption_ids,
                        t5_output_caption_ids=pairs_t5_output_caption_ids)
            if loss is not None:
                if n_gpu > 1:
                    loss = loss.mean()
                total_loss += float(loss)

            # Keep eval generation conditioning consistent with training (text + visual).
            input_ids_shaped = input_ids.view(-1, input_ids.shape[-1])
            segment_ids_shaped = segment_ids.view(-1, segment_ids.shape[-1])
            input_mask_shaped = input_mask.view(-1, input_mask.shape[-1])
            text_layers, _ = model.bert(
                input_ids_shaped,
                segment_ids_shaped,
                input_mask_shaped,
                output_all_encoded_layers=True,
            )
            sequence_output = text_layers[-1]

            video_mask = video_mask.view(-1, video_mask.shape[-1])

            beam_size = max(1, getattr(model, "beam_size", 1))
            max_length = getattr(model, "max_txt_len", args.max_words)
            generated_ids = model.generate_caption_ids(
                visual_output,
                video_mask,
                num_beams=beam_size,
                max_length=max_length,
                sequence_output=sequence_output,
                attention_mask=input_mask_shaped,
            )
            all_result_lists.extend(model.t5_tokenizer.batch_decode(generated_ids, skip_special_tokens=True))

            pairs_output_caption_ids = pairs_output_caption_ids.view(-1, pairs_output_caption_ids.shape[-1])
            caption_list = pairs_output_caption_ids.cpu().detach().numpy()

            for re_idx, re_list in enumerate(caption_list):
                decode_text = decode_tokens_to_text(re_list, tokenizer)
                all_caption_lists.append(decode_text)

    # Calculate and log average validation loss
    avg_val_loss = total_loss / len(test_dataloader)
    logger.info("  Average Validation Loss: {:.4f}".format(avg_val_loss)) 
    
    complete_results_path = save_complete_results(
        all_result_lists,
        test_set,
        args.output_dir,
        filename=complete_results_filename,
    )
    if complete_results_path:
        logger.info("File of complete results is saved in {}".format(complete_results_path))

    save_predictions(
        all_result_lists,
        all_caption_lists,
        args.output_dir,
        pred_filename=pred_filename,
        ref_filename=ref_filename,
    )

    if args.datatype == "msrvtt":
        all_caption_lists = []
        sentences_dict = test_dataloader.dataset.sentences_dict
        video_sentences_dict = test_dataloader.dataset.video_sentences_dict
        for idx in range(len(sentences_dict)):
            video_id, _ = sentences_dict[idx]
            sentences = video_sentences_dict[video_id]
            all_caption_lists.append(sentences)
        all_caption_lists = [list(itms) for itms in zip(*all_caption_lists)]
    else:
        all_caption_lists = [all_caption_lists]

    if nlgEvalObj is not None:
        # Apply PTBTokenizer for consistency with SCST training CIDEr computation
        ptb_tokenizer = PTBTokenizer()

        # Tokenize predictions
        cands_dict = {idx: [{'caption': c}] for idx, c in enumerate(all_result_lists)}
        cands_tokenized = ptb_tokenizer.tokenize(cands_dict)
        hyp_list_tokenized = [cands_tokenized[idx][0] for idx in range(len(all_result_lists))]

        # Tokenize references
        if isinstance(all_caption_lists[0], list):
            # Multiple reference sets (e.g., MSRVTT)
            refs_dict = {}
            for idx in range(len(all_caption_lists[0])):
                refs_dict[idx] = [{'caption': all_caption_lists[j][idx]}
                                  for j in range(len(all_caption_lists))]
            refs_tokenized = ptb_tokenizer.tokenize(refs_dict)
            ref_list_tokenized = []
            for j in range(len(all_caption_lists)):
                ref_list_tokenized.append(
                    [refs_tokenized[idx][j] for idx in range(len(all_caption_lists[0]))]
                )
        else:
            # Single reference set (e.g., YoucookII, already wrapped in list)
            refs_dict = {idx: [{'caption': c}] for idx, c in enumerate(all_caption_lists[0])}
            refs_tokenized = ptb_tokenizer.tokenize(refs_dict)
            ref_list_tokenized = [[refs_tokenized[idx][0] for idx in range(len(all_caption_lists[0]))]]

        metrics_nlg = nlgEvalObj.compute_metrics(ref_list=ref_list_tokenized, hyp_list=hyp_list_tokenized)
        log_metrics(logger, metrics_nlg)
        CIDEr = metrics_nlg.get("CIDEr", 0.0)
        bleu_4 = metrics_nlg.get("Bleu_4", 0.0)
    else:
        logger.warning("Evaluation metrics skipped (pycocoevalcap not available)")
        CIDEr = 0.0
    
    return CIDEr, avg_val_loss, bleu_4
