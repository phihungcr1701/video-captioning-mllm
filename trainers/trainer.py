import torch
import time


def train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer, scheduler,
                global_step, logger, local_rank=0):
    torch.cuda.empty_cache()
    model.train()
    log_step = args.n_display
    start_time = time.time()
    total_loss = 0

    # Accumulators for per-loss logging
    loss_accum = {}
    loss_count = 0

    for step, batch in enumerate(train_dataloader):
        batch = tuple(t.to(device=device, non_blocking=True) for t in batch)

        input_ids, input_mask, segment_ids, video, video_mask, \
        pairs_masked_text, pairs_token_labels, masked_video, video_labels_index,\
        pairs_input_caption_ids, pairs_decoder_mask, pairs_output_caption_ids = batch

        output = model(input_ids, segment_ids, input_mask, video, video_mask,
                     pairs_masked_text=pairs_masked_text, pairs_token_labels=pairs_token_labels,
                     masked_video=masked_video, video_labels_index=video_labels_index,
                     input_caption_ids=pairs_input_caption_ids, decoder_mask=pairs_decoder_mask,
                     output_caption_ids=pairs_output_caption_ids)

        # Support both dict and scalar output
        if isinstance(output, dict):
            loss = output["loss"]
            # Accumulate detached metrics for logging
            for k, v in output.items():
                if k != "loss" and torch.is_tensor(v):
                    if k not in loss_accum:
                        loss_accum[k] = 0.0
                    loss_accum[k] += float(v)
        else:
            loss = output

        if n_gpu > 1:
            loss = loss.mean()
        if args.gradient_accumulation_steps > 1:
            loss = loss / args.gradient_accumulation_steps

        loss.backward()

        total_loss += float(loss)
        loss_count += 1

        if (step + 1) % args.gradient_accumulation_steps == 0:

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            if scheduler is not None:
                scheduler.step()

            optimizer.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % log_step == 0 and local_rank == 0:
                # Base log line
                lr_str = "-".join([str('%.6f' % itm) for itm in sorted(list(set(optimizer.get_lr())))])
                logger.info("Epoch: %d/%s, Step: %d/%d, Lr: %s, Loss: %f, Time/step: %f", epoch + 1,
                            args.epochs, step + 1,
                            len(train_dataloader), lr_str,
                            float(loss),
                            (time.time() - start_time) / (log_step * args.gradient_accumulation_steps))

                # Loss breakdown log (if available)
                if loss_accum and loss_count > 0:
                    parts = []
                    for k in ["decoder_loss", "itc_loss", "itm_loss", "itm_acc",
                              "itc_temp", "itc_top1", "itm_pos_mean", "itm_neg_mean"]:
                        if k in loss_accum:
                            avg_val = loss_accum[k] / loss_count
                            parts.append("{}={:.4f}".format(k, avg_val))
                    if parts:
                        logger.info("  [Loss breakdown] %s", ", ".join(parts))
                    # Reset accumulators
                    loss_accum = {}
                    loss_count = 0

                start_time = time.time()

    total_loss = total_loss / len(train_dataloader)
    return total_loss, global_step
