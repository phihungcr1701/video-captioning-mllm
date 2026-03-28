# Adapted from main_task_caption_t5.py for Q-Former training.
# Changes vs original:
#   - Solution B: filters out cross.* weights from pretrained checkpoint so
#     Q-Former (module_cross_qformer) initialises from scratch.
#   - Freeze/unfreeze: bert + visual are frozen for the first N epochs
#     (default 5), then unfrozen and the optimizer is rebuilt to include them.
#   - Default epochs set to 20.

from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import os
import argparse

from metrics import CaptionEvaluator, PYCOCOEVALCAP_AVAILABLE
from modules.tokenization import BertTokenizer
from transformers import T5Tokenizer

from utils.setup_utils import set_seed_logger, init_device
from utils.model_utils_t5 import save_model, load_model
from utils.optimizer_utils import prep_optimizer
from data.dataloader_factory_t5 import DATALOADER_DICT
from trainers.trainer import train_epoch
from inference.caption_generator_t5 import eval_epoch

# ── Q-Former-specific init_model (Solution B) ──────────────────────────────
from modules.file_utils import PYTORCH_PRETRAINED_BERT_CACHE
from modules.modeling_t5 import UniVL_T5


def init_model_qformer(args, device, n_gpu, local_rank, logger):
    """Like init_model but filters cross.* weights from the pretrained
    checkpoint so the Q-Former cross module is randomly initialised."""
    model_state_dict = None
    if args.init_model:
        raw_state = torch.load(args.init_model, map_location='cpu')
        cross_keys = [k for k in raw_state if k.startswith('cross.')]
        if cross_keys:
            if local_rank == 0:
                logger.info("Q-Former: removing %d cross.* keys from checkpoint "
                            "(will be randomly initialised).", len(cross_keys))
            model_state_dict = {k: v for k, v in raw_state.items()
                                if not k.startswith('cross.')}
        else:
            model_state_dict = raw_state

    cache_dir = (args.cache_dir if args.cache_dir
                 else os.path.join(str(PYTORCH_PRETRAINED_BERT_CACHE), 'distributed'))
    model = UniVL_T5.from_pretrained(
        args.bert_model, args.visual_model, args.cross_model, args.decoder_model,
        cache_dir=cache_dir, state_dict=model_state_dict, task_config=args)
    model.to(device)
    return model
# ────────────────────────────────────────────────────────────────────────────


# ── Freeze / unfreeze helpers ──────────────────────────────────────────────
def freeze_bert_visual(model, logger, local_rank=0):
    """Set requires_grad=False for bert.* and visual.* parameters."""
    base = model.module if hasattr(model, 'module') else model
    count = 0
    for name, p in base.named_parameters():
        if name.startswith('bert.') or name.startswith('visual.') or name.startswith('normalize_video.'):
            p.requires_grad = False
            count += 1
    if local_rank == 0:
        logger.info("Froze %d bert/visual/normalize_video parameters.", count)


def unfreeze_all(model, logger, local_rank=0):
    """Set requires_grad=True for ALL parameters."""
    base = model.module if hasattr(model, 'module') else model
    count = 0
    for _, p in base.named_parameters():
        if not p.requires_grad:
            p.requires_grad = True
            count += 1
    if local_rank == 0:
        logger.info("Unfroze %d parameters.", count)


def _update_loss_weights(args, epoch, logger, local_rank=0):
    """Update effective ITC/ITM weights based on warmup + ramp schedule.
    Sets args.current_itc_weight and args.current_itm_weight so the model reads them."""
    warmup = getattr(args, 'it_aux_warmup_epochs', 1)
    ramp = getattr(args, 'it_aux_ramp_epochs', 2)

    if epoch < warmup:
        factor = 0.0
    elif epoch < warmup + ramp:
        factor = float(epoch - warmup + 1) / float(ramp)
    else:
        factor = 1.0

    itc_target = getattr(args, 'itc_weight', 0.05)
    itm_target = getattr(args, 'itm_weight', 0.05)

    args.current_itc_weight = itc_target * factor
    args.current_itm_weight = itm_target * factor

    if local_rank == 0:
        use_itc = getattr(args, 'use_itc_loss', False)
        use_itm = getattr(args, 'use_itm_loss', False)
        logger.info("Epoch %d: ITC weight=%.4f%s, ITM weight=%.4f%s (factor=%.2f, warmup=%d, ramp=%d)",
                    epoch + 1,
                    args.current_itc_weight, " [ON]" if use_itc else " [OFF]",
                    args.current_itm_weight, " [ON]" if use_itm else " [OFF]",
                    factor, warmup, ramp)
# ────────────────────────────────────────────────────────────────────────────


# Initialize distributed training only if environment is properly configured
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl")


def get_args(description='UniVL Q-Former on Caption Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run pretraining.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev set.")

    parser.add_argument('--train_csv', type=str, default='data/youcookii_singlef_train.csv', help='')
    parser.add_argument('--val_csv', type=str, default='data/youcookii_singlef_val.csv', help='')
    parser.add_argument('--data_path', type=str, default='data/youcookii_caption_transcript.pickle',
                        help='caption and transcription pickle file path')
    parser.add_argument('--features_path', type=str, default='data/youcookii_videos_feature.pickle',
                        help='feature path for 2D features')

    parser.add_argument('--num_thread_reader', type=int, default=1, help='')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--epochs', type=int, default=20, help='upper epoch limit')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    parser.add_argument('--batch_size_val', type=int, default=3500, help='batch size eval')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate exp epoch decay')
    parser.add_argument('--n_display', type=int, default=100, help='Information display frequence')
    parser.add_argument('--video_dim', type=int, default=1024, help='video feature dimension')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--max_words', type=int, default=20, help='')
    parser.add_argument('--max_frames', type=int, default=100, help='')
    parser.add_argument('--feature_framerate', type=int, default=1, help='')
    parser.add_argument('--min_time', type=float, default=5.0, help='Gather small clips')
    parser.add_argument('--margin', type=float, default=0.1, help='margin for loss')
    parser.add_argument('--hard_negative_rate', type=float, default=0.5, help='rate of intra negative sample')
    parser.add_argument('--negative_weighting', type=int, default=1, help='Weight the loss for intra negative')
    parser.add_argument('--n_pair', type=int, default=1, help='Num of pair to output from data loader')

    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True,
                        help="Bert pre-trained model for the text encoder")
    parser.add_argument("--t5_model", default="t5-base", type=str, required=False,
                        help="T5 model name for the decoder (default: t5-base)")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False,
                        help="Decoder config (used for interface compatibility; weights replaced by T5)")
    parser.add_argument("--init_model", default=None, type=str, required=False,
                        help="Initial model checkpoint (e.g. the CIDEr=58 checkpoint).")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased BERT model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level.")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `caption` to finetune.")
    parser.add_argument("--datatype", default="msrvtt", type=str, help="Dataset: msrvtt or youcook.")

    parser.add_argument("--world_size", default=0, type=int, help="distributed training")
    parser.add_argument("--local_rank", default=None, type=int, help="distributed training")
    parser.add_argument('--coef_lr', type=float, default=0.1, help='coefficient for bert branch.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL.")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, high priority.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder (compat).")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")

    # ── Q-Former-specific arguments ─────────────────────────────────────────
    parser.add_argument('--freeze_epochs', type=int, default=5,
                        help="Number of initial epochs to freeze bert+visual (default: 5). "
                             "Set to 0 to train everything from the start.")

    # ── ITC / ITM loss arguments ────────────────────────────────────────────
    parser.add_argument('--use_itc_loss', action='store_true',
                        help="Enable ITC (Image-Text Contrastive) loss.")
    parser.add_argument('--itc_weight', type=float, default=0.05,
                        help="Target weight for ITC loss (default: 0.05).")
    parser.add_argument('--itc_proj_dim', type=int, default=256,
                        help="ITC projection dimension (default: 256).")
    parser.add_argument('--itc_init_temp', type=float, default=0.07,
                        help="Initial temperature for ITC (default: 0.07).")
    parser.add_argument('--itc_gather_distributed', action='store_true',
                        help="All-gather ITC embeddings across DDP ranks.")
    parser.add_argument('--use_itm_loss', action='store_true',
                        help="Enable ITM (Image-Text Matching) loss.")
    parser.add_argument('--itm_weight', type=float, default=0.05,
                        help="Target weight for ITM loss (default: 0.05).")
    parser.add_argument('--itm_num_hard_negatives', type=int, default=1,
                        help="Number of hard negatives per sample (default: 1).")
    parser.add_argument('--itm_use_hard_negative', action='store_true', default=True,
                        help="Use hard negatives from ITC similarity (default: True).")
    parser.add_argument('--itm_use_richer_fusion', action='store_true',
                        help="Use richer 4-way ITM feature fusion (default: False).")
    parser.add_argument('--it_aux_warmup_epochs', type=int, default=1,
                        help="Epochs to keep ITC/ITM weight at 0 (default: 1).")
    parser.add_argument('--it_aux_ramp_epochs', type=int, default=2,
                        help="Epochs to linearly ramp ITC/ITM weight to target (default: 2).")
    # ────────────────────────────────────────────────────────────────────────

    args = parser.parse_args()

    if args.local_rank is None:
        args.local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))
    if not args.do_train and not args.do_eval:
        raise ValueError("At least one of `do_train` or `do_eval` must be True.")

    args.batch_size = int(args.batch_size / args.gradient_accumulation_steps)

    return args


def main():
    args = get_args()
    args, logger = set_seed_logger(args)
    device, n_gpu = init_device(args, args.local_rank, logger)

    # ── Tokenizers ──────────────────────────────────────────────────────────
    bert_tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    t5_model_name = getattr(args, 't5_model', 't5-base')
    t5_tokenizer = T5Tokenizer.from_pretrained(t5_model_name)
    tokenizer = (bert_tokenizer, t5_tokenizer)
    # ────────────────────────────────────────────────────────────────────────

    # ── Model init (Solution B: cross.* weights are filtered out) ──────────
    model = init_model_qformer(args, device, n_gpu, args.local_rank, logger)
    # ────────────────────────────────────────────────────────────────────────

    assert args.task_type == "caption"

    if PYCOCOEVALCAP_AVAILABLE:
        nlgEvalObj = CaptionEvaluator()
        if args.local_rank == 0:
            logger.info("Using pycocoevalcap for evaluation metrics")
    else:
        nlgEvalObj = None
        if args.local_rank == 0:
            logger.warning("pycocoevalcap not available. Evaluation metrics will be skipped.")

    assert args.datatype in DATALOADER_DICT
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, logger)
    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        freeze_epochs = args.freeze_epochs

        # ── Phase 1: Freeze bert+visual, build optimizer for cross+decoder only ──
        if freeze_epochs > 0:
            freeze_bert_visual(model, logger, args.local_rank)

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps,
                                                     device, n_gpu, args.local_rank, coef_lr=coef_lr)

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)
            if freeze_epochs > 0:
                logger.info("  Freeze schedule: bert+visual frozen for first %d epochs, "
                            "then fine-tune all for remaining %d epochs.",
                            freeze_epochs, args.epochs - freeze_epochs)

        best_score = 0.00001
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            # ── Update ITC / ITM loss weights (warmup + ramp) ────────────────
            _update_loss_weights(args, epoch, logger, args.local_rank)

            # ── Phase 2: Unfreeze at the designated epoch ────────────────────
            if freeze_epochs > 0 and epoch == freeze_epochs:
                if args.local_rank == 0:
                    logger.info("=" * 60)
                    logger.info("Epoch %d: UNFREEZING bert + visual — rebuilding optimizer.",
                                epoch + 1)
                    logger.info("=" * 60)

                # Unwrap DDP to get base model
                base_model = model.module if hasattr(model, 'module') else model
                unfreeze_all(base_model, logger, args.local_rank)

                # Rebuild optimizer with ALL params now trainable.
                # Use coef_lr for bert branch so it trains at a lower rate.
                remaining_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                   / args.gradient_accumulation_steps) * (args.epochs - epoch)
                optimizer, scheduler, model = prep_optimizer(
                    args, base_model, remaining_steps,
                    device, n_gpu, args.local_rank, coef_lr=args.coef_lr)
            # ─────────────────────────────────────────────────────────────────

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, logger, local_rank=args.local_rank)

            # Rank 0: log loss and save checkpoint
            output_model_file = None
            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(epoch, args, model, logger, type_name="")

            # ALL ranks run eval simultaneously so no rank sits idle → no NCCL timeout.
            _nlg = nlgEvalObj if args.local_rank == 0 else None
            Bleu_4, _ = eval_epoch(args, model, test_dataloader, t5_tokenizer,
                                   device, n_gpu, logger, nlgEvalObj=_nlg)
            if args.local_rank == 0:
                if best_score <= Bleu_4:
                    best_score = Bleu_4
                    best_output_model_file = output_model_file
                logger.info("The best model is: {}, the Bleu_4 is: {:.4f}".format(
                    best_output_model_file, best_score))

        # All ranks reload best model and run final eval
        if args.local_rank == 0:
            model = load_model(-1, args, n_gpu, device, logger, model_file=best_output_model_file)
        _nlg = nlgEvalObj if args.local_rank == 0 else None
        Bleu_4, _ = eval_epoch(args, model, test_dataloader, t5_tokenizer,
                               device, n_gpu, logger, nlgEvalObj=_nlg)
    elif args.do_eval:
        _nlg = nlgEvalObj if args.local_rank == 0 else None
        Bleu_4, _ = eval_epoch(args, model, test_dataloader, t5_tokenizer,
                               device, n_gpu, logger, nlgEvalObj=_nlg)


if __name__ == "__main__":
    main()
