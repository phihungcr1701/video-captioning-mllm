from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from __future__ import print_function

import torch
import os
import argparse

from metrics import CaptionEvaluator, PYCOCOEVALCAP_AVAILABLE
from modules.tokenization import BertTokenizer

from utils.setup_utils import set_seed_logger, init_device
from utils.model_utils import init_model, save_model, load_model
from utils.optimizer_utils import prep_optimizer
from utils.cider_utils import CorpusCider
from data.dataloader_factory import DATALOADER_DICT
from trainers.trainer import train_epoch
from inference.caption_generator import eval_epoch

# Initialize distributed training only if environment is properly configured
if not torch.distributed.is_initialized():
    torch.distributed.init_process_group(backend="nccl")


def get_args(description='UniVL on Caption Task'):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--do_pretrain", action='store_true', help="Whether to run training.")
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
    parser.add_argument("--bert_model", default="bert-base-uncased", type=str, required=True, help="Bert pre-trained model")
    parser.add_argument("--visual_model", default="visual-base", type=str, required=False, help="Visual module")
    parser.add_argument("--cross_model", default="cross-base", type=str, required=False, help="Cross module")
    parser.add_argument("--decoder_model", default="decoder-base", type=str, required=False, help="Decoder module")
    parser.add_argument("--init_model", default=None, type=str, required=False, help="Initial model.")
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")
    parser.add_argument("--warmup_proportion", default=0.1, type=float,
                        help="Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--n_gpu', type=int, default=1, help="Changed in the execute process.")

    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")

    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")

    parser.add_argument("--task_type", default="caption", type=str, help="Point the task `caption` to finetune.")
    parser.add_argument("--datatype", default="youcook", type=str, help="Point the dataset `youcook` to finetune.")

    parser.add_argument("--world_size", default=0, type=int, help="distribted training")
    parser.add_argument("--local_rank", default=None, type=int, help="distribted training")
    parser.add_argument('--coef_lr', type=float, default=1.0, help='coefficient for bert branch.')
    parser.add_argument('--lr_qformer', type=float, default=5e-5, help='Learning rate for QFormer parameters.')
    parser.add_argument('--lr_lora', '--lr_t5_decoder', dest='lr_lora', type=float, default=1e-5,
                        help='Learning rate for T5 LoRA/decoder parameters.')
    parser.add_argument('--use_mil', action='store_true', help="Whether use MIL as Miech et. al. (2020).")
    parser.add_argument('--sampled_use_mil', action='store_true', help="Whether use MIL, has a high priority than use_mil.")

    parser.add_argument('--text_num_hidden_layers', type=int, default=12, help="Layer NO. of text.")
    parser.add_argument('--visual_num_hidden_layers', type=int, default=6, help="Layer NO. of visual.")
    parser.add_argument('--cross_num_hidden_layers', type=int, default=2, help="Layer NO. of cross.")
    parser.add_argument('--decoder_num_hidden_layers', type=int, default=3, help="Layer NO. of decoder.")

    parser.add_argument('--freeze_vit', action='store_true', help="Freeze vision encoder parameters.")
    parser.add_argument('--scst', action='store_true', help="Enable SCST training for caption loss.")
    parser.add_argument('--beam_size', type=int, default=5, help="Beam size for SCST decoding.")
    parser.add_argument('--t5_model', type=str, default='google/flan-t5-xl', help="T5 model name.")
    parser.add_argument('--max_txt_len', type=int, default=32, help="Maximum text length for T5 tokenizer.")
    parser.add_argument('--num_query_token', type=int, default=32, help="Number of Qformer query tokens.")
    parser.add_argument('--qformer_vision_width', type=int, default=768,
                        help="Encoder feature width expected by QFormer cross-attention.")
    parser.add_argument('--qformer_checkpoint', type=str, default='',
                        help="Optional local path or Hugging Face repo id for QFormer weights.")
    parser.add_argument('--qformer_checkpoint_file', type=str, default='',
                        help="Optional exact checkpoint filename inside the QFormer checkpoint repo/path.")
    parser.add_argument('--qformer_checkpoint_local_files_only', action='store_true',
                        help="Load QFormer checkpoint from local Hugging Face cache only.")
    parser.add_argument('--lora', action='store_true', help="Enable LoRA for T5.")
    parser.add_argument('--lora_r', type=int, default=16, help="LoRA rank.")
    parser.add_argument('--lora_alpha', type=int, default=32, help="LoRA alpha.")
    parser.add_argument('--lora_dropout', type=float, default=0.05, help="LoRA dropout.")

    parser.add_argument('--stage_two', action='store_true', help="Whether training with decoder.")
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

    tokenizer = BertTokenizer.from_pretrained(args.bert_model, do_lower_case=args.do_lower_case)
    model = init_model(args, device, n_gpu, args.local_rank)

    # Get T5 tokenizer from model for dataloader T5 tokenization (used by SCST)
    t5_tokenizer = getattr(model, 't5_tokenizer', None)

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
    test_dataloader, test_length = DATALOADER_DICT[args.datatype]["val"](args, tokenizer, logger, t5_tokenizer=t5_tokenizer)
    if args.local_rank == 0:
        logger.info("***** Running test *****")
        logger.info("  Num examples = %d", test_length)
        logger.info("  Batch size = %d", args.batch_size_val)
        logger.info("  Num steps = %d", len(test_dataloader))

    if args.do_train:
        train_dataloader, train_length, train_sampler = DATALOADER_DICT[args.datatype]["train"](args, tokenizer, t5_tokenizer=t5_tokenizer)
        num_train_optimization_steps = (int(len(train_dataloader) + args.gradient_accumulation_steps - 1)
                                        / args.gradient_accumulation_steps) * args.epochs

        coef_lr = args.coef_lr
        if args.init_model:
            coef_lr = 1.0
        optimizer, scheduler, model = prep_optimizer(args, model, num_train_optimization_steps, device, n_gpu, args.local_rank, coef_lr=coef_lr)

        # P1: Pre-compute corpus-level IDF for SCST CIDEr reward
        if args.scst and hasattr(train_dataloader.dataset, 'video_sentences_dict'):
            model_inner = model.module if hasattr(model, 'module') else model
            corpus_cider = CorpusCider()
            corpus_cider.init_corpus_df(train_dataloader.dataset.video_sentences_dict)
            model_inner._cider_scorer = corpus_cider

        if args.local_rank == 0:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_length)
            logger.info("  Batch size = %d", args.batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps * args.gradient_accumulation_steps)

        best_cider = float('-inf')
        best_bleu_4 = float('-inf')
        best_output_model_file = None
        global_step = 0
        for epoch in range(args.epochs):
            train_sampler.set_epoch(epoch)

            tr_loss, global_step = train_epoch(epoch, args, model, train_dataloader, device, n_gpu, optimizer,
                                               scheduler, global_step, logger, local_rank=args.local_rank)

            if args.local_rank == 0:
                logger.info("Epoch %d/%s Finished, Train Loss: %f", epoch + 1, args.epochs, tr_loss)
                output_model_file = save_model(0, args, model, logger, type_name="last")
                if epoch >= 0:
                    CIDEr, _, Bleu_4 = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, logger, nlgEvalObj=nlgEvalObj)

                    is_better = (CIDEr > best_cider) or (CIDEr == best_cider and Bleu_4 > best_bleu_4)
                    if is_better:
                        best_cider = CIDEr
                        best_bleu_4 = Bleu_4
                        best_output_model_file = save_model(0, args, model, logger, type_name="best")

                    logger.info(
                        "The best model is: %s, CIDEr: %.4f, BLEU_4: %.4f",
                        best_output_model_file,
                        best_cider,
                        best_bleu_4,
                    )
                else:
                    logger.warning("Skip the evaluation after {}-th epoch.".format(epoch+1))

        if args.local_rank == 0:
            model = load_model(-1, args, n_gpu, device, logger, model_file=best_output_model_file)
            CIDEr, _, _ = eval_epoch(
                args,
                model,
                test_dataloader,
                tokenizer,
                device,
                n_gpu,
                logger,
                nlgEvalObj=nlgEvalObj,
                pred_filename="hyp_best.txt",
                ref_filename="ref_best.txt",
                complete_results_filename="hyp_complete_results_best.txt",
            )
    elif args.do_eval:
        if args.local_rank == 0:
            CIDEr, _, _ = eval_epoch(args, model, test_dataloader, tokenizer, device, n_gpu, logger, nlgEvalObj=nlgEvalObj)


if __name__ == "__main__":
    main()
