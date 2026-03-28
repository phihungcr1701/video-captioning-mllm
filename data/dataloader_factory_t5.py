# Adapted from data/dataloader_factory.py
# Changes: dataloader_msrvtt_train / dataloader_msrvtt_test use
#          MSRVTT_Caption_T5_DataLoader and accept a (bert_tokenizer, t5_tokenizer) tuple.

import torch
from torch.utils.data import DataLoader, SequentialSampler
from dataloaders.dataloader_youcook_caption import Youcook_Caption_DataLoader
from dataloaders.dataloader_msrvtt_caption_t5 import MSRVTT_Caption_T5_DataLoader


def dataloader_youcook_train(args, tokenizer):
    # YoucookII dataloader is unchanged — not modifying youcook for this experiment
    youcook_dataset = Youcook_Caption_DataLoader(
        csv=args.train_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(youcook_dataset)
    dataloader = DataLoader(
        youcook_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(youcook_dataset), train_sampler


def dataloader_youcook_test(args, tokenizer, logger):
    youcook_testset = Youcook_Caption_DataLoader(
        csv=args.val_csv,
        data_path=args.data_path,
        features_path=args.features_path,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        tokenizer=tokenizer,
        max_frames=args.max_frames,
    )

    test_sampler = SequentialSampler(youcook_testset)
    dataloader_youcook = DataLoader(
        youcook_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
    )

    if args.local_rank == 0:
        logger.info('YoucookII validation pairs: {}'.format(len(youcook_testset)))
    return dataloader_youcook, len(youcook_testset)


def dataloader_msrvtt_train(args, tokenizer):
    """
    Args:
        tokenizer: tuple (bert_tokenizer, t5_tokenizer)
    """
    bert_tokenizer, t5_tokenizer = tokenizer
    msrvtt_dataset = MSRVTT_Caption_T5_DataLoader(
        csv_path=args.train_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        bert_tokenizer=bert_tokenizer,
        t5_tokenizer=t5_tokenizer,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        split_type="train",
    )

    train_sampler = torch.utils.data.distributed.DistributedSampler(msrvtt_dataset)
    dataloader = DataLoader(
        msrvtt_dataset,
        batch_size=args.batch_size // args.n_gpu,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        drop_last=True,
    )
    return dataloader, len(msrvtt_dataset), train_sampler


def dataloader_msrvtt_test(args, tokenizer, logger=None, split_type="test"):
    """
    Args:
        tokenizer: tuple (bert_tokenizer, t5_tokenizer)
    """
    bert_tokenizer, t5_tokenizer = tokenizer
    msrvtt_testset = MSRVTT_Caption_T5_DataLoader(
        csv_path=args.val_csv,
        json_path=args.data_path,
        features_path=args.features_path,
        bert_tokenizer=bert_tokenizer,
        t5_tokenizer=t5_tokenizer,
        max_words=args.max_words,
        feature_framerate=args.feature_framerate,
        max_frames=args.max_frames,
        split_type=split_type,
    )

    test_sampler = SequentialSampler(msrvtt_testset)
    dataloader_msrvtt = DataLoader(
        msrvtt_testset,
        sampler=test_sampler,
        batch_size=args.batch_size_val,
        num_workers=args.num_thread_reader,
        pin_memory=False,
        drop_last=False,
    )
    return dataloader_msrvtt, len(msrvtt_testset)


DATALOADER_DICT = {
    "youcook": {
        "train": dataloader_youcook_train,
        "val": dataloader_youcook_test
    },
    "msrvtt": {
        "train": dataloader_msrvtt_train,
        "val": dataloader_msrvtt_test
    }
}
