torchrun --nproc_per_node=2 --standalone \
main_task_caption_t5.py \
--do_train --num_thread_reader=4 \
--epochs=9 --batch_size=64 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path /kaggle/input/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/msrvtt_clip_vitl14_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased \
--t5_model t5-base \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 32 --visual_num_hidden_layers 6 \
--datatype msrvtt --stage_two \
--init_model /kaggle/input/datasets/phihungcrr1701/best-model-t5-stage-1/pytorch_model.bin.8 \
--gradient_accumulation_steps=2 \
--video_dim 768
