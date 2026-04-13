torchrun --nproc_per_node=1 --standalone \
main_task_caption.py \
--do_train --stage_two --task_type caption --datatype msrvtt \
--num_thread_reader=4 \
--epochs=15 --batch_size=256 \
--n_display=50 \
--train_csv data/msrvtt/MSRVTT_train.9k.csv \
--val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv \
--data_path data/msrvtt/MSRVTT_data.json \
--features_path /workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle \
--output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased \
--do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 \
--batch_size_val 256 --visual_num_hidden_layers 6 \
--freeze_vit \
--init_model /workspace/project/datasets/trnphmngcminh/checkpoint-qformer-t5-model/versions/2/pytorch_model.bin.best.0 \
--gradient_accumulation_steps=2 \
--video_dim 768 \
--qformer_vision_width 1408 \
--qformer_checkpoint Salesforce/blip2-opt-6.7b-coco \
--lora --lr_qformer 2e-5 --lr_lora 1e-5 --lora_r=32 --lora_alpha=64

# torchrun --nproc_per_node=1 --standalone main_task_caption.py --do_train --stage_two --task_type caption --datatype msrvtt --num_thread_reader=4 --epochs=1 --batch_size=128 --n_display=50 --train_csv data/msrvtt/MSRVTT_train.9k.csv --val_csv data/msrvtt/MSRVTT_JSFUSION_test.csv --data_path data/msrvtt/MSRVTT_data.json --features_path /workspace/project/datasets/phihungcrr1701/msrvtt-clip-vitl14-features/versions/1/msrvtt_clip_vitl14_features.pickle --output_dir ckpts/ckpt_msrvtt_caption --bert_model bert-base-uncased --do_lower_case --lr 3e-5 --max_words 48 --max_frames 48 --batch_size_val 128 --visual_num_hidden_layers 6 --freeze_vit --init_model /workspace/project/video-captioning/ckpts/ckpt_msrvtt_caption/pytorch_model.bin.best_v1.0 --gradient_accumulation_steps=2 --video_dim 768 --qformer_vision_width 1408 --qformer_checkpoint Salesforce/blip2-opt-6.7b-coco --lora --lr_qformer 2e-5 --lr_lora 1e-5 --lora_r=32 --lora_alpha=64 --scst --max_txt_len 20 
