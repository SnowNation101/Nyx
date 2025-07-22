#!/usr/bin/env bash

MODEL_NAME_OR_PATH="/fs/archive/share/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME_OR_PATH="/fs/archive/share/Nyx-3B-Pretrained"


if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="./checkpoint/ft_$(date +%F-%H%M.%S)"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="./data/"
fi

if [ -z "$MODEL_BACKBONE" ]; then
  MODEL_BACKBONE="qwen2_5_vl"
fi

if [ -z "$PROCESSOR_NAME" ]; then
  PROCESSOR_NAME="/fs/archive/share/Qwen2.5-VL-3B-Instruct"

fi

export SWANLAB_PROJ_NAME="Nyx"

deepspeed --include localhost:0,1,4,5,6,7 --master_port 12345 train.py \
  --deepspeed "ds_config.json" \
  --image_dir "/fs/archive/share/mm_datasets/mmE5" \
  --dataset_name "/fs/archive/share/mm_datasets/Nyx-mmE5-MMEB" \
  --subset_name TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
  --synthetic_dataset_name "/fs/archive/share/mm_datasets/Nyx-mmE5-Synthetic" \
  --synthetic_subset_name Retrieval VQA\
  --mm_dataset_path  "/fs/archive/share/mm_datasets/NyxQA/train_hardneg_new.json"\
  --feedback_dataset_path "/home/u2024001042/workspace/nyx/outputs/feedback/final_feedback.json" \
  --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
  --num_sample_per_subset 30000 \
  --dataloader_num_workers 4 \
  --gradient_checkpointing True \
  --num_train_epochs 1 \
  --lora --lora_r 16 \
  --max_len 999999 --output_dir "${OUTPUT_DIR}" --logging_steps 4 \
  --lr_scheduler_type linear --learning_rate 1e-5 --max_grad_norm 5.0 \
  --warmup_ratio 0.05 --save_steps 25 --save_total_limit 3 --normalize True \
  --temperature 0.02 \
  --model_backbone "${MODEL_BACKBONE}" \
  --processor_name "${PROCESSOR_NAME}" \
  --resume_from_checkpoint "${OUTPUT_DIR}" \
  --per_device_train_batch_size 28 --gradient_accumulation_steps 8 \
  --negative_ratio 1 \
  --report_to swanlab
  # --resume_from_checkpoint "checkpoint/ft_2025-07-21-1856.20"
