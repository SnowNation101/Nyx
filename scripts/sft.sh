#!/usr/bin/env bash

MODEL_NAME_OR_PATH="/fs/archive/share/Qwen2.5-VL-3B-Instruct"
# MODEL_NAME_OR_PATH="/fs/archive/share/Nyx-3B-Pretrained"


if [[ $# -ge 1 && ! "$1" == "--"* ]]; then
    MODEL_NAME_OR_PATH=$1
    shift
fi

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="/fs/archive/share/nyx-ckpt/ft_$(date +%F-%H%M.%S)"
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


deepspeed --include localhost:0,1,2,3,4,5,6,7 --master_port 12345 train.py \
  --deepspeed "ds_config.json" \
  --image_dir "/fs/archive/share/mm_datasets/mmE5" \
  --dataset_name "/fs/archive/share/mm_datasets/Nyx-mmE5-MMEB" \
  --subset_name TAT-DQA ArxivQA InfoSeek_it2t InfoSeek_it2it ImageNet_1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO \
  --synthetic_dataset_name "/fs/archive/share/mm_datasets/Nyx-mmE5-Synthetic" \
  --synthetic_subset_name Retrieval VQA \
  --t2t_dataset_name "/fs/archive/share/mm_datasets/Nyx-T2T-Data" \
  --t2t_subset_name 2wikimultihopqa hotpotqa musique \
  --mm_dataset_path  "/fs/archive/share/mm_datasets/NyxQA/filtered_train_hardneg.json"\
  --feedback_dataset_path "outputs/feedback/filtered_feedback.json" \
  --model_name "${MODEL_NAME_OR_PATH}" --bf16 --pooling last \
  --num_sample_per_subset 30000 \
  --dataloader_num_workers 2 \
  --gradient_checkpointing True \
  --num_train_epochs 1 \
  --lora --lora_r 8 \
  --max_len 999999999 \
  --output_dir "${OUTPUT_DIR}" \
  --lr_scheduler_type linear --learning_rate 1e-5 --max_grad_norm 5.0 \
  --warmup_ratio 0.05  --normalize True \
  --temperature 0.02 \
  --model_backbone "${MODEL_BACKBONE}" \
  --processor_name "${PROCESSOR_NAME}" \
  --logging_steps 5 \
  --save_steps 25 --save_total_limit 999999999 \
  --per_device_train_batch_size 20 --gradient_accumulation_steps 4 \
  --negative_ratio 1 \
  --report_to swanlab \
  --resume_from_checkpoint "/fs/archive/share/nyx-ckpt/ft_2025-07-29-1832.47"
  # --resume_from_checkpoint "checkpoint/ft_2025-07-24-1603.25"
  # --resume_from_checkpoint "checkpoint/ft_2025-07-24-1120.10"
  # --resume_from_checkpoint "${OUTPUT_DIR}"
  # --resume_from_checkpoint "checkpoint/ft_2025-07-22-1629.58"
  # --resume_from_checkpoint "checkpoint/ft_2025-07-23-2026.37"

