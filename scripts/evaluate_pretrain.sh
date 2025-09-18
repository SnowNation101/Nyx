#!/usr/bin/env bash

if [ -z "$MODEL_NAME" ]; then
    MODEL_NAME="/fs/archive/share/u2024001042/Nyx-3B-Pretrained"
fi
if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="./outputs/eval_pt"
fi
if [ -z "$BATCH_SIZE" ]; then
    BATCH_SIZE=16
fi
if [ -z "$MODEL_BACKBONE" ]; then
    MODEL_BACKBONE="qwen2_5_vl"
fi

CUDA_VISIBLE_DEVICES=7 python3 evaluate.py \
    --model_name "${MODEL_NAME}" \
    --encode_output_path "${OUTPUT_DIR}" \
    --pooling last --normalize True \
    --dataloader_num_workers 4 \
    --dataset_name "/fs/archive/share/mm_datasets/MMEB-eval" \
    --subset_name Wiki-SS-NQ Visual7W-Pointing RefCOCO RefCOCO-Matching ImageNet-1K N24News HatefulMemes SUN397 VOC2007 InfographicsVQA ChartQA A-OKVQA DocVQA OK-VQA Visual7W VisDial CIRR NIGHTS WebQA VisualNews_i2t VisualNews_t2i MSCOCO_i2t MSCOCO_t2i MSCOCO Place365 ImageNet-A ImageNet-R ObjectNet Country211 ScienceQA VizWiz GQA TextVQA OVEN FashionIQ EDIS \
    --dataset_split test \
    --per_device_eval_batch_size ${BATCH_SIZE} \
    --image_dir "/fs/archive/share/mm_datasets/mmE5/images/eval_images" \
    --model_backbone "${MODEL_BACKBONE}"
