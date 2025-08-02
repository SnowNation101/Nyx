import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from collections import Counter
import re
import string

from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os
from typing import List, Optional

# =========== configuration ===========
generated_dir = "generated/qwenvl_da"

os.makedirs(generated_dir, exist_ok=True)

# =========== generation ===========

model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

with open("/fs/archive/share/mm_datasets/MMQA/MMQA_dev.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]


for item in tqdm(dataset, desc="Generating answers for dev"):
    question = item['question']

    messages = [
        {
            "role": "system",
            "content": "Answer the question. Only give me the answer and do not output any other words."
        },
        {
            "role": "user",
            "content": question
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = processor(
        text=prompt,
        return_tensors="pt",
        padding=True,
    ).to(model.device)

    generated_ids = model.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
        )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    item['generated_answer'] = output_text

with open(f"{generated_dir}/dev_generated_answers.json", "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/dev_generated_answers.json")

del model, processor
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ============evaluation============

def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    def white_space_fix(text):
        return ' '.join(text.split())
    def lower(text):
        return str(text).lower()
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def compute_exact(a_pred, a_true):
    return int(normalize_answer(a_pred) == normalize_answer(a_true))

def compute_f1(a_pred, a_true):
    pred_tokens = normalize_answer(a_pred).split()
    true_tokens = normalize_answer(a_true).split()
    common = Counter(pred_tokens) & Counter(true_tokens)
    num_same = sum(common.values())
    if len(pred_tokens) == 0 or len(true_tokens) == 0:
        return int(pred_tokens == true_tokens)
    if num_same == 0:
        return 0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(true_tokens)
    f1 = 2 * precision * recall / (precision + recall)
    return f1

with open(f"{generated_dir}/dev_generated_answers.json", "r") as f:
    dataset = json.load(f)

total_em = 0
total_f1 = 0
count = 0
for item in tqdm(dataset, desc=f"evaluating dev answers"):
    pred = item["generated_answer"].strip()
    gold_list = [ans["answer"] for ans in item["answers"]]
    em_scores = [compute_exact(pred, g) for g in gold_list]
    f1_scores = [compute_f1(pred, g) for g in gold_list]
    total_em += max(em_scores)
    total_f1 += max(f1_scores)
    count += 1
avg_em = total_em / count
avg_f1 = total_f1 / count
print(f"\nEvaluation:")
print(f"  Exact Match: {avg_em:.4f}")
print(f"  F1 Score:    {avg_f1:.4f}")