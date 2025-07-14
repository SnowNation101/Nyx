from datasets import load_dataset
import json
import os

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


generated_dir = "generated/qwenvl_da"
os.makedirs(generated_dir, exist_ok=True)

dataset_dir = "/fs/archive/share/mm_datasets/ScienceQA"
test_data = load_dataset(
    dataset_dir,
    split="test",
)


model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)



results = []
for item in tqdm(test_data, desc="Generating answers for test"):
    question = item["question"]
    choices = item["choices"]

    prompt = f"Question: {question}\nChoices: {', '.join(choices)}"

    if item['lecture'] != "":
        prompt = f"{question}\nLecture: {item['lecture']}" + prompt

    user_content = (
        [{"type": "image", "image": item['image']}] + [{"type": "text", "text": prompt}]
        if item.get('image')
        else [{"type": "text", "text": prompt}]
    )

    messages = [
        {
            "role": "system",
            "content": "Answer the mulitple-choice question. Only give me the answer and do not output any other words."
        },
        {
            "role": "user",
            "content": user_content
        }
    ]

    prompt = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    image_inputs, _ = process_vision_info(messages)

    inputs = processor(
        text=prompt,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    item['generated_answer'] = output_text

    results.append({
        "question": question,
        "choices": choices,
        "answer": choices[item['answer']],
        "generated_answer": output_text
    })

with open(f"{generated_dir}/test_generated_answers.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/test_generated_answers.json")

# =============evaluation============

with open(f"{generated_dir}/test_generated_answers.json", "r") as f:
    dataset = json.load(f)

total_acc = 0
count = 0
for item in dataset:
    pred = item["generated_answer"].strip()
    true = item["answer"]
    
    # Normalize answers
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
    
    pred_norm = normalize_answer(pred)
    true_norm = normalize_answer(true)

    if pred_norm == true_norm:
        total_acc += 1
    count += 1

avg_acc = total_acc / count
print(f"\nEvaluation:")
print(f"  Accuracy: {avg_acc:.4f}")