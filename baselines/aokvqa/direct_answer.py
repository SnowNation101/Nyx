import json
from tqdm import tqdm
from datasets import load_dataset
from generator import MMGenerator
from PIL import Image

import re
import string
from collections import Counter
import os

from load_aokvqa import get_aokvqa_data


model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
vlm = MMGenerator(model_path=model_path)

_, dev_datset, test_dataset = get_aokvqa_data()

os.makedirs("generated/direct", exist_ok=True)


for item in tqdm(test_dataset, desc=f"Generating answers for test set"):
    image_path = item["image_path"]
    question = item["question"]
    choices = item["choices"]
    images = [Image.open(image_path)]
    docs = []
    question_choice = "<|vision_start|><|image_pad|><|vision_end|> Question: " + question + "\nChoices: " + ", ".join(choices)
    question_direct = "<|vision_start|><|image_pad|><|vision_end|> Question: " + question

    response_choice = vlm.generate(docs, question_choice, images)
    response_direct = vlm.generate(docs, question_direct, images)

    item["response_choice"] = response_choice
    item["response_direct"] = response_direct


output_path = f"generated/direct/test_response.json"
with open(output_path, "w") as f:
    json.dump(test_dataset, f, indent=2, ensure_ascii=True)
print(f"Saved: {output_path}")


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in string.punctuation)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

with open("generated/direct/test_response.json", "r") as f:
    dataset = json.load(f)

total = 0
correct_choice = 0
correct_direct = 0

for item in dataset:
    pred_choice = normalize_answer(item['response_choice'])
    ans_choice = item['choices'][item['correct_choice_idx']]
    total += 1
    if pred_choice == ans_choice:
        correct_choice += 1
    pred_direct = normalize_answer(item['response_direct'])
    ans_direct = item['direct_answers']
    if  pred_direct in ans_direct:
        correct_direct += 1

print(f"Multiple Choice Accuracy: {correct_choice / total:.4f} ({correct_choice}/{total})")
print(f"Direct Answer Accuracy: {correct_direct / total:.4f} ({correct_direct}/{total})")