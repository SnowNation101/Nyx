import json
from tqdm import tqdm
from datasets import load_dataset
from generator import MMGenerator

import re
import string
from collections import Counter

subsets = ["hotpotqa", "2wikimultihopqa", "bamboogle", "musique"]
split = "test"

model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
vlm = MMGenerator(model_path=model_path)

dataset_path = "/fs/archive/share/mm_datasets/Nyx-T2T-Data"
for subset in subsets:
    print(f"\nProcessing {subset}...")
    dataset = load_dataset(dataset_path, subset, split=split)
    new_data = []
    for item in tqdm(dataset, desc=f"Generating answers for {subset}"):
        question = item["qry"]
        docs = []
        response = vlm.generate(docs, question, images=None)
        new_item = dict(item)
        new_item["response"] = response
        new_item.pop("neg_text", None)
        new_data.append(new_item)
    output_path = f"generated/direct/{subset}_response.json"
    with open(output_path, "w") as f:
        json.dump(new_data, f, indent=2, ensure_ascii=True)
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

for subset in subsets:
    with open(f"generated/direct/{subset}_response.json", "r") as f:
        dataset = json.load(f)

    total_em = 0
    total_f1 = 0
    count = 0
    for item in tqdm(dataset, desc=f"evaluating {subset} answers"):
        pred = item["response"].strip()
        gold_list = item["ans"]

        em_scores = [compute_exact(pred, g) for g in gold_list]
        f1_scores = [compute_f1(pred, g) for g in gold_list]

        total_em += max(em_scores)
        total_f1 += max(f1_scores)
        count += 1

    avg_em = total_em / count
    avg_f1 = total_f1 / count
    print(f"\n{subset} Evaluation:")
    print(f"  Exact Match: {avg_em:.4f}")
    print(f"  F1 Score:    {avg_f1:.4f}")