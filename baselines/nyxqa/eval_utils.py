import json
import string
import re
from collections import Counter


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


def evaluate(path):
    with open(path, "r") as f:
        dataset = json.load(f)

    total_acc = 0
    count = 0
    for item in dataset:
        pred = item["generated_answer"].strip()
        true = item["choices"][item["right_choice"]].strip()
        
        pred_norm = normalize_answer(pred)
        true_norm = normalize_answer(true)

        if pred_norm == true_norm:
            total_acc += 1
        count += 1

    avg_acc = total_acc / count
    print(f"\nEvaluation:")
    print(f"  Accuracy: {avg_acc:.4f}")