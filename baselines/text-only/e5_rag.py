import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from collections import Counter
import re
import string
import torch.nn.functional as F

from torch import Tensor
from transformers import AutoTokenizer, AutoModel
from datasets import load_dataset
from generator import MMGenerator

# ===========configuration===========
subsets = ["hotpotqa", "2wikimultihopqa", "bamboogle", "musique"]
split = "test"
model_name = "/fs/archive/share/e5-base-v2"
corpus_path = "t2t_corpus.json"
index_path = "index/e5.faiss"
dataset_path = "/fs/archive/share/mm_datasets/Nyx-T2T-Data"
retrieved_dir = "retrieved/e5"
generated_dir = "generated/e5"
top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# ===========embedding===========
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")


with open(corpus_path, "r") as f:
    corpus = json.load(f)

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        input_texts = [item]  # Assuming item is a single text entry
        batch_dict = tokenizer(
            input_texts, 
            max_length=512, 
            padding=True, 
            truncation=True, 
            return_tensors='pt').to(model.device)
        with torch.no_grad():
            outputs = model(**batch_dict)
            output_embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0]
            embeddings.append(output_embedding.float().cpu().numpy())

    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
else:
    index = faiss.read_index(index_path)
    print(f"Index loaded from {index_path}")

# ===========retrieval===========


def retrieve(index, corpus, query, top_k=10):
    input_texts = [query]
    batch_dict = tokenizer(
        input_texts, 
        max_length=512, 
        padding=True, 
        truncation=True, 
        return_tensors='pt').to(model.device)
    
    with torch.no_grad():
        outputs = model(**batch_dict)
        query_embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0].float().cpu().numpy()
    
    _, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [corpus[i] for i in I[0]]

for subset in subsets:
    print(f"\nProcessing {subset} retrieval...")
    
    dataset = load_dataset(dataset_path, subset, split=split)
    results = []
    
    for item in tqdm(dataset, desc=f"retrieving {subset}"):
        item["retrieved_docs"] = retrieve(index, corpus, item["qry"], top_k=top_k)
        results.append(dict(item))
    
    output_path = f"{retrieved_dir}/{subset}_retrieved_docs.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Saved: {output_path}")

del tokenizer, model, index
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ===========generation===========

vlm = MMGenerator(model_path="/fs/archive/share/Qwen2.5-VL-7B-Instruct")

for subset in subsets:
    with open(f"{retrieved_dir}/{subset}_retrieved_docs.json", "r") as f:
        dataset = json.load(f)
    for item in tqdm(dataset, desc=f"generating {subset} answers"):
        question = item["qry"]
        retrieved_docs = item["retrieved_docs"][:top_k]
        response = vlm.generate(docs=retrieved_docs, question=question, images=None)
        item["response"] = response
        item.pop("neg_text", None)
    with open(f"{generated_dir}/{subset}_response.json", "w") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=True)
    print(f"Saved: {generated_dir}/{subset}_response.json")

# ===========evaluating===========

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
    with open(f"{generated_dir}/{subset}_response.json", "r") as f:
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

    

