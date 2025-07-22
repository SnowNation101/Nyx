import json
import os
import numpy as np
from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm
from datasets import load_dataset
from collections import Counter
import re
import string
from generator import MMGenerator

# ===========configuration===========
subsets = ["hotpotqa", "2wikimultihopqa", "bamboogle", "musique"]
split = "test"
corpus_path = "t2t_corpus.json"
index_path = "index/visret.faiss"
dataset_path = "/fs/archive/share/mm_datasets/Nyx-T2T-Data"
retrieved_dir = "retrieved/visret"
generated_dir = "generated/visret"
top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)

# ===========embedding===========
model_name = "/fs/archive/share/VisRAG-Ret"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda()
model.eval()

def weighted_mean_pooling(hidden, attention_mask):
    attention_mask_ = attention_mask * attention_mask.cumsum(dim=1)
    s = torch.sum(hidden * attention_mask_.unsqueeze(-1).float(), dim=1)
    d = attention_mask_.sum(dim=1, keepdim=True).float()
    reps = s / d
    return reps

@torch.no_grad()
def encode(text_list=None, image_list=None):
    all_embeddings = []

    if text_list:
        inputs_text = {
            "text": text_list,
            "image": [None] * len(text_list),
            "tokenizer": tokenizer
        }
        outputs_text = model(**inputs_text)
        reps_text = weighted_mean_pooling(outputs_text.last_hidden_state, outputs_text.attention_mask)
        reps_text = F.normalize(reps_text, p=2, dim=1).detach().cpu().numpy()
        all_embeddings.extend(reps_text)

    if image_list:
        inputs_image = {
            "text": [''] * len(image_list),
            "image": image_list,
            "tokenizer": tokenizer
        }
        outputs_image = model(**inputs_image)
        reps_image = weighted_mean_pooling(outputs_image.last_hidden_state, outputs_image.attention_mask)
        reps_image = F.normalize(reps_image, p=2, dim=1).detach().cpu().numpy()
        all_embeddings.extend(reps_image)

    if not all_embeddings:
        raise ValueError("At least one of text_list or image_list must be provided.")

    all_embeddings = np.array(all_embeddings)
    final_embedding = np.mean(all_embeddings, axis=0)

    return final_embedding

corpus = json.load(open(corpus_path, "r"))

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        outputs = encode(text_list=[item])
        embeddings.append(outputs)
    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
else:
    index = faiss.read_index(index_path)
    print(f"Index loaded from {index_path}")


# ===========retrieving===========
def retrieve(index, corpus, text, top_k=10):
    query_embedding = encode(text_list=[text]).reshape(1, -1)
    _, I = index.search(query_embedding, top_k)
    return [corpus[i] for i in I[0]]

for subset in subsets:
    print(f"\nProcessing {subset} retrieval...")
    dataset = load_dataset(dataset_path, subset, split=split)
    results = []
    for item in tqdm(dataset, desc=f"retrieving {subset}"):
        text = item['qry']
        item["retrieved_docs"] = retrieve(index, corpus, text)
        results.append(dict(item))
    with open(f"{retrieved_dir}/{subset}_retrieved_docs.json", "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Retrieved documents for {subset} saved to {retrieved_dir}/{subset}_retrieved_docs.json")

del model, tokenizer

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