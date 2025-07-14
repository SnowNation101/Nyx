import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from collections import Counter
import re
import string
from PIL import Image
from FlagEmbedding import BGEM3FlagModel
from generator import MMGenerator

# ===========configuration===========
model_name = "/fs/archive/share/bge-m3"
corpus_path = "mmqa_corpus_with_captions.json"

retrieved_dir = "retrieved/bgem3"
generated_dir = "generated/bgem3"
index_dir = "index"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# ===========embedding & indexing===========
model = BGEM3FlagModel(model_name, use_fp16=True)

with open(corpus_path, "r") as f:
    corpus = json.load(f)

index_path = f"{index_dir}/bgem3.faiss"

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        text = item["text"].replace("\n<|image|>", "")
        if item.get("caption", ""):
            text += " " + item["caption"]
        output = model.encode(
            text,
            batch_size=12,
            max_length=8192
        )['dense_vecs']
        embeddings.append(output)

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
    query_embedding = model.encode(
        query,
        batch_size=12,
        max_length=8192
    )['dense_vecs'].reshape(1, -1)
    
    _, I = index.search(query_embedding, top_k)
    return [corpus[i] for i in I[0]]

with open("/fs/archive/share/mm_datasets/MMQA/MMQA_dev.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

for item in tqdm(dataset, desc=f"retrieving dev set"):
    question = "Please retrieve the most relevant document to answer the question " + item["question"]
    item['retrieved_docs'] = retrieve(
        index=index,
        corpus=corpus,
        query=question,
        top_k=retrieve_top_k
    )

with open(f"{retrieved_dir}/dev_retrieved_docs.json", "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
print(f"Retrieved documents saved to {retrieved_dir}/dev_retrieved_docs.json")

del index, model
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ===========generation===========
vlm = MMGenerator(model_path="/fs/archive/share/Qwen2.5-VL-7B-Instruct")

with open(os.path.join(retrieved_dir, "dev_retrieved_docs.json"), "r") as f:
    dataset = json.load(f)

for item in tqdm(dataset, desc="Generating answers for MMQA items"):
    question = item["question"]
    retrieved_docs = item["retrieved_docs"]
    docs = []
    for doc in retrieved_docs[:generate_top_k]:
        doc = doc['text'].replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")
        docs.append(doc)

    images = [Image.open(os.path.join("/fs/archive/share/mm_datasets/MMQA/images", doc['image'])) for doc in retrieved_docs[:generate_top_k] if doc['image'] != ""]
    if not images:
        images = None

    answer = vlm.generate(
        docs=docs,
        images=images,
        question=question,
    )

    item["generated_answer"] = answer

with open(os.path.join(generated_dir, "dev_generated_answers.json"), "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/dev_generated_answers.json")

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
print(f"\nBGEM3 RAG dev Evaluation:")
print(f"  Exact Match: {avg_em:.4f}")
print(f"  F1 Score:    {avg_f1:.4f}")
