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

from generator import MMGenerator
from transformers import CLIPProcessor, CLIPModel

# ===========configuration===========
model_name = "/fs/archive/share/clip-vit-base-patch32"
corpus_path = "mmqa_corpus_with_captions.json"

retrieved_dir = "retrieved/clip"
generated_dir = "generated/clip"
index_dir = "index"
retrieved_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# ==============embedding===========

model = CLIPModel.from_pretrained(model_name).to("cuda")
model.eval()
processor = CLIPProcessor.from_pretrained(model_name)

with open(corpus_path, "r") as f:
    corpus = json.load(f)


if not os.path.exists(f"{index_dir}/clip.faiss"):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        text = item['text'].replace("\n<|image|>", "")
        text_inputs = processor(
            text=text, 
            return_tensors="pt",
            truncation=True,
        ).to("cuda")
        images_inputs = None
        if item['image'] != "":
            image = Image.open(os.path.join("/fs/archive/share/mm_datasets/MMQA/images", item['image'])).convert("RGBA")
            # Ensure the image dimensions are at least 4x4
            width, height = image.size
            new_width = max(width, 4)
            new_height = max(height, 4)
            if width != new_width or height != new_height:
                image = image.resize((new_width, new_height))
            images = [image]
            images_inputs = processor(
                images=images, 
                return_tensors="pt",
                truncation=True,
            ).to("cuda")
        with torch.no_grad():
            text_outputs = model.get_text_features(**text_inputs)
            if images_inputs is not None:
                image_outputs = model.get_image_features(**images_inputs)
                outputs = (text_outputs + image_outputs) / 2
            else:
                outputs = text_outputs
            embeddings.append(outputs.float().cpu().numpy())
    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, f"{index_dir}/clip.faiss")
    print(f"Index saved to {index_dir}/clip.faiss")
else:
    index = faiss.read_index(f"{index_dir}/clip.faiss")
    print(f"Index loaded from {index_dir}/clip.faiss")

# ===========retrieval===========

def retrieve(index, corpus, text, top_k=10):
    inputs = processor(
        text=text, 
        return_tensors="pt",
        truncation=True,
    ).to("cuda")
    with torch.no_grad():
        outputs = model.get_text_features(**inputs).float().cpu().numpy()
    
    _, I = index.search(outputs, top_k)
    retrieved_items = [corpus[i] for i in I[0]]
    return retrieved_items

with open("/fs/archive/share/mm_datasets/MMQA/MMQA_dev.jsonl", "r") as f:
    dataset = [json.loads(line) for line in f]

for item in tqdm(dataset, desc="Retrieving MMQA items"):
    question = "Please retrieve the most relevant document to answer the question: " + item["question"]
    item['retrieved_docs'] = retrieve(
        index=index,
        corpus=corpus,
        text=question,
        top_k=retrieved_top_k
    )

with open(os.path.join(retrieved_dir, "dev_retrieved_docs.json"), "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)
print(f"Retrieved documents saved to {retrieved_dir}/dev_retrieved_docs.json")

del model, processor
torch.cuda.empty_cache()
torch.cuda.ipc_collect()

# ===========generation===========

vlm = MMGenerator(model_path="/fs/archive/share/Qwen2.5-VL-7B-Instruct")

with open(os.path.join(retrieved_dir, "dev_retrieved_docs.json"), "r") as f:
    dataset = json.load(f)

for item in tqdm(dataset, desc="Generating MMQA answers"):
    question = item["question"]
    retrieved_docs = item["retrieved_docs"]
    docs = []
    for doc in retrieved_docs[:generate_top_k]:
        doc = doc['text'].replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")
        docs.append(doc)
    
    # Prepare the context from retrieved documents
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

del vlm

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
print(f"\nCLIP RAG dev Evaluation:")
print(f"  Exact Match: {avg_em:.4f}")
print(f"  F1 Score:    {avg_f1:.4f}")
