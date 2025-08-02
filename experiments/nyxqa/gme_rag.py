from transformers import AutoModel, AutoProcessor
from transformers.models.qwen2_vl.modeling_qwen2_vl import Qwen2VLModel
from PIL import Image
import os
import json
import numpy as np
import faiss
from tqdm import tqdm
import torch
from eval_utils import evaluate
from generator import MMGenerator
from qwen_vl_utils import process_vision_info

IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

# ===========configuration===========

model_name = "/fs/archive/share/gme-Qwen2-VL-2B-Instruct"
dataset_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

index_path = "index/gme.faiss"
index_dir = "index"
retrieved_dir = "retrieved/gme"
generated_dir = "generated/gme"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# ===========embedding===========
model_path = "/fs/archive/share/gme-Qwen2-VL-2B-Instruct"
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype="float16", device_map='cuda', trust_remote_code=True
)

corpus = json.load(open(os.path.join(dataset_dir, "corpus.json"), "r"))

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        text = item['text'].replace(IMAGE_TOKEN, "")
        # Only encode the first image
        if item['images']:
            image = Image.open(os.path.join(image_dir, item['images'][0])).convert("RGB")
            images = [image]
        else:
            images = None
        outputs = model.get_fused_embeddings(texts=[text], images=images)
        embeddings.append(outputs)
    
    embeddings = np.vstack(embeddings).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
    print(f"Index saved to {index_path}")
else:
    index = faiss.read_index(index_path)
    print(f"Index loaded from {index_path}")

# ===========retrieval===========
def retrieve(index, corpus, text, images, top_k=10):
    query_embedding = model.get_fused_embeddings(texts=[text], images=images)
    _, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [corpus[i] for i in I[0]]

test_data = json.load(open(os.path.join(dataset_dir, "test.json"), "r"))

for item in tqdm(test_data, desc="Retrieving"):
    question = item['qry']
    choices = item['choices']
    image_paths = item['qry_image_path']

    text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
    text = text + "\nChoices: " + ", ".join(choices)
    text = text.replace(IMAGE_TOKEN, "")

    if image_paths:
        images = [Image.open(os.path.join(image_dir, image_paths[0])).convert("RGB")]
    else:
        images = None

    item['retrieved_docs'] = retrieve(
        index=index,
        corpus=corpus,
        text=text,
        images=images,
        top_k=retrieve_top_k
    )

with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "w") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f"Retrieved docs saved: {retrieved_dir}/test_retrieved_docs.json")

# ===========generation===========

vlm = MMGenerator("/fs/archive/share/Qwen2.5-VL-7B-Instruct")

with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "r") as f:
    test_data = json.load(f)

for item in tqdm(test_data, desc="Generating answers for test"):
    question = item["qry"]
    choices = item["choices"]

    text = f"Question: {question}\nChoices: {', '.join(choices)}"
    text = text.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)

    retrieved_docs = item["retrieved_docs"]

    docs = []
    for doc in retrieved_docs[:generate_top_k]:
        doc_text = doc['text'].replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)
        docs.append(doc_text)
    
    images = []
    for doc in retrieved_docs[:generate_top_k]:
        if doc['images']:
            for image_path in doc['images']:
                image = Image.open(os.path.join(image_dir, image_path)).convert("RGBA")
                images.append(image)
    if item['qry_image_path']:
        for image_path in item['qry_image_path']:
            image = Image.open(os.path.join(image_dir, image_path)).convert("RGBA")
            images.append(image)
    
    if not images:
        images = None
    
    answer = vlm.generate(
        docs=docs,
        images=images,
        question=text,
    )

    item["generated_answer"] = answer

with open(os.path.join(generated_dir, "test_generated_answers.json"), "w") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/test_generated_answers.json")

# ============evaluation============

evaluate(os.path.join(generated_dir, "test_generated_answers.json"))