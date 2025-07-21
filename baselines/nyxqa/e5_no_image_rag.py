import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from eval_utils import evaluate
from PIL import Image
from generator import MMGenerator

from torch import Tensor
from transformers import AutoTokenizer, AutoModel

IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

# ===========configuration===========
model_name = "/fs/archive/share/e5-base-v2"
dataset_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

index_path = "index/e5_no_img.faiss"
index_dir = "index"
retrieved_dir = "retrieved/e5"
generated_dir = "generated/e5"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# ===========embedding===========
def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name).to("cuda")

corpus = json.load(open(os.path.join(dataset_dir, "corpus.json"), "r"))
captions = json.load(open(os.path.join(dataset_dir, "nyx_caption.json"), "r"))

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        text = item['text']
        images = item['images']

        text.replace(IMAGE_TOKEN, "")

        input_texts = [text]  # Assuming item is a single text entry
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

    embeddings = np.array(embeddings).astype(np.float32)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, index_path)
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
        output_embedding = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])[0]
        query_embedding = output_embedding.float().cpu().numpy()
    
    _, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [corpus[i] for i in I[0]]

test_data = json.load(open(os.path.join(dataset_dir, "test.json"), "r"))

for item in tqdm(test_data, desc="Retrieving documents for test"):
    question = item['qry']
    choices = item['choices']
    image_paths = item['qry_image_path']

    text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
    text = text + "\nChoices: " + ", ".join(choices)
    
    text = text.replace(IMAGE_TOKEN, "")

    item['retrieved_docs'] = retrieve(
        index=index,
        corpus=corpus,
        query=text,
        top_k=retrieve_top_k
    )

with open(os.path.join(retrieved_dir, "test_retrieved_docs_no_img.json"), "w") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f"Retrieved docs saved: {retrieved_dir}/test_retrieved_docs_no_img.json")

# ===========generation===========

vlm = MMGenerator("/fs/archive/share/Qwen2.5-VL-7B-Instruct")

with open(os.path.join(retrieved_dir, "test_retrieved_docs_no_img.json"), "r") as f:
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

with open(os.path.join(generated_dir, "test_generated_answers_no_img.json"), "w") as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/test_generated_answers_no_img.json")

# ============evaluation============

evaluate(os.path.join(generated_dir, "test_generated_answers_no_img.json"))