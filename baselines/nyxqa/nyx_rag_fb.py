import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from eval_utils import evaluate
from PIL import Image
from generator import MMGenerator

from transformers import Qwen2_5_VLModel, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"


# ===========configuration===========

model_name = "/fs/archive/share/Nyx-3B-Feedback0"
dataset_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

index_path = "index/nyx_fb.faiss"
index_dir = "index"
retrieved_dir = "retrieved/nyx_fb"
generated_dir = "generated/nyx_fb"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)


def process_images(images):
    if not images:
        return None
    pseudo_message = [{
        "content": [{"type": "image", "image": image} for image in images]
    }]
    images, _ = process_vision_info(pseudo_message)
    return images

# ===========embedding===========
def last_pooling(last_hidden_state, attention_mask, normalize=True):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

processor = Qwen2_5_VLProcessor.from_pretrained(model_name, use_fast=True)
model = Qwen2_5_VLModel.from_pretrained(
    model_name, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
).eval()

with open(os.path.join(dataset_dir, "corpus.json"), "r") as f:
    corpus = json.load(f)

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        text = item['text'].replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")

        images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in item['images']]
        images = process_images(images)

        inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = last_pooling(
                model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1],
                inputs['attention_mask']
                )
        embeddings.append(outputs.float().cpu().numpy())
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
    inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
    with torch.no_grad():
        query_embedding = last_pooling(
            model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1],
            inputs['attention_mask']
        ).float().cpu().numpy()
    _, I = index.search(query_embedding, top_k)
    return [corpus[i] for i in I[0]]

with open(os.path.join(dataset_dir, "test.json"), "r") as f:
    test_data = json.load(f)

for item in tqdm(test_data, desc="Retrieving documents for test"):
    question = item['qry']
    choices = item['choices']
    image_paths = item['qry_image_path']

    text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
    text = text + "\nChoices: " + ", ".join(choices)
    text = text.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)

    images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in image_paths]
    images = process_images(images)

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
    
    images = process_images(images)
    
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