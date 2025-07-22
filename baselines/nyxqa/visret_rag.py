from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import os
import faiss
from tqdm import tqdm
from eval_utils import evaluate
from generator import MMGenerator


IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

# ===========configuration===========
model_name = "/fs/archive/share/VisRAG-Ret"
dataset_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

index_path = "index/visret.faiss"
index_dir = "index"
retrieved_dir = "retrieved/visret"
generated_dir = "generated/visret"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# ===========embedding===========
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


with open(os.path.join(dataset_dir, "corpus.json"), "r") as f:
    corpus = json.load(f)

if not os.path.exists(index_path):
    embeddings = []
    for item in tqdm(corpus, desc="Indexing"):
        text = item['text'].replace(IMAGE_TOKEN, "")
        images = [Image.open(os.path.join(image_dir, path)).convert("RGB") for path in item['images']]
        if not images:
            images = None
        outputs = encode(text_list=[text], image_list=images)
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
    query_embedding = encode(text_list=[text], image_list=images)
    _, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [corpus[i] for i in I[0]]

test_data = json.load(open(os.path.join(dataset_dir, "test.json"), "r"))

for item in tqdm(test_data, desc="Retrieving"):
    question = item['qry']
    choices = item['choices']
    image_paths = item['qry_image_path']

    text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
    text = text + "\nChoices: " + ", ".join(choices)
    text = text.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)

    images = [Image.open(os.path.join(image_dir, path)).convert("RGB") for path in image_paths]
    if not images:
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

del model, tokenizer

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