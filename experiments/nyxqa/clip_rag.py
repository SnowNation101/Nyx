import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
from eval_utils import evaluate

from generator import MMGenerator
from transformers import CLIPProcessor, CLIPModel

IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

# ===========configuration===========
model_name = "/fs/archive/share/clip-vit-base-patch32"
data_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

index_path = "index/clip.faiss"
index_dir = "index"
retrieved_dir = "retrieved/clip"
generated_dir = "generated/clip"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# # ===========embedding===========
# model = CLIPModel.from_pretrained(model_name).to("cuda")
# model.eval()
# processor = CLIPProcessor.from_pretrained(model_name)

# corpus = json.load(open(os.path.join(data_dir, "corpus.json"), "r"))


# if not os.path.exists(index_path):
#     embeddings = []
#     for item in tqdm(corpus, desc="Indexing corpus"):
#         text = item["text"]
#         text_inputs = processor(
#             text=text, 
#             return_tensors="pt",
#             truncation=True,
#         ).to("cuda")
#         images = []
#         for image_path in item["images"]:
#             image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
#             image = image.resize((max(image.width, 4), max(image.height, 4)))
#             images.append(image)
#         if images:
#             images_inputs = processor(
#                 images=images, 
#                 return_tensors="pt",
#                 truncation=True,
#             ).to("cuda")

#         with torch.no_grad():
#             all_embeds = []
#             text_outputs = model.get_text_features(**text_inputs)
#             all_embeds.extend(text_outputs.float().cpu().numpy())
#             image_outputs = model.get_image_features(**images_inputs) if images else None
#             if image_outputs is not None:
#                 all_embeds.extend(image_outputs.float().cpu().numpy())
#             output = np.array(all_embeds)
#             output = np.mean(output, axis=0, keepdims=True)
#         embeddings.append(output)

#     embeddings = np.vstack(embeddings).astype("float32")
#     index = faiss.IndexFlatIP(embeddings.shape[1])
#     index.add(embeddings)
#     faiss.write_index(index, index_path)
#     print(f"Index saved to {index_path}")
# else:
#     index = faiss.read_index(index_path)
#     print(f"Index loaded from {index_path}")

# # ===========retrieval===========
# def retrieve(index, corpus, text, images, top_k=10):
#     text_inputs = processor(
#         text=text, 
#         return_tensors="pt",
#         truncation=True,
#     ).to("cuda")
    
#     if images:
#         images_inputs = processor(
#             images=images, 
#             return_tensors="pt",
#             truncation=True,
#         ).to("cuda")
    
#     with torch.no_grad():
#         all_embeds = []
#         text_outputs = model.get_text_features(**text_inputs)
#         all_embeds.extend(text_outputs.float().cpu().numpy())
        
#         if images:
#             image_outputs = model.get_image_features(**images_inputs)
#             all_embeds.extend(image_outputs.float().cpu().numpy())
        
#         query_embedding = np.mean(np.array(all_embeds), axis=0, keepdims=True)

#     _, I = index.search(query_embedding, top_k)
#     return [corpus[i] for i in I[0]]

# test_data = json.load(open(os.path.join(data_dir, "test.json"), "r"))

# for item in tqdm(test_data, desc="Retrieving documents for test"):
#     question = item['qry']
#     choices = item['choices']
#     image_paths = item['qry_image_path']

#     text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
#     text = text + "\nChoices: " + ", ".join(choices)
#     text = text.replace(IMAGE_TOKEN, "")

#     if image_paths:
#         images = []
#         for image_path in image_paths:
#             image = Image.open(os.path.join(image_dir, image_path)).convert("RGB")
#             image = image.resize((max(image.width, 4), max(image.height, 4)))
#             images.append(image)
#     else:
#         images = None
    
#     item['retrieved_docs'] = retrieve(
#         index=index,
#         corpus=corpus,
#         text=text,
#         images=images,
#         top_k=retrieve_top_k
#     )
    
# with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "w") as f:
#     json.dump(test_data, f, indent=2, ensure_ascii=True)
# print(f"Retrieved docs saved: {retrieved_dir}/test_retrieved_docs.json")

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