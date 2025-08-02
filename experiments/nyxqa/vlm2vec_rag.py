import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from eval_utils import evaluate
from PIL import Image
from generator import MMGenerator
from transformers import AutoModelForCausalLM, AutoProcessor
from qwen_vl_utils import process_vision_info

IMAGE_TOKEN = "<|image|>"
PHI_IMAGE_TOKEN = "<|image_1|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

def process_images(images):
    if not images:
        return None
    pseudo_message = [{
        "content": [{"type": "image", "image": image} for image in images]
    }]
    images, _ = process_vision_info(pseudo_message)
    return images

# ===========configuration===========

model_name = "/fs/archive/share/VLM2Vec-Full"
dataset_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

index_path = "index/vlm2vec.faiss"
index_dir = "index"
retrieved_dir = "retrieved/vlm2vec"
generated_dir = "generated/vlm2vec"
retrieve_top_k = 10
generate_top_k = 1

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

# # ===========embedding===========
# def last_pooling(last_hidden_state, attention_mask, normalize=True):
#     sequence_lengths = attention_mask.sum(dim=1) - 1
#     batch_size = last_hidden_state.shape[0]
#     reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
#     if normalize:
#         reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
#     return reps

# processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
# model = AutoModelForCausalLM.from_pretrained(
#     model_name,
#     device_map="auto",
#     trust_remote_code=True,
#     torch_dtype="auto",
#     _attn_implementation="flash_attention_2"
# )
# model.eval()

# corpus = json.load(open(os.path.join(dataset_dir, "corpus.json"), "r"))

# if not os.path.exists(index_path):
#     embeddings = []
#     for item in tqdm(corpus, desc="Indexing"):
#         text = item['text']
        
#         images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in item['images']]
#         images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in item['images']]
#         images = process_images(images)

#         n_images = text.count(IMAGE_TOKEN)
#         for idx in range(1, n_images + 1):
#             text = text.replace(IMAGE_TOKEN, f"<|image_{idx}|>", 1)

#         inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
#         with torch.no_grad():
#             outputs = last_pooling(
#                 model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1],
#                 inputs['attention_mask']
#                 )
#         embeddings.append(outputs.float().cpu().numpy())
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
#     inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
#     with torch.no_grad():
#         query_embedding = last_pooling(
#             model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1],
#             inputs['attention_mask']
#         ).float().cpu().numpy()
#     _, I = index.search(query_embedding, top_k)
#     return [corpus[i] for i in I[0]]

# with open(os.path.join(dataset_dir, "test.json"), "r") as f:
#     test_data = json.load(f)

# for item in tqdm(test_data, desc="Retrieving documents for test"):
#     question = item['qry']
#     choices = item['choices']
#     image_paths = item['qry_image_path']

#     text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
#     text = text + "\nChoices: " + ", ".join(choices)
    

#     images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in image_paths]
#     images = process_images(images)

#     n_images = text.count(IMAGE_TOKEN)
#     for idx in range(1, n_images + 1):
#         text = text.replace(IMAGE_TOKEN, f"<|image_{idx}|>", 1)

#     item['retrieved_docs'] = retrieve(
#         index=index,
#         corpus=corpus,
#         text=text,
#         images=images,
#         top_k=retrieve_top_k
#     )

# with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "w") as f:
#     json.dump(test_data, f, indent=2, ensure_ascii=False)
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