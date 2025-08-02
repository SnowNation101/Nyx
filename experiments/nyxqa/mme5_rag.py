import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from eval_utils import evaluate
from PIL import Image
from generator import MMGenerator

from transformers import MllamaForConditionalGeneration, AutoProcessor

# ===========configuration===========

model_name = "/fs/archive/share/mmE5-mllama-11b-instruct"
dataset_dir = "/fs/archive/share/mm_datasets/NyxQA"
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"
index_path = "index/mmE5.faiss"
retrieved_dir = "retrieved/mmE5"
generated_dir = "generated/mmE5"
index_dir = "index"
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

# processor = AutoProcessor.from_pretrained(model_name)
# model = MllamaForConditionalGeneration.from_pretrained(
#     model_name, torch_dtype=torch.bfloat16
# ).to("cuda")
# model.eval()

# with open(os.path.join(dataset_dir, "corpus.json"), "r") as f:
#     corpus = json.load(f)

# if not os.path.exists(index_path):
#     embeddings = []
#     for item in tqdm(corpus, desc="Indexing"):
#         text = item['text']
#         if item['images']:
#             images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in item['images']]
#         else:
#             images = None
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

#     if image_paths:
#         images = [Image.open(os.path.join(image_dir, path)).convert("RGBA") for path in image_paths]
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
#     json.dump(test_data, f, indent=2, ensure_ascii=False)
# print(f"Retrieved docs saved: {retrieved_dir}/test_retrieved_docs.json")

# ===========generation===========

# vlm = MMGenerator("/fs/archive/share/Qwen2.5-VL-7B-Instruct")

# with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "r") as f:
#     test_data = json.load(f)

# for item in tqdm(test_data, desc="Generating answers for test"):
#     question = item["qry"]
#     choices = item["choices"]

#     text = f"Question: {question}\nChoices: {', '.join(choices)}"
#     text = text.replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")

#     retrieved_docs = item["retrieved_docs"]

#     docs = []
#     for doc in retrieved_docs[:generate_top_k]:
#         doc_text = doc['text'].replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")
#         docs.append(doc_text)
    
#     images = []
#     for doc in retrieved_docs[:generate_top_k]:
#         if doc['images']:
#             for image_path in doc['images']:
#                 image = Image.open(os.path.join(image_dir, image_path)).convert("RGBA")
#                 images.append(image)
#     if item['qry_image_path']:
#         for image_path in item['qry_image_path']:
#             image = Image.open(os.path.join(image_dir, image_path)).convert("RGBA")
#             images.append(image)
#     if not images:
#         images = None
    
#     answer = vlm.generate(
#         docs=docs,
#         images=images,
#         question=text,
#     )

#     item["generated_answer"] = answer

# with open(os.path.join(generated_dir, "test_generated_answers.json"), "w") as f:
#     json.dump(test_data, f, indent=2, ensure_ascii=False)
# print(f"Generated answers saved to {generated_dir}/test_generated_answers.json")

# ============evaluation============

evaluate(os.path.join(generated_dir, "test_generated_answers.json"))