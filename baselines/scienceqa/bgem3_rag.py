import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm

import re
import string
from PIL import Image
from FlagEmbedding import BGEM3FlagModel
from generator import MMGenerator

# ===========configuration===========
model_name = "/fs/archive/share/bge-m3"
data_dir = "/fs/archive/share/mm_datasets/ScienceQA"

lecture_path = "scienceqa_lecture_corpus.json"
example_qa_path = "scienceqa_example_qa_corpus.json"
caption_path = "scienceqa_captions.json"

retrieved_dir = "retrieved/bgem3"
generated_dir = "generated/bgem3"
index_dir = "index"
retrieve_top_k = 10
generate_top_k_lec = 1
generate_top_k_qa = 2

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

with open(os.path.join(data_dir, "problems.json"), "r") as f:
    problems = json.load(f)
with open(os.path.join(data_dir, "pid_splits.json"), "r") as f:
    pid_splits = json.load(f)

train_pids = pid_splits['train']
val_pids = pid_splits['val']
test_pids = pid_splits['test']


with open(caption_path, "r") as f:
    captions = json.load(f)

def corpus_get_caption(image_path):
    id = image_path.split("/")[1]
    return captions[id]

def question_get_caption(pid):
    return captions[str(pid)]


# # ===========embedding===========
# model = BGEM3FlagModel(model_name, use_fp16=True)

# with open(lecture_path, "r") as f:
#     lecture_corpus = json.load(f)
# with open(example_qa_path, "r") as f:
#     example_qa_corpus = json.load(f)


# if not os.path.exists(f"{index_dir}/bgem3_lecture.faiss"):
#     embeddings = []
#     for item in tqdm(lecture_corpus, desc="Indexing lectures"):
#         text = item
#         output = model.encode(
#             text,
#             batch_size=12,
#             max_length=8192
#         )['dense_vecs']
#         embeddings.append(output)

#     embeddings = np.vstack(embeddings).astype("float32")
#     lecture_index = faiss.IndexFlatIP(embeddings.shape[1])
#     lecture_index.add(embeddings)
#     faiss.write_index(lecture_index, f"{index_dir}/bgem3_lecture.faiss")
#     print(f"Lecture index saved to {index_dir}/bgem3_lecture.faiss")
# else:
#     lecture_index = faiss.read_index(f"{index_dir}/bgem3_lecture.faiss")
#     print(f"Lecture index loaded from {index_dir}/bgem3_lecture.faiss")

# # Indexing example QAs
# if not os.path.exists(f"{index_dir}/bgem3_example_qa.faiss"):
#     embeddings = []
#     for item in tqdm(example_qa_corpus, desc="Indexing example QAs"):
#         text = item['text'].replace("\n<|image|>", "")
#         if item['image']:
#             text = corpus_get_caption(item['image']) + " " + text
#         output = model.encode(
#             text,
#             batch_size=12,
#             max_length=8192
#         )['dense_vecs']
#         embeddings.append(output)

#     embeddings = np.vstack(embeddings).astype("float32")
#     example_qa_index = faiss.IndexFlatIP(embeddings.shape[1])
#     example_qa_index.add(embeddings)
#     faiss.write_index(example_qa_index, f"{index_dir}/bgem3_example_qa.faiss")
#     print(f"Example QA index saved to {index_dir}/bgem3_example_qa.faiss")
# else:
#     example_qa_index = faiss.read_index(f"{index_dir}/bgem3_example_qa.faiss")
#     print(f"Example QA index loaded from {index_dir}/bgem3_example_qa.faiss")

# # ===========retrieving===========
# def retrieve(index, corpus, query, top_k=10):
#     query_embedding = model.encode(
#         query,
#         batch_size=12,
#         max_length=8192
#     )['dense_vecs'].reshape(1, -1)
    
#     _, I = index.search(query_embedding, top_k)
#     return [corpus[i] for i in I[0]]

# retrieved = []
# for pid in tqdm(test_pids, desc="Retrieving test set"):
#     item = problems[pid]
#     question = item['question']
#     choices = item['choices']
#     image = item['image']

#     if image:
#         caption = question_get_caption(pid)
#         question = f"Question: {caption} {question}"
#     else:
#         question = f"Question: {question}"
    
#     question += f" Choices: {', '.join(choices)}"

#     text = f"Please retrieve the most relevant lecture to answer the question: {question}"
#     lectures = retrieve(lecture_index, lecture_corpus, text, top_k=retrieve_top_k)

#     text = f"Please retrieve the most relevant example Q&A to answer the question: {question}"
#     example_qas = retrieve(example_qa_index, example_qa_corpus, text, top_k=retrieve_top_k)

#     item['pid'] = pid
#     item['retrieved_lecture'] = lectures
#     item['retrieved_example_qa'] = example_qas

#     retrieved.append(item)

# with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "w") as f:
#     json.dump(retrieved, f, indent=2, ensure_ascii=False)
# print(f"Saved: {retrieved_dir}/test_retrieved_docs.json")

# del model
# torch.cuda.empty_cache()
# torch.cuda.ipc_collect()

# ===========generating===========
with open(os.path.join(retrieved_dir, "test_retrieved_docs.json"), "r") as f:
    retrieved = json.load(f)

vlm = MMGenerator(model_path="/fs/archive/share/Qwen2.5-VL-7B-Instruct")

genereted = []
for item in tqdm(retrieved, desc="Generating answers for test set"):
    question = item["question"]
    choices = item["choices"]
    question = f"Question: {question}\nChoices: {', '.join(choices)}\nAnswer:"
    
    retrieved_lecture = item["retrieved_lecture"]
    retrieved_example_qa = item["retrieved_example_qa"]

    lectures = []
    for lecture in retrieved_lecture[:generate_top_k_lec]:
        lectures.append(f"Lecture: {lecture}")

    example_qas = []
    for example_qa in retrieved_example_qa[:generate_top_k_qa]:
        example_qa = example_qa['text'].replace("<|image|>", "<|vision_start|><|image_pad|><|vision_end|>")
        example_qas.append(f"Example Q&A: {example_qa}")

    docs = lectures + example_qas

    question_image = item['image']
    if question_image:
        question_image = Image.open(f"/fs/archive/share/mm_datasets/ScienceQA/images/test/{item['pid']}/{question_image}")
        question = "<|vision_start|><|image_pad|><|vision_end|>" + question
    images = []
    images += [Image.open(os.path.join(data_dir, "images", doc['image'])) for doc in retrieved_example_qa[:generate_top_k_qa] if doc['image']]
    images = images.append(question_image) if question_image else images
    images = images if images else None
    
    answer = vlm.generate(
        docs=docs,
        images=images,
        question=question,
    )
    
    item["generated_answer"] = answer
    genereted.append(item)

with open(os.path.join(generated_dir, "test_generated_answers.json"), "w") as f:
    json.dump(genereted, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/test_generated_answers.json")

# ============evaluation============

with open(f"{generated_dir}/test_generated_answers.json", "r") as f:
    dataset = json.load(f)

total_acc = 0
count = 0
for item in dataset:
    pred = item["generated_answer"].strip()
    true = item['choices'][item["answer"]]
    
    # Normalize answers
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
    
    pred_norm = normalize_answer(pred)
    true_norm = normalize_answer(true)

    if pred_norm == true_norm:
        total_acc += 1
    count += 1

avg_acc = total_acc / count
print(f"\nEvaluation:")
print(f"  Accuracy: {avg_acc:.4f}")