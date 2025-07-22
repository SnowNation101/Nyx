from transformers import AutoModel, AutoTokenizer
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
import json
import os
import faiss
from tqdm import tqdm
from generator import MMGenerator
import re
import string


IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

# ===========configuration===========
model_name = "/fs/archive/share/VisRAG-Ret"
data_dir = "/fs/archive/share/mm_datasets/NyxQA"
lecture_path = "scienceqa_lecture_corpus.json"
example_qa_path = "scienceqa_example_qa_corpus.json"

index_dir = "index"
retrieved_dir = "retrieved/visret"
generated_dir = "generated/visret"
retrieve_top_k = 10
generate_top_k_lec = 1
generate_top_k_qa = 2

os.makedirs(retrieved_dir, exist_ok=True)
os.makedirs(generated_dir, exist_ok=True)
os.makedirs(index_dir, exist_ok=True)

problems = json.load(open(os.path.join(data_dir, "problems.json"), "r"))
pid_splits = json.load(open(os.path.join(data_dir, "pid_splits.json"), "r"))

train_pids = pid_splits['train']
val_pids = pid_splits['val']
test_pids = pid_splits['test']

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

lecture_corpus = json.load(open(lecture_path, "r"))
example_qa_corpus = json.load(open(example_qa_path, "r"))

# Index lecture corpus
if not os.path.exists(os.path.join(index_dir, "visret_lecture.faiss")):
    embeddings = []
    for item in tqdm(lecture_corpus, desc="Indexing lecture corpus"):
        text = item
        outputs = encode(text_list=[text])
        embeddings.append(outputs)
    embeddings = np.vstack(embeddings).astype("float32")
    lecture_index = faiss.IndexFlatIP(embeddings.shape[1])
    lecture_index.add(embeddings)
    faiss.write_index(lecture_index, os.path.join(index_dir, "visret_lecture.faiss"))
    print(f"Lecture corpus indexed and saved to {os.path.join(index_dir, 'visret_lecture.faiss')}")
else:
    lecture_index = faiss.read_index(os.path.join(index_dir, "visret_lecture.faiss"))
    print(f"Lecture corpus loaded from {os.path.join(index_dir, 'visret_lecture.faiss')}")

# Index example QA corpus
if not os.path.exists(os.path.join(index_dir, "visret_example_qa.faiss")):
    embeddings = []
    for item in tqdm(example_qa_corpus, desc="Indexing example QA corpus"):
        text = item['text'].replace(IMAGE_TOKEN, "")
        if item['image']:
            image = Image.open(os.path.join(data_dir, "images", item['image'])).convert("RGB")
            images = [image]
        else:
            images = None
        outputs = encode(text_list=[text], image_list=images)
        embeddings.append(outputs)
    embeddings = np.vstack(embeddings).astype("float32")
    example_qa_index = faiss.IndexFlatIP(embeddings.shape[1])
    example_qa_index.add(embeddings)
    faiss.write_index(example_qa_index, os.path.join(index_dir, "visret_example_qa.faiss"))
    print(f"Example QA corpus indexed and saved to {os.path.join(index_dir, 'visret_example_qa.faiss')}")
else:
    example_qa_index = faiss.read_index(os.path.join(index_dir, "visret_example_qa.faiss"))
    print(f"Example QA corpus loaded from {os.path.join(index_dir, 'visret_example_qa.faiss')}")

# ===========retrieving===========
def retrieve(index, corpus, text, images, top_k=10):
    query_embedding = encode(text_list=[text], image_list=images)
    _, I = index.search(query_embedding.reshape(1, -1), top_k)
    return [corpus[i] for i in I[0]]

retrieved = []
for pid in tqdm(test_pids, desc="Retrieving"):
    item = problems[pid]
    question = item["question"]
    choices = item["choices"]
    image = item["image"]
    
    question += f"\nChoices: " + ", ".join(choices)
    if image:
        image_path = f"{data_dir}/images/test/{pid}/{image}"
        images = [Image.open(image_path)]
    else:
        images = None

    # retrieve lectures
    text = f"Please retrieve the most relevant lecture to answer the question:\n Question: {question}"
    lectures = retrieve(
        index=lecture_index,
        corpus=lecture_corpus,
        text=text,
        images=images,
        top_k=retrieve_top_k
    )
    
    # retrieve example Q&A
    text = f"Please retrieve the most relevant example Q&A to answer the question:\n Question: {question}"
    example_qas = retrieve(
        index=example_qa_index,
        corpus=example_qa_corpus,
        text=text,
        images=images,
        top_k=retrieve_top_k
    )

    item['pid'] = pid
    item['retrieved_lecture'] = lectures
    item['retrieved_example_qa'] = example_qas

    retrieved.append(item)

with open(f"{retrieved_dir}/test_retrieved_docs.json", "w") as f:
    json.dump(retrieved, f, indent=2, ensure_ascii=False)
print(f"Retrieved docs saved: {retrieved_dir}/test_retrieved_docs.json")

del model, tokenizer

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