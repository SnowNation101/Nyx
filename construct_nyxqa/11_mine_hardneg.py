import torch
import json
import os
import numpy as np
import faiss

from PIL import Image
from transformers import MllamaForConditionalGeneration, AutoProcessor
from tqdm import tqdm

# Pooling and Normalization
def last_pooling(last_hidden_state, attention_mask, normalize=True):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), 
                             sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps

model_name = "/fs/archive/share/mmE5-mllama-11b-instruct"

# Load Processor and Model
processor = AutoProcessor.from_pretrained(model_name)
model = MllamaForConditionalGeneration.from_pretrained(
    model_name, torch_dtype=torch.bfloat16
).to("cuda")
model.eval()

with open("NyxQA/corpus.json", "r") as f:
    generated_qa = json.load(f)
image_dir = "/fs/archive/share/mm_datasets/NyxQA/images"

# Indexing the corpus
if not os.path.exists("all_data/index.faiss"):
    embeddings = []
    for item in tqdm(generated_qa, desc="Indexing"):
        images = [Image.open(os.path.join(image_dir, image)) for image in item["images"]] or None

        if images:
            images = [
                img.resize((
                    max(img.width, 4),
                    max(img.height, 4)
                )) for img in images
            ]

        inputs = processor(
            text=item["text"],
            images=images,
            return_tensors="pt").to("cuda")
        with torch.no_grad():
            outputs = last_pooling(
                model(
                    **inputs, 
                    return_dict=True, 
                    output_hidden_states=True
                ).hidden_states[-1], 
                inputs['attention_mask'])
            embeddings.append(outputs.float().cpu().numpy())

    embeddings = np.vstack(embeddings).astype("float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, "all_data/index.faiss")
    print(f"Index saved to all_data/index.faiss")
else:
    index = faiss.read_index("all_data/index.faiss")
    print(f"Index loaded from all_data/index.faiss")


# Retrieve hard negatives
with open("NyxQA/train.json", "rb") as f:
    train_data = json.load(f)

for item in tqdm(train_data, desc="retrieving hard negs"):
    images = [Image.open(os.path.join(image_dir, image)) for image in item["qry_image_path"]] or None

    if images:
        images = [
            img.resize((
                max(img.width, 4),
                max(img.height, 4)
            )) for img in images
        ]

    inputs = processor(
        text="Please retrieve the most relevant document to answer the question.\n" + item["qry"],
        images=images,
        return_tensors="pt").to("cuda")
    
    with torch.no_grad():
        outputs = last_pooling(
            model(
                **inputs, 
                return_dict=True, 
                output_hidden_states=True
            ).hidden_states[-1], 
            inputs['attention_mask'])
    
    q_reps = outputs.float().cpu().numpy()
    scores, indices = index.search(q_reps, k=20)
    
    retrieved_data = []
    for i, idx in enumerate(indices[0]):
        retrieved_data.append({
            "text": generated_qa[idx]["text"],
            "images": generated_qa[idx]["images"]
            })
    
    neg_text = []
    neg_image_path = []
    for i, data in enumerate(retrieved_data[10:]):
        if data["text"] != item["pos_text"]:
            neg_text.append(data["text"])
            neg_image_path.append(data["images"])
    
    item["neg_text"] = neg_text[:5]
    item["neg_image_path"] = neg_image_path[:5]

for item in train_data:
    item["qry_image_path"] = ["images/NyxQA/" + img for img in item["qry_image_path"]]
    item["pos_image_path"] = ["images/NyxQA/" + img for img in item["pos_image_path"]]
    item["neg_image_path"] = [["images/NyxQA/" + img for img in img_list] for img_list in item["neg_image_path"]]

with open("NyxQA/train_hardneg.json", "w") as f:
    json.dump(train_data, f, ensure_ascii=True, indent=2)