import math
import numpy as np
import torch
import torchvision.transforms as T
from collections import Counter
import re
import string
from datasets import load_dataset

from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
import os
import json
from tqdm import tqdm

# =========== configuration ===========
generated_dir = "generated/internvl_da"
os.makedirs(generated_dir, exist_ok=True)

# =========== internvl utils ===========

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map

# =========== generation ===========

model_path = "/fs/archive/share/InternVL3-8B"
device_map = split_model(model_path)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    load_in_8bit=False,
    low_cpu_mem_usage=True,
    use_flash_attn=True,
    trust_remote_code=True,
    device_map=device_map).eval()
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False)

generation_config = dict(
    pad_token_id = tokenizer.eos_token_id,
    max_new_tokens=4096, 
    do_sample=True)

with open("/fs/archive/share/mm_datasets/ScienceQA/problems.json", "r") as f:
    problems = json.load(f)
with open("/fs/archive/share/mm_datasets/ScienceQA/pid_splits.json", "r") as f:
    pid_splits = json.load(f)

train_pids = pid_splits['train']
val_pids = pid_splits['val']
test_pids = pid_splits['test']

results = []
for pid in tqdm(test_pids, desc="Generating answers for test"):
    item = problems[pid]
    item['pid'] = pid
    question = item["question"]
    choices = item["choices"]

    # Prepare the prompt
    system = "Answer the multiple-choice question. Only give me the answer and do not output any other words.\n"
    image_tag = "<image>\n" if item.get('image') else ""
    question_block = f"Question: {question}\nChoices: {', '.join(choices)}"
    prompt = f"{system}{image_tag}{question_block}"

    pixel_values = None
    if item.get('image'):
        image_path = f"/fs/archive/share/mm_datasets/ScienceQA/images/test/{pid}/{item['image']}"
        pixel_values = load_image(image_path, max_num=12).to(torch.bfloat16).cuda()

    response = model.chat(tokenizer, pixel_values, prompt, generation_config)

    item['generated_answer'] = response
    results.append(item)

with open(f"{generated_dir}/test_generated_answers.json", "w") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print(f"Generated answers saved to {generated_dir}/test_generated_answers.json")

# =========== evaluation ===========

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