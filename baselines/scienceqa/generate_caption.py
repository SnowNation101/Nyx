from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
from PIL import Image
import json

model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
image_dir = "/fs/archive/share/mm_datasets/ScienceQA/images"
output_json = "scienceqa_captions.json"

# Load processor and model
processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True)

with open("/fs/archive/share/mm_datasets/ScienceQA/problems.json", "r") as f:
    problems = json.load(f)

# Prepare image + prompt batches
llm_inputs = []
pids_with_images = []

print(f"Processing all images to llm_inputs...")
for pid, item in tqdm(problems.items(), desc="Processing images"):
    if item['image']:
        image_path = os.path.join(image_dir, item['split'], pid, item['image'])
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Failed to load image for pid {pid}: {e}")
            continue

        messages = [
            {"role": "system", "content": "You are a professional assistant specialized in visual understanding."},
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": "Please describe the image."}
            ]}
        ]

        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, _ = process_vision_info(messages)

        llm_inputs.append({
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
        })
        pids_with_images.append(pid)

# Initialize model
llm = LLM(
    model=model_path,
    limit_mm_per_prompt={"image": 1},
)

sampling_params = SamplingParams(
    temperature=0.1,
    top_p=0.001,
    repetition_penalty=1.05,
    max_tokens=4096,
    stop_token_ids=[],
)

# Run batched inference
print(f"Generating captions for {len(pids_with_images)} images...")
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

# Match outputs with pids
captions_dict = {}
for pid, output in zip(pids_with_images, outputs):
    try:
        text = output.outputs[0].text.strip()
        captions_dict[pid] = text
    except Exception as e:
        print(f"Warning: Failed to extract output for pid {pid}: {e}")

# Save to JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(captions_dict, f, indent=2, ensure_ascii=False)

print(f"Saved {len(captions_dict)} captions to {output_json}")