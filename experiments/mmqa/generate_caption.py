from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
from PIL import Image
import json

model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
image_dir = "/fs/archive/share/mm_datasets/MMQA/MMQA/final_dataset_images"
input_json = "mmqa_corpus.json"
output_json = "mmqa_corpus_with_captions.json"

# Load processor and model
processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True)

# Load data
with open(input_json, "r") as f:
    corpus = json.load(f)

# Prepare image + prompt batches
image_items = []
llm_inputs = []

print(f"Processing corpus to llm_inputs...")
for item in tqdm(corpus, desc="Processing corpus"):
    if item['image'] != "":
        image_path = os.path.join(image_dir, item['image'])
        image = Image.open(image_path)

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
        image_items.append(item)

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

# Run batched inference with vLLM
print(f"Generating captions for {len(image_items)} images...")
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

# Insert captions into original corpus
for item, output in zip(image_items, outputs):
    item["caption"] = output.outputs[0].text.strip()

# Fill empty captions for items without image
for item in corpus:
    if "caption" not in item:
        item["caption"] = ""

# Save result
with open(output_json, "w") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)

print(f"Saved captioned corpus to: {output_json}")
