from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import os
from tqdm import tqdm
from PIL import Image
import json

model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
image_dir = "/fs/archive/share/mm_datasets/obelics_images"
output_json = "./NyxQA/nyx_caption.json"


# Load processor and model
processor = AutoProcessor.from_pretrained(
    model_path,
    use_fast=True)

# Prepare image + prompt batches
llm_inputs = []
image_files = []

print(f"Processing images in directory to llm_inputs...")

# Get all files in the directory (no subdirectories)
for file in tqdm(os.listdir(image_dir), desc="Processing images"):
    if file.lower().endswith('.jpg'):
        image_path = os.path.join(image_dir, file)
        try:
            image = Image.open(image_path).convert("RGB")
            
            messages = [
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
            image_files.append(file)
            processed_count += 1
        except Exception as e:
            print(f"Warning: Failed to process image {file}: {e}")
            continue
        

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
print(f"Generating captions for {len(image_files)} images...")
outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

# Match outputs with image files
captions_dict = {}
for image_file, output in zip(image_files, outputs):
    try:
        image_id = image_file.split('.')[0]
        text = output.outputs[0].text.strip()
        captions_dict[image_id] = text
    except Exception as e:
        print(f"Warning: Failed to extract output for image {image_file}: {e}")

# Save to JSON
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(captions_dict, f, indent=2, ensure_ascii=False)

print(f"Saved {len(captions_dict)} captions to {output_json}")