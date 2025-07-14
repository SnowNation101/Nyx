import os
import json
import math
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams

from PIL import Image, ImageFile
import pillow_avif
# import avif

os.environ["TOKENIZERS_PARALLELISM"] = "false"

ImageFile.LOAD_TRUNCATED_IMAGES = True

# 1、生成包含多个image的query
# 2、生成的query上下文无关
# 3、生成的answer简短
# 4、问题的格式
MULTIMODAL_PROMPT = f"""You are given a document containing text and images, please analyze the content and raise no more than five questions along with their corresponding answers. Requirement: 
1、The question must be independent of the context, that is, it cannot rely on background information that is not mentioned.
2、Each question should contain at least two images, and you need to clearly indicate them like:“Based on the following images, <image1>, <image2> ...” or “Considering both images, <image1> and <image3>, ...” etc. 
3、Please shorten the answer in each of the QA pairs to no more than 5 words, while ensuring the accuracy and conciseness of the answer. 

Example:
Q: Considering both images, <image2> and <image3>, what is the primary focus of the organizations represented?
A: Agriculture and farmer support.

Output format:
[Q1:…,A1:…],[Q2:…,A2:…]…
"""

IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 1024 * 28 * 28
MAX_RATIO = 200

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor


def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor

def smart_resize(
    height: int, width: int, 
    factor: int = IMAGE_FACTOR, 
    min_pixels: int = MIN_PIXELS, 
    max_pixels: int = MAX_PIXELS
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:
        1. Both dimensions (height and width) are divisible by 'factor'.
        2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
        3. The aspect ratio of the image is maintained as closely as possible.
    """
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar



def main():
    model_path = "/fs/archive/share/InternVL3-78B"
    dataset_path = "./all_data/obelics_10k.json"

    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    print(len(dataset))
    
    multi_image_dataset = []
    for data in dataset:
        if len(data["images"]) > 1:
            multi_image_dataset.append(data)
    with open("all_data/multi_image_dataset.json", "w") as f:
        json.dump(multi_image_dataset, f, indent=4)

    dataset = multi_image_dataset

    images_dir = "/fs/archive/share/mm_datasets/obelics_images"

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 5, "video": 0},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        tensor_parallel_size=4,
        seed=42,
        gpu_memory_utilization=0.8
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    stop_tokens = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|end|>"]
    stop_token_ids = [
        token_id for token in stop_tokens
        if (token_id := tokenizer.convert_tokens_to_ids(token)) is not None
    ]

    vlm = LLM(**asdict(engine_args))
    sampling_params = SamplingParams(
        temperature=0.0, max_tokens=4096, stop_token_ids=stop_token_ids
    )

    llm_inputs = []
    raw_metadata = []
    for data in dataset:
        doc = data["text"].replace("<|image|>", "<image>")
        images = []
        for image_path in data['images']:
            image = Image.open(os.path.join(images_dir, image_path))
            new_h, new_w = smart_resize(image.height, image.width)
            image = image.resize((new_w, new_h))
            images.append(image)
        user_prompt = MULTIMODAL_PROMPT
        messages = [{"role": "user", "content": f"{user_prompt}\n{doc}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_entry = {"prompt": prompt}
        input_entry["multi_modal_data"] = {"image": images}

        llm_inputs.append(input_entry)
        raw_metadata.append({"text": data["text"], "images": data["images"]})

    batch_outputs = vlm.generate(llm_inputs, sampling_params=sampling_params)

    results = []
    for meta, output in zip(raw_metadata, batch_outputs):
        results.append({
            "doc": meta["text"],
            "images": meta["images"],
            "generated_qa": output.outputs[0].text,
        })

    with open("all_data/generated_multi_image_qa.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()