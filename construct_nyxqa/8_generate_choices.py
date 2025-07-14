import os
import json
import math
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from PIL import Image,ImageFile


os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = True

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
    images_dir = "/fs/archive/share/mm_datasets/obelics_images"
    generated_qa_path = "all_data/qa_flattened.json"

    with open(generated_qa_path, "r") as f:
        generated_qa = json.load(f)

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 10, "video": 0},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        tensor_parallel_size=4,
        seed=42,
        # gpu_memory_utilization=0.7
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

    for data in generated_qa:
        doc = data["pos_text"].replace("<|image|>", "<image>")
        images = []
        for image_path in data['pos_image_path']:
            image = Image.open(os.path.join(images_dir, image_path))
            new_h, new_w = smart_resize(image.height, image.width)
            image = image.resize((new_w, new_h))
            images.append(image)

        for image_path in data['qry_image_path']:
            image = Image.open(os.path.join(images_dir, image_path))
            new_h, new_w = smart_resize(image.height, image.width)
            image = image.resize((new_w, new_h))
            images.append(image)

        qry = data["qry"].replace("<|image|>", "<image>")
        ans = data["ans"]

        user_prompt = f"""Given a document and a corresponding generated question + correct answer pair, create 3 plausible but incorrect options to form a multiple-choice question (4 options total). Place the correct answer at a random position and output in the format: ["option1", "option2", "option3", "option4"].

        Requirements:
        1. Distractors: Avoid obvious or absurd choices (e.g., "eating a sandwich" for a political speech).
        2. Positioning: Randomize the correct answer’s location (e.g., not always Option A).
        3. Output in the right format.
        """

        messages = [{"role": "user", "content": f"{user_prompt}\nDocument:{doc}\nQuestion:{qry}\nRight answer:{ans}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_entry = {"prompt": prompt}
        if images:
            input_entry["multi_modal_data"] = {"image": images}

        llm_inputs.append(input_entry)

    batch_outputs = vlm.generate(llm_inputs, sampling_params=sampling_params)

    results = []
    for data, output in zip(generated_qa, batch_outputs):
        
        results.append({
            "qry": data["qry"],
            "ans": data["ans"],
            "choices": output.outputs[0].text,
            "pos_image_path": data["pos_image_path"],
            "qry_image_path": data["qry_image_path"],
            "img_num": data["img_num"]
        })

    os.makedirs("all_data", exist_ok=True)
    with open("all_data/generated_choices.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()