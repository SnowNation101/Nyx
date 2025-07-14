import os
import json
import math
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams
from PIL import Image,ImageFile


os.environ["TOKENIZERS_PARALLELISM"] = "false"
ImageFile.LOAD_TRUNCATED_IMAGES = False

TEXT_PROMPT = f"""You are given a text and some QA pairs based on the text. Please shorten the answer in each of the QA pairs to no more than 5 words, while ensuring the accuracy and conciseness of the answer. 

Requirements:
1、Retain the core information and remove redundant descriptions.
2、If the original answer exceeds 5 words, prioritize retaining key nouns or phrases.
3、Ensure the shortened answer still directly answers the corresponding question.
4、If the question cannot be answered without contextual information or the answer cannot be shortened without losing critical information, delete the entire QA pair.

Example:
Text:
", they broke the law, but it\u2019s not a felony. It\u2019s an act of love. It\u2019s an act of commitment to your family. I honestly think that that is a different kind of crime that there should be a price paid, but it shouldn\u2019t rile people up that people are actually coming to this country to provide for their families.\u201d\n\n21 thoughts on \u201cUnethical Quote of the Month: Jeb Bush\u201d"

Shortened QA:
Q: What type of crime does Jeb Bush describe as being committed by people coming to the country to provide for their families?
Original Answer: Jeb Bush describes it as an act of love and commitment to family, not a felony.
Shortened Answer: Love and commitment to family

Deleted QA:
Q: What does Jeb Bush's comments mainly express?
A: Jeb Bush's comments mainly express a sympathetic and compassionate view toward undocumented immigrants who enter the country illegally to support their families.
(The question cannot be answered without contextual information: Jeb Bush's comments)

Output format:
[Q1:…,A1:…],[Q2:…,A2:…]…
"""

MULTIMODAL_PROMPT = f"""You are given a document containing text and images and some QA pairs based on the document. Please shorten the answer in each of the QA pairs to no more than 5 words, while ensuring the accuracy and conciseness of the answer. 

Requirements:
1、Retain the core information and remove redundant descriptions.
2、If the original answer exceeds 5 words, prioritize retaining key nouns or phrases.
3、Ensure the shortened answer still directly answers the corresponding question.
4、If the question cannot be answered without contextual information or the answer cannot be shortened without losing critical information, delete the entire QA pair.

Example:
Document:
"<|image|>The statement by Jeb Bush has its sunny side, I suppose: with any luck, it should ensure that we don\u2019t have a Bush-Clinton contest in 2016. Maybe that was Jeb\u2019s intent. Otherwise, his comments are irresponsible attacks on the rule of law, common sense, fairness and national sovereignty.\n\n\u201cThere are means by which we can control our border better than we have. And there should be penalties for breaking the law.But the way I look at this \u2014 and I\u2019m going to say this, and it\u2019ll be on tape and so be it. The way I look at this is someone who comes to our country because they couldn\u2019t come legally, they come to our country because their families \u2014 the dad who loved their children \u2014 was worried that their children didn\u2019t have food on the table. And they wanted to make sure their family was intact, and they crossed the border because they had no other means to work to be able to provide for their family. Yes”
Shortened QA:
Q: Based on the image, <image1>, what is Jeb Bush doing in the image?
Original Answer: Jeb Bush is speaking at a podium.
Shortened Answer: Speaking at a podium

Deleted QA:
Q: What is the main concern expressed about Jeb Bush's comments?
A: The main concern is that his comments are irresponsible attacks on the rule of law, common sense, fairness, and national sovereignty.
(The question cannot be answered without contextual information: Jeb Bush's comments)

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
    images_dir = "/fs/archive/share/mm_datasets/obelics_images"
    generated_qa_path = "./all_data/generated_qa.json"

    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    
    with open(generated_qa_path, "r") as f:
        generated_qa = json.load(f)

    # dataset = dataset[:50]  # For testing purposes
    # generated_qa = generated_qa[:50]

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 5, "video": 0},
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
    raw_metadata = []
    for idx,data in enumerate(dataset):
        doc = data["text"].replace("<|image|>", "<image>")
        images = []
        for image_path in data['images']:
            image = Image.open(os.path.join(images_dir, image_path))
            new_h, new_w = smart_resize(image.height, image.width)
            image = image.resize((new_w, new_h))
            images.append(image)

        qa = generated_qa[idx]["generated_qa"] # str
        if images:
            user_prompt = MULTIMODAL_PROMPT
        else:
            user_prompt = TEXT_PROMPT

        messages = [{"role": "user", "content": f"{user_prompt}\n{doc}\nQA:{qa}"}]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        input_entry = {"prompt": prompt}
        if images:
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

    os.makedirs("all_data", exist_ok=True)
    with open("all_data/simplified_qa.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()