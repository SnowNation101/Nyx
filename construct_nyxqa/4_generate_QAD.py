import os
import json
import math
from dataclasses import asdict
from transformers import AutoProcessor, AutoTokenizer
from vllm import LLM, EngineArgs, SamplingParams

from PIL import Image, ImageFile

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Some images in the dataset are truncated,
# importing this to avoid errors when resizing them.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Some images in the dataset are AVIF format,
# which requires the pillow_avif package to be loaded.


TEXT_PROMPT = f"""Given a text, please analyze the content of the text and raise no more than five questions along with their corresponding answers. Requirement: 
1、The question must be independent of the context, that is, it cannot rely on background information that is not mentioned.
2、The questions raised can be answered in concise language.

Example:
Text:
", they broke the law, but it\u2019s not a felony. It\u2019s an act of love. It\u2019s an act of commitment to your family. I honestly think that that is a different kind of crime that there should be a price paid, but it shouldn\u2019t rile people up that people are actually coming to this country to provide for their families.\u201d\n\n21 thoughts on \u201cUnethical Quote of the Month: Jeb Bush\u201d"

Incorrect question:
What does the speaker think about this crime? (Without specifying who the "speaker" is)

Correct question:
What type of crime does Jeb Bush describe as being committed by people coming to the country to provide for their families?

Answer:
Jeb Bush describes it as an act of love and commitment to family, not a felony.

Output format:
[Q1:…,A1:…],[Q2:…,A2:…]…
"""



MULTIMODAL_PROMPT = f"""You are given a document containing text and images, please analyze the content and raise no more than five questions along with their corresponding answers. Requirement: 
1、The question must be independent of the context, that is, it cannot rely on background information that is not mentioned.
2、You can ask questions about the images in the document, but you need to clearly indicate them like:“Based on the image, <image2>, ...” or “Considering both images, <image1> and <image3>, ...” etc. 
3、The questions raised can be answered in concise language.

Example:
Document:
"<|image|>The statement by Jeb Bush has its sunny side, I suppose: with any luck, it should ensure that we don\u2019t have a Bush-Clinton contest in 2016. Maybe that was Jeb\u2019s intent. Otherwise, his comments are irresponsible attacks on the rule of law, common sense, fairness and national sovereignty.\n\n\u201cThere are means by which we can control our border better than we have. And there should be penalties for breaking the law.But the way I look at this \u2014 and I\u2019m going to say this, and it\u2019ll be on tape and so be it. The way I look at this is someone who comes to our country because they couldn\u2019t come legally, they come to our country because their families \u2014 the dad who loved their children \u2014 was worried that their children didn\u2019t have food on the table. And they wanted to make sure their family was intact, and they crossed the border because they had no other means to work to be able to provide for their family. Yes”

Incorrect question:
Considering both the text and <image1>, what might be the context of Jeb Bush's speech? (The question cannot be answered without context)

Correct question:
In the image, <image1>, what might be the context of Jeb Bush's speech? 

Incorrect question:
What is the main concern expressed about Jeb Bush's comments? (Without specifying what the "comments" is)

Correct question:
What is the main concern expressed about Jeb Bush's comments "someone who comes to our country because they couldn\u2019t come legally, they come to our country because their families"?

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

    with open(dataset_path, "r") as f:
        dataset = json.load(f)

    # dataset = dataset[:50]  # For testing purposes

    engine_args = EngineArgs(
        model=model_path,
        trust_remote_code=True,
        max_model_len=32768,
        limit_mm_per_prompt={"image": 5, "video": 0},
        mm_processor_kwargs={"max_dynamic_patch": 4},
        tensor_parallel_size=4,
        seed=42,
        gpu_memory_utilization=0.7
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

        if images:
            user_prompt = MULTIMODAL_PROMPT
        else:
            user_prompt = TEXT_PROMPT

        messages = [{"role": "user", "content": f"{user_prompt}\n{doc}"}]
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

    with open("./all_data/generated_qa.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()