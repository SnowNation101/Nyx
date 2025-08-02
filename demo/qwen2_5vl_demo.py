from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os

path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# default processer
processor = AutoProcessor.from_pretrained(path)

# To avoid flash attn 2 error
processor.tokenizer.padding_side = "left"

# The default range for the number of visual tokens per image in the model is 4-16384.
# You can set min_pixels and max_pixels according to your needs, such as a token range of 256-1280, to balance performance and cost.
# min_pixels = 256*28*28
# max_pixels = 1280*28*28
# processor = AutoProcessor.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", 
#     min_pixels=min_pixels, 
#     max_pixels=max_pixels
#     )

image = Image.open("/fs/archive/share/mm_datasets/GAOKAO-MM/Data/2010-2023_Chemistry_MCQs/2010-2023_Chemistry_MCQs_52_0.png")


# Sample messages for batch inference
messages1 = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "image", "image": image},
            {"type": "text", "text": "What are the common elements in these pictures?"},
        ],
    }
]
messages2 = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Who are you?"},
]
# Combine messages for batch processing
messages = [messages1, messages2]

# Preparation for batch inference
texts = [
    processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    for msg in messages
]

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=texts,
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    truncation=True,
    max_length=3000,
    return_tensors="pt",
)

# Batch Inference
generated_ids = model.generate(**inputs, max_new_tokens=128)
generated_ids_trimmed = [
    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_texts = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)
print(output_texts)

