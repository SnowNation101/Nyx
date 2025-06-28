from transformers import AutoProcessor
from transformers import Qwen2_5_VLForConditionalGeneration
import torch
from PIL import Image
import json
import os
import sys
from qwen_vl_utils import process_vision_info

from load_aokvqa import get_aokvqa_data

model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"


vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

train_dataset, val_dataset = get_aokvqa_data()

outputs = []
for data in train_dataset[:5]:
    image_path = data['image_path']
    question = data['question']
    answers = data['direct_answers']

    unique_answers = list(set(answers))
    merged_answer = "Possible Answers: " + ", ".join(unique_answers)

    rationales = data['rationales']
    merged_rationale = "Rationale: " + " ".join(rationales)

    message = [
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": (
                    "Generate a coherent and informative document that integrates the provided question, answer, and rationale. "
                    "The document MUST explicitly include the image using the placeholder '<|image|>'. "
                    "You may insert the image at the beginning, in the middle, or at the end of the document. "
                    "However, do not treat the image as a standalone object to be described. Instead, make sure that the image and the text are meaningfully connected — "
                    "for example, the image can provide visual support for the reasoning, or be referenced when elaborating on the answer. "
                    "The goal is to create a well-structured and unified document where the image and the text complement each other."

                    "\n\nExample:\n"
                    "A woman stands at a podium addressing a crowd, as seen in <|image|>. She appears confident and well-prepared, likely delivering an important speech. "
                    "Given the presence of microphones and the formal setting, she might be a political figure or a keynote speaker at a public event. "
                    "This visual context helps support the answer that she is giving a speech, and the rationale that her body language and setting suggest a formal, intentional communication."
                )
            },
            {"type": "text", "text": "Question: "},
            {"type": "image", "image": Image.open(image_path)},
            {"type": "text", "text": question},
            {"type": "text", "text": merged_answer},
            {"type": "text", "text": merged_rationale},
        ]
    }
]


    text = processor.apply_chat_template(
        message, tokenize=False, add_generation_prompt=True,
    )
    print(text, "\n\n")


    image_inputs, _ = process_vision_info(message)

    inputs = processor(
        text=text,
        images=image_inputs,
        return_tensors="pt",
        padding=True,
    ).to("cuda")

    generated_ids = vlm.generate(
        **inputs,
        max_new_tokens=4096,
    )
    generated_ids_trimmed = [
        out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_texts = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )[0]

    outputs.append({
        "image_path": image_path,
        "question": question,
        "answers": unique_answers,
        "rationales": rationales,
        "doc": output_texts
    })

    print("Generated Document:")
    print(output_texts, "\n")
    print("Original Question:", question)
    print("Original Answers:", unique_answers)
    print("Original Rationale:", rationales)

    print("==============================\n")

with open("generated_docs.json", "w") as f:
    json.dump(outputs, f, indent=4)