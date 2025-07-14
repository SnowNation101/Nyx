from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os
from typing import List, Optional

class MMGenerator:
    def __init__(self, model_path: str):
        self.vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map="auto",
        )
        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)

    def _build_messages(self, docs: List[str], question: str) -> List[dict]:
        """Constructs the prompt messages for the model."""
        base_system_prompt = (
            "Answer the question based on the given document. "
            "Only give me the answer and do not output any other words.\n"
            "The following are given documents."
        )
        user_content = [
            {"type": "text", "text": f"{doc}\n\n"} for doc in docs
        ] + [{"type": "text", "text": f"Question: {question}"}]

        return [
            {"role": "system", "content": base_system_prompt},
            {"role": "user", "content": user_content},
        ]

    def _prepare_images(self, images: Optional[List[Image.Image]]) -> Optional[List[dict]]:
        """Preprocess images if provided."""
        if not images:
            return None
        return process_vision_info([{"type": "image", "image": img} for img in images])

    def generate(
        self,
        docs: List[str],
        question: str,
        images: Optional[List[Image.Image]] = None
    ) -> str:
        """Generates a response based on documents and optional images."""
        messages = self._build_messages(docs, question)
        prompt = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs = self._prepare_images(images)
        inputs = self.processor(
            text=prompt,
            images=image_inputs,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        generated_ids = self.vlm.generate(
            **inputs,
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.1,
            top_p=0.001,
        )
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        return output_text