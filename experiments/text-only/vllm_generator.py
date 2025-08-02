from typing import List, Optional
from vllm import LLM, EngineArgs, SamplingParams
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
from dataclasses import asdict


class MMGenerator:
    def __init__(self, 
                 model_path: str, 
                 max_model_len: int = 32768, 
                 seed: int = 42):
        self.engine_args = EngineArgs(
            model=model_path,
            trust_remote_code=True,
            max_model_len=max_model_len,
            limit_mm_per_prompt={"image": 5, "video": 0},
            mm_processor_kwargs={"max_dynamic_patch": 4},
            seed=seed,
        )

        self.processor = AutoProcessor.from_pretrained(model_path, use_fast=True)
        self.vlm = LLM(**asdict(self.engine_args))


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

        llm_inputs = {
            "prompt": prompt,
            "multi_modal_data": {"image": image_inputs},
        }

        sampling_params = SamplingParams(
            temperature=0.1,
            top_p=0.001,
            repetition_penalty=1.05,
            max_tokens=4096,
            stop_token_ids=[]
        )

        outputs = self.vlm.generate([llm_inputs], sampling_params=sampling_params)
        return outputs[0].outputs[0].text.strip()