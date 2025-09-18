import logging

from PIL import ImageFile
from typing import List
from dataclasses import dataclass
from transformers import ProcessorMixin

ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

@dataclass
class TrainCollator:
    processor: ProcessorMixin

    def __call__(self, examples):
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        neg_inputs = self._get_batch_inputs(examples, 4, 5)

        return qry_inputs, pos_inputs, neg_inputs
    
    def _get_batch_inputs(self, examples, text_idx, image_idx):
        texts = []
        images = []
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            if isinstance(text, List):
                for txt in text:
                    # Since text must not be None, we can safely append it
                    texts.append(txt)
            else:
                texts.append(text)
            for img in image:
                if img is not None:
                    images.append(img)

        inputs = self.processor(
            text=texts, 
            images=images if images else None, 
            return_tensors="pt", 
            padding=True,
            truncation=True,
        )
        return inputs
    
    
@dataclass
class EvalCollator:
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        texts = []
        images = []
        for example in examples:
            text, image = example
            text = text.replace("<|image_1|>", "<|vision_start|><|image_pad|><|vision_end|>")
            if isinstance(text, List):
                for txt in text:
                    # Since text must not be None, we can safely append it
                    texts.append(txt)
            else:
                texts.append(text)
            if image is not None:
                images.append(image)


        inputs = self.processor(
            text=texts, 
            images=images if images else None, 
            return_tensors="pt", 
            padding=True,
            truncation=True
        )
        return inputs