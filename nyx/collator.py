import logging
import torch

from PIL import ImageFile
from typing import List
from dataclasses import dataclass
from transformers import ProcessorMixin

from nyx.arguments import DataArguments, ModelArguments
from qwen_vl_utils import process_vision_info


ImageFile.LOAD_TRUNCATED_IMAGES = True

logger = logging.getLogger(__name__)

@dataclass
class TrainCollator:
    data_args: DataArguments
    model_args: ModelArguments
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
            # print(example)
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
            max_length=self.data_args.max_len,
            padding=True,
            truncation=True,
        )
        return inputs
    
    
@dataclass
class EvalCollator:
    data_args: DataArguments
    model_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, image_sizes = [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image is None:
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=None, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, None, return_tensors="pt", max_length=self.data_args.max_len,
                                            truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                image_exist = True
                if self.model_args.model_backbone == "llava_next":
                    inputs = self.processor(images=image, text=text, return_tensors="pt")
                else:
                    inputs = self.processor(text, [image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                image_sizes.append(inputs['image_sizes'])

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            if self.model_args.model_backbone == "llava_next":
                pixel_values_shape = list(set(v.shape for v in pixel_values if v is not None))[0]
                pixel_values_list = [v if v is not None else torch.zeros(pixel_values_shape) for v in pixel_values]
                pixel_values = torch.cat(pixel_values_list, dim=0)
            else:
                pixel_values = torch.cat(pixel_values, dim=0)
            if self.model_args.model_backbone == "llava_next":
                image_sizes_shape = list(set(v.shape for v in image_sizes if v is not None))[0]
                image_sizes = [v if v is not None else torch.ones(image_sizes_shape) for v in image_sizes]
            image_sizes = torch.cat(image_sizes, dim=0)
            
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'image_sizes': image_sizes,
            }

        return inputs


def first_non_int_element(lst):
    for element in lst:
        if not isinstance(element, int):
            return element
    return None

def convert_zero_tensor(tensor_list, need_zero=False, seq_len=None):
    if not tensor_list:
        raise ValueError("The tensor_list is empty. Cannot infer tensor properties.")
    
    first_tensor = first_non_int_element(tensor_list)
    tensor_shape = first_tensor.shape
    if seq_len is not None:
        tensor_shape = torch.Size([seq_len, *tensor_shape[1:]])
    dtype = first_tensor.dtype
    device = first_tensor.device

    if need_zero:
        zero_tensor = torch.zeros(tensor_shape, dtype=dtype, device=device)
    else:
        zero_tensor = torch.ones(tensor_shape, dtype=dtype, device=device)
    
    return zero_tensor


@dataclass
class LlamaCollator:
    data_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        qry_inputs = self._get_batch_inputs(examples, 0, 1)
        pos_inputs = self._get_batch_inputs(examples, 2, 3)
        neg_inputs = self._get_batch_inputs(examples, 4, 5)

        return qry_inputs, pos_inputs, neg_inputs
    
    def _get_batch_inputs(self, examples, text_idx, image_idx):
        text = []
        images = []
        for example in examples:
            text, image = example[text_idx], example[image_idx]
            text.append(text)
            if isinstance(image, List):
                images.extend(image)
            else:
                images.append(image)
        inputs = self.processor(
            text=text, 
            images=[image], 
            return_tensors="pt", 
            max_length=self.data_args.max_len, 
            truncation=True)
        return inputs

@dataclass
class LlamaEvalCollator:
    data_args: DataArguments
    model_args: DataArguments
    processor: ProcessorMixin

    def __call__(self, examples):
        """
        :param examples: qry, qry_image, pos_text, pos_image
        """
        inputs = self._get_batch_inputs(examples)
        return inputs

    def _get_batch_inputs(self, examples):
        input_ids, pixel_values, aspect_ratio_ids, aspect_ratio_mask, batch_cross_attention_mask = [], [], [], [], []
        image_exist = False
        for example in examples:
            text, image = example
            if image == None:
                text = text.replace("<|image_1|>\n", "<|begin_of_text|>")
            else:
                text = text.replace("<|image_1|>\n", "<|image|><|begin_of_text|>")
            if image is None:
                inputs = self.processor(text=text, images=None, return_tensors="pt", max_length=self.data_args.max_len,
                                        truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
            else:
                image_exist = True
                inputs = self.processor(text=text, images=[image], return_tensors="pt", max_length=self.data_args.max_len, truncation=True)
                input_ids.append(inputs["input_ids"].squeeze(0).unsqueeze(1))
                pixel_values.append(inputs['pixel_values'])
                aspect_ratio_ids.append(inputs["aspect_ratio_ids"])
                aspect_ratio_mask.append(inputs["aspect_ratio_mask"])
                batch_cross_attention_mask.append(inputs["cross_attention_mask"].squeeze(0))

        input_ids = torch._C._nn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.processor.tokenizer.pad_token_id
        ).squeeze(2)
        attention_mask = input_ids.ne(self.processor.tokenizer.pad_token_id)

        if not image_exist:
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
            }
        else:
            pixel_values = torch.cat(pixel_values, dim=0)
            aspect_ratio_ids = torch.cat(aspect_ratio_ids, dim=0)
            aspect_ratio_mask = torch.cat(aspect_ratio_mask, dim=0)
            cross_attention_mask = torch._C._nn.pad_sequence(
                batch_cross_attention_mask, batch_first=True, padding_value=0
            )
            inputs = {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'pixel_values': pixel_values,
                'aspect_ratio_ids': aspect_ratio_ids,
                'aspect_ratio_mask': aspect_ratio_mask,
                'cross_attention_mask': cross_attention_mask,
            }

        return inputs
