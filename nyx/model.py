import torch
import torch.distributed as dist
import random
import torch.nn.functional as F

from typing import Dict, Optional
from torch import nn, Tensor
from transformers import (
    PreTrainedModel, AutoConfig,
    AutoModelForCausalLM,
    MllamaForConditionalGeneration, 
    LlavaNextForConditionalGeneration,
    Qwen2_5_VLModel,

    )
from peft import LoraConfig, get_peft_model, PeftModel

from nyx.arguments import ModelArguments, TrainingArguments

import os

class MMEBModel(nn.Module):
    def __init__(self,
                 encoder: PreTrainedModel,
                 pooling: str = 'cls',
                 normalize: bool = False,
                 temperature: float = 1.0,
                 training_args: TrainingArguments = None,
                 model_args: ModelArguments = None,
                 ):
        super().__init__()
        self.config = encoder.config
        if hasattr(self.config, 'hidden_size'):
            self.hidden_size = self.config.hidden_size
        else:
            self.hidden_size = self.config.text_config.hidden_size
        self.encoder = encoder
        self.pooling = pooling
        self.normalize = normalize
        self.temperature = temperature
        self.cross_entropy = nn.CrossEntropyLoss(reduction='mean')
        self.is_ddp = dist.is_initialized()
        self.training_args = training_args
        self.model_args = model_args

        self.use_mrl = True
        self.matryoshka_dims = [2048, 1024, 512, 256]
        self.matryoshka_weights = [1, 1, 0.2, 0.2]
        self.n_dims_per_step = -1


        print(f"DDP: {self.is_ddp}")
        if self.is_ddp:
            self.process_rank = dist.get_rank()
            self.world_size = dist.get_world_size()
            print(f"Process rank: {self.process_rank}, World size: {self.world_size}")

    def encode_inputs(self, inputs):
        hidden_states = self.encoder(
            **inputs, return_dict=True, 
            output_hidden_states=True
            )
        hidden_states = hidden_states.hidden_states[-1]
        pooled_output = self._pooling(hidden_states, inputs['attention_mask'])
        return pooled_output

    def _pooling(self, last_hidden_state, attention_mask):
        if self.pooling == 'last' or self.pooling == 'eos':
            sequence_lengths = attention_mask.sum(dim=1) - 1
            batch_size = last_hidden_state.shape[0]
            reps = last_hidden_state[
                    torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
        else:
            raise NotImplementedError
        if self.normalize:
            reps = F.normalize(reps, p=2, dim=-1)
        return reps

    def gradient_checkpointing_enable(self, gradient_checkpointing_kwargs: Optional[Dict] = None):
        gradient_checkpointing_kwargs={'use_reentrant': False}
        if self.training_args.bf16:
            self.encoder.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)
        else:
            self.encoder.base_model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs)

    @classmethod
    def build(cls, model_args: ModelArguments, training_args: TrainingArguments, **hf_kwargs):
        # Loading the base model
        force_download = dist.get_world_size() != 1
        config = AutoConfig.from_pretrained(
            model_args.model_name,
            trust_remote_code=True,
            force_download=force_download)
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        elif hasattr(config, 'text_config'):
            config.text_config.use_cache = False

        config.padding_side = "right"

        if model_args.model_backbone == "llava_next":
            config.use_cache = False
            config.padding_side = "left"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                model_args.model_name,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            base_model.padding_side = "left"
        elif model_args.model_backbone == "phi35v":
            config._attn_implementation = "eager"
            base_model = AutoModelForCausalLM.from_pretrained(
                model_args.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
            )
        elif model_args.model_backbone == "mllama":
            base_model = MllamaForConditionalGeneration.from_pretrained(
                model_args.model_name, **hf_kwargs, config=config, 
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            base_model.padding_side = "right"
        elif model_args.model_backbone == "qwen2_5_vl":
            base_model = Qwen2_5_VLModel.from_pretrained(
                model_args.model_name, 
                config=config,
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )

        if hasattr(base_model.config, 'text_config'):
            base_model.config.hidden_size = base_model.config.text_config.hidden_size
            base_model.config.text_config.use_cache = False

        if model_args.lora:
            lora_config = LoraConfig(
                r=model_args.lora_r,
                lora_alpha=model_args.lora_alpha,
                target_modules=model_args.lora_target_modules.split(','),
                lora_dropout=model_args.lora_dropout,
                init_lora_weights="gaussian",
                use_dora=True,
                inference_mode=False
            )
            lora_model = get_peft_model(base_model, lora_config)
            model = cls(
                encoder=lora_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                training_args=training_args,
                model_args=model_args
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                temperature=model_args.temperature,
                training_args=training_args,
                model_args=model_args
            )
        
        if training_args.gradient_checkpointing:
            base_model.enable_input_require_grads()
        
        return model

    @classmethod
    def load(cls, model_args: ModelArguments, **hf_kwargs):
        # Loading the base model
        config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
        if hasattr(config, 'use_cache'):
            config.use_cache = False
        config.padding_side = "right"

        checkpoint_path = model_args.checkpoint_path if model_args.checkpoint_path else model_args.model_name
        
        if model_args.model_backbone == "llava_next":
            config.use_cache = False
            config.padding_side = "right"
            base_model = LlavaNextForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True
            )
            base_model.padding_side = "right"
        elif model_args.model_backbone == "phi35v":
            # Loading the base model
            config = AutoConfig.from_pretrained(model_args.model_name, trust_remote_code=True)
            config.use_cache = False
            config.padding_side = "right"
            config._attn_implementation = "eager"
            base_model = AutoModelForCausalLM.from_pretrained(model_args.model_name, **hf_kwargs, config=config,
                                                          torch_dtype=torch.bfloat16, trust_remote_code=True)
            base_model.padding_side = "right"
        elif model_args.model_backbone == "mllama":
            base_model = MllamaForConditionalGeneration.from_pretrained(
                checkpoint_path, **hf_kwargs, config=config, 
                attn_implementation="sdpa",
                torch_dtype=torch.bfloat16, 
                trust_remote_code=True
            )
            base_model.padding_side = "right"
        elif model_args.model_backbone == "qwen2_5_vl":
            base_model = Qwen2_5_VLModel.from_pretrained(
                model_args.model_name, 
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
                trust_remote_code=True
            )

        # Building the model on top of the base
        if model_args.lora:
            lora_config = LoraConfig.from_pretrained(checkpoint_path)
            lora_model = PeftModel.from_pretrained(base_model, checkpoint_path, config=lora_config)
            
            merged_model = lora_model.merge_and_unload()
            model = cls(
                encoder=merged_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                model_args=model_args
            )
        else:
            model = cls(
                encoder=base_model,
                pooling=model_args.pooling,
                normalize=model_args.normalize,
                model_args=model_args
            )
        return model

    def save(self, output_dir: str):
        self.encoder.save_pretrained(output_dir)

    def _shrink(self, tensor: Tensor, dim: int) -> Tensor:
        tensor_dim = tensor.shape[-1]
        if dim > tensor_dim:
            raise ValueError(
                f"Dimension {dim} in matryoshka_dims cannot exceed embedding dim {tensor_dim}"
            )
        return F.normalize(tensor[..., :dim], p=2, dim=-1)

    def forward(
        self,
        qry: Dict[str, Tensor] = None,
        tgt: Dict[str, Tensor] = None,
        neg: Dict[str, Tensor] = None
    ):
        """
        Forward function for contrastive learning with optional Matryoshka Representation Learning (MRL).

        Args:
            qry (Dict[str, Tensor]): Query input features.
            tgt (Dict[str, Tensor]): Target (positive) input features.
            neg (Dict[str, Tensor], optional): Negative input features.

        Returns:
            loss (Tensor): Computed contrastive loss.
        """
        

        # Encode inputs into dense representations
        qry_reps = self.encode_inputs(qry) if qry else None
        tgt_reps = self.encode_inputs(tgt) if tgt else None
        neg_reps = self.encode_inputs(neg) if neg else None

        if qry_reps is None or tgt_reps is None:
            return {"qry_reps": qry_reps, "tgt_reps": tgt_reps}

        # Gather tensors across devices (if using DistributedDataParallel)
        if self.is_ddp:
            qry_reps = self._dist_gather_tensor(qry_reps)
            tgt_reps = self._dist_gather_tensor(tgt_reps)
            neg_reps = self._dist_gather_tensor(neg_reps) if neg_reps is not None else None

        # ---------- Matryoshka Representation Learning ----------
        if getattr(self, "use_mrl", False):
            dims = getattr(self, "matryoshka_dims", [])
            weights = getattr(self, "matryoshka_weights", None)
            n_dims_per_step = getattr(self, "n_dims_per_step", -1)

            if weights is None:
                weights = [1.0] * len(dims)

            if len(weights) != len(dims):
                raise ValueError("Length of matryoshka_weights must match matryoshka_dims.")

            dim_indices = range(len(dims))
            if 0 < n_dims_per_step < len(dims):
                dim_indices = sorted(random.sample(dim_indices, n_dims_per_step))

            total_loss = 0.0

            for idx in dim_indices:
                dim = dims[idx]
                weight = weights[idx]

                qry_slice = self._shrink(qry_reps, dim)
                tgt_slice = self._shrink(tgt_reps, dim)
                neg_slice = self._shrink(neg_reps, dim) if neg_reps is not None else None

                # Positive pair scores
                pos_scores = self.compute_similarity(qry_slice, tgt_slice)
                pos_scores = pos_scores.view(qry_slice.size(0), -1)
                scores = pos_scores.clone()

                # Negative scores
                if neg_slice is not None:
                    batch_size = qry_slice.size(0)
                    neg_ratio = int(neg_slice.shape[0] / batch_size)
                    neg_scores = torch.sum(
                        qry_slice.unsqueeze(1) * neg_slice.view(batch_size, neg_ratio, -1), dim=-1
                    )
                    scores = torch.cat([pos_scores, neg_scores], dim=1)

                # Compute target and loss
                target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
                target = target * (qry_slice.size(0) // tgt_slice.size(0))
                slice_loss = self.cross_entropy(scores / self.temperature, target)
                total_loss += weight * slice_loss
                
            return total_loss

        # ---------- Standard Contrastive Learning ----------
        pos_scores = self.compute_similarity(qry_reps, tgt_reps)
        pos_scores = pos_scores.view(qry_reps.size(0), -1)
        scores = pos_scores.clone()

        if neg_reps is not None:
            batch_size = qry_reps.size(0)
            neg_ratio = int(neg_reps.shape[0] / batch_size)
            neg_scores = torch.sum(
                qry_reps.unsqueeze(1) * neg_reps.view(batch_size, neg_ratio, -1), dim=-1
            )
            scores = torch.cat([pos_scores, neg_scores], dim=1)

        target = torch.arange(scores.size(0), device=scores.device, dtype=torch.long)
        target = target * (qry_reps.size(0) // tgt_reps.size(0))
        return self.cross_entropy(scores / self.temperature, target)


    def _dist_gather_tensor(self, t: Tensor):
        t = t.contiguous()
        all_tensors = [torch.empty_like(t) for _ in range(self.world_size)]
        dist.all_gather(all_tensors, t)
        all_tensors[self.process_rank] = t
        all_tensors = torch.cat(all_tensors, dim=0)
        return all_tensors

    def compute_similarity(self, q_reps, p_reps):
        return torch.matmul(q_reps, p_reps.transpose(0, 1))
