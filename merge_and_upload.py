from peft import PeftModel
import torch
from transformers import Qwen2_5_VLModel, Qwen2_5_VLProcessor


base_model_path = "/fs/archive/share/Qwen2.5-VL-3B-Instruct"
processor = Qwen2_5_VLProcessor.from_pretrained(base_model_path)

# ------------------------------------------------------------------------------

ckpt_path = "./checkpoint/ft_2025-07-10-2228.30"
save_path = "./Nyx-3B-Pretrained"

base_model = Qwen2_5_VLModel.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ckpt_path, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

model.push_to_hub("SnowNation/Nyx-3B-Pretrained")
processor.push_to_hub("SnowNation/Nyx-3B-Pretrained")

# ------------------------------------------------------------------------------

ckpt_path = "./checkpoint/ft_2025-07-29-2341.31"
save_path = "./Nyx-3B-Feedback"

base_model = Qwen2_5_VLModel.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ckpt_path, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

model.save_pretrained(save_path)
processor.save_pretrained(save_path)

model.push_to_hub("SnowNation/Nyx-3B-Feedback")
processor.push_to_hub("SnowNation/Nyx-3B-Feedback")

# ------------------------------------------------------------------------------

ckpt_path = "./checkpoint/ft_2025-09-19-1126.05"
save_path = "./Nyx-3B-MMEB-mmE5"

base_model = Qwen2_5_VLModel.from_pretrained(
    base_model_path,
    dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)
model = PeftModel.from_pretrained(base_model, ckpt_path, torch_dtype=torch.bfloat16)
model = model.merge_and_unload()

model.save_pretrained(save_path)
processor.save_pretrained(save_path)