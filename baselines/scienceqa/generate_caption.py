from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
from PIL import Image
import os
import json
from tqdm import tqdm


with open("/fs/archive/share/mm_datasets/ScienceQA/problems.json", "r") as f:
    problems = json.load(f)