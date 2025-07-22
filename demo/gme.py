from transformers import AutoModel, AutoProcessor
from transformers.utils.versions import require_version
from PIL import Image


texts = [
    "The Tesla Cybertruck is a battery electric pickup truck built by Tesla, Inc. since 2023.",
    "Alibaba office.",
]
images = [
    Image.open("/fs/archive/share/mm_datasets/NyxQA/images/111.jpg"),
    Image.open("/fs/archive/share/mm_datasets/NyxQA/images/222.jpg"),
]


gme = AutoModel.from_pretrained(
    "/fs/archive/share/gme-Qwen2-VL-2B-Instruct",
    torch_dtype="float16", device_map='cuda', trust_remote_code=True
)

embedding = gme.get_fused_embeddings(
    texts=texts,
    images=images,
)

print(embedding.shape)