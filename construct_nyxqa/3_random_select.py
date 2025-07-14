import json
import random
from tqdm import tqdm

with open("./all_data/obelics_verified_image_dataset.json", "r") as f:
    dataset = json.load(f)

text_only = []
single_img = []
multi_img = []

for data in tqdm(dataset, desc="分类数据"):
    img_num = len(data["images"])
    if img_num == 0:
        text_only.append(data)
    elif img_num == 1:
        single_img.append(data)
    else:
        multi_img.append(data)

total = 10000
text_only_num = int(total * 0.6329)
single_img_num = int(total * 0.2985)
multi_img_num = total - text_only_num - single_img_num

print(f"text_only: {text_only_num}, single_img: {single_img_num},multi_img: {multi_img_num}")


random.seed(42) 

sampled_text = random.sample(text_only, text_only_num)
sampled_single = random.sample(single_img, single_img_num)
sampled_multi = random.sample(multi_img, multi_img_num)


filtered_dataset = sampled_text + sampled_single + sampled_multi
random.shuffle(filtered_dataset) 

cnt = {
    "total": 0,
    "text_only": 0,
    "single_img": 0,
    "multi_img": 0,
}

for data in filtered_dataset:
    img_num = len(data["images"])
    cnt["total"] += 1
    if img_num == 0:
        cnt["text_only"] += 1
    elif img_num == 1:
        cnt["single_img"] += 1
    else:
        cnt["multi_img"] += 1

print(f"Total items: {cnt['total']}")
print(f"Text only: {cnt['text_only']} ({cnt['text_only'] / cnt['total'] * 100:.2f}%)")
print(f"Single image: {cnt['single_img']} ({cnt['single_img'] / cnt['total'] * 100:.2f}%)")
print(f"Multiple images: {cnt['multi_img']} ({cnt['multi_img'] / cnt['total'] * 100:.2f}%)")

output_file = "./all_data/obelics_10k.json"
with open(output_file, "w") as f:
    json.dump(filtered_dataset, f, indent=2)
