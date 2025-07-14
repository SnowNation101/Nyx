import json
import random
import os
import shutil
from tqdm import tqdm


with open("/fs/archive/share/u2024001042/nyxqa/construct_nyxqa/all_data/choices_dataset.json", "r") as f:
    all_data = json.load(f)

text_only = []
single_img = []
multi_img = []

for item in tqdm(all_data, desc="Grouping items"):
    if item["img_num"] == 0:
        text_only.append(item)
    elif item["img_num"] == 1:
        single_img.append(item)
    else:
        multi_img.append(item)


random.seed(42)
random.shuffle(text_only)
random.shuffle(single_img)
random.shuffle(multi_img)

def split_data(group, num_dev, num_test):
    dev = group[:num_dev]
    test = group[num_dev:num_dev + num_test]
    train = group[num_dev + num_test:]
    return train, dev, test

text_train, text_dev, text_test = split_data(text_only, 200, 200)
single_train, single_dev, single_test = split_data(single_img, 200, 200)
multi_train, multi_dev, multi_test = split_data(multi_img, 200, 200)


train_set = text_train + single_train + multi_train
dev_set = text_dev + single_dev + multi_dev
test_set = text_test + single_test + multi_test


random.shuffle(train_set)

output_dir = "./NyxQA"
os.makedirs(output_dir, exist_ok=True)

# with open(os.path.join(output_dir, "train.json"), "w") as f:
#     json.dump(train_set, f, indent=2, ensure_ascii=False)

# with open(os.path.join(output_dir, "dev.json"), "w") as f:
#     json.dump(dev_set, f, indent=2, ensure_ascii=False)

# with open(os.path.join(output_dir, "test.json"), "w") as f:
#     json.dump(test_set, f, indent=2, ensure_ascii=False)

# print(f"Split completed. Train: {len(train_set)}, Dev: {len(dev_set)}, Test: {len(test_set)}")

with open("all_data/obelics_verified_image_dataset.json", "r") as f:
    corpus_data = json.load(f)

with open(os.path.join(output_dir, "corpus.json"), "w") as f:
    json.dump(corpus_data, f, indent=2, ensure_ascii=False)

src = "all_data/obelics_verified_image_dataset.json"
dst = os.path.join(output_dir, "corpus.json")
os.makedirs(os.path.dirname(dst), exist_ok=True)
shutil.copy(src, dst)