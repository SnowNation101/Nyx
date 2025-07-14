import json
from tqdm import tqdm
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = False

with open("./all_data/obelics_chunked_dataset.json", "r") as f:
    dataset = json.load(f)

filtered_dataset = []
error_count = 0
skipped_count = 0
for data in dataset:
    corrupted = False
    for image in data['images']:
        try:
            image_path = f"/fs/archive/share/mm_datasets/obelics_images/{image}"
            with Image.open(image_path) as img:
                img.verify()  # Check format
            with Image.open(image_path) as img:
                img.load()   # Check decoding (will raise if truncated)
        except Exception as e:
            # print(f"Error with image {image}: {e}")
            error_count += 1
            corrupted = True
            break  # Skip the whole item
    if not corrupted:
        filtered_dataset.append(data)
    else:
        skipped_count += 1
print(f"Total corrupted/truncated images: {error_count}")
print(f"Total skipped items: {skipped_count}")

print(f"dataset length: {len(filtered_dataset)}")

cnt = {
    "total": 0,
    "text_only": 0,
    "single_img": 0,
    "multi_img": 0,
}

for data in tqdm(filtered_dataset, desc="Counting items"):
    img_num = len(data["images"])
    cnt["total"] += 1
    if img_num == 0:
        cnt["text_only"] += 1
    elif img_num == 1:
        cnt["single_img"] += 1
    else:
        cnt["multi_img"] += 1

print("Count Summary:")
print(f"Total items: {cnt['total']}")
print(f"Text only: {cnt['text_only']} ({cnt['text_only'] / cnt['total'] * 100:.2f}%)")
print(f"Single image: {cnt['single_img']} ({cnt['single_img'] / cnt['total'] * 100:.2f}%)")
print(f"Multiple images: {cnt['multi_img']} ({cnt['multi_img'] / cnt['total'] * 100:.2f}%)")


with open("./all_data/obelics_verified_image_dataset.json", "w") as f:
    json.dump(filtered_dataset, f, indent=4)

