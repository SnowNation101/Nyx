import json
import os

IMAGE_TOKEN = "<|image|>"
PHI_IMAGE_TOKEN = "<|image_1|>"

with open("process_obelics/obelics_hardneg.json", "r") as f:
    dataset = json.load(f)

for item in dataset:
    item["qry"] = item["qry"].replace(IMAGE_TOKEN, PHI_IMAGE_TOKEN)
    item["pos_text"] = item["pos_text"].replace(IMAGE_TOKEN, PHI_IMAGE_TOKEN)
    item["neg_text"] = [neg_text.replace(IMAGE_TOKEN, PHI_IMAGE_TOKEN) for neg_text in item["neg_text"]]
    item["qry_image_path"] = [os.path.join("images/OBELICS", path) for path in item["qry_image_path"]]
    item["pos_image_path"] = [os.path.join("images/OBELICS", path) for path in item["pos_image_path"]]
    item["neg_image_path"] = [[os.path.join("images/OBELICS", path) for path in neg_image] for neg_image in item["neg_image_path"]]

with open("process_obelics/mm_dataset.json", "w") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=True)