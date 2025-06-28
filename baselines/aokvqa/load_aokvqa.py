import os
import json
import random

coco_dir = "/fs/archive/share/mm_datasets/MSCOCO2017"
aokvqa_dir = "/fs/archive/share/mm_datasets/A-OKVQA"


def load_aokvqa(aokvqa_dir, split, version='v1p0'):
    assert split in ['train', 'val', 'test', 'test_w_ans']
    dataset = json.load(open(
        os.path.join(aokvqa_dir, f"aokvqa_{version}_{split}.json")
    ))
    for data in dataset:
        data.update({
            'image_path': get_coco_path(split, data['image_id'], coco_dir)
        })
    return dataset


def get_coco_path(split, image_id, coco_dir):
    return os.path.join(coco_dir, f"{split}2017", f"{image_id:012}.jpg")


def get_aokvqa_data():
    # Load full training set
    full_train_dataset = load_aokvqa(aokvqa_dir, 'train')
    
    # Set random seed and sample 2000 examples for dev set
    random.seed(42)
    dev_dataset = random.sample(full_train_dataset, 2000)

    # Remove dev examples from training set
    dev_ids = set(id(data) for data in dev_dataset)
    train_dataset = [data for data in full_train_dataset if id(data) not in dev_ids]

    # Load validation set
    val_dataset = load_aokvqa(aokvqa_dir, 'val')

    return train_dataset, dev_dataset, val_dataset


if __name__ == "__main__":
    train_dataset, dev_dataset, val_dataset = get_aokvqa_data()
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")

    print(f"Sample train data: {train_dataset[0]}")
    print(f"Sample dev data: {dev_dataset[0]}")
    print(f"Sample val data: {val_dataset[0]}")

    sample_train_data = train_dataset[0]
    print("Train data fields and their data types:")
    for field, value in sample_train_data.items():
        if isinstance(value, list):
            print(f"{field}: list containing elements of type {type(value[0]) if value else 'unknown (empty list)'}")
        else:
            print(f"{field}: {type(value)}")
