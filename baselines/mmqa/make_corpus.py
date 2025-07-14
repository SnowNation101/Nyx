import json
from pathlib import Path
from tabulate import tabulate
from PIL import Image

def convert_mmqa_table_to_string(data):
    headers = [col.get('column_name', '') for col in data['table']['header']]
    rows = [[cell.get('text', '') for cell in row] for row in data['table']['table_rows']]
    return tabulate(rows, headers=headers, tablefmt='grid')


def load_jsonl_file(file_path, formatter):
    results = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            result = formatter(data)
            if result is not None:
                results.append(result)
    return results


images_dir = Path("/fs/archive/share/mm_datasets/downloads/MMQA/final_dataset_images")
def process_image_entry(d):
    image_file = images_dir / d["path"]
    try:
        with Image.open(image_file) as img:
            img.verify()
            width, height = img.size
            if width * height > 16384 * 28 * 28:
                return None
        return {
            "text": f"{d['title']}\n<|image|>",
            "image": d["path"]
        }
    except Exception:
        print(f"Invalid image file: {image_file}")
        return None


def main():
    root_dir = Path("/fs/archive/share/mm_datasets/MMQA/MMQA")
    output_path = Path("mmqa_corpus.json")
    dataset = []


    image_path = root_dir / "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_images.jsonl"
    dataset += load_jsonl_file(image_path, lambda d: process_image_entry(d))
    
    table_path = root_dir / "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_tables.jsonl"
    dataset += load_jsonl_file(table_path, lambda d: {
        "text": f"{d['title']}\n{convert_mmqa_table_to_string(d)}",
        "image": ""
    })

    text_path = root_dir / "multimodalqa_final_dataset_pipeline_camera_ready_MMQA_texts.jsonl"
    dataset += load_jsonl_file(text_path, lambda d: {
        "text": f"{d['title']}\n{d['text']}",
        "image": ""
    })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)



if __name__ == "__main__":
    main()
