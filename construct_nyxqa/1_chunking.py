from transformers import AutoProcessor
from transformers import PreTrainedTokenizerBase
import json

processor = AutoProcessor.from_pretrained(
    "/fs/archive/share/Qwen2.5-VL-7B-Instruct",
    use_fast=True)
tokenizer: PreTrainedTokenizerBase = processor.tokenizer

special_token = "<|image|>"
if special_token not in tokenizer.get_vocab():
    tokenizer.add_special_tokens({"additional_special_tokens": [special_token]})

def chunk_dataset_by_token_length(dataset, max_tokens=200):
    new_dataset = []

    for entry in dataset:
        text = entry["text"]
        images = entry["images"]

        input_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)

        current_chunk = []
        current_image_count = 0

        for token in tokens:
            current_chunk.append(token)
            if token == special_token:
                current_image_count += 1

            if len(current_chunk) >= max_tokens:
                chunk_text = tokenizer.convert_tokens_to_string(current_chunk)
                chunk_images = images[:current_image_count]
                images = images[current_image_count:]
                new_dataset.append({
                    "text": chunk_text,
                    "images": chunk_images
                })

                current_chunk = []
                current_image_count = 0

    return new_dataset

def main():

    dataset = []
    with open("/fs/archive/share/mm_datasets/obelics_processed.jsonl", "r") as f:
        for line in f:
            data = json.loads(line) 
            dataset.append(data) 
    chunked_dataset = chunk_dataset_by_token_length(dataset,200)
    token_num_less_than_200 = 0
    for data in chunked_dataset:
        text = data["text"]
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        tokens = tokenizer.convert_ids_to_tokens(input_ids)
        if(len(tokens) < 200):
            token_num_less_than_200 += 1
    print("trunked data num:",len(chunked_dataset))
    print("token_num_less_than_200:",token_num_less_than_200)

    chunked_dataset = [data for data in chunked_dataset if len(data['images']) <= 5]
    print("trunked data num:",len(chunked_dataset))
    with open("./all_data/obelics_chunked_dataset.json", "w") as f:
        json.dump(chunked_dataset, f, indent=4)

if __name__ == "__main__":
    main()

