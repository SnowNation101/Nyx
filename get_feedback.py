import torch
import json
import os
import numpy as np
import faiss
from tqdm import tqdm
from PIL import Image
from datasets import load_dataset
from vllm import LLM, SamplingParams

from transformers import Qwen2_5_VLModel, Qwen2_5_VLProcessor
from qwen_vl_utils import process_vision_info

IMAGE_TOKEN = "<|image|>"
QWEN_IMAGE_TOKEN = "<|vision_start|><|image_pad|><|vision_end|>"

retrieve_top_k = 30

retrieval_dir = "outputs/retrieval"
os.makedirs(retrieval_dir, exist_ok=True)
generation_dir = "outputs/generation"
os.makedirs(generation_dir, exist_ok=True)

def process_images(images):
    if not images:
        return None
    pseudo_message = [{
        "content": [{"type": "image", "image": image} for image in images]
    }]
    images, _ = process_vision_info(pseudo_message)
    return images

def last_pooling(last_hidden_state, attention_mask, normalize=True):
    sequence_lengths = attention_mask.sum(dim=1) - 1
    batch_size = last_hidden_state.shape[0]
    reps = last_hidden_state[torch.arange(batch_size, device=last_hidden_state.device), sequence_lengths]
    if normalize:
        reps = torch.nn.functional.normalize(reps, p=2, dim=-1)
    return reps


# ===============RETRIEVAL================
def retrieval():
    text_index = faiss.read_index("baselines/text-only/index/nyx.faiss")
    text_corpus = json.load(open("baselines/text-only/t2t_corpus.json", "r"))

    mmqa_index = faiss.read_index("baselines/mmqa/index/nyx.faiss")
    mmqa_corpus = json.load(open("baselines/mmqa/mmqa_corpus.json", "r"))

    scienceqa_lecture_index = faiss.read_index("baselines/scienceqa/index/nyx_lecture.faiss")
    scienceqa_qa_index = faiss.read_index("baselines/scienceqa/index/nyx_example_qa.faiss")
    scienceqa_lecture_corpus = json.load(open("baselines/scienceqa/scienceqa_lecture_corpus.json", "r"))
    scienceqa_qa_corpus = json.load(open("baselines/scienceqa/scienceqa_example_qa_corpus.json", "r"))

    nyxqa_index = faiss.read_index("baselines/nyxqa/index/nyx.faiss")
    nyxqa_corpus = json.load(open("/fs/archive/share/mm_datasets/NyxQA/corpus.json", "r"))

    model_path = "/fs/archive/share/Nyx-3B-Pretrained"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=True)
    model = Qwen2_5_VLModel.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map="auto"
    ).eval()

    def retrieve(index, corpus, text, images=None, top_k=30):
        inputs = processor(text=text, images=images, return_tensors="pt").to("cuda")
        with torch.no_grad():
            query_embedding = last_pooling(
                model(**inputs, return_dict=True, output_hidden_states=True).hidden_states[-1],
                inputs['attention_mask']
            ).float().cpu().numpy()
        _, I = index.search(query_embedding, top_k)
        return [corpus[i] for i in I[0]]

    # 1. Text-only retrieval
    print("Start retrieving for text-only datasets...")
    text_subsets = ["hotpotqa", "2wikimultihopqa", "musique"]
    text_dataset_path = "/fs/archive/share/mm_datasets/Nyx-T2T-Data"
    for subset in text_subsets:
        dataset = load_dataset(text_dataset_path, subset, split="train")
        results = []
        for item in tqdm(dataset, desc=f"Retrieving {subset}"):
            text = "Please retrieve the most relevant document to answer the question.\n" + item["qry"]
            item["retrieved_docs"] = retrieve(text_index, text_corpus, item["qry"])
            results.append(dict(item))
        with open(os.path.join(retrieval_dir, f"{subset}_retrieved.json"), "w") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Retrieved documents for {subset} saved to {retrieval_dir}/{subset}_retrieved.json")

    # 2. MMQA retrieval
    print("Start retrieving for MMQA dataset...")

    with open("/fs/archive/share/mm_datasets/MMQA/MMQA_train.jsonl", "r") as f:
        dataset = [json.loads(line) for line in f]

    results = []
    for item in tqdm(dataset, desc="Retrieving MMQA"):
        text = "Please retrieve the most relevant document to answer the question.\n" + item["question"]
        item["retrieved_docs"] = retrieve(
            index=mmqa_index,
            corpus=mmqa_corpus, 
            text=text,
            images=None,
            top_k=retrieve_top_k
        )
        results.append(item)

    with open(os.path.join(retrieval_dir, "mmqa_retrieval.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Retrieved documents for MMQA saved to {retrieval_dir}/mmqa_retrieval.json")

    # 3. ScienceQA retrieval
    print("Start retrieving for ScienceQA dataset...")

    problems = json.load(open("/fs/archive/share/mm_datasets/ScienceQA/problems.json", "r"))
    pid_splits = json.load(open("/fs/archive/share/mm_datasets/ScienceQA/pid_splits.json", "r"))
    train_pids = pid_splits["train"]

    results = []
    for pid in tqdm(train_pids, desc="Retrieving ScienceQA"):
        item = problems[pid]
        question = item["question"]
        choices = item["choices"]
        image_path = item["image"]

        text = "Question: " + question + "\nChoices: " + ", ".join(choices)

        images = []
        if image_path:
            image = Image.open(os.path.join("/fs/archive/share/mm_datasets/ScienceQA/images/train/", pid, image_path)).convert("RGBA")
            images.append(image)
            text = QWEN_IMAGE_TOKEN + text
        images = process_images(images)

        text1 = "Please retrieve the most relevant lecture to answer the question.\n" + text
        item['retrieved_lecture'] = retrieve(
            index=scienceqa_lecture_index,
            corpus=scienceqa_lecture_corpus,
            text=text1,
            images=images,
            top_k=retrieve_top_k
        )

        text2 = "Please retrieve the most relevant example Q&A to answer the question.\n" + text
        item['retrieved_example_qa'] = retrieve(
            index=scienceqa_qa_index,
            corpus=scienceqa_qa_corpus,
            text=text2,
            images=images,
            top_k=retrieve_top_k
        )

        item['pid'] = pid

        results.append(item)

    with open(os.path.join(retrieval_dir, "scienceqa_retrieval.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Retrieved documents for ScienceQA saved to {retrieval_dir}/scienceqa_retrieval.json")

    # 4. NyxQA retrieval
    print("Start retrieving for NyxQA dataset...")

    dataset = json.load(open("/fs/archive/share/mm_datasets/NyxQA/train.json", "r"))

    results = []
    for item in tqdm(dataset, desc="Retrieving NyxQA"):
        question = item['qry']
        choices = item['choices']
        image_paths = item['qry_image_path']

        text = "Please retrieve the most relevant document to answer the question.\nQuestion: " + question
        text = text + "\nChoices: " + ", ".join(choices)
        text = text.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)

        images = [Image.open(os.path.join("/fs/archive/share/mm_datasets/NyxQA/images", path)).convert("RGBA") for path in image_paths]
        images = process_images(images)

        item['retrieved_docs'] = retrieve(
            index=nyxqa_index,
            corpus=nyxqa_corpus,
            text=text,
            images=images,
            top_k=retrieve_top_k
        )
        results.append(item)

    with open(os.path.join(retrieval_dir, "nyxqa_retrieval.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Retrieved documents for NyxQA saved to {retrieval_dir}/nyxqa_retrieval.json")

# ===============GENERATION================
def generation():
    model_path = "/fs/archive/share/Qwen2.5-VL-7B-Instruct"
    processor = Qwen2_5_VLProcessor.from_pretrained(model_path, use_fast=True)

    llm = LLM(
        model=model_path,
        limit_mm_per_prompt={"image": 10, "video": 0},
    )

    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=256,
        stop_token_ids=[],
    )

    # # 1. Text-only generation
    # for subset in ["hotpotqa", "2wikimultihopqa", "musique"]:
    #     print(f"Generating responses for {subset}...")
    #     llm_inputs = []
    #     dataset = json.load(open(os.path.join(retrieval_dir, f"{subset}_retrieval.json"), "r"))
        
    #     window_length = 1 # Number of documents to consider in each sliding window
    #     window_cnt = retrieve_top_k - window_length + 1  # Number of sliding windows for each query

    #     for item in dataset:
    #         question = item["qry"]
    #         retrieved_docs = item["retrieved_docs"]

    #         sys_prompt = (
    #             "Answer the question based on the given document. "
    #             "Only give me the answer and do not output any other words.\n"
    #             "The following are given documents."
    #         )

    #         question_text = f"Question: {question}\n"

    #         # Sliding window over retrieved documents
    #         # Here, window_length = 1, meaning only one document is used at a time
    #         for doc in retrieved_docs:
    #             prompt = f"Document: {doc}\n" + question_text
    #             messages = [
    #                 {"role": "system", "content": sys_prompt},
    #                 {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #             ]

    #             input_text = processor.apply_chat_template(
    #                 messages, tokenize=False, add_generation_prompt=True
    #             )

    #             llm_inputs.append({
    #                 "prompt": input_text
    #             })

    #     # Batch generate responses
    #     outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

    #     # Combine generated responses with original data
    #     results = []
    #     output_index = 0  # Used to track the current index of outputs
    #     for item in dataset:
    #         response_list = []
    #         for window_idx in range(window_cnt):
    #             response = outputs[output_index].outputs[0].text  # Get the text of the current output
    #             response_list.append({
    #                 "docs": [{
    #                     "text": item["retrieved_docs"][window_idx],
    #                     "images": [],
    #                 }],
    #                 "response": response
    #             })
    #             output_index += 1  # Move to the next output

    #         results.append({
    #             "qry": item["qry"],
    #             "qry_image_path": [],
    #             "answer": item["ans"],
    #             "responses": response_list
    #         })

    #     # Save results to file
    #     with open(os.path.join(generation_dir, f"{subset}_generation.json"), "w") as f:
    #         json.dump(results, f, indent=2, ensure_ascii=False)
    #     print(f"Generated responses for {subset} saved to {generation_dir}/{subset}_generation.json")
            

    # # 2. MMQA generation
    # print("Generating responses for MMQA...")
    # dataset = json.load(open(f"{retrieval_dir}/mmqa_retrieval.json", "r"))
    # mmqa_img_dir = "/fs/archive/share/mm_datasets/MMQA/images"

    # window_length = 1
    # window_cnt = retrieve_top_k - window_length + 1

    # sys_prompt = (
    #     "Answer the question based on the given document. "
    #     "Only give me the answer and do not output any other words.\n"
    #     "The following are given documents."
    # )

    # llm_inputs = []
    # for item in dataset:
    #     question = item["question"]
    #     retrieved_docs = item["retrieved_docs"]

    #     question_text = f"Question: {question}\n"

    #     for doc in retrieved_docs:
    #         images = []
    #         if doc['image']:
    #             image = Image.open(os.path.join(mmqa_img_dir, doc['image'])).convert("RGBA")
    #             images.append(image)
    #         images = process_images(images)

    #         prompt = f"Document: {doc['text']}\n" + question_text
    #         prompt = prompt.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)
    #         messages = [
    #             {"role": "system", "content": sys_prompt},
    #             {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #         ]

    #         input_text = processor.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )

    #         entry = {"prompt": input_text}
    #         if images:
    #             entry["multi_modal_data"] = {"image": images}
    #         llm_inputs.append(entry)

    # outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

    # results = []
    # output_index = 0
    # for item in dataset:
    #     response_list = []
    #     for window_idx in range(window_cnt):
    #         response = outputs[output_index].outputs[0].text
    #         doc = item["retrieved_docs"][window_idx]
    #         text = doc['text']
    #         images = [doc['image']] if doc['image'] else []
    #         response_list.append({
    #             "docs": [{"text": text, "images": images}],
    #             "response": response
    #         })
    #         output_index += 1

    #     answers = [ans['answer'] for ans in item['answers']]
    #     results.append({
    #         "qry": item["question"],
    #         "qry_image_path": [],
    #         "answer": answers,
    #         "responses": response_list
    #     })
    
    # with open(os.path.join(generation_dir, "mmqa_generation.json"), "w") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print(f"Generated responses for MMQA saved to {generation_dir}/mmqa_generation.json")

    # # 3. ScienceQA generation
    # print("Generating responses for ScienceQA...")
    # dataset = json.load(open(f"{retrieval_dir}/scienceqa_retrieval.json", "r"))
    
    # # ScienceQA has two types of documents: lecture and example Q&A
    # # Each question we concat one lecture and two example Q&A
    # window_length = 2
    # window_cnt = retrieve_top_k - window_length + 1

    # sys_prompt = (
    #     "Answer the question. You may refer to the following lecture and example QA to help you answer. "
    #     "Only give me the answer and do not output any other words.\n"
    #     "The following are the lecture and example QAs."
    # )

    # llm_inputs = []
    # for item in tqdm(dataset, desc="Preparing ScienceQA inputs"):
    #     question = item['question']
    #     choices = item['choices']
    #     retrieved_lecture = item['retrieved_lecture']
    #     retrieved_example_qa = item['retrieved_example_qa']

    #     qry_images = []
    #     if item['image']:
    #         image = Image.open(os.path.join("/fs/archive/share/mm_datasets/ScienceQA/images/train", item['pid'], item['image'])).convert("RGBA")
    #         qry_images.append(image)
    #         question = QWEN_IMAGE_TOKEN + question
        
    #     for idx in range(window_cnt):
    #         images = qry_images.copy()
    #         lecture = retrieved_lecture[idx]
    #         prompt = f"Lecture: {lecture}\n"
    #         for qa_idx in range(window_length):
    #             example_qa = retrieved_example_qa[qa_idx]
    #             prompt += f"Example Q&A {qa_idx + 1}: {example_qa['text']}\n"
    #             if example_qa['image']:
    #                 image = Image.open(os.path.join("/fs/archive/share/mm_datasets/ScienceQA/images", example_qa['image'])).convert("RGBA")
    #                 images.append(image)
    #         prompt += f"Question: {question}\nChoices: {', '.join(choices)}\n"
    #         prompt = prompt.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)
    #         messages = [
    #             {"role": "system", "content": sys_prompt},
    #             {"role": "user", "content": [{"type": "text", "text": prompt}]}
    #         ]
    #         input_text = processor.apply_chat_template(
    #             messages, tokenize=False, add_generation_prompt=True
    #         )

    #         entry = {"prompt": input_text}
    #         if images:
    #             entry["multi_modal_data"] = {"image": images}
    #         llm_inputs.append(entry)
    
    # outputs = llm.generate(llm_inputs, sampling_params=sampling_params)

    # results = []
    # output_index = 0
    # for item in dataset:
    #     response_list = []
    #     for window_idx in range(window_cnt):
    #         response = outputs[output_index].outputs[0].text
    #         lecture = item["retrieved_lecture"][window_idx]
    #         response_list.append({
    #             "lectures": [lecture],
    #             "example_qas": [item["retrieved_example_qa"][window_idx], item["retrieved_example_qa"][window_idx + 1]],
    #             "response": response
    #         })
    #         output_index += 1

    #     answer = item['choices'][item['answer']]
    #     results.append({
    #         "qry": item["question"],
    #         "qry_image_path": [],
    #         "answer": answer,
    #         "responses": response_list
    #     })
    
    # with open(os.path.join(generation_dir, "scienceqa_generation.json"), "w") as f:
    #     json.dump(results, f, indent=2, ensure_ascii=False)
    # print(f"Generated responses for ScienceQA saved to {generation_dir}/scienceqa_generation.json")
        

    # 4. NyxQA generation
    # To avoid memory issues, we process NyxQA in batches
    print("Generating responses for NyxQA...")
    dataset = json.load(open(f"{retrieval_dir}/nyxqa_retrieval.json", "r"))

    sys_prompt = (
        "Answer the question based on the given document. "
        "Only give me the answer and do not output any other words.\n"
        "The following are given documents."
    )

    llm_inputs = []
    results = []
    batch_size = 1000  # Define the batch size based on the number of items
    batch_items = []  # To store the current batch of items

    for item in tqdm(dataset, desc="Preparing NyxQA inputs"):
        question = item['qry']
        choices = item['choices']
        retrieved_docs = item['retrieved_docs']

        qry_images = []
        for path in item['qry_image_path']:
            image = Image.open(os.path.join("/fs/archive/share/mm_datasets/NyxQA/images", path)).convert("RGBA")
            qry_images.append(image)

        for doc in retrieved_docs:
            images = qry_images.copy()
            for path in doc['images']:
                image = Image.open(os.path.join("/fs/archive/share/mm_datasets/NyxQA/images", path)).convert("RGBA")
                images.append(image)
            images = process_images(images)

            prompt = f"Document: {doc['text']}\nQuestion: {question}\nChoices: {', '.join(choices)}\n"
            prompt = prompt.replace(IMAGE_TOKEN, QWEN_IMAGE_TOKEN)
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": [{"type": "text", "text": prompt}]}
            ]

            input_text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            entry = {"prompt": input_text}
            if images:
                entry["multi_modal_data"] = {"image": images}
            llm_inputs.append(entry)

        batch_items.append(item)  # Add the current item to the batch

        # Process the batch when it reaches the batch size
        if len(batch_items) == batch_size:
            outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
            output_index = 0
            for batch_item in batch_items:
                response_list = []
                for window_idx in range(len(batch_item["retrieved_docs"])):
                    response = outputs[output_index].outputs[0].text
                    doc = batch_item["retrieved_docs"][window_idx]
                    text = doc['text']
                    images = doc['images']
                    response_list.append({
                        "docs": [{"text": text, "images": images}],
                        "response": response
                    })
                    output_index += 1

                answer = batch_item['choices'][batch_item['right_choice']]
                results.append({
                    "qry": batch_item["qry"],
                    "qry_image_path": batch_item.get("qry_image_path", []),
                    "answer": answer,
                    "responses": response_list
                })
            llm_inputs = []  # Clear the batch inputs
            batch_items = []  # Clear the batch items

    # Process any remaining items in the last batch
    if batch_items:
        outputs = llm.generate(llm_inputs, sampling_params=sampling_params)
        output_index = 0
        for batch_item in batch_items:
            response_list = []
            for window_idx in range(len(batch_item["retrieved_docs"])):
                response = outputs[output_index].outputs[0].text
                doc = batch_item["retrieved_docs"][window_idx]
                text = doc['text']
                images = doc['images']
                response_list.append({
                    "docs": [{"text": text, "images": images}],
                    "response": response
                })
                output_index += 1

            answer = batch_item['choices'][batch_item['right_choice']]
            results.append({
                "qry": batch_item["qry"],
                "qry_image_path": batch_item.get("qry_image_path", []),
                "answer": answer,
                "responses": response_list
            })

    # Save results to file
    with open(os.path.join(generation_dir, "nyxqa_generation.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"Generated responses for NyxQA saved to {generation_dir}/nyxqa_generation.json")
    

def feedback():
    for subset in ["hotpotqa", "2wikimultihopqa", "musique"]:
        


if __name__ == "__main__":
    # retrieval()
    # generation()
    feedback()