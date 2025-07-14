from datasets import load_dataset
import json
import os
from tqdm import tqdm

dataset_dir = "/fs/archive/share/mm_datasets/ScienceQA"

with open(os.path.join(dataset_dir, "problems.json"), "r") as f:
    problems = json.load(f)
with open(os.path.join(dataset_dir, "pid_splits.json"), "r") as f:
    pid_splits = json.load(f)

train_pids = pid_splits['train']
val_pids = pid_splits['val']
test_pids = pid_splits['test']

corpus_set = set()

for pid in train_pids:
    item = problems[pid]
    if item['lecture']:
        corpus_set.add(item['lecture'])
for pid in val_pids:
    item = problems[pid]
    if item['lecture']:
        corpus_set.add(item['lecture'])
for pid in test_pids:
    item = problems[pid]
    if item['lecture']:
        corpus_set.add(item['lecture'])

corpus = list(corpus_set)

with open("scienceqa_lecture_corpus.json", "w") as f:
    json.dump(corpus, f, indent=2, ensure_ascii=False)
print("Saved scienceqa_lecture_corpus.json with", len(corpus), "lectures.")

example_qa_corpus = []
for pid in train_pids:
    item = problems[pid]

    question = item['question']
    answer = item['choices'][item['answer']]
    hint = item['hint']
    solution = item['solution']

    text = ""

    if item["image"]:
        image = f"train/{pid}/{item['image']}"
        text += f"Question: <|image|>{question}\n"
    else:
        image = None
        text += f"Question: {question}\n"

    if hint:
        text += f"Hint: {hint}\n"
    text += f"Choices: {', '.join(item['choices'])}\n"
    text += f"Answer: {answer}\n"


    if solution:
        text += f"Solution: {solution}"

    example_qa_corpus.append({
        "text": text,
        "image": image,
    })

with open("scienceqa_example_qa_corpus.json", "w") as f:
    json.dump(example_qa_corpus, f, indent=2, ensure_ascii=False)
print("Saved scienceqa_example_qa_corpus.json with", len(example_qa_corpus), "example Q&A pairs.")