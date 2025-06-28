from load_aokvqa import get_aokvqa_data

train_dataset, _, _ = get_aokvqa_data()

import json
import os

corpus = []
for data in train_dataset:
    image_path = data['image_path']
    question = data['question']
    answers = data['direct_answers']

    unique_answers = list(set(answers))
    merged_answer = "Possible Answers: " + ", ".join(unique_answers)

    rationales = data['rationales']
    merged_rationale = "Rationale: " + " ".join(rationales)

    doc = (
        "Question: " + question + "\n" +
        merged_answer + "\n" +
        merged_rationale + "\n"
    )

    corpus.append({
        "image_path": image_path,
        "doc": doc
    })

with open("aokvqa_corpus.json", "w") as f:
    json.dump(corpus, f, indent=2)

