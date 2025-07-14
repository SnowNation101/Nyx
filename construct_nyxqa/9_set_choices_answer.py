import json
import ast
from sentence_transformers import SentenceTransformer
import numpy as np
import re
import random
# model = SentenceTransformer("/home/u2024001059/PLJP/model/BAAI_bge-m3")

# generated_qa_path = "all_data/qa_flattened.json"

# with open(generated_qa_path, "r") as f:
#     generated_qa = json.load(f)

# generated_choices_path = "all_data/generated_choices.json"
# with open(generated_choices_path, "r") as f:
#     generated_choices = json.load(f)

# results = []
# for data, choices_data in zip(generated_qa, generated_choices):
#     choices = choices_data["choices"]
#     try:
#         choices_list = ast.literal_eval(choices)
#     except (SyntaxError, ValueError):
#         try:
#             choices_list = json.loads(choices)
#         except json.JSONDecodeError:
#             choices_list = re.findall(r'"([^"]*)"', choices)
#             if not choices_list:  
#                 print(f"Failed to parse choices: {choices}")
#                 continue

#     query_embedding = model.encode(data["ans"], normalize_embeddings=True)
#     choice_embeddings = model.encode(choices_list, normalize_embeddings=True)

#     similarities = np.dot(choice_embeddings, query_embedding)
    
#     correct_index = np.argmax(similarities)
#     best_match = choices_list[correct_index]
#     similarity_score = similarities[correct_index]
    
#     right_choice = chr(65 + correct_index)  # 65 is ASCII for 'A'
    
#     results.append({
#         "qry": data["qry"],
#         "ans": data["ans"],
#         "choices": choices,
#         "right_choice": right_choice,
#         "pos_image_path": data["pos_image_path"],
#         "qry_image_path": data["qry_image_path"],
#         "img_num": data["img_num"]
#     })

# with open("all_data/multiple_choices_qa_dataset.json", "w") as f:
#     json.dump(results, f, indent=4)

# # shuffle
# with open("all_data/multiple_choices_qa_dataset.json", "r") as f:
#     dataset = json.load(f)

# for data in dataset:
#     choices = data["choices"]
#     try:
#         choices_list = ast.literal_eval(choices)
#     except (SyntaxError, ValueError):
#         try:
#             choices_list = json.loads(choices)
#         except json.JSONDecodeError:
#             choices_list = re.findall(r'"([^"]*)"', choices)
#             if not choices_list:  
#                 print(f"Failed to parse choices: {choices}")
#                 continue
#     correct_answer = choices_list[ord(data["right_choice"])-ord('A')]
#     choices_list = list(choices_list)
#     random.shuffle(choices_list)
    
#     new_correct_index = choices_list.index(correct_answer)
    
#     data["right_choice"] = chr(65 + new_correct_index)  # 65 = 'A'
    
#     data["choices"] = choices_list

# with open("all_data/multiple_choices_qa_dataset_shuffled.json", "w") as f:
#     json.dump(dataset, f, indent=4)

with open("all_data/qa_flattened.json", "r") as f:
    qa_dataset = json.load(f)
with open("all_data/multiple_choices_qa_dataset_shuffled.json", "r") as f:
    dataset = json.load(f)

new_data = []
for qa,data in zip(qa_dataset,dataset):
    new_data.append({
        "qry": data["qry"],
        "ans": data["ans"],
        "pos_text": qa["pos_text"],
        "choices": data["choices"],
        "right_choice": ord(data["right_choice"])-ord('A'),
        "pos_image_path": data["pos_image_path"],
        "qry_image_path": data["qry_image_path"],
        "img_num": data["img_num"]
    })

with open("all_data/choices_dataset.json", "w") as f:
    json.dump(new_data, f, indent=4)