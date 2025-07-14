import json
import re
import os
import random

def split_qa(qa_string):
    qa_entries = re.findall(r'\[(Q\d+:.*?, A\d+:.*?)\]', qa_string)

    qa_list = []

    for entry in qa_entries:
        q_match = re.search(r'Q\d+: (.*?), A\d+:', entry)
        a_match = re.search(r'A\d+: (.*)', entry)
        
        if q_match and a_match:
            question = q_match.group(1).strip()
            answer = a_match.group(1).strip()
            qa_list.append({"q": question, "a": answer})
    return qa_list

def extract_image(text):
    matches = re.findall(r'<image(\d+)>', text)
    if matches:
        image_numbers = list(map(int, matches))
        return image_numbers
    else:
        return None

def main():
    simplified_qa_file =  "./all_data/simplified_qa.json"
    with open(simplified_qa_file, "r") as f:
        dataset1 = json.load(f)
    generated_multi_image_qa = "./all_data/generated_multi_image_qa.json"
    with open(generated_multi_image_qa, "r") as f:
        dataset2 = json.load(f)
    
    merged_data = dataset1 + dataset2
    random.shuffle(merged_data)

    dataset = merged_data

    new_data = []
    pattern = re.compile(r'<image\d+>')
    for idx,data in enumerate(dataset):
        doc = data["doc"]
        images = data["images"]
        qa_string = data["generated_qa"]
        qa_list = split_qa(qa_string)
        for idx,qa in enumerate(qa_list):
            query_images = []
            if "text" not in qa["q"] and "author" not in qa["q"]:
                if "image" not in qa["q"]:
                    new_data.append(
                            {
                                "qry": qa["q"],
                                "ans": qa["a"],
                                "pos_text": doc,
                                "pos_image_path":images,
                                "qry_image_path":query_images
                            }
                        )
                else:
                    if pattern.search(qa["q"]):
                        for number in extract_image(qa['q']):
                            if number <= len(images):
                                query_images.append(images[number-1])
                        if len(extract_image(qa['q'])) == len(query_images):
                            new_data.append(
                                {
                                    "qry": qa["q"],
                                    "ans": qa["a"],
                                    "pos_text": doc,
                                    "pos_image_path":images,
                                    "qry_image_path":query_images
                                }
                            )
    for item in new_data:
        item["qry"] = re.sub(r"<image(\d+)>", r"<|image|>", item["qry"])
        item["img_num"] = len(item["pos_image_path"])

    with open("all_data/qa_flattened.json", "w") as f:
        json.dump(new_data, f, indent=4)

if __name__ == "__main__":
    # print(split_qa("[Q1: What does Jeb Bush describe as an act of love and commitment to family?, A1: People providing for families.],\n[Q2: According to Jeb Bush, what type of crime is committed by people coming to the country to provide for their families?, A2: Not a felony.],\n[Q3: What does Jeb Bush suggest should be the consequence for the type of crime he describes?, A3: A price paid.],\n[Q4: What does Jeb Bush imply should not rile people up?, A4: People providing for families.],\n[Q5: What is the title of the article that includes the quote from Jeb Bush?, A5: \"Unethical Quote of the Month: Jeb Bush\"]"))
    # print(extract_image("Q2: Based on <image1>, what is Jeb Bush doing in the image?, A2: Speaking at a podium."))
    main()