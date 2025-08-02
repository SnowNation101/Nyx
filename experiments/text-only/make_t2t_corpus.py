import json

to_wiki_path = "/fs/archive/share/mm_datasets/t2t_data/2wikimultihopqa/dev_with_retrieved_docs.jsonl"
hotpotqa_path = "/fs/archive/share/mm_datasets/t2t_data/hotpotqa/dev_with_retrieved_docs.jsonl"
musique_path = "/fs/archive/share/mm_datasets/t2t_data/musique/dev_with_retrieved_docs.jsonl"
bamboogle_path = "/fs/archive/share/mm_datasets/t2t_data/bamboogle/test_with_retrieved_docs.jsonl"

corpus = set()

def add_docs(path):
    with open(path, "r") as f:
        for line in f:
            item = json.loads(line)
            for doc in item["retrieved_docs"][:20]:
                corpus.add(doc)

add_docs(to_wiki_path)
add_docs(hotpotqa_path)
add_docs(musique_path)
add_docs(bamboogle_path)

print(f"Total unique documents in T2T corpus: {len(corpus)}")

with open("baselines/t2t_corpus.json", "w") as f:
    json.dump(list(corpus), f, ensure_ascii=True, indent=2)
