import json
import random

from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("./hf_models/gpt2-xl")

def longest_common_subarray(arr1, arr2):
    m = len(arr1)
    n = len(arr2)
    
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    max_length = 0
    end_index = -1

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if arr1[i - 1] == arr2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
                if dp[i][j] > max_length:
                    max_length = dp[i][j]
                    end_index = i
            else:
                dp[i][j] = 0

    if max_length == 0:
        return []

    start_index = end_index - max_length
    longest_subarray = arr1[start_index:end_index]
    
    return longest_subarray

def nq_iter(nq_data):
    for line in nq_data:
        yield line["loc"], line["loc_ans"]

file = "./data/PARAREL/data_for_completion.json"
nq_file = "./data/ZSRE/zsre_mend_eval.json"
output_file = "./data/LAMA/lama_qa.json"

if __name__ == "__main__":
    random.seed(42)
    with open(file, "r", encoding="utf-8") as f:
        data = json.load(f)
    with open(nq_file, "r", encoding="utf-8") as f:
        nq_data = json.load(f)
    nq_length = len(nq_data)
    nq_iterator = nq_iter(nq_data)
    qa_dset = []

    for key, bags in data.items():
        for bag in bags:
            if len(bag) <= 1:
                continue
            rewrite_query, ans, relation = bag[0]
            rewrite_tokens = tok.encode(" " + rewrite_query)
            subject = None
            
            for query, a, r in bag[1:]:
                subject = longest_common_subarray(rewrite_tokens, tok.encode(" " + query))
            
            subject = tok.decode(subject).strip()
                
            if (subject == rewrite_query) or (subject not in rewrite_query) or (subject == ''):
                continue

            pharaphrased_query = random.choice([x[0] for x in bag[1:]])
            
            line = {
                "subject": subject,
                "src": rewrite_query,
                "pred": "",
                "rephrase": pharaphrased_query,
                "alt": ans,
                "answers": [""],
                "loc": None,
                "loc_ans": None,
                "relation": relation,
            }
            qa_dset.append(line)
    random.shuffle(qa_dset)
    for i, (loc, loc_ans) in enumerate(nq_iterator):
        if i >= nq_length:
            break
        qa_dset[i]["loc"] = loc
        qa_dset[i]["loc_ans"] = loc_ans
    qa_dset = qa_dset[:nq_length]
    print(f"Collecting {len(qa_dset)} examples.", )
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(qa_dset, f, ensure_ascii=False, indent=2)