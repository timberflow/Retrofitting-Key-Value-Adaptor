import pandas as pd
import nltk
import numpy as np
import os
import jsonlines
import json
from torch.utils.data import Dataset
import logging
import hydra
from datasets import load_dataset, load_from_disk

from baselines.melo.utils import *

LOG = logging.getLogger(__name__)
ANS_TRIGGER = " Answer:"

class SCOTUS(Dataset):
    def __init__(self, split):
        if split == "train":
            data = load_dataset("tomh/grace-scotus", split="train")
        elif split == "edit":
            data = load_dataset("tomh/grace-scotus", split="test")

        text = data['text']
        labels = data['label']
        self.data = [{
            "text": x,
            "labels": y,
        } for x, y in zip(text, labels)]
        print(f"Loaded SCOTUS {split}: {len(data)} data items.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class NQ(Dataset):
    def __init__(self, path=f"{hydra.utils.get_original_cwd()}/data/nq_train.json"):
        with open(path, "r") as f:
            NQ = json.load(f)

        questions, answers = NQ["questions"], NQ["answers"]
        self.data = []
        for x, y in zip(questions[:1000], answers[:1000]):
            self.data.append({
                "text": x,
                "labels": y
            })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class zsRE(Dataset):
    def __init__(self, path=f"{hydra.utils.get_original_cwd()}/data/ZSRE/zsre_mend_eval.json", split="edit"):
        questions, answers = self.load_zsre(path)

        edits = []
        for x, y in zip(questions, answers):
            edits.append({
                "text": x,
                "labels": y
            })

        n_edits = min(10000, len(questions))

        np.random.seed(42)
        shuffle_ix = np.random.choice(n_edits, n_edits, replace=False)
        shuffle_edit, shuffle_holdout = shuffle_ix[:(n_edits // 2)], shuffle_ix[(n_edits // 2):]
        # shuffle_edit, shuffle_holdout = shuffle_ix[:5000], shuffle_ix[5000:]
        edit_batches = [edits[i] for i in shuffle_edit]
        edit_batches_holdout = [edits[i] for i in shuffle_holdout]
        print(f"Loaded {len(edit_batches)} possible edits and {len(edit_batches_holdout)} holdouts.")

        if split == "edit":
            self.data = edit_batches
        elif split == "holdout":
            self.data = edit_batches_holdout
        else:
            print(f"split '{split}' undefined")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

    def load_zsre(self, data_path):
        questions = []
        answers = []
        with jsonlines.open(data_path) as f:
            for d in f:
                ex = {k: d[k] for k in ["input", "prediction", "alternatives", "filtered_rephrases", "output"]}
                questions.append(ex["input"])
                answers.append(ex["output"][0]["answer"])
                if len(ex["filtered_rephrases"]) >= 10:  # Only use samples for which there are 10

                    for rephrase in ex["filtered_rephrases"][:10]:  # Only use the first 10 rephrasings
                        questions.append(rephrase)
                        answers.append(ex["output"][0]["answer"])
        return questions, answers


class zsRE_balanced(Dataset):
    def __init__(self, path=f"{hydra.utils.get_original_cwd()}/data/ZSRE/zsre_mend_eval.json", split="edit", n_edits= 5000):
        # inner = 1*question + (n-1)*rephrases
        # outer = n*rephrases
        inner_questions, inner_answers, outer_questions, outer_answers, loc_questions, loc_answers = \
            self.load_zsre(path, n_edits=n_edits)


        edits = []
        hold_outs = []
        upstreams = []
        for x, y in zip(inner_questions, inner_answers):
            edits.append({
                "text": x,
                "labels": y
            })

        for x, y in zip(outer_questions, outer_answers):
            hold_outs.append({
                "text": x,
                "labels": y
            })

        for x, y in zip(loc_questions, loc_answers):
            upstreams.append({
                "text": x,
                "labels": y
            })

        n_edits = min(n_edits, len(inner_questions))

        # np.random.seed(42)
        # shuffle_edit = np.random.choice(n_edits, n_edits, replace=False)
        # shuffle_holdout = np.random.choice(len(outer_questions), len(outer_questions), replace=False)
        # edit_batches = [edits[i] for i in shuffle_edit]
        # edit_batches_holdout = [hold_outs[i] for i in shuffle_holdout]
        # print(f"Loaded {len(edit_batches)} possible edits and {len(edit_batches_holdout)} holdouts.")

        if split == "edit":
            self.data = edits
        elif split == "holdout":
            self.data = hold_outs
        elif split == "upstream":
            self.data = upstreams
        else:
            print(f"split '{split}' undefined")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



    def load_zsre(self, data_path, n_edits=10000):
        inner_questions = []
        inner_answers = []
        outer_questions = []
        outer_answers = []
        loc_questions = []
        loc_answers = []
        with open(data_path, "r", encoding = "utf8") as f:
            ds = json.load(f)
            for d in ds:
                inner_questions.append(d["src"] + ANS_TRIGGER)
                inner_answers.append(" " + d["alt"])

                outer_questions.append(d["rephrase"] + ANS_TRIGGER)
                outer_answers.append(" " + d["alt"])
                 
                assert (
                    "nq question: " in d["loc"]
                ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
                loc_questions.append(d["loc"] + "?" + ANS_TRIGGER)
                loc_answers.append(" " + d["loc_ans"])

                if len(inner_answers) >= n_edits:
                    break
        # assert len(inner_answers) > n_edits, "Not enough balanced data"
        return inner_questions, inner_answers, outer_questions, outer_answers, loc_questions, loc_answers
    
class LAMA_balanced(Dataset):
    def __init__(self, path=f"{hydra.utils.get_original_cwd()}/data/LAMA/lama_qa.json", split="edit", n_edits = 10000):
        # inner = 1*question + (n-1)*rephrases
        # outer = n*rephrases
        inner_questions, inner_answers, outer_questions, outer_answers, loc_questions, loc_answers = \
            self.load_lama(path, n_edits=n_edits)


        edits = []
        hold_outs = []
        upstreams = []
        for x, y in zip(inner_questions, inner_answers):
            edits.append({
                "text": x,
                "labels": y
            })

        for x, y in zip(outer_questions, outer_answers):
            hold_outs.append({
                "text": x,
                "labels": y
            })

        for x, y in zip(loc_questions, loc_answers):
            upstreams.append({
                "text": x,
                "labels": y
            })

        n_edits = min(n_edits, len(inner_questions))

        if split == "edit":
            self.data = edits
        elif split == "holdout":
            self.data = hold_outs
        elif split == "upstream":
            self.data = upstreams
        else:
            print(f"split '{split}' undefined")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]



    def load_lama(self, data_path, n_edits=10000):
        inner_questions = []
        inner_answers = []
        outer_questions = []
        outer_answers = []
        loc_questions = []
        loc_answers = []
        with open(data_path, "r", encoding = "utf8") as f:
            ds = json.load(f)
            for d in ds:
                inner_questions.append(d["src"])
                inner_answers.append(" " + d["alt"])

                outer_questions.append(d["rephrase"])
                outer_answers.append(" " + d["alt"])
                 
                assert (
                    "nq question: " in d["loc"]
                ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
                loc_questions.append(d["loc"] + "?")
                loc_answers.append(" " + d["loc_ans"])

                if len(inner_answers) >= n_edits:
                    break
        return inner_questions, inner_answers, outer_questions, outer_answers, loc_questions, loc_answers


class WebText10k(Dataset):
    def __init__(self):
        data = load_dataset('stas/openwebtext-10k')['train']
        upstream = data["text"][:1000]
        self.text = [{"text": s,
                      "labels": [],
                      "concept": []} for s in upstream]

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]


class Hallucination(Dataset):
    def __init__(self, split):
        self.data = pd.DataFrame(load_dataset("potsawee/wiki_bio_gpt3_hallucination")["evaluation"])
        self.base_dir  = hydra.utils.get_original_cwd()

        concept_path = os.path.join(self.base_dir,'wiki_bio_concepts.txt')
        concepts = self.load_concepts(concept_path)
        self.concepts = [s.strip() for s in concepts]

        edit_batches, accurates, originals = self.get_edits(self.data, self.concepts)

        if split == "edit":
            self.text = edit_batches
            print(f"Loaded {len(self.text)} edits")
        elif split == "accurate":
            self.text = accurates
            print(f"Loaded {len(self.text)} accurates")
        elif split == "original":
            self.text = originals
            print(f"Loaded {len(self.text)} originals")
        elif split == "pretrain":
            upstream = WebText10k()
            self.text = accurates + originals + upstream.text[
                                                :200]  # Add 200 samples from webtext to make sure GPT2 stays good on its training data (200 seems ad hoc but it's actually pretty robust to this choice)
            self.text = [{
                "text": x["text"],
                "labels": len(self.text) * [],
                "concept": len(self.text) * [],
            } for x in self.text]
            print(f"Loaded {len(self.text)} pretraining instances")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return self.text[idx]

    def load_concepts(self, PATH):
        if not os.path.exists(PATH):
            concepts = self.generate_concepts()
        else:
            with open(PATH, 'r') as f:
                concepts = f.readlines()

        # Regenerate if existing concepts are diff shape (this dataset keeps getting updated)
        if len(concepts) != len(self.data):
            concepts = self.generate_concepts()
        return concepts

    def generate_concepts(self):
        wikibio = load_dataset("wiki_bio")
        bio_idx = self.data["wiki_bio_test_idx"]
        concepts = [wikibio["test"]["input_text"][i]["context"].strip().replace("-lrb- ", "").replace(" -rrb-", "") for
                    i in bio_idx]
        with open('./wiki_bio_concepts.txt', 'w') as f:
            f.write('\n'.join(concepts))
        return concepts

    def get_edits(self, data, concepts):
        edits = []
        originals = []
        accurates = []
        for i in range(len(self.data)):
            header = f"This is a Wikipedia passage about {concepts[i]}."
            annotations = self.data["annotation"][i]
            correct_sentences = nltk.sent_tokenize(self.data["wiki_bio_text"][i])[:len(annotations)]
            for j, annotation in enumerate(annotations):
                if "inaccurate" in annotation:
                    prompt = " ".join(self.data["gpt3_sentences"][i][:j])
                    request = {
                        "text": f"{header} {prompt}",
                        "labels": correct_sentences[min(j, len(correct_sentences) - 1)],
                        "concept": concepts[i],
                    }
                    edits.append(request)
                    request = {
                        "text": f"{header} {prompt}",
                        "labels": self.data["gpt3_sentences"][i][j],
                        "concept": concepts[i],
                    }
                    originals.append(request)
                else:
                    prompt = " ".join(self.data["gpt3_sentences"][i][:j])
                    request = {
                        "text": f"{header} {prompt}",
                        "labels": self.data["gpt3_sentences"][i][j],
                        "concept": concepts[i],
                    }
                    accurates.append(request)

        return edits, accurates, originals

