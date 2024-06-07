import json
from pathlib import Path
from typing import List, Dict, Optional, Union

import torch
from transformers import AutoTokenizer

from util.globals import *

ANS_TRIGGER = ""

class PARARELQADataset:
    def __init__(self, data_dir: Union[List[Dict], str, Path], tok: AutoTokenizer, size=10000, ans_trigger = "", *args, **kwargs):
        if isinstance(data_dir, str) or isinstance(data_dir, Path):
            data_dir = Path(data_dir)
            lama_loc = data_dir / "PARAREL" / "pararel_qa.json"

            with open(lama_loc, "r") as f:
                raw = json.load(f)
        else:
            raw = data_dir

        data = []
        for i, record in enumerate(raw):
            assert (
                "nq question: " in record["loc"]
            ), f"Neighborhood prompt missing `nq question:`. Check for errors?"
            ans_toks = tok(" " + record["loc_ans"])["input_ids"]
            data.append(
                {
                    "case_id": i,
                    "requested_rewrite": {
                        "prompt": record["src"].replace(record["subject"], "{}") + ans_trigger,
                        "subject": record["subject"],
                        "target_new": {"str": record["alt"]},
                        "target_true": {"str": "<|endoftext|>"},
                    },
                    "paraphrase_prompts": [record["rephrase"] + ans_trigger],
                    "neighborhood_prompts": [
                        {
                            "prompt": record["loc"] + "?" + ans_trigger + tok.decode(ans_toks[:i]),
                            "target": tok.decode(ans_toks[i]),
                        }
                        for i in range(len(ans_toks))
                    ],
                    "attribute_prompts": [],
                    "generation_prompts": [],
                }
            )
        self._data = data[:size]

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

class PARARELTrainDataset(object):
    def __init__(
        self, 
        data_dir: Union[List[Dict], str], 
        tok: AutoTokenizer, 
        stats_dir: str = None, 
        max_len = 512, 
        bsz = 1, 
        size = 10000
    ):
        self.tok = tok
        self.max_len = max_len
        self.bsz = bsz
        if isinstance(data_dir, str):
            data_dir = Path(data_dir)
            lama_loc = data_dir / "PARAREL" / "pararel_qa.json"

            with open(lama_loc, "r") as f:
                raw = json.load(f)
        else:
            raw = data_dir
 
        stats = None
        if stats_dir:
            stats = torch.load(stats_dir)
        
        if size and size >= 0:
            size = size if size > 1 else int(size * len(raw))
            raw = raw[:size]
            if stats is not None:
                stats = stats[:size]

        self.data = self.get_prefix_example(raw, stats)
        self._reset_()

    def _data_iter(self):
        for item in self.data:
            yield item

    def _reset_(self):
        self.data_iter = self._data_iter()

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self

    def collate_fn(self, batch):
        assert self.bsz == 1, "Support batch size = 1 only"
        return batch[0]

    def __next__(self):
        batch = []
        for idx in range(self.bsz):
            try:
                item = next(self.data_iter)
                batch += [item]
            except StopIteration:
                break

        if not batch:
            raise StopIteration
        
        return self.collate_fn(batch)

    def get_prefix_example(self, raw, stats = None):
        data = []
        for i, line in enumerate(raw):
            src, rep, tgt, alt = line["src"], line["rephrase"], line["pred"], line["alt"]
            src += ANS_TRIGGER
            rep += ANS_TRIGGER
            
            if tgt == "":
                tgt = self.tok.eos_token
            elif "GPT2" in type(self.tok).__name__:
                tgt = " " + tgt

            if alt == "":
                alt = self.tok.eos_token
            elif "GPT2" in type(self.tok).__name__:
                alt = " " + alt

            src_encoded = self.tok(
                text = [src], 
                return_tensors = "pt", 
                padding = "max_length", 
                max_length = self.max_len, 
                truncation = True
            )
            rep_encoded = self.tok(
                text = [rep], 
                return_tensors = "pt", 
                padding = "max_length", 
                max_length = self.max_len, 
                truncation = True
            )
            tgt_encoded = self.tok(
                text = [tgt], 
                return_tensors = "pt", 
                padding = "max_length", 
                max_length = self.max_len, 
                truncation = True, 
                add_special_tokens=False
            )
            alt_encoded = self.tok(
                text = [alt], 
                return_tensors = "pt", 
                padding = "max_length", 
                max_length = self.max_len, 
                truncation = True, 
                add_special_tokens=False
            )
 
            line = {"src": src_encoded, "rep": rep_encoded, "tgt": tgt_encoded, "alt": alt_encoded, "aux": None}
            if stats is not None:
                line["aux"] = stats[i]["states"]
            data += [line]

            alt_ids = self.tok.encode(alt)
            for i in range(1, len(alt_ids)):
                prefix = self.tok.decode(alt_ids[:i])
                target = self.tok.decode(alt_ids[i:])
                src_prefix = src + prefix
                rep_prefix = rep + prefix
                src_prefix_encoded = self.tok(
                    text = [src_prefix], 
                    return_tensors = "pt", 
                    padding = "max_length", 
                    max_length = self.max_len, 
                    truncation = True
                )
                rep_prefix_encoded = self.tok(
                    text = [rep_prefix], 
                    return_tensors = "pt", 
                    padding = "max_length", 
                    max_length = self.max_len, 
                    truncation = True
                )
                alt_prefix_encoded = self.tok(
                    text = [target], 
                    return_tensors = "pt", 
                    padding = "max_length", 
                    max_length = self.max_len, 
                    truncation = True, 
                    add_special_tokens=False
                )
                line = {"src": src_prefix_encoded, "rep": rep_prefix_encoded, "tgt": None, "alt": alt_prefix_encoded, "aux": None}
                if stats is not None:
                    line["aux"] = stats[i]["states"]
                data += [line]
                
        return data