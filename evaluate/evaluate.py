import json
import tqdm
import datetime
import pathlib
from typing import List, Dict, Union, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

from others.logging import init_logger
from evaluate.summarize_utils_zsre import summarize_rewrite_quality_zsre
from data import DS_DICT, EVAL_METHOD_DICT

logger = init_logger()

hf_cache = "./hf_models/gpt2-xl"
data_dir = "./data/ZSRE/"
save_dir = "./results/"

def eval_qa(
    model: AutoModelForCausalLM, 
    records: Union[List[Dict], str], 
    tok: AutoTokenizer, 
    ds_name: str,
    dir_to_save: str,
    summarize: Optional[bool] = True,
    metric_type: Optional[str] = "quality",
):
    
    if isinstance(records, str):
        records = DS_DICT[ds_name](tok = tok, data_dir = records)
    results = []
    for record in tqdm.tqdm(records):
        ret = EVAL_METHOD_DICT[ds_name](
            model = model,
            tok = tok,
            record = record,
            metric_type = metric_type,
        )
        ret["case_id"] = record["case_id"]
        results += [ret]

    current_time = datetime.datetime.now()
    time_str = current_time.strftime("%Y%m%d%H%M%S")

    fn = f"result-{ds_name}-{time_str}.json"
    fp = pathlib.Path(dir_to_save) / fn

    logger.info(f"Saving results at {fp}.")
    with open(fp, "w", encoding="utf8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    if summarize:
        summarize_rewrite_quality_zsre(results)

    return results

if __name__ == "__main__":
    model = AutoModelForCausalLM.from_pretrained("")
    tok = AutoTokenizer.from_pretrained("")

