import json
import os

import shutil
from pathlib import Path
from time import time
from typing import Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import DS_DICT
from data.zsre import MENDQADataset
from evaluate.evaluate import eval_qa
from evaluate.summarize_utils import summarize_rewrite_quality
from util import nethook
from util.globals import *
from baselines.rome import (
    ROMEHyperParams, 
    apply_rome_to_model, 
    MEMITHyperParams, 
    apply_memit_to_model,
    FTHyperParams,
    apply_ft_to_model,
)
from baselines.mend import (
    MENDHyperParams,
    MendRewriteExecutor,
)

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

ALG_DICT = {
    "ROME": (ROMEHyperParams, apply_rome_to_model),
    "MEMIT": (MEMITHyperParams, apply_memit_to_model),
    "FT": (FTHyperParams, apply_ft_to_model),
    "MEND": (MENDHyperParams, MendRewriteExecutor().apply_to_model),
}

def main(
    alg_name: str,
    model_name: Union[str, Tuple],
    hparams_fname: str,
    ds_name: str,
    dataset_size_limit: int,
    continue_from_run: str,
    conserve_memory: bool,
    dir_name: str,
):
    # Set algorithm-specific variables
    params_class, apply_algo = ALG_DICT[alg_name]

    # Determine run directory
    if continue_from_run is not None:
        run_dir = RESULTS_DIR / dir_name / continue_from_run
        assert (
            run_dir.exists()
        ), f"If continuing from run, {continue_from_run} must exist!"
    else:
        alg_dir = RESULTS_DIR / dir_name
        if alg_dir.exists():
            id_list = [
                int(str(x).split("_")[-1])
                for x in alg_dir.iterdir()
                if str(x).split("_")[-1].isnumeric()
            ]
            run_id = 0 if not id_list else max(id_list) + 1
        else:
            run_id = 0
        run_dir = RESULTS_DIR / dir_name / f"run_{str(run_id).zfill(3)}"
        run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Results will be stored at {run_dir}")

    # Get run hyperparameters
    params_path = hparams_fname
    hparams = params_class.from_json(params_path)
    if not (run_dir / "params.json").exists():
        shutil.copyfile(params_path, run_dir / "params.json")
    print(f"Executing {alg_name} with parameters {hparams}")

    # Instantiate vanilla model
    print("Instantiating model")
    if type(model_name) is str:
        model = AutoModelForCausalLM.from_pretrained(model_name).cuda()
        tok = AutoTokenizer.from_pretrained(model_name)
        tok.pad_token = tok.eos_token
    else:
        model, tok = model_name

    # Load data
    print("Loading dataset, attribute snippets, tf-idf data")

    ds_class = DS_DICT[ds_name]
    ds_inputs = {"data_dir": DATA_DIR, "size": dataset_size_limit, "tok": tok}
    if ds_name == "counterfact":
        ds_inputs["ans_trigger"] = ""
    ds = ds_class(**ds_inputs)

    # Iterate through dataset
    num_edits = 10000 if alg_name in ("MEMIT", "FT") else 1
    start = time()
    for i, record_chunks in enumerate(chunks(ds, num_edits)):
        # Compute weight changes + record weights that changed
        print(f"Chunk {i} starts.")
        args_conserve_memory = (
            dict(return_orig_weights_device="cuda")
            if conserve_memory
            else dict()
        )
        edited_model, _ = apply_algo(
            model,
            tok,
            [{"case_id": record["case_id"], **record["requested_rewrite"]} for record in record_chunks],
            hparams,
            copy=False,
            return_orig_weights=True,
            **args_conserve_memory,
        )
    exec_time = time() - start
    print("Execution took", exec_time)

    # Empty CUDA cache
    print("Empty cuda cache...")
    torch.cuda.empty_cache()
    # Execute evaluation suite
    start = time()
    metrics = eval_qa(
        model = edited_model,
        tok = tok,
        records = ds,
        ds_name = ds_name,
        dir_to_save = run_dir,
    )
    summarize_rewrite_quality(metrics)

    print("Evaluation took", time() - start)

def chunks(arr, n):
    """Yield successive n-sized chunks from arr."""
    for i in range(0, len(arr), n):
        yield arr[i : i + n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["ROME", "MEMIT", "FT", "MEND"],
        default="ROME",
        help="Editing algorithm to use. Results are saved in results/<alg_name>/<run_id>, "
        "where a new run_id is generated on each run. "
        "If continuing from previous run, specify the run_id in --continue_from_run.",
        required=True,
    )
    parser.add_argument(
        "--model_name",
        default="gpt2-xl",
        help="Model to edit.",
        required=True,
    )
    parser.add_argument(
        "--hparams_fname",
        type=str,
        default="gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        default="zsre",
        help="Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--continue_from_run",
        type=str,
        default=None,
        help="If continuing from previous run, set to run_id. Otherwise, leave as None.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--skip_generation_tests",
        dest="skip_generation_tests",
        action="store_true",
        help="Only run fast probability-based tests without slow generation tests. "
        "Useful for quick debugging and hyperparameter sweeps.",
    )
    parser.add_argument(
        "--conserve_memory",
        dest="conserve_memory",
        action="store_true",
        help="Reduce memory usage during evaluation at the cost of a minor slowdown. "
        "Backs up model weights on CPU instead of GPU.",
    )
    parser.set_defaults(skip_generation_tests=False, conserve_memory=False)
    args = parser.parse_args()

    main(
        args.alg_name,
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.dataset_size_limit,
        args.continue_from_run,
        args.skip_generation_tests,
        args.conserve_memory,
        args.alg_name,
    )
