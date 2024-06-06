import os
import json

from transformers import AutoTokenizer, AutoModelForCausalLM

from kv.edit_model import Hparams
from kv.alg import apply_alg_to_model

from data import DS_DICT, TRAIN_DS_DICT

from evaluate.evaluate import eval_qa
from evaluate.summarize_utils import summarize_rewrite_quality


def kv_main(
    hf_cache: str,
    hparams_path: str,
    dataset: str,
    data_dir: str,
    data_size: int,
    stats_dir: str,
    cache_dir: str,
    seed: int,
):
    
    with open(hparams_path, "r") as f:
        hparams = json.load(f)
        hparams = Hparams(**hparams)

    if not os.path.exists(hf_cache):
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        hf_cache = os.path.join(cache_dir, hf_cache)

    model = AutoModelForCausalLM.from_pretrained(hf_cache).cuda()
    tok = AutoTokenizer.from_pretrained(hf_cache)

    # set pad_token_id
    if tok.pad_token_id is None:
        tok.add_special_tokens({'pad_token': tok.eos_token})
    tok.padding_side = "right"

    train_dset = TRAIN_DS_DICT[dataset](
        data_dir = data_dir,
        stats_dir = stats_dir,
        tok = tok,
        max_len = hparams.max_len,
        size = data_size,
    )

    eval_dset = DS_DICT[dataset](
        data_dir = data_dir,
        tok = tok,
        max_len = hparams.max_len,
        size = data_size,
    )

    apply_alg_to_model(
        model = model,
        tok = tok,
        requests = train_dset,
        hparams = hparams_path,
        seed = seed,
    )

    results = eval_qa(
        model = model,
        tok = tok,
        records = eval_dset,
        ds_name = dataset,
        dir_to_save = hparams.result_path,
    )

    summarize_rewrite_quality(results)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--alg_name",
        choices=["KV"],
        default="KV",
        help="Editing algorithm to use.",
        required=False,
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
        default="./hparams/kv/gpt2-xl.json",
        help="Name of hyperparameters file, located in the hparams/<alg_name> folder.",
        required=True,
    )
    parser.add_argument(
        "--ds_name",
        default="zsre",
        help="Dataset to perform evaluations on.",
    )
    parser.add_argument(
        "--data_dir",
        default="./data",
        help="Path of dataset.",
    )
    parser.add_argument(
        "--dataset_size_limit",
        type=int,
        default=10000,
        help="Truncate CounterFact to first n records.",
    )
    parser.add_argument(
        "--stats_dir",
        type=str,
        default=None,
        help="Auxiliary states directory used by ReVa.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./hf_models",
        help="Saving path of hugging face cache.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--visible_cudas",
        type=str,
        default="7",
        help="Visible cuda devices.",
    )

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.visible_cudas

    kv_main(
        args.model_name,
        args.hparams_fname,
        args.ds_name,
        args.data_dir,
        args.dataset_size_limit,
        args.stats_dir,
        args.cahce_dir,
        args.seed,
    )