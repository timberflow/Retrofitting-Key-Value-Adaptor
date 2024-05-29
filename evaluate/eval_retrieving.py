import json
import tqdm
from typing import Optional, List, Dict, Union
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from data import DS_DICT, EVAL_METHOD_DICT
from kv_src.edit_model import Hparams
from kv_src.pytorch_utils import get_attr
from .evaluate import eval_qa

def eval_kr(
    edited_model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    hparams: Hparams,
    index_map: Union[List[int], str],
    data_dir: Optional[str] = "./data",
    save_dir: Optional[str] = "./results",
    ds_name: Optional[str] = "zsre",
    model: Optional[AutoModelForCausalLM] = None,
    pre_edit_result: Optional[Union[List[Dict], str]] = None,
    post_edit_result: Optional[Union[List[Dict], str]] = None,
):
    ds_class, eval_method = DS_DICT[ds_name], EVAL_METHOD_DICT[ds_name]
    records = ds_class(
        data_dir = data_dir,
        tok = tok,
    )

    if isinstance(index_map, str):
        with open(index_map, "r", encoding="utf-8") as f:
            index_map = json.load(f)
    index_map = index_map[1]

    if pre_edit_result == None:
        assert model is not None, "Requiring pre-edit model to evaluate!"
        pre_edit_result = eval_qa(
            model = model,
            records = records,
            tok = tok,
            ds_name = ds_name,
            dir_to_save = save_dir,
            metric_type = "prediction",
        )
    elif isinstance(pre_edit_result, str):
        with open(pre_edit_result, "r", encoding = "utf-8") as f:
            pre_edit_result = json.load(f)

    if post_edit_result == None:
        post_edit_result = eval_qa(
            model = edited_model,
            records = records,
            tok = tok,
            ds_name = ds_name,
            dir_to_save = save_dir,
            metric_type = "prediction",
        )
    elif isinstance(post_edit_result, str):
        with open(post_edit_result, "r", encoding = "utf-8") as f:
            post_edit_result = json.load(f)
    
    pre_post_difference = np.mean([compute_kr_success(x, y) for x, y in zip(pre_edit_result, post_edit_result)])
    print("Pre and post edit difference: {:.4f}".format(pre_post_difference))

    retrieve_success, retrieve_consistency = [], []

    for i, record in tqdm.tqdm(enumerate(records), total=len(records)):
        v = frozen_kv_neuron(
            model = edited_model,
            hparams = hparams,
            frozen_idx = index_map[i],
        )
        result_dict = eval_method(
            model = edited_model,
            tok = tok,
            record = record,
            metric_type = "prediction",
        )
        retrieve_success += [not compute_kr_success(result_dict, post_edit_result[i])]
        retrieve_consistency += [compute_kr_success(result_dict, pre_edit_result[i])]
        recover_kv_neuron(
            model = edited_model,
            hparams = hparams,
            frozen_idx = index_map[i],
            cached_vector = v,
        )

    print("Retrieve success rate: {}  Retrieve consistency rate: {}".format(
        np.mean(retrieve_success), 
        np.mean(retrieve_consistency),
    ))
    
def frozen_kv_neuron(
    model: AutoModelForCausalLM,
    hparams: Hparams,
    frozen_idx: int,
):
    edit_layer = hparams.edit_layer
    edit_block = get_attr(model, hparams.mlp_layers[edit_layer])
    # extract value vectors
    weight_name = hparams.weight_names[1]
    value_weight = get_attr(edit_block, weight_name)
    # frozen value weight at index i
    value_cache = value_weight[torch.tensor(frozen_idx)].detach().clone()
    value_weight[torch.tensor(frozen_idx)] = torch.tensor(0., device = next(model.parameters()).device)

    return value_cache

def recover_kv_neuron(
    model: AutoModelForCausalLM,
    hparams: Hparams,
    frozen_idx: int,
    cached_vector: torch.FloatTensor,
):
    edit_layer = hparams.edit_layer
    edit_block = get_attr(model, hparams.mlp_layers[edit_layer])
    # extract value vectors
    weight_name = hparams.weight_names[1]
    value_weight = get_attr(edit_block, weight_name)
    # frozen value weight at index i
    value_weight[torch.tensor(frozen_idx)] = cached_vector.to(next(model.parameters()).device)


def compute_kr_success(
    r1: Dict,
    r2: Dict,
):
    return (r1["rewrite_prompts_correct"] == r2["rewrite_prompts_correct"])
