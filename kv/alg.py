import json
from typing import List, Dict, Union, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from kv.edit_model import Hparams, EditModel
from kv.pytorch_utils import set_seed
from kv.kn_trainer import TrainArgs
from data import TRAIN_DS_DICT

def apply_alg_to_model(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    requests: object,
    hparams: Union[Hparams, str],
    seed: int = 42,
    **kwargs,
) -> AutoModelForCausalLM:
    # manual random seed = 42
    set_seed(seed)

    # get hparams
    if isinstance(hparams, str):
        with open(hparams, "r") as f:
            hparams = json.load(f)
            hparams = Hparams(**hparams)

    # frozen parameters
    model.requires_grad_(False)
    
    # set training arguments
    train_args = TrainArgs()

    # build editing model
    edit_model = EditModel(
        hparams = hparams,
        train_args = train_args,
        model = model,
        tokenizer = tok,
        device = hparams.device,
    )

    # start editing
    edit_model.run_edit(requests)

    # convert to evaluation mode
    edit_model.eval()

    return model