import random
import tqdm
import torch
import pathlib
from typing import List, Dict, Union, Tuple
from transformers import AutoModelForCausalLM, AutoTokenizer

from kv.edit_model import Hparams, EditModel
from kv.pytorch_utils import get_attr

def generate_aux_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    hparams: Hparams,
    ref_records: List[Dict],
    records_name: str,
    num_examples_in_subject: int = 2,
    num_examples_out_subject: int = 2,
    num_examples_itc: int = 4,
    length: Union[int, Tuple[int, int]] = (2,10),
    temperature: float = 1.,
):
    edit_model = EditModel(
        hparams = hparams,
        model = model,
        tokenizer = tokenizer,
        device = "cuda",
    )
    edit_block = get_attr(model, hparams.mlp_layers[hparams.edit_layer])

    aux_examples = []
    for i, record in tqdm.tqdm(enumerate(ref_records)):
        # in subject examples
        in_subject_text = in_subject_examples(
            model = model,
            tokenizer = tokenizer,
            record = record,
            length = length,
            num_return_sequences = num_examples_in_subject,
            temperature = temperature,
        )
        in_subject_states = edit_block.cached_state.squeeze()
        # out subject examples
        out_subject_text = out_subject_examples(
            model = model,
            tokenizer = tokenizer,
            length = length,
            num_return_sequences = num_examples_out_subject,
            temperature = temperature,
        )
        out_subject_states = edit_block.cached_state.squeeze()
        # contrastive learning examples
        idxs = [idx for idx in range(len(ref_records)) if idx != i]
        itc_records = [ref_records[idx] for idx in random.sample(idxs, k = num_examples_itc)]
        itc_text = itc_examples(
            model = model,
            tokenizer = tokenizer,
            records = itc_records
        )
        itc_subject_states = edit_block.cached_state[:, -1]

        aux_text = in_subject_text + out_subject_text + itc_text
        aux_states = torch.cat((in_subject_states, out_subject_states, itc_subject_states), dim = 0)

        aux_examples += [{"text": aux_text, "states": aux_states}]
    
    edit_model.logger.info(f"Saving auxiliary states at {hparams.aux_path}.")
    fn = "_".join([records_name, hparams.name, str(hparams.edit_layer), "stats.pt"])
    torch.save(aux_examples, pathlib.Path(hparams.aux_path) / fn)
        


def in_subject_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    record: Dict,
    length: Union[int, Tuple[int, int]],
    num_return_sequences: int = 2,
    temperature: float = 1.,
):
    if isinstance(length, tuple):
        length = random.randint(length[0], length[1])

    example_trigger = record["requested_rewrite"]["subject"]
    encoded_trigger = tokenizer(example_trigger, return_tensors = "pt").to("cuda")
    output_ids = model.generate(
        **encoded_trigger,
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens = length,
        do_sample = True,
        temperature = temperature,
        num_return_sequences = num_return_sequences,
    )
    output_string = [tokenizer.decode(x, skip_special_tokens = True) for x in output_ids]
    return output_string

def out_subject_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    length: Union[int, Tuple[int, int]],
    num_return_sequences: int = 2,
    temperature: float = 1.,
):
    if isinstance(length, tuple):
        length = random.randint(length[0], length[1])

    example_trigger = tokenizer.bos_token
    encoded_trigger = tokenizer(example_trigger, return_tensors = "pt").to("cuda")
    output_ids = model.generate(
        **encoded_trigger,
        pad_token_id = tokenizer.eos_token_id,
        max_new_tokens = length,
        do_sample = True,
        temperature = temperature,
        num_return_sequences = num_return_sequences,
    )
    output_string = [tokenizer.decode(x, skip_special_tokens = True) for x in output_ids]
    return output_string

def itc_examples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    records: List[Dict],
):
    itc_examples = [
        record["requested_rewrite"]["prompt"].format(record["requested_rewrite"]["subject"])
        for record in records
    ]
    encoded_itc = tokenizer(itc_examples, return_tensors = "pt", padding = "longest").to("cuda")
    _ = model.forward(
        **encoded_itc,
    )
    return itc_examples