from typing import List, Union

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

def single_decode(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    prompt_tokens: List[str],
):
    params = model.config
    device = next(model.parameters()).device
    # tokenize prompts
    prompt_tokens = [tokenizer.encode(x) for x in prompt_tokens]
    bsz = len(prompt_tokens)

    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.n_positions
    total_len = min(params.n_positions, 1 + max_prompt_len)

    pad_id = tokenizer.pad_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        # padding from left side
        tokens[k, :len(t)] = torch.tensor(t, dtype=torch.long, device=device)
    input_text_mask = tokens != pad_id

    with torch.no_grad():
        logits = model.forward(input_ids = tokens, attention_mask = input_text_mask).logits

    last_non_masked = input_text_mask.sum(1) - 1
    to_gather = last_non_masked.unsqueeze(1).repeat(1, logits.size(-1)).unsqueeze(1)
    gathered = torch.gather(logits, 1, to_gather).squeeze(1)
    out_tokens = torch.argmax(gathered, dim=-1)

    return [tokenizer.decode(x) for x in out_tokens], logits

def generate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer, 
    prompt_tokens: List[str],
    max_gen_len: int,
    temperature: float = 0.0,
    top_p: float = 0.9,
    echo: bool = False,
):
    params = model.config
    # get model's device
    device = next(model.parameters()).device
    # tokenize prompts
    prompt_tokens = [tokenizer.encode(x) for x in prompt_tokens]
    bsz = len(prompt_tokens)

    min_prompt_len = min(len(t) for t in prompt_tokens)
    max_prompt_len = max(len(t) for t in prompt_tokens)
    assert max_prompt_len <= params.n_positions
    total_len = min(params.n_positions, max_gen_len + max_prompt_len)

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    tokens = torch.full((bsz, total_len), pad_id, dtype=torch.long, device=device)
    for k, t in enumerate(prompt_tokens):
        tokens[k, : len(t)] = torch.tensor(t, dtype=torch.long, device=device)

    prev_pos = 0
    eos_reached = torch.tensor([False] * bsz, device=device)
    input_text_mask = tokens != pad_id
    if min_prompt_len == total_len:
        logits = model.forward(tokens, prev_pos)

    for cur_pos in range(min_prompt_len, total_len):
        # with embedding cache
        # logits = model.forward(tokens[:, prev_pos:cur_pos])[0]
        # without embedding cache
        logits = model.forward(input_ids = tokens[:, :cur_pos]).logits
        if temperature > 0:
            probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
            next_token = sample_top_p(probs, top_p)
        else:
            next_token = torch.argmax(logits[:, -1], dim=-1)

        next_token = next_token.reshape(-1)
        # only replace token if prompt has already been generated
        next_token = torch.where(
            input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
        )
        tokens[:, cur_pos] = next_token
        eos_reached |= (~input_text_mask[:, cur_pos]) & (
            next_token == eos_id
        )
        prev_pos = cur_pos
        if all(eos_reached):
            break
    out_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        # cut to max gen len
        start = 0 if echo else len(prompt_tokens[i])
        toks = toks[start : len(prompt_tokens[i]) + max_gen_len]
        # cut to eos tok if any
        if eos_id in toks:
            eos_idx = toks.index(eos_id)
            toks = toks[:eos_idx]
        out_tokens.append(toks)
    return [tokenizer.decode(x) for x in out_tokens], out_tokens

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token