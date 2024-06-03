import math
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

from typing import List, Set, Tuple

# Activation
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def gelu_new(x):
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

def swish(x):
    return x * torch.sigmoid(x)

def shifted_swish(x, bias):
    return swish(x - bias)

def shifted_gelu(x, bias):
    return gelu_new(x - bias)

def shifted_relu(x, bias):
    return torch.nn.functional.relu(x - bias)



ACT2FN = {
    "gelu": gelu, 
    "relu": torch.nn.functional.relu, 
    "swish": swish, 
    "silu": swish, 
    "gelu_new": gelu_new, 
    "shifted_swish": shifted_swish,
    "shifted_gelu": shifted_gelu,
    "shifted_relu": shifted_relu,
}

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def rmsnorm(x, dim = -1, eps = 1e-8):
    mu = x.mean(dim = dim, keepdim = True)
    var = x.var(dim = dim, keepdim = True)
    return (x - mu).div(torch.sqrt(var + eps))

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

def prune_conv1d_layer(layer: Conv1D, index: torch.LongTensor, dim: int = 1) -> Conv1D:
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer

def find_pruneable_heads_and_indices(
    heads: List[int], n_heads: int, head_size: int, already_pruned_heads: Set[int]
) -> Tuple[Set[int], torch.LongTensor]:
    mask = torch.ones(n_heads, head_size)
    heads = set(heads) - already_pruned_heads  # Convert to set and remove already pruned heads
    for head in heads:
        # Compute how many pruned heads are before the head and move the index accordingly
        head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index: torch.LongTensor = torch.arange(len(mask))[mask].long()
    return heads, index

def build_pred_mask(src_mask, tgt_mask, max_len = None):
    device = src_mask.device
    max_len = max_len if max_len else src_mask.size(-1)
    bsz = src_mask.size(0)
    prompt_len = src_mask.sum(-1) + tgt_mask.sum(-1)
    max_len = max_len if max_len else torch.max(prompt_len) + 1
    pred_mask = torch.zeros((bsz, max_len), dtype = torch.int64).to(device)
    pred_mask[torch.arange(max_len).to(device)[None,:].repeat(bsz,1) < prompt_len[:,None]] = 1
    pred_mask = pred_mask - src_mask
    # left shift mask for one position
    shifted_mask = torch.cat((pred_mask[:,1:], torch.zeros((bsz, 1), device=device)), dim = 1)
    return pred_mask, shifted_mask

def concat_masked_indexes(src, src_mask, tgt, tgt_mask, max_len):
    pred_mask, shifted_mask = build_pred_mask(src_mask, tgt_mask, max_len)
    concat = torch.zeros_like(pred_mask)
    concat_mask = src_mask + pred_mask
    concat[src_mask == 1] = src[src_mask == 1]
    concat[pred_mask == 1] = tgt[tgt_mask == 1]
    return concat, concat_mask, shifted_mask

def retain_first_unmasked(mask):
    first_one_indices = mask.argmax(dim=1, keepdim=True)
    retained_mask = torch.zeros_like(mask)
    retained_mask.scatter_(1, first_one_indices, 1)
    return retained_mask

def retain_last_unmasked(mask):
    last_non_masked = mask.sum(-1) - 1
    retained_mask = torch.zeros_like(mask)
    retained_mask.scatter_(1, last_non_masked.unsqueeze(1), 1)
    return retained_mask

def cal_entropy(x, dim = -1, eps = 1e-8):
    x_clamped = torch.clamp(x, eps, 1.0 - eps)
    return -torch.sum(x * torch.log(x_clamped), dim=dim)

class SeqGenCrossEntropyLoss(nn.Module):
    def __init__(self, label_smoothing = 0):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss(label_smoothing = label_smoothing)
        self.softmax = nn.Softmax(-1)

    def forward(self, logits, target, pred_mask, tgt_mask):
        logits = logits[pred_mask == 1]
        target = target[tgt_mask == 1]
        # probs = self.softmax(logits)
        # tgt_probs = probs.gather(-1, target[..., None]).squeeze(-1)
        # loss = -torch.log(tgt_probs).mean()
        loss = self.cross_entropy(logits, target)
        return loss

def get_attr(module, param_name, global_prefix = ""):
    param_name = global_prefix + param_name
    attrs = param_name.split('.')
    current_module = module
    for attr in attrs:
        if attr.isdigit():
            current_module = current_module[int(attr)]
        else:
            current_module = getattr(current_module, attr)
    return current_module

def set_attr(module, param_name, value, global_prefix = ""):
    param_name = global_prefix + param_name
    attrs = param_name.split('.')
    current_module = module
    for attr in attrs[:-1]:
        if attr.isdigit():
            current_module = current_module[int(attr)]
        else:
            current_module = getattr(current_module, attr)
    setattr(current_module, attrs[-1], value)

def build_cached_forward(model, hparams):
    '''
    convert model.forward to a cached forward method, which can save inputs
    in sub-modules' properties to allow them accessing original inputs during
    forward stage.
    this function can be replaced with torch.nn.register_pre_hook()
    '''
    def modify_forward(func):
        def wrapper(self, input_ids, attention_mask = None, **inputs):
            if "pred_mask" not in inputs:
                pred_mask = retain_last_unmasked(attention_mask)
            else:
                pred_mask = inputs["pred_mask"]
            if "pred_mask" in inputs:
                del inputs["pred_mask"]
            for block_name in hparams.mlp_layers:
                mlp_block = get_attr(self, block_name)
                mlp_block.input_dict = copy.deepcopy(inputs)
                mlp_block.input_dict.update({"pred_mask": pred_mask})
            return func(input_ids = input_ids, attention_mask = attention_mask, **inputs)
        return wrapper
    model.forward = modify_forward(model.forward).__get__(model, AutoModelForCausalLM)

