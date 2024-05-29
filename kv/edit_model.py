import json
import random
import copy
import tqdm
import inspect
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from others import logging
from kv.pytorch_utils import (
    ACT2FN, 
    rmsnorm,
    SeqGenCrossEntropyLoss, 
    get_attr, 
    set_attr,
    build_pred_mask,
    concat_masked_indexes,
    build_cached_forward,
)
from kv.kn_trainer import KNStepTrainer

class Hparams:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.mlp_layers = [self.mlp_layers.format(i) for i in range(self.num_layers)]


class EditedMLPLayer(nn.Module):
    def __init__(
        self, 
        mlp_block, 
        d_model, 
        actv, 
        forward_method, 
        alpha,
        max_pooling = True,
        seq_sampling = True,
    ):
        super().__init__()
        self.mlp = mlp_block
        self.LoRA_A = nn.Parameter(requires_grad=False)
        self.LoRA_B = nn.Parameter(requires_grad=False)
        self.kn_bias = nn.Parameter(requires_grad=False)

        self.d_model = d_model
        self.actv = actv
        self.use_bias = "bias" in inspect.signature(actv).parameters
        self.forward_method = forward_method
        self.alpha = alpha
        self.max_pooling = max_pooling
        self.seq_sampling = seq_sampling

        self.new_LoRA_A = nn.Parameter(requires_grad=False)
        self.new_LoRA_B = nn.Parameter(requires_grad=False)
        self.new_kn_bias = nn.Parameter(requires_grad=False)

        self.cached_state = None
        
        self.device = next(mlp_block.parameters()).device
        self.to(self.device)
        # prevent recursive functional call
        self._train = False

    def to_train(self):
        self._train = True
    
    def to_eval(self):
        self.eval()
        self._train = False

    def auto_activation(self, x, bias = None):
        if self.use_bias:
            if bias is None:
                bias = 0 if self.kn_bias.size == torch.Size([0]) else self.kn_bias
            return self.actv(x, bias)
        else:
            return self.actv(x)

    def init_LoRA_AB(self, rank = 1, A_weight = None, B_weight = None, bias = None):
        # Initialize learnable parameters
        if self.new_LoRA_A.requires_grad:
            self.new_LoRA_A.data = torch.cat(
                (self.new_LoRA_A.data, (A_weight if A_weight is not None else torch.rand(self.d_model, rank)).to(self.device)), dim = 1)
            self.new_LoRA_B.data = torch.cat(
                (self.new_LoRA_B.data, (B_weight if B_weight is not None else torch.rand(rank, self.d_model)).to(self.device)), dim = 0)
            self.new_kn_bias.data = torch.cat((self.new_kn_bias.data, torch.tensor([0 if bias is None else bias]).to(self.device)), dim = 0)
        else:
            self.new_LoRA_A.data = (torch.rand(self.d_model, rank) if A_weight is None else A_weight).to(self.device)
            self.new_LoRA_B.data = (torch.rand(rank, self.d_model) if B_weight is None else B_weight).to(self.device)
            self.new_kn_bias.data = (torch.tensor([0] * rank) if bias is None else bias).to(self.device)
        # set requires_grad = True for new parameters
        self.new_LoRA_A.requires_grad_(True)
        self.new_LoRA_B.requires_grad_(True)
        
    def frozen_LoRA_AB(self):
        # append new parameters into existed parameters
        if self.LoRA_A.size() == torch.Size([0]):
            self.LoRA_A.data = self.new_LoRA_A.data
            self.LoRA_B.data = self.new_LoRA_B.data
            self.kn_bias.data = self.new_kn_bias.data
        else:
            self.LoRA_A.data = torch.cat((self.LoRA_A.data, self.new_LoRA_A.data), dim = 1)
            self.LoRA_B.data = torch.cat((self.LoRA_B.data, self.new_LoRA_B.data), dim = 0)
            self.kn_bias.data = torch.cat((self.kn_bias.data, self.new_kn_bias.data), dim = 0)
        # set requires_grad = False for fine-tuned parameters
        self.new_LoRA_A.requires_grad_(False)
        self.new_LoRA_B.requires_grad_(False)

    def clear_LoRA_AB(self):
        # set extra parameters to None
        self.LoRA_A.data = torch.empty(0).to(self.device)
        self.LoRA_B.data = torch.empty(0).to(self.device)
        self.kn_bias.data = torch.empty(0).to(self.device)
        self.new_LoRA_A.data = torch.empty(0).to(self.device)
        self.new_LoRA_B.data = torch.empty(0).to(self.device)
        self.new_kn_bias.data = torch.empty(0).to(self.device)
        # set requires_grad = False for extra parameters
        self.new_LoRA_A.requires_grad_(False)
        self.new_LoRA_B.requires_grad_(False)


    def forward(self, x):
        # cache mlp input x
        self.cached_state = x.clone().detach()
        # original forward() function
        base_output = self.mlp(x)
        edit_output = 0
        x_norm = rmsnorm(x)
        self.norm_cached_state = x_norm.clone().detach()
        if self._train:
            if self.new_LoRA_A.size() != torch.Size([0]):
                # run forward() with extra knowledge neuron
                edit_intermediate = self.actv(torch.matmul(x_norm, self.new_LoRA_A), self.new_kn_bias)
                # No dropout
                edit_output = torch.matmul(edit_intermediate, self.new_LoRA_B)
                # run forward with previous editted paramaters
                if self.forward_method == "iterative" and self.LoRA_A.size() != torch.Size([0]):
                    prev_edit_intermediate = self.actv(torch.matmul(x_norm, self.LoRA_A), self.kn_bias)
                    edit_output += torch.matmul(prev_edit_intermediate, self.LoRA_B)
        elif self.LoRA_A.size() != torch.Size([0]):
            if self.max_pooling:
                # only activate one neuron during evaluation
                activated_states = self.actv(torch.matmul(x_norm, self.LoRA_A), self.kn_bias)
                activated_neuron_value, activated_neuron_idx = activated_states.max(dim = -1)
                edit_output += activated_neuron_value.unsqueeze(-1) * self.LoRA_B[activated_neuron_idx]
            else:
                # activate all neurons during evaluation
                activated_states = self.actv(torch.matmul(x_norm, self.LoRA_A), self.kn_bias)
                edit_output += torch.matmul(activated_states, self.LoRA_B)
        
        # apply new knowledge neurons to tokens marked by pred_mask
        if hasattr(self, "input_dict"):
            attention_mask = self.input_dict["pred_mask"]
            # during inferrence stage, sequence length = 1
            if x.size(1) == attention_mask.size(1) and self.seq_sampling:
                edit_output *= attention_mask.unsqueeze(-1).float()
        output_states = base_output + self.alpha * edit_output
        return output_states

class EditModel(object):
    def __init__(
        self, 
        hparams, 
        model, 
        tokenizer, 
        device,
        train_args = None, 
        logger = logging.init_logger()
    ):
        self.hparams = hparams
        self.train_args = train_args
        self.model = model
        self.tok = tokenizer
        self.device = device
        self.logger = logger

        if self.train_args:
            train_args.learning_rate = self.hparams.edit_lr
            train_args.num_epoch = self.hparams.edit_epoch

        self.init_model()

    def init_model(self):
        # replace MLP layers of transformer with KN-adapted MLP Block
        self.replace_with_edited_block()
        #
        build_cached_forward(self.model, self.hparams)

    def run_edit(self, dataloader):
        # reset iteration
        dataloader._reset_()
        # output the global configuration of editing
        self.logger.info(
            f"Editing Configuration:\n" + "\n".join(
                [f"{key}: {value}" for key, value in self.hparams.__dict__.items() if key != "mlp_layers"]
            )
        )

        trainer = KNStepTrainer(
            model = self.model,
            tokenizer = self.tok,
            loss_func = SeqGenCrossEntropyLoss(label_smoothing = self.hparams.label_smoothing),
            logger = self.logger,
            device = self.device,
            hparams = self.hparams,
            train_args = self.train_args,
        )

        # start editing
        for i, batch in enumerate(dataloader):
            self.logger.info(f"Example {i}/{len(dataloader)}:")
            # locate the layer to be edited
            layer, key_states, value_states, shifts = trace_back_golden(
                model = self.model, 
                hparams = self.hparams,
                prompt = batch["src"], 
                alter = batch["alt"],
                device = self.device,
            )
            self.logger.info(f"Editing parameters at layer {layer}.")
            for idx, (key_state, value_state, bias) in enumerate(zip(key_states, value_states, shifts)):
                edit_block = get_attr(self.model, self.hparams.mlp_layers[layer])
                # initialize learnable parameters
                key_state = key_state.unsqueeze(1).repeat([1, self.hparams.prank])
                value_state = value_state.unsqueeze(0).repeat([self.hparams.prank, 1])
                bias = bias.repeat([self.hparams.prank])
                if not self.hparams.weight_init:
                    key_state = None
                    value_state = None
                if not self.hparams.bias_init:
                    bias = 0
                edit_block.init_LoRA_AB(
                    rank = self.hparams.prank, 
                    A_weight = key_state, 
                    B_weight = value_state,
                    bias = bias,
                )
                if self.hparams.edit_tokens == "first":
                    break
            
            if self.hparams.fine_tuning:
                self.train()
                base_loss_items, loc_loss_items, recon_loss_items, pred_acc = trainer.train_examples(batch["src"], batch["alt"], batch["aux"])
                self.logger.info(
                    "Example {}/{} finished, Loss: {:.4f} + {:.4f} + {:.4f} = {:.4f} -> {:.4f} + {:.4f} + {:.4f} = {:.4f},"
                    " Accuracy: {:.4f}".format(
                    i, len(dataloader), 
                    base_loss_items[0], loc_loss_items[0], recon_loss_items[0], base_loss_items[0] + loc_loss_items[0] + recon_loss_items[0], 
                    base_loss_items[-1], loc_loss_items[-1], recon_loss_items[-1], base_loss_items[-1] + loc_loss_items[-1] + recon_loss_items[-1], 
                    pred_acc
                ))
                # frozen learnable parameters
                edit_block = get_attr(self.model, self.hparams.mlp_layers[layer])
                edit_block.frozen_LoRA_AB()

            else:
                self.eval()
                edit_block = get_attr(self.model, self.hparams.mlp_layers[layer])
                edit_block.frozen_LoRA_AB()

        # save checkpoints
        self.save_lora_weight()


    def replace_with_edited_block(self):
        for block_name in self.hparams.mlp_layers:
            mlp_block = get_attr(self.model, block_name)
            if not isinstance(mlp_block, EditedMLPLayer):
                edited_mlp_block = EditedMLPLayer(
                    mlp_block = mlp_block,
                    d_model = self.hparams.d_model,
                    actv = ACT2FN[self.hparams.actv],
                    forward_method = self.hparams.forward_method,
                    alpha = self.hparams.alpha,
                    max_pooling = self.hparams.max_pooling,
                    seq_sampling = self.hparams.seq_sampling,
                )
                set_attr(self.model, block_name, edited_mlp_block)

    def clear_edited_block(self):
        for block_name in self.hparams.mlp_layers:
            mlp_block = get_attr(self.model, block_name)
            if isinstance(mlp_block, EditedMLPLayer):
                mlp_block.clear_LoRA_AB()

    def save_lora_weight(self):
        saved_dict = {}
        for name, param in self.model.named_parameters():
            if "LoRA" in name or "kn" in name:
                saved_dict.update({
                name: param
            })
        self.logger.info("Saving weight at {}.".format(self.hparams.saved_path))
        torch.save(saved_dict, self.hparams.saved_path)

    def merge_lora_weight(self, weight_path):
        self.logger.info("Loading weight from {}.".format(weight_path))
        state_dict = torch.load(weight_path, map_location=self.device)
        for name, param in state_dict.items():
            block = get_attr(self.model, name)
            block.data = param

    def write_result(self, path, obj):
        self.logger.info(f"Writing results at {path}.")
        with open(path, "w+", encoding = "utf-8") as f:
            json.dump(obj, f, indent = 2, ensure_ascii=False)

    def train(self):
        for block_name in self.hparams.mlp_layers:
            mlp_block = get_attr(self.model, block_name)
            mlp_block.to_train()
    
    def eval(self):
        for block_name in self.hparams.mlp_layers:
            mlp_block = get_attr(self.model, block_name)
            mlp_block.to_eval()

def trace_back_golden(model, hparams, prompt, alter, device):
    # Get embedding weight
    emb = get_attr(model, hparams.embedding_weight)
    
    src = copy.deepcopy(prompt).to(device)
    alt = copy.deepcopy(alter).to(device)
    # Assume the batch_size = 1
    src_len = src.attention_mask.sum()
    alt_len = alt.attention_mask.sum()

    src.input_ids, src.attention_mask, _ = concat_masked_indexes(
        src = src.input_ids,
        src_mask = src.attention_mask,
        tgt = alt.input_ids,
        tgt_mask = alt.attention_mask,
        max_len = src.input_ids.size(-1),
    )
    
    with torch.no_grad():
        hidden_states = model.forward(**src, output_hidden_states=True).hidden_states[1:]

        
    # tgt_layers, key_states, value_states, shifts = [], [], [], []
    # golden answer is successfully predicted if its probability > 0.1
    
        
    tgt_layer = hparams.edit_layer
    tgt_block = get_attr(model, hparams.mlp_layers[tgt_layer])
    key_states = tgt_block.cached_state[0, src_len - 1:src_len - 1 + alt_len]
    # apply length penalty
    value_states = torch.index_select(emb, 0, alt.input_ids[alt.attention_mask == 1])

    # normalize
    key_states = rmsnorm(key_states)

    full_shifts = torch.matmul(key_states, key_states.T).diag()
    shifts = full_shifts * hparams.shifted_bias_factor

    return tgt_layer, key_states, value_states, shifts