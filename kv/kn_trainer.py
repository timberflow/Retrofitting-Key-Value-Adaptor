import torch
import torch.optim as optim

from kv.pytorch_utils import (
    retain_last_unmasked,
    get_attr,
    rmsnorm
)

class TrainArgs:
    optimizer = "AdamW"
    learning_rate = 5e-2
    betas = (0.9,0.99)
    num_epoch = 5
    weight_decay = 0.
    max_len = 512

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class KNStepTrainer(object):
    def __init__(
        self, 
        model,
        tokenizer,
        loss_func,
        logger,
        device,
        hparams,
        train_args
    ):
        self.model = model
        self.tok = tokenizer
        self.loss_func = loss_func
        self.logger = logger
        self.device = device
        self.hparams = hparams
        self.args = train_args

    def train_examples(self, inputs, target, loc_stats = None):
        # to cuda
        inputs, target = (
            inputs.to(self.device), 
            target.to(self.device), 
        )
        # build optimizer for each example
        opt = optim.AdamW(
            self.model.parameters(),
            lr = self.args.learning_rate,
            betas = self.args.betas,
        )
        # compute prediction mask
        pred_mask = retain_last_unmasked(inputs.attention_mask)

        target.attention_mask[:,1:] = 0

        pred_mask = pred_mask.to(self.device)

        edit_block = get_attr(self.model, self.hparams.mlp_layers[self.hparams.edit_layer])
        k_init = get_attr(edit_block, self.hparams.ft_weight_names[0]).clone().detach()
        
        inputs["pred_mask"] = pred_mask
        basic_loss_items, loc_loss_items, recon_loss_items = [], [], []
        for _ in range(self.args.num_epoch):
            opt.zero_grad()
            logits = self.model(**inputs).logits
            basic_loss = self.loss_func(
                logits = logits, 
                target = target.input_ids, 
                tgt_mask = target.attention_mask,
                pred_mask = pred_mask,
            )

            # calculate locality loss
            loc_loss = torch.tensor(0)
            
            if loc_stats is not None and self.hparams.loc_coefficient > 0:
                edit_key_stats = get_attr(edit_block, self.hparams.ft_weight_names[0])
                edit_kn_bias = get_attr(edit_block, self.hparams.ft_bias_name)
                margin_loss = rmsnorm(loc_stats).matmul(edit_key_stats) - edit_kn_bias
                loc_loss = torch.maximum(margin_loss, torch.zeros_like(margin_loss)).mean()
                loc_loss *= self.hparams.loc_coefficient

            recon_loss = torch.tensor(0)
            if self.hparams.recon_coefficient > 0:
                edit_key_stats = get_attr(edit_block, self.hparams.ft_weight_names[0])
                input_x = edit_block.norm_cached_state[edit_block.input_dict["pred_mask"] == 1]
                recon_loss = torch.square(input_x @ edit_key_stats - input_x @ k_init).mean()
                recon_loss *= self.hparams.recon_coefficient
        
            loss = basic_loss + loc_loss + recon_loss

            loss.backward()
            opt.step()

            loc_loss_items += [loc_loss.cpu().item()]
            basic_loss_items += [basic_loss.cpu().item()]
            recon_loss_items += [recon_loss.cpu().item()]
        
        pred = torch.argmax(logits, dim = -1)[pred_mask == 1]
        gold = target.input_ids[target.attention_mask == 1]
        self.logger.info("gold: {}".format(self.tok.convert_ids_to_tokens(gold)))
        self.logger.info("pred: {}".format(self.tok.convert_ids_to_tokens(pred)))
        pred_acc = torch.sum(pred == gold) / pred_mask.sum()
        return basic_loss_items, loc_loss_items, recon_loss_items, pred_acc.cpu().item()