import random
import importlib
import logging
import hydra
from omegaconf import OmegaConf,open_dict
import numpy as np
import torch
from torch.utils.data import DataLoader
from baselines.melo.utils import *
import baselines.melo.models as models
from baselines.melo.trainer import zsre_trainer

from evaluate.evaluate import eval_qa
from evaluate.summarize_utils import summarize_rewrite_quality
from data import DS_DICT

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

OmegaConf.register_new_resolver("uuid", lambda: uuid())
LOG = logging.getLogger(__name__)
@hydra.main(config_path='./melo/config', config_name='config')
def run(config):
    grace_config_keys = ['edit_lr','init_radius','expand_mode','key_id','num_edit_per_block','num_block','num_rank_per_block']
    model_config_keys = ['target_modules','grace_layer']
    GRACE_CONFIG = dict(config.grace)
    MODEL_CONFIG = dict(config.model)

    for k in grace_config_keys:
        LOG.info(f'[-GRACE CONFIG-]  {k}: {GRACE_CONFIG[k]}')
    for k in model_config_keys:
        LOG.info(f'[-MODEL CONFIG-]  {k}: {MODEL_CONFIG[k]}')

    base_dir = hydra.utils.get_original_cwd()
    with open_dict(config):
        config.base_dir = base_dir

    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    if config.task == "pararel" or config.task == "zsre":
        model = models.get_hf_model(config)
    else:
        print(f"{config.task} task not found")

    model.to(config.device)
    tokenizer = models.get_tokenizer(config)

    '''
    Load Dataset
    '''
    if config.task == "pararel" or config.task == "zsre":
        from baselines.melo.dataset import zsRE_balanced, PARAREL_balanced
        from baselines.melo.metrics import gen_ACC
        ds_dict = {"zsre": zsRE_balanced, "pararel": PARAREL_balanced}
        balanced_dataset = ds_dict[config.task]

        edits = balanced_dataset(split="edit", n_edits=config.max_n_edits)
        edit_holdouts = balanced_dataset(split="holdout", n_edits=config.max_n_edits)
        upstream = balanced_dataset(split="upstream", n_edits=config.max_n_edits)

        '''Get Loaders
        '''
        batch_size = config.grace.num_edit_per_block
        edit_loader = DataLoader(edits, batch_size=batch_size, shuffle=True)
        edit_holdout_loader = DataLoader(edit_holdouts, batch_size=batch_size, shuffle=False)
        upstream_loader = DataLoader(upstream, batch_size=batch_size, shuffle=False)
        '''Define Metrics
        '''
        metric = gen_ACC # Measure QA F1
        tokenize = tokenize_qa
    else:
        print(f"{config.task} task not found")

    alg_module = importlib.import_module(f'baselines.melo.algs.{config.alg}')
    AlgClass = getattr(alg_module,config.alg.upper())
    alg = AlgClass(model,config,tokenizer)
    alg.to(config.device)

    # Trainer
    if config.task == "pararel" or config.task == "zsre":
        trainer = zsre_trainer(config,alg,tokenize,metric,edit_loader,upstream_loader,edit_holdout_loader)

    # trainer.pre_editing_analyse()
    torch.cuda.empty_cache()
    alg = trainer.run_edit()

    ds = DS_DICT[config.task](
        data_dir = config.eval_data_dir,
        tok = alg.model_tok
    )
    metrics = eval_qa(
        model = alg,
        tok = alg.model_tok,
        records = ds,
        ds_name = config.task,
        dir_to_save = config.eval_result_dir,
    )
    summarize_rewrite_quality(metrics)


if __name__ == '__main__':
    run()