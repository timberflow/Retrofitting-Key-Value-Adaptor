# Initializing and Retrofitting Key Value Adaptor
Coming soon!

To reproduce the main result of iReVa, you can execute the following command and pay attention to subsequent points.
```
# zsRE
python run_editing.py                             \
--model_name gpt2-xl                              \
--hparams_fname ./hparams/kv/gpt2-xl_zsRE.json    \
--dataset_size_limit 10000

# PARAREL
python run_editing.py                             \
--model_name gpt2-xl                              \
--hparams_fname ./hparams/kv/gpt2-xl_PARAREL.json \
--dataset_size_limit 10000    
```

1. The weights of base model is not provided, running the command directly will automatically download the corresponding files into the ./hf_models folder. If you prefer to download the weights manually, you need to place them in the ./hf_models folder and modify the value of --model_name to the path where you download the weights. Make sure that the weight file of model xxx is saved at ./hf_models/xxx (e.g. ./hf_models/gpt2-xl).

2. The checkpoint of iReVa will be saved at ./checkpoints, make sure that it does not exist before you reproduce iReVa to avoid latent risk.

3. To modify hyper-parameters, you can change the specific configuration in ./hparams/kv/. For example, you can convert the value of "fine_tuning" to false in ./hparams/kv/gpt2-xl_zsRE.json to undo the fine-tuning setup in iReVa.


To reproduce baselines, please check the setup procedures in ./baselines. After completing the preparations, run following example commands in this directory to see baselines' behaviours on zsRE and PARAREL. Moreover, you can modify the argmument "--alg_name" and "--hparams_fname" to try different baselines and "--ds_name" to experiment with different benchmarks for FT, MEND, ROME MEMIT; for MELO, modify the "+experiment" simply.
```
# Fine Tuning
python -m baselines.run_baselines           \
--model_name gpt2-xl                        \
--alg_name FT                               \
--ds_name zsre                              \
--hparams_fname ./hparams/ft/gpt2-xl.json   \
--dataset_size_limit 10000

# MEND
python -m baselines.run_baselines           \
--model_name gpt2-xl                        \
--alg_name MEND                             \
--ds_name zsre                              \
--hparams_fname ./hparams/mend/gpt2-xl.json \
--dataset_size_limit 10000

# ROME
python -m baselines.run_baselines           \
--model_name gpt2-xl                        \
--alg_name ROME                             \
--ds_name zsre                              \
--hparams_fname ./hparams/rome/gpt2-xl.json \
--dataset_size_limit 10000

# MEMIT
python -m baselines.run_baselines           \
--model_name gpt2-xl                        \
--alg_name MEMIT                            \
--ds_name zsre                              \
--hparams_fname ./hparams/memit/gpt2-xl.json\
--dataset_size_limit 10000

# MELO
python3 -m baselines.run_melo +alg=lora +experiment=zsre +model=gpt2xl
```
