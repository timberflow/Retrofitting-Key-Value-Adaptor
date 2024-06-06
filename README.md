# Initializing and Retrofitting Key Value Adaptor
Coming soon!

To reproduce the main result of iReVa, you can execute the following command and pay attention to subsequent points.
```
python run_editing.py --model_name gpt2-xl --hparams_fname ./hparams/kv/gpt2-xl_zsRE.json --dataset_size_limit 10000       # zsRE
python run_editing.py --model_name gpt2-xl --hparams_fname ./hparams/kv/gpt2-xl_PARAREL.json --dataset_size_limit 10000    # PARAREL
```

1. The weights of base model is not provided, running the command directly will automatically download the corresponding files into the ./hf_models folder. If you prefer to download the weights manually, you need to place them in the ./hf_models folder and modify the value of --model_name to the path where you download the weights. Make sure that the weight file of model xxx is saved at ./hf_models/xxx (e.g. ./hf_models/gpt2-xl).

2. The checkpoint of iReVa will be saved at ./checkpoints, make sure that it does not exist before you reproduce iReVa to avoid latent risk.


To reproduce baselines, please check the setup procedures in ./baselines.
