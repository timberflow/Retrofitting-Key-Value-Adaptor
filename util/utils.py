import os
import glob
import pathlib
import random
import numpy as np
import torch

def set_seed(seed=42):  # torch.manual_seed(42) is all u need
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

def is_dir_empty(directory):
    path = pathlib.Path(directory)
    if path.exists() and path.is_dir() and not any(path.iterdir()):
        return True
    return False

def load_vec_states(states_dir, num_layers, num_part = 2):
    base_dir = pathlib.Path(states_dir)
    dir_list = glob.glob(str(base_dir / "part-[0-9]*"))

    meta_data = []
    embedding_data = [[] for _ in range(num_layers)]

    random.shuffle(dir_list)
    # load hidden states
    for i, direc in enumerate(dir_list):
        if i >= num_part:
            break

        file_iter = pathlib.Path(direc).iterdir()
        for file in file_iter:
            data = torch.load(file)
            if file.name == "labels.pt":
                meta_data.append(data)
            else:
                n_layer = int(file.name.split(".")[1])
                embedding_data[n_layer].append(data)

    meta_data = torch.cat(meta_data, dim = 0)
    for i in range(len(embedding_data)):
        embedding_data[i] = torch.cat(embedding_data[i], dim = 0).unsqueeze(1)
    embedding_data = torch.cat(embedding_data, dim = 1)

    return embedding_data, meta_data