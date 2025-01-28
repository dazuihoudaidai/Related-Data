import json
import os
import pickle
import subprocess
import numpy as np
import torch


def compute_sample_weights(target, class_weights):
    return torch.Tensor([class_weights[t.int()] for t in target])


def adjust_negative_edges(target):
    return target - 1, [i for i, t in enumerate(target) if t.item() != 0]


def save_dict_to_json(data_dict, file_path):
    if os.path.isfile(file_path):
        with open(file_path, encoding="utf-8") as f:
            existing_data = json.load(f)
    else:
        existing_data = {}
    
    existing_data.update({str(k): v for k, v in data_dict.items()})
    
    with open(file_path, "w+", encoding="utf-8") as f:
        json.dump(existing_data, f, ensure_ascii=False, indent=4)


def vectorize_dataframe_with_numpy(df, columns: list):
    return np.stack([df[col].to_numpy() for col in columns], axis=-1).tolist()


def scale_to_unit_range(x):
    return (x - x.min()) / (x.max() - x.min())


def create_directories(directories):
    assert isinstance(directories, list)
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def save_model_state(model, directory, file_name):
    model_path = os.path.join(directory, f"{file_name}.pt")
    torch.save(model.state_dict(), model_path)
    print(f"Model saved at {model_path}.")


def load_model_state(model, checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))


def remove_files_from_directory(directory_path):
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)
        if os.path.isfile(item_path):
            os.remove(item_path)
        else:
            remove_files_from_directory(item_path)
