import os
import json
import numpy as np


def read_json(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No path is located at {path}")
    with open(path, "r") as file:
        json_data = json.load(file)
    return json_data

def read_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"No path is located at {path}")
    array = np.load(path)
    return array['data']
