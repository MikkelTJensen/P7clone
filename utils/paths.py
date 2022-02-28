import os
from natsort import natsorted

def get_folders_in_folder(path: str) -> list:
    return natsorted([os.path.join(path, name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))])

# Same like the first one, just returns only inner folder names
def get_folders(path: str) -> list:
    return natsorted([os.path.join(name)
            for name in os.listdir(path)
            if os.path.isdir(os.path.join(path, name))])
