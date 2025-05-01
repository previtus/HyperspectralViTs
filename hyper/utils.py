import torch
from typing import Union, Dict
from torch import Tensor
import os
import os.path
import json
import numpy as np
import pandas as pd
from distutils.dir_util import copy_tree
import shutil

def to_device(x: Union[Dict[str, Tensor], Tensor], device: torch.device) -> Union[Dict[str, Tensor], Tensor]:
    if torch.is_tensor(x):
        return x.to(device)
    elif hasattr(x, "keys"):
        return {k: to_device(v, device) for k, v in x.items()}
    return x

def check_headless():
    # check if we are running locally or on cluster (naturally, rename to your hostname...)
    return not os.popen('hostname').read().strip() == 'vitek-blade'

def diff_dicts(a, b, missing=KeyError):
    # https://stackoverflow.com/questions/32815640/how-to-get-the-difference-between-two-dictionaries-in-python
    """
    Find keys and values which differ from `a` to `b` as a dict.

    If a value differs from `a` to `b` then the value in the returned dict will
    be: `(a_value, b_value)`. If either is missing then the token from
    `missing` will be used instead.

    :param a: The from dict
    :param b: The to dict
    :param missing: A token used to indicate the dict did not include this key
    :return: A dict of keys to tuples with the matching value from a and b
    """
    return {
        key: (a.get(key, missing), b.get(key, missing))
        for key in dict(
            set(a.items()) ^ set(b.items())
        ).keys()
    }

def compare_settings(setting, loaded_setting):
    print("Comparing current settings with the loaded one:")
    avoid_keys = ["experiment_path", "experiment_name"]
    avoid_subkeys = ["load_path"]

    same = True
    for key in setting.keys():
        if key in avoid_keys: continue

        if setting[key] == loaded_setting[key]:
            continue

        differences = diff_dicts(setting[key], loaded_setting[key])
        diff_keys = [k for k in differences.keys() if k not in avoid_subkeys]
        if len(diff_keys)>0:
            print(key, "differs in:")
            for k in differences.keys():
                print(" - ", k, "Currently", differences[k][0], ", but in the previous:", differences[k][1])

                same = False
            # print(differences)
            print("---")
    print("\ Comparison finished.")
    print("")
    return same

class CustomJSONEncoder(json.JSONEncoder):

    def default(self, obj_to_encode):
        """Pandas and Numpy have some specific types that we want to ensure
        are coerced to Python types, for JSON generation purposes. This attempts
        to do so where applicable.
        """
        # Pandas dataframes have a to_json() method, so we'll check for that and
        # return it if so.
        if hasattr(obj_to_encode, "to_json"):
            return obj_to_encode.to_json()
        # Numpy objects report themselves oddly in error logs, but this generic
        # type mostly captures what we're after.
        if isinstance(obj_to_encode, np.generic):
            return obj_to_encode.item()
        # ndarray -> list, pretty straightforward.
        if isinstance(obj_to_encode, np.ndarray):
            return obj_to_encode.tolist()
        # if isinstance(obj_to_encode, Polygon):
        #     return mapping(obj_to_encode)
        if isinstance(obj_to_encode, pd.Timestamp):
            return obj_to_encode.isoformat()
        # if isinstance(obj_to_encode, datetime):
        #     return obj_to_encode.isoformat()
        # torch or tensorflow -> list, pretty straightforward.
        if hasattr(obj_to_encode, "numpy"):
            return obj_to_encode.numpy().tolist()
        # If none of the above apply, we'll default back to the standard JSON encoding
        # routines and let it work normally.
        return super().default(obj_to_encode)

def file_exists(file_path):
    return os.path.isfile(file_path)
def path_exists(p):
    return os.path.exists(p)

def copy_files_folders(source_files, to_directory):
    for source in source_files:
        if os.path.isdir(source):
            # if folder doesn't exist (we don't check the contents...)
            if not path_exists(to_directory):
                copy_tree(source, to_directory)
        else:
            # if we don't have the same file already?
            if not path_exists(os.path.join(to_directory, os.path.basename(source))):
                shutil.copy(source, to_directory)

def find_latest_checkpoint_v2(expetiment_folder):
    # Directly use the "last.ckpt"
    last_candidate = os.path.join(expetiment_folder,"last.ckpt")
    if file_exists(last_candidate):
        print("Found the last checkpoint in -> ", last_candidate)
        return True, last_candidate
    else:
        print("Didn't find the last checkpoint candidates!")
        return False, []

def safely_delete_file(file_path):
    try:
        os.remove(file_path)
    except Exception as e:
        print("Failed deleting", file_path)
        print("with", e)

def copy_file(source, target):
    try:
        shutil.copyfile(source, target)
    except Exception as e:
        print("Failed copying!", source, target)
        print("with", e)
