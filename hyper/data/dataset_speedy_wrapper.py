# Wrapper Dataset
# ! Experimental only implementation

import json
import os.path

import tqdm
import numpy as np
import torch
#from hyper.data.dataset import HyperDataset
from hyper.data.data_utils import mkdir, file_exists

class SpeedyWrapperDataset(object):
    def __init__(self, obj, wrapper_path=""):
        self._wrapped_obj = obj

        self.wrapper_path = wrapper_path
        print("WrapperInit with Scratch path:", self.wrapper_path)
        mkdir(self.wrapper_path)

        self.memory_keys = None

    def __getattr__(self, attr):
        if attr in self.__dict__:
            return getattr(self, attr)
        return getattr(self._wrapped_obj, attr)

    def __len__(self):
        return self._wrapped_obj.__len__()

    def __getitem__(self, idx: int):
        loaded_dict = self.load_sample(idx)
        return loaded_dict

    def save_all_samples_as_files(self):
        L = self.__len__()
        #memory_data = []
        for idx in tqdm.tqdm(range(L)):
            save_path = os.path.join(self.wrapper_path, str(idx)) + ".npz"
            if self.memory_keys is not None:  # allow at least one save - to get to know the keys
                if file_exists(save_path):
                    return True

            sample = self._wrapped_obj[idx]
            self.save_sample(idx, sample, save_path)
            #memory_data.append(memory)
        #self.memory_data = memory_data

    def save_all_samples_memory(self):
        # Would save all to the RAM
        L = self.__len__()
        samples = []
        for idx in tqdm.tqdm(range(L)):
            sample = self._wrapped_obj[idx]
            samples.append(sample)
        self.memory_data = samples

    def save_sample(self, idx, sample, save_path):
        save_path_json = os.path.join(self.wrapper_path, str(idx)) + ".json"
        save_path_csv = os.path.join(self.wrapper_path, str(idx)) + ".csv"

        data_dict = {}
        memory_dict = {}
        for k in sample.keys():
            # print(k, type(sample[k]), type(sample[k]).__name__)
            v = sample[k]
            name = type(v).__name__
            if name == "Tensor":
                data_dict[k] = v.numpy().astype(np.float32) # x, y
            elif name == "ndarray":
                data_dict[k] = v # valid_mask
            else:
                memory_dict[k] = v

        # save main data
        np.savez(save_path, **data_dict)

        # save the small data
        if self.memory_keys is None:
            self.memory_keys = memory_dict.keys()
        csv_data = "|".join(map(str, memory_dict.values()))
        with open(save_path_csv, "w") as f:
            f.write(csv_data)

    def load_sample(self, idx):
        load_path = os.path.join(self.wrapper_path, str(idx)) + ".npz"
        load_path_json = os.path.join(self.wrapper_path, str(idx)) + ".json"
        load_path_csv = os.path.join(self.wrapper_path, str(idx)) + ".csv"
        container = np.load(load_path)
        sample = {name: container[name] for name in container}

        for k in sample.keys():
            sample[k] = torch.from_numpy(sample[k].astype(np.float32))

        with open(load_path_csv, "r") as f:
            csv_data = f.read()
        vs = csv_data.split("|")
        ks = self.memory_keys # times', 'id', 'qplume_fulltile'
        # print(vs, ks)
        for k_i, k in enumerate(ks):
            # print(k, vs[k], type(vs[k]))
            if k == "times":
                sample[k] = int(vs[k_i])
            elif k == "id":
                sample[k] = str(vs[k_i])
            elif k == "qplume_fulltile":
                sample[k] = float(vs[k_i])

        return sample

