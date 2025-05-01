# Pytorch Lightning Datamodule wrapper around the Dataset
import os.path

import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from hyper.data.dataset import HyperDataset
from hyper.data.dataset_speedy_wrapper import SpeedyWrapperDataset
from timeit import default_timer as timer
from types import MethodType

import pandas as pd
import numpy as np

def add_sample_weight(dataframe:pd.DataFrame) -> pd.DataFrame:
    # Assumes the information in "isplume" is correct - and it is when we recompute these
    plume_fraction = np.sum(dataframe["has_plume"]) / dataframe.shape[0]
    plume_weight = 1 / plume_fraction
    non_plume_weight = 1 / (1 - plume_fraction)
    dataframe["sample_weight"] = dataframe["has_plume"].apply(
        lambda x: plume_weight if x else non_plume_weight)
    return dataframe


class HyperDataModule(pl.LightningDataModule):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings

        self.num_workers = settings.dataloader.num_workers
        self.train_batch_size = settings.dataloader.train_batch_size
        self.val_batch_size = settings.dataloader.val_batch_size

        self.has_validation = False # Then it just uses train + test
        if settings.dataset.val_csv != "":
            self.has_validation = True # Will also use val

        self.weighted_random_sampler = settings.model.weighted_random_sampler

        self.save_hyperparameters()

    def setup(self, stage):
        """Initialize the main ``Dataset`` objects.
        This method is called once per GPU per run.
        Args:
            stage: stage to set up
        """

    def prepare_data(self):
        """
        Make sure that the dataset is downloaded.
        This method is only called once per run.
        """

        use_wrapper = self.settings.dataset.presave_to_scratch
        if use_wrapper:
            if self.settings.dataset.path_to_scratch == "":
                path_to_scratch = os.environ['SCRATCH']
                print("Scratch path found:", path_to_scratch)
                self.settings.dataset.path_to_scratch = path_to_scratch
            wrapper_path = self.settings.dataset.path_to_scratch

        if use_wrapper:
            self.train_dataset = SpeedyWrapperDataset(HyperDataset(self.settings, "train"),
                                                      wrapper_path=os.path.join(wrapper_path, "train"))
            if self.has_validation:
                self.val_dataset = SpeedyWrapperDataset(HyperDataset(self.settings, "val", sort_df_by_plume_size=True),
                                                        wrapper_path=os.path.join(wrapper_path, "val"))
            self.test_dataset = SpeedyWrapperDataset(HyperDataset(self.settings, "test", sort_df_by_plume_size=True),
                                                 wrapper_path=os.path.join(wrapper_path, "test"))

        else:
            self.train_dataset = HyperDataset(self.settings, "train")
            if self.has_validation:
                self.val_dataset = HyperDataset(self.settings, "val", sort_df_by_plume_size=True)
            self.test_dataset = HyperDataset(self.settings, "test", sort_df_by_plume_size=True)

        i, a, o = self.train_dataset.normaliser.get_params() # these may be fitted on the train data

        if self.has_validation:
            self.val_dataset.normaliser.set_params(i, a, o)

        self.test_dataset.normaliser.set_params(i, a, o)

        if use_wrapper:
            # Cook samples of wrapper ...
            start = timer()
            self.train_dataset.save_all_samples_as_files()
            if self.has_validation:
                self.val_dataset.save_all_samples_as_files()
            self.test_dataset.save_all_samples_as_files()
            end = timer()
            time = (end - start)
            print("Wrapping dataset took: "+str(time)+"s ("+str(time/60.0)+"min)")

        # This seems hacky, but otherwise pl wants to use the val dataset even if we don't have anyone set...
        # So we basically override the function with this only if its needed ...
        if self.has_validation:
            self.val_dataloader = MethodType(outside_val_dataloader, self)


    def train_dataloader(self, num_workers = None, batch_size = None, shuffle=True):
        """Initializes and returns the training dataloader"""
        self.train_batch_size = batch_size or self.train_batch_size
        num_workers = num_workers or self.num_workers

        if self.weighted_random_sampler:
            # Set weight per sample
            train_dataframe = add_sample_weight(self.train_dataset.dataframe)

            weight_random_sampler = WeightedRandomSampler(train_dataframe["sample_weight"].values,
                                                          num_samples=len(self.train_dataset),
                                                          replacement=True)  # Must be true otherwise we should lower num_samples
            print("Using WeightedRandomSampler in the train dataset.")
            shuffle = False
        else:
            weight_random_sampler = None
            # keep shuffle from the argument (default = True)

        return DataLoader(self.train_dataset, batch_size=self.train_batch_size,
                          num_workers=num_workers, sampler=weight_random_sampler,
                          shuffle=shuffle)

    def test_dataloader(self, num_workers = None, batch_size = None):
        """Initializes and returns the test dataloader"""
        num_workers = num_workers or self.num_workers
        self.test_batch_size = batch_size or self.val_batch_size # use same from the val setting...
        return DataLoader(self.test_dataset, batch_size=self.test_batch_size,
                            shuffle=False, num_workers=num_workers)

    def summarise(self):
        print("[Data] / Train (batch size", self.train_batch_size,")")
        self.train_dataset.summarise()
        if self.has_validation:
            print("[Data] / Val (batch size", self.train_batch_size,")")
            self.val_dataset.summarise()
        print("[Data] / Test (batch size", self.train_batch_size,")")
        self.test_dataset.summarise()

def outside_val_dataloader(self, num_workers = None, batch_size = None):
    """Initializes and returns the validation dataloader"""
    num_workers = num_workers or self.num_workers
    self.val_batch_size = batch_size or self.val_batch_size
    return DataLoader(self.val_dataset, batch_size=self.val_batch_size,
                        shuffle=False, num_workers=num_workers)
