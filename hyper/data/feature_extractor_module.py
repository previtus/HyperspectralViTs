# Various feature extraction from existing data. (Ex: ratio between two bands)
import torch
from torch.nn import Module
from hyper.features.cook_features import matched_filter

class DataFeatureExtractor(Module):
    def __init__(self, settings, dataset):
        self.settings = settings
        self.dataset = dataset

        self.feature_extractor = self.settings.dataset.feature_extractor

        self.active = len(self.feature_extractor) > 0 # active, if we have any ops on the list

    def process(self, dataframe):
        print("Running Feature Extractor for dataset", self.dataset.mode, "with settings:", self.feature_extractor)

        for operation in self.feature_extractor:
            name = operation[0]
            print("Feature:", name)

            if name == "matched_filter":
                matched_filter(dataframe, operation, self.settings, self.dataset) # note: caching occurs inside

        return dataframe

