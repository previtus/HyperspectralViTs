# Augmentation of data
import torch
from torch.nn import Module
import kornia.augmentation as K

class DataAugmentor(Module):
    def __init__(self, settings, dataset):
        super().__init__()

        self.settings = settings
        self.dataset = dataset

        if dataset.mode == "train":
            # augmentations:
            self.spatial_augmentations = K.AugmentationSequential(
                K.RandomRotation(p=0.5, degrees=90),
                K.RandomHorizontalFlip(p=0.5),
                K.RandomVerticalFlip(p=0.5),
                keepdim=True,
                data_keys=["input"], # , model_output_type] + extra_types,
            )
        else:
            self.spatial_augmentations = []

    def augment(self, products):
        if len(self.spatial_augmentations) == 0:
            return products

        else:
            lenghts_of_products = [len(a) for a in products]
            pre_aug_data = torch.cat(products, dim=0).float()

            # print("pre_aug_data", pre_aug_data.shape)
            post_aug_data = self.spatial_augmentations(pre_aug_data)[0]
            # print("post_aug_data", post_aug_data.shape)

            products_split = []
            start_idx = 0
            for prod_len in lenghts_of_products:
                p = post_aug_data[start_idx:start_idx + prod_len]
                products_split.append(p)
                start_idx += prod_len

            return products_split