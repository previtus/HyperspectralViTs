# Normalises data given information from which band they are
# - can do multiple normalisation schemes

import torch
from torch.nn import Module
from hyper.parameters.normaliser_values import BAND_NORMALIZATION
from hyper.data.data_utils import aviris_band_name_to_wv, emit_band_name_to_wv, file_exists_and_not_empty
import numpy as np
from tqdm import tqdm
import os
from hyper.data.data_utils import load_band_names_as_numpy, load_emit_data_as_tensors


class DataNormaliser(Module):
    def __init__(self, settings, dataset, input_product_names, output_product_names, auxiliary_product_names):
        super().__init__()
        self.debug_normaliser = False

        self.settings = settings
        self.dataset = dataset

        self.BAND_NORMALIZATION = BAND_NORMALIZATION

        if self.dataset.mode != "train":
            print("Not a train dataset, won't set the normalisation at all! Later load these from the train set!")
            return None

        override_products = settings.dataset.normalisation.override_products
        if self.settings.dataset.normalisation.mode_input == "from_data" and len(override_products) > 0:
            print("Will exclude these specific products from the 'from data' cooking")
            print("Products to exclude:", override_products)
            print("While we have input_product_names:", input_product_names)

            input_product_names = [prod for prod in input_product_names if prod not in override_products]
            params_for_overriden = self.init_params(override_products)
            print("^ reduced input_product_names to", input_product_names)

        if self.settings.dataset.normalisation.mode_input == "hardcoded":
            print("Using hardcoded normalisation params")

            # load input normalisation parameters from hardcoded values
            self.params_inputs_np = self.init_params(input_product_names)

        elif self.settings.dataset.normalisation.mode_input == "from_data":
            print("This is train dataset, will set the normalisation params")

            self.max_style = self.settings.dataset.normalisation.max_style
            # compute input normalisation parameters from data
            cache_file = os.path.join(settings.dataset.feature_folder, settings.dataset.normalisation.save_load_from+".npy")
            already_computed = file_exists_and_not_empty(cache_file)
            print("searching for normalisation file in:", cache_file, "found?", already_computed)
            if self.settings.debug.recalculate_normalisation: already_computed = False
            if already_computed:
                self.params_inputs_np = np.load(cache_file)
                print("loaded computed statistics from", cache_file)
            else:
                self.params_inputs_np = self.compute_params_prep(input_product_names)
                print("saved computed statistics to", cache_file)
                np.save(cache_file, self.params_inputs_np)

        if self.settings.dataset.normalisation.mode_input == "from_data" and len(override_products) > 0:
            print("after getting the from data normalisation values, we will also add these:", params_for_overriden.shape, "for", override_products)
            self.params_inputs_np = np.concatenate((params_for_overriden, self.params_inputs_np), axis=1)

        print("debug 1st band params ~", self.params_inputs_np[:,0])

        self.params_inputs = self.use_params(self.params_inputs_np)

        self.params_aux_np = self.init_params(auxiliary_product_names)
        self.params_aux = self.use_params(self.params_aux_np)

        self.params_outputs_np = self.init_params(output_product_names)
        self.params_outputs = self.use_params(self.params_outputs_np)

    def get_params(self):
        return self.params_inputs_np, self.params_aux_np, self.params_outputs_np

    def set_params(self, params_inputs_np, params_aux_np, params_outputs_np):
        self.params_inputs = self.use_params(params_inputs_np)
        self.params_aux = self.use_params(params_aux_np)
        self.params_outputs = self.use_params(params_outputs_np)

    def init_params(self, product_names):
        if not len(product_names) > 0:
            return None

        offsets = []
        factors = []
        clip_min = []
        clip_max = []

        for p in product_names:
            # example: TOA_AVIRIS_640nm
            if "TOA_AVIRIS_" in p:
                p = aviris_band_name_to_wv(p)
            elif "EMIT_" in p:
                p = emit_band_name_to_wv(p) # this catches both specific products and band ranges

            p_dbg = p
            if p not in self.BAND_NORMALIZATION.keys():
                if not isinstance(p,int) and "mf_" in p:
                    # cooked features for matched filter use different normalisation params
                    p = 'mag1c_default'
                else:
                    p = 'default'

            if self.debug_normaliser:
                print("for product", p_dbg, "i found", BAND_NORMALIZATION[p])
            offsets.append(BAND_NORMALIZATION[p]["offset"])
            factors.append(BAND_NORMALIZATION[p]["factor"])
            clip_min.append(BAND_NORMALIZATION[p]["clip"][0])
            clip_max.append(BAND_NORMALIZATION[p]["clip"][1])

        params_np = np.zeros((4,len(offsets)))
        params_np[0] = offsets
        params_np[1] = factors
        params_np[2] = clip_min
        params_np[3] = clip_max
        return params_np

    def compute_params_prep(self, product_names):
        dataframe = self.dataset.dataframe
        root_folder = self.settings.dataset.root_folder

        # Note: seems like most data has min at 0, or small value above, with 0 also being no-data value
        num_bands = len(product_names)
        num_samples = len(dataframe)
        max_values = np.zeros( (num_bands, num_samples) )

        for idx in tqdm(range(len(dataframe))):
            # note: still before tiling
            row = dataframe.iloc[idx]
            if "folder" in row.keys():
                event_name = row["folder"].split("/")[-1]  # for older AVIRIS csvs
            elif "event_id" in row.keys():
                event_name = row["event_id"]  # for newer EMIT csvs

            event_folder = os.path.join(root_folder, event_name)
            window = None

            if self.dataset.format == "AVIRIS":
                Xs = load_band_names_as_numpy(product_names, window, event_folder)
            elif self.dataset.format == "EMIT" or self.dataset.format == "EMIT_MINERALS":
                x_tensors = load_emit_data_as_tensors(product_names, window, event_folder, alternative_folder=None, available_bands=self.dataset.available_bands)
                x_tensors = torch.nan_to_num(x_tensors, nan=0.0)
                Xs = x_tensors.numpy()

            for band_i in range(len(Xs)):
                if self.max_style == "max":
                    max_values[band_i][idx] = float(np.max(Xs[band_i]))
                elif self.max_style == "max_outliers":
                    p = self.settings.dataset.normalisation.max_outlier_percentile
                    max_values[band_i][idx] = float(np.max(no_outliers(Xs[band_i], percentile=p)))

        max_value = np.max(max_values, axis=1)
        min_value = np.zeros_like(max_value)

        clip_min = np.zeros_like(max_value)
        clip_max = np.ones_like(max_value) * 2

        # normalise between 0-1 - with clip between 0,2
        #  data = (data - min(data)) / (max(data) - min(data))
        #  offsets = min(data), factors = max(data) - min(data)

        offsets = min_value
        factors = max_value #- min_value
        params_np = np.zeros((4,len(offsets)))
        params_np[0] = offsets
        params_np[1] = factors
        params_np[2] = clip_min
        params_np[3] = clip_max
        return params_np

    def use_params(self, params_np):
        if params_np is None: return None
        offsets = torch.nn.Parameter(torch.from_numpy(np.array(params_np[0])[:, None, None]),
                                                requires_grad=False)  # (C, 1, 1)
        factors = torch.nn.Parameter(torch.from_numpy(np.array(params_np[1])[:, None, None]),
                                                requires_grad=False)
        clip_min = torch.nn.Parameter(torch.from_numpy(np.array(params_np[2])[:, None, None]),
                                                 requires_grad=False)
        clip_max = torch.nn.Parameter(torch.from_numpy(np.array(params_np[3])[:, None, None]),
                                                 requires_grad=False)

        params = [offsets, factors, clip_min, clip_max]
        return params

    def normalize_x(self, x):
        offsets, factors, clip_min, clip_max = self.params_inputs
        return torch.clamp((x-offsets) / factors, clip_min, clip_max).float()

    def denormalize_x(self, x):
        offsets, factors, clip_min, clip_max = self.params_inputs
        return (x * factors) + offsets

    def normalize_y(self, y):
        offsets, factors, clip_min, clip_max = self.params_outputs
        return torch.clamp((y-offsets) / factors, clip_min, clip_max).float()

    def denormalize_y(self, y):
        offsets, factors, clip_min, clip_max = self.params_outputs
        return (y * factors) + offsets

    def normalize_aux(self, aux):
        if self.params_aux is None:
            return aux

        offsets, factors, clip_min, clip_max = self.params_aux
        return torch.clamp((aux-offsets) / factors, clip_min, clip_max).float()

    def denormalize_aux(self, aux):
        if self.params_aux is None:
            return aux

        offsets, factors, clip_min, clip_max = self.params_aux
        return (aux * factors) + offsets


def no_outliers(d, percentile=5):
    upper_quartile = np.percentile(d, 100-percentile)
    lower_quartile = np.percentile(d, percentile)
    return d[np.where((d >= lower_quartile) & (d <= upper_quartile))]
