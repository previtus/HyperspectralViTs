# ! Experimental only implementation

import os
import torch
import matplotlib.pyplot as plt

from hyper.data.data_utils import input_products_from_settings_AVIRIS, input_products_from_settings_EMIT, load_band_names_as_tensors, load_emit_data_as_tensors
from hyper.data.data_utils import aviris_band_name_to_wv, emit_band_name_to_wv, file_exists_and_not_empty, save_data_to_path, mkdir
import numpy as np
from hyper.parameters.aviris_wavelenghts import methane_target, get_idx_from_nearest_band

from hyper.features.magic_utils import magic_per_tile, magic_per_column
from tqdm import tqdm

def matched_filter(dataframe, operation_settings, settings, dataset, verbose = False, debug_jpgs=True):
    verbose = True
    # self.settings, self.dataset
    # from data in dataframe compute matched filter results on the fly
    name, band_ranges, filter_type, target_types = operation_settings # ["matched_filter", [2100, 2400], "tile", "original_signal"],
    name = "mf" # make it shorter ...
    print("Matched filter on", band_ranges, filter_type, target_types)

    band_ranges_str = ""
    for r in band_ranges:
        band_ranges_str += str(r[0])+"-"+str(r[1])+"_"

    available_bands = dataset.available_bands
    # Folders:
    root_folder = settings.dataset.root_folder
    feature_folder = settings.dataset.feature_folder
    mkdir(feature_folder)

    # Settings:
    num_iter = settings.matched_filter.num_iter

    if dataset.format == "AVIRIS":
        print("available_aviris_bands: ", len(available_bands), available_bands)
        input_products_from_ranges = input_products_from_settings_AVIRIS([], band_ranges, available_bands)
        print("input_products_from_ranges", len(input_products_from_ranges), input_products_from_ranges)
        band_indices = np.asarray([get_idx_from_nearest_band(aviris_band_name_to_wv(b)) for b in input_products_from_ranges])  # < not pretty
        print("selected band_indices", len(band_indices))

    elif dataset.format == "EMIT":
        print("available_emit_bands: ", available_bands)
        input_products_from_ranges = input_products_from_settings_EMIT([], band_ranges, available_bands)
        print("input_products_from_ranges", len(input_products_from_ranges), input_products_from_ranges)
        band_indices = np.asarray([get_idx_from_nearest_band(emit_band_name_to_wv(b)) for b in input_products_from_ranges])  # < not pretty
        print("selected band_indices", len(band_indices))

    ### STEPS:
    # Go through all rows in this dataset, and for each use the available bands
    # print("dataset.input_product_names", dataset.input_product_names)

    new_product_names = []

    # Get target signatures:
    target_type = target_types[0]
    if target_type == "methane":
        full_target = methane_target
        target = full_target[band_indices, 1]
        targets = [target]

    elif target_type == "load_signatures_from":
        path_to_targets = target_types[1]
        loaded_targets = np.load(path_to_targets)
        print("From", path_to_targets, "loaded following targets:", loaded_targets.shape)

        targets = []
        for s in loaded_targets:
            targets.append(s[band_indices, 1])

        # adjust the name:
        target_type = "load_" + path_to_targets.split("/")[-1].replace(".npy", "")

    else:
        assert False, print("unknown target signal", target_type)

    for idx in tqdm(range(len(dataframe))):
        row = dataframe.iloc[idx]
        if "folder" in row.keys():
            event_name = row["folder"].split("/")[-1] # for older AVIRIS csvs
        elif "event_id" in row.keys():
            event_name = row["event_id"] # for newer EMIT csvs

        event_folder = os.path.join(root_folder, event_name)
        target_folder = os.path.join(feature_folder, event_name)

        x_loaded = False
        x = None

        for target_i, target in enumerate(targets):
            # Naming of the features (also to check if we already computed them)
            num_str = ""
            if len(targets) > 1:
                num_str = "_" + str(target_i).zfill(2) # add numbers only if this filter has multiple signals
            unique_name = name + "_" + band_ranges_str + filter_type + "_" + target_type + num_str + ".tif"
            #print("unique_name name will be:", unique_name)
            # matched_filter_2200_2400_tile_methane_00.tif

            if idx == 0:
                new_product_names.append(unique_name)

            full_path = os.path.join(target_folder, unique_name)
            mkdir(target_folder)
            exists_already = file_exists_and_not_empty(full_path)

            if exists_already:
                if verbose: print("Exists, skipping.")
                pass

            else:
                if verbose: print("Cooking file", full_path)

                if not x_loaded:
                    window = None
                    if dataset.format == "AVIRIS":
                        x = load_band_names_as_tensors(input_products_from_ranges, window, event_folder)
                        x = x.permute((1, 2, 0))
                    elif dataset.format == "EMIT":
                        x = load_emit_data_as_tensors(input_products_from_ranges, window, event_folder, feature_folder,available_bands=dataset.available_bands)
                        x = torch.nan_to_num(x, nan=0.0)
                        x = x.permute((1, 2, 0))

                    if verbose: print("Idx", idx, event_name, "loaded:", x.shape, " (W,H,C) from", len(input_products_from_ranges), "names.")
                    if verbose: print("with use target", target.shape)
                    x_loaded = True

                # From x compute matched filter according to the filter_type
                if filter_type == "tile":
                    magic, _ = magic_per_tile(x, target, num_iter=num_iter)
                elif "column" in filter_type:
                    columns_width = int(filter_type.split("_")[1])
                    magic, _ = magic_per_column(x, target, columns_width=columns_width, num_iter=num_iter)

                if verbose: print("Computed matched filter product", magic.shape, "from", filter_type, target_type)
                geo_source = os.path.join(event_folder, "mag1c.tif") # geo ref from the magic product
                # geo_source = os.path.join(event_folder, input_products_from_ranges[0])
                save_data_to_path(magic, full_path, geo_source)

                # Also save visualisation into jpg (rough guidance how good this feature could be)
                if debug_jpgs:
                    vis_magic = np.clip(magic / 1750., 0, 2)
                    plt.imshow(vis_magic)
                    plt.savefig(full_path.replace(".tif", "_vis.jpg"))
                    plt.close()


    # Add what we cooked to the usual input products
    dataset.input_product_names = dataset.input_product_names + new_product_names
    # print("dataset.input_product_names now:", dataset.input_product_names)

    return dataframe