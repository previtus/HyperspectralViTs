# Tiling smaller areas from larger geo data, while keeping repeatability
# - can do multiple tiling schemes, is reproducible and fixed across trainings

import random
import torch
import os
from torch.nn import Module
import pandas as pd
import numpy as np
from tqdm import tqdm

from hyper.data.tiling_utils import create_windows_regular_tiling
from hyper.data.data_utils import simple_load
from rasterio.windows import Window

def get_values_at_window_locations(full_data, window_col_off, window_row_off, window_width, window_height):
    assert len(full_data.shape) == 3 # 1, W, H
    return full_data[:, window_row_off:window_row_off+window_height, window_col_off:window_col_off+window_width]

def load_label_for_row(row, settings):
    if "folder" in row.keys():
        event_name = row["folder"].split("/")[-1]  # for older AVIRIS csvs
    elif "event_id" in row.keys():
        event_name = row["event_id"]  # for newer EMIT csvs
    else:
        print("Can't resolve the id from:", row)
        print("Check the csv formatting...")
        assert False

    # Support different output names:
    y_name = settings.dataset.output_products[0] # if we later have more than one label, this will need to be done for all
    label_path = os.path.join(settings.dataset.root_folder, event_name, y_name+".tif")
    window = Window(row["window_col_off"], row["window_row_off"], row["window_width"], row["window_height"])

    label_data = simple_load(label_path, window=window)
    return label_data

def update_labels(dataframe, settings):
    # Update the "has_plume" values inside a dataframe from actual labels each row is aiming at (using the newest window info)
    for i, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
        label_data = load_label_for_row(row, settings)
        has_label = "TRUE" if np.max(label_data) == 1 else "FALSE"
        dataframe.at[i, "has_plume"] = has_label

    return dataframe

def check_num_of_valid_px(row, settings, thr_for_valid_data_in_tile):
    if "folder" in row.keys():
        # for older AVIRIS csvs
        return True, None # AVIRIS didn't need checking ...

    elif "event_id" in row.keys():
        event_name = row["event_id"]  # for newer EMIT csvs
    else:
        assert False

    # Load data:
    if settings.dataset.format == "EMIT":
        valid_path = os.path.join(settings.dataset.root_folder, event_name, "B_EMIT_641nm.tif") # we assume that we always have at least this band in the datasets ...
    elif settings.dataset.format == "EMIT_MINERALS":
        valid_path = os.path.join(settings.dataset.root_folder, event_name, "C") # load valid mask from the full data? (or pre-computed valid mask rather?)

    window = Window(row["window_col_off"], row["window_row_off"], row["window_width"], row["window_height"])
    valid_data = simple_load(valid_path, bands_to_load=[0], window=window)
    # detects NaNs:
    valid_mask = np.where(valid_data == valid_data, 1, 0) # 1 valid, 0 invalid
    _, w,h = valid_mask.shape
    valid_perc = int(np.sum(valid_mask.flatten())) / (w*h)

    if valid_perc < thr_for_valid_data_in_tile:
        return False, valid_mask
    else:
        return True, valid_mask

class DataTiler(Module):
    def __init__(self, settings, dataset):
        self.settings = settings
        self.tiler_settings = settings.dataset.tiler
        self.dataset = dataset

        self.mode = self.tiler_settings.mode
        self.perpixel_how_many_from_each = self.tiler_settings.perpixel_how_many_from_each
        pass

    def tile_dataframe(self, dataframe):
        # For each row, take the 512x512 data and tile it with
        tile_size = self.tiler_settings.tile_size
        tile_overlap = self.tiler_settings.tile_overlap

        root_folder = self.settings.dataset.root_folder
        name_csv, ext = os.path.splitext(self.dataset.csv_path)

        mode_marker = "tiled"
        if self.mode == "per_pixel":
            mode_marker = "perpixel"

        dataset_path_tiled = os.path.join(root_folder, f"{name_csv}_{mode_marker}_{tile_size}_{tile_overlap}{ext}")

        if not os.path.exists(dataset_path_tiled):
            print(f"Tiled dataset {dataset_path_tiled} not found. Generating...")
            if self.mode == "regular":
                dataframe = self.regular_tile_dataframe(dataframe, tile_size, tile_overlap)
                if self.settings.dataset.format == "EMIT_MINERALS":
                    for i, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
                        dataframe.at[i, "has_plume"] = False # false to all... won't influence the weighting
                else:
                    dataframe = update_labels(dataframe, self.settings)

            elif self.mode == "per_pixel":
                # Careful - as the area sampling is random, each call like this will make it's own .csv file with different mini-tiles
                dataframe = self.perpixel_tile_dataframe(dataframe, tile_size, tile_overlap)

            dataframe.to_csv(dataset_path_tiled)
        else:
            print(f"Loading tiled dataset from {dataset_path_tiled}")
            dataframe = self.dataset.load_dataframe(dataset_path_tiled)

        return dataframe

    def regular_tile_dataframe(self, dataframe, tile_size, tile_overlap, check_validity_of_tiles=True):
        dataframe_tiled_list = []
        n_before_tiling = len(dataframe)

        for row in tqdm(dataframe.reset_index().to_dict(orient="records"), total=n_before_tiling):
            del_keys = ["window_row_off", "window_col_off", "window_width", "window_height", "window_labels"]
            for k in del_keys:
                if k in row.keys():
                    del row[k]

            dim = self.tiler_settings.input_size
            windows = create_windows_regular_tiling((dim,dim), (tile_size,tile_size), (tile_overlap,tile_overlap))
            for w in windows:
                row_copy = dict(row)
                row_copy["window_row_off"] = w.row_off
                row_copy["window_col_off"] = w.col_off
                row_copy["window_width"] = w.width
                row_copy["window_height"] = w.height
                row_copy["id"] = f"{row['id']}_r{w.row_off}_c{w.col_off}_w{w.width}_h{w.height}"

                tile_is_valid = True
                if check_validity_of_tiles:
                    # Check if the data in the tile contains valid data?
                    tile_is_valid, _ = check_num_of_valid_px(row_copy, self.settings, self.settings.dataset.tiler.emit_thr_for_valid_data_in_tile)

                if tile_is_valid:
                    dataframe_tiled_list.append(row_copy)
                # else:
                #     print("excluding tile", row_copy["id"], "it didn't have enough valid px!")

        dataframe_tiled = pd.DataFrame(dataframe_tiled_list)
        dataframe_tiled = dataframe_tiled.set_index("id")

        n_after_tiling = len(dataframe_tiled)
        print("Tiled the dataset, from", n_before_tiling, "full tiles (512x512) into", n_after_tiling,"smaller tiles ("+str(tile_size)+"x"+str(tile_size)+" with overlap of "+str(tile_overlap)+")")

        return dataframe_tiled

    def perpixel_tile_dataframe(self, dataframe, tile_size, tile_overlap):
        dataframe_tiled_list_POS = []
        dataframe_tiled_list_NEG = []
        n_before_tiling = len(dataframe)

        __internal_load_style = "each_window_separate"
        __internal_load_style = "windows_from_full" # much faster!

        for row in tqdm(dataframe.reset_index().to_dict(orient="records"), total=n_before_tiling):
            del_keys = ["window_row_off", "window_col_off", "window_width", "window_height", "window_labels"]
            for k in del_keys:
                if k in row.keys():
                    del row[k]

            dim = self.tiler_settings.input_size
            windows = create_windows_regular_tiling((dim,dim), (tile_size,tile_size), (tile_overlap,tile_overlap))
            random.shuffle(windows)

            # Speed up - load the entire tile instead:
            # if __internal_load_style == "windows_from_full":
            if True:
                row_full = dict(row)
                row_full["window_row_off"] = 0
                row_full["window_col_off"] = 0
                row_full["window_width"] = self.settings.dataset.tiler.input_size
                row_full["window_height"] = self.settings.dataset.tiler.input_size
                _, full_valid = check_num_of_valid_px(row_full, self.settings, 1.0)
                # print("full_valid:", full_valid.shape)
                full_label = load_label_for_row(row_full, self.settings)
                # print("full_label:", full_label.shape)

            print("we have", len(windows),"potential windows -> we will subselect only", self.perpixel_how_many_from_each, "from them!")
            we_want_from_each_class = int(self.perpixel_how_many_from_each / 2)
            add_POS = []
            add_NEG = []
            for w in windows:
                row_copy = dict(row)
                row_copy["window_row_off"] = w.row_off
                row_copy["window_col_off"] = w.col_off
                row_copy["window_width"] = w.width
                row_copy["window_height"] = w.height
                row_copy["id"] = f"{row['id']}_r{w.row_off}_c{w.col_off}_w{w.width}_h{w.height}"

                # For this window:
                # - check validity, and label
                if __internal_load_style == "each_window_separate":
                    tile_is_valid, _ = check_num_of_valid_px(row_copy, self.settings, 1.0)
                if __internal_load_style == "windows_from_full":
                    validity_data = get_values_at_window_locations(full_valid, w.col_off, w.row_off, w.width, w.height)
                    tile_is_valid = np.min(validity_data) == 1.0

                # assert tile_is_valid1==tile_is_valid2
                # tile_is_valid = tile_is_valid2

                if not tile_is_valid:
                    # early reject
                    continue

                if __internal_load_style == "each_window_separate":
                    label_data = load_label_for_row(row_copy, self.settings)
                if __internal_load_style == "windows_from_full":
                    label_data = get_values_at_window_locations(full_label, w.col_off, w.row_off, w.width, w.height)

                # here we consider only full tiles - all pixels in the label must be the same class
                # (alt would be to allow boundary ones too)

                if np.min(label_data) != np.max(label_data):
                    # not the same
                    continue

                has_label = "TRUE" if np.max(label_data) == 1 else "FALSE"

                # also update the row directly ...
                row_copy["has_plume"] = has_label

                # - if valid add
                if has_label == "TRUE":
                    add_POS.append(row_copy)
                else:
                    add_NEG.append(row_copy)
                # - stop if we can make the desired number
                # - at the end balance

                if len(add_POS) >= we_want_from_each_class and len(add_NEG) >= we_want_from_each_class:
                    break

            # Now we sample balanced samples
            we_want_from_each_class = min(len(add_POS), len(add_NEG)) # < adjust if we found less samples
            print("we selected POS", len(add_POS), "and NEG", len(add_NEG))
            add_POS = random.sample(add_POS, we_want_from_each_class)
            add_NEG = random.sample(add_NEG, we_want_from_each_class)
            print("subselected ... POS", len(add_POS), "and NEG", len(add_NEG))

            dataframe_tiled_list_POS += add_POS
            dataframe_tiled_list_NEG += add_NEG

        dataframe_tiled = pd.DataFrame(dataframe_tiled_list_POS + dataframe_tiled_list_NEG)
        dataframe_tiled = dataframe_tiled.set_index("id")

        n_after_tiling = len(dataframe_tiled)
        print("Tiled the dataset, from", n_before_tiling, "full tiles (512x512) into", n_after_tiling,"smaller tiles ("+str(tile_size)+"x"+str(tile_size)+" with overlap of "+str(tile_overlap)+")")

        return dataframe_tiled

