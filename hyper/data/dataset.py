# Pytorch Dataset, holds everything inside.
import os.path

import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

from hyper.data.normaliser_module import DataNormaliser
from hyper.data.tiler_module import DataTiler
from hyper.data.augmentor_module import DataAugmentor
from hyper.data.feature_extractor_module import DataFeatureExtractor
from hyper.data.visualiser_module import DataVisualiser
from hyper.data.data_utils import get_available_aviris_bands, get_available_emit_bands, load_band_names_as_tensors, load_emit_data_as_tensors
from hyper.data.data_utils import input_products_from_settings_AVIRIS, input_products_from_settings_EMIT, center_crop_lists, get_validity_mask
from hyper.data.data_utils import emit_adjust_base, file_exists
from rasterio.windows import Window

class HyperDataset(Dataset):
    def __init__(self, settings, mode = "train", sort_df_by_plume_size=False):
        print("Creating", mode, "dataset")
        self.mode = mode
        self.settings = settings
        self.sort_df_by_plume_size = sort_df_by_plume_size

        if self.mode == "train":
            csv_path = os.path.join(settings.dataset.root_folder, settings.dataset.train_csv)
        elif self.mode == "custom":
            csv_path = settings.dataset.custom_csv
        elif self.mode == "test":
            csv_path = os.path.join(settings.dataset.root_folder, settings.dataset.test_csv)
        elif self.mode == "val":
            csv_path = os.path.join(settings.dataset.root_folder, settings.dataset.val_csv)

        self.csv_path = csv_path
        self.dataframe = self.load_dataframe(self.csv_path)

        # Format of data
        self.format = settings.dataset.format # either AVIRIS, EMIT or EMIT_MINERALS
        if self.format == "EMIT" or self.format == "EMIT_MINERALS":
            # check the settings for features not suported by EMIT...
            if len(settings.dataset.feature_extractor) > 0:
                print("! EMIT DATASET DOESNT SUPPORT FEATURE EXTRACTION ON THE FLY !")
                assert False

        # Dataset modules:
        self.tiler = DataTiler(settings, self)
        self.feature_extractor = DataFeatureExtractor(settings, self)
        self.augmentor = DataAugmentor(settings, self)

        self.augmentation_active = self.settings.dataset.augment

        # Multi-temporal initialisation
        self.multitemporal = False
        self.multitemporal_bases = [""]
        if settings.dataset.multitemporal.enabled:
            self.multitemporal = True
            self.multitemporal_bases = settings.dataset.multitemporal.bases

            self.multitemporal_idx_as_y = None
            if settings.dataset.multitemporal.multitemporal_idx_as_y != "None":
                self.multitemporal_idx_as_y = int(settings.dataset.multitemporal.multitemporal_idx_as_y)

        # Setup functions, run these once upon creation ...
        self.band_names_initialised = False
        self.init_band_names()

        # Extract features
        if self.feature_extractor.active:
            self.feature_extractor.process(self.dataframe) # before tiling

        # Init normaliser after we know the band names
        self.normaliser = DataNormaliser(settings, self, self.input_product_names, self.output_product_names, self.auxiliary_product_names)

        # Find the index we can use for validity mask (first non mag1c band)
        # TODO: in the future when we have bi-temporal samples - compare one from A and one from B - alternatively even save it as a .tif file...
        self.validity_band_idx = 0
        while self.input_product_names[self.validity_band_idx] in ["mag1c.tif", "A_magic30_tile.tif", "B_magic30_tile.tif"]:
            if self.validity_band_idx > len(self.input_product_names):
                print("Can't get mask like this!")
                assert False
            self.validity_band_idx += 1
            #print("self.validity_band_idx", self.validity_band_idx)

        # Extract smaller tiles
        if not settings.debug.no_tiling:
            if self.mode == "train" and self.settings.dataset.tiler.train:
                self.dataframe = self.tiler.tile_dataframe(self.dataframe)

            if self.mode == "test" and self.settings.dataset.tiler.test:
                self.dataframe = self.tiler.tile_dataframe(self.dataframe)

            if self.mode == "val" and self.settings.dataset.tiler.val:
                self.dataframe = self.tiler.tile_dataframe(self.dataframe)

        # load auxiliary (like 'mag1c'), probably turned off for training, but useful for plotting
        self.load_auxiliary = len(self.settings.dataset.auxiliary_products) > 0

        # Internals:
        self.preloaded_in_memory = False

        self.normalisation_active = True
        if settings.debug.no_normalisation:
            self.normalisation_active = False

        self.visualiser = DataVisualiser(settings, self)

        # Sorting the dataframe
        if self.sort_df_by_plume_size:
            self.sort_dataframe()

    def load_dataframe(self, path):
        dataframe = pd.read_csv(path)
        dataframe = dataframe.set_index("id")
        return dataframe

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx: int):
        row = self.dataframe.iloc[idx]

        try:
            if "folder" in row.keys():
                event_name = row["folder"].split("/")[-1] # for older AVIRIS csvs
            elif "event_id" in row.keys():
                event_name = row["event_id"] # for newer EMIT csvs

            event_folder = os.path.join(self.settings.dataset.root_folder, event_name)
            feature_folder = os.path.join(self.settings.dataset.feature_folder, event_name)

            # windowed read
            # special: in the training dataset, we sometimes augment with rotations - this is a feature to on purpose load larger area around the tile before the rotation
            extend_window_by_extra = 0
            if self.mode == "train" and self.settings.dataset.augment_rotation_load_surrounding_area > 0:
                extend_window_by_extra = int(self.settings.dataset.augment_rotation_load_surrounding_area * row["window_width"] / 2.0)

            # only train has this:
            if "window_col_off" in row.keys():
                window = Window(row["window_col_off"], row["window_row_off"], row["window_width"], row["window_height"])
            else: # test and val won't
                window = None

            # Actually load
            # TODO: load depends on the regime - loading each band separately (AVIRIS) -vs- as one big file altogether (EMIT)
            if self.format == "AVIRIS":
                x = load_band_names_as_tensors(self.input_product_names, window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra)
                y = load_band_names_as_tensors(self.output_product_names, window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra)

                multiple_x = []
                multiple_x.append(x)

            elif self.format == "EMIT" or self.format == "EMIT_MINERALS":
                multiple_x = []
                if self.multitemporal:
                    for base in self.multitemporal_bases:
                        base_input_products = emit_adjust_base(self.input_product_names, base, first_base=self.multitemporal_bases[0])
                        x = load_emit_data_as_tensors(base_input_products, window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra, available_bands=self.available_bands)
                        multiple_x.append(x)
                else:
                    x = load_emit_data_as_tensors(self.input_product_names, window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra, available_bands=self.available_bands)
                    multiple_x.append(x)

                # additional fill in nans needed here
                for time_idx in range(len(multiple_x)):
                    multiple_x[time_idx] = torch.nan_to_num(multiple_x[time_idx], nan=0.0)

                CH_over_keep_all = False
                if self.format == "EMIT_MINERALS": CH_over_keep_all = True

                y = load_band_names_as_tensors(self.output_product_names, window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra, CH_over_keep_all=CH_over_keep_all)

            if self.settings.model.multiply_loss_by_mag1c:
                weight_mag1c = load_band_names_as_tensors(["weight_mag1c.tif"], window, event_folder, extend_window_by_extra=extend_window_by_extra)

            aux = None
            if self.load_auxiliary:
                aux = load_band_names_as_tensors(self.auxiliary_product_names, window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra)

            # if we have valid_mask file?
            preloaded_valid_mask = None
            if file_exists(os.path.join(event_folder,"valid_mask.tif")):
                preloaded_valid_mask = load_band_names_as_tensors(["valid_mask.tif"], window, event_folder, feature_folder, extend_window_by_extra=extend_window_by_extra)

            # Data augment
            if self.augmentation_active:
                augment_products = multiple_x
                y_idx = len(multiple_x)
                if y is not None:
                    augment_products.append(y)
                if aux is not None:
                    augment_products.append(aux)
                    aux_idx = len(augment_products)-1
                if preloaded_valid_mask is not None:
                    augment_products.append(preloaded_valid_mask)
                    valid_mask_idx = len(augment_products)-1
                if self.settings.model.multiply_loss_by_mag1c:
                    augment_products.append(weight_mag1c)
                    weight_mag1c_idx = len(augment_products)-1

                augmented_products = self.augmentor.augment(augment_products)

                if extend_window_by_extra > 0:
                    # if we extended the read area before augmentations, we should now crop the center with the right dims
                    augmented_products = center_crop_lists(augmented_products, row["window_width"], row["window_height"])

                multiple_x = augmented_products[0:y_idx]
                if y is not None:
                    y = augmented_products[y_idx]
                if aux is not None:
                    aux = augmented_products[aux_idx]
                if preloaded_valid_mask is not None:
                    # print("debug searchin for valid mask at idx", valid_mask_idx)
                    preloaded_valid_mask = augmented_products[valid_mask_idx]
                if self.settings.model.multiply_loss_by_mag1c:
                    weight_mag1c = augmented_products[weight_mag1c_idx]

            # Get validity mask (after all augmentations)
            if preloaded_valid_mask is not None:
                mask_nodata = preloaded_valid_mask # might not work with time series though
            else:
                global_mask_nodata = None
                for time_idx in range(len(multiple_x)):
                    mask_nodata = get_validity_mask(multiple_x[time_idx][[self.validity_band_idx]])
                    if global_mask_nodata is None:
                        global_mask_nodata = mask_nodata
                    else:
                        # merge the two masks
                        global_mask_nodata = np.where(mask_nodata == 1, global_mask_nodata, 0)  # 1 valid, 0 invalid
                        # ^ where the new one is valid, look at the previous one
                mask_nodata = global_mask_nodata
                # ^ note, not as torch right now!

            # Note: we should apply the validity map to all products in x (namely in EMIT if we use bi-temporal data)
            for time_idx in range(len(multiple_x)):
                multiple_x[time_idx] = multiple_x[time_idx] * mask_nodata

            # Normalise
            if self.normalisation_active:
                for time_idx in range(len(multiple_x)):
                    multiple_x[time_idx] = self.normaliser.normalize_x(multiple_x[time_idx])
                if y is not None:
                    y = self.normaliser.normalize_y(y)

            out_dict = {
                "y": y,
                "times": len(multiple_x),
                "id": event_name,
                "qplume_fulltile": row["qplume"],
                "valid_mask": mask_nodata,
            }
            if not self.multitemporal:
                out_dict["x"] = multiple_x[0] # later as [batch size, channels, W, H]
            else:
                if self.multitemporal_idx_as_y is not None:
                    # use the target idx as y
                    kept_indices = [kept_idx for kept_idx in list(range(len(multiple_x))) if kept_idx != self.multitemporal_idx_as_y]
                    as_y = multiple_x[self.multitemporal_idx_as_y]
                    as_x = []
                    for kept_idx in kept_indices:
                        as_x.append(multiple_x[kept_idx])
                    # X: special case - if there's just one time sample now, just show it as a single input
                    if len(as_x) == 1:
                        out_dict["x"] = as_x[0]
                    else:
                        out_dict["x"] = torch.stack(as_x,0)

                    # Y: append to the y we have right now ...
                    if y is None:
                        out_dict["y"] = as_y
                    else:
                        #print("Y as of now:", y.shape)
                        # for example [1, 128, 128] with [86, 128, 128]
                        out_dict["y"] = torch.cat([y,as_y], 0) # cat two different shapes
                else:
                    # normal functioning
                    # later as [batch size, time, channels, W, H]
                    out_dict["x"] = torch.stack(multiple_x,0)

            if self.settings.model.multiply_loss_by_mag1c:
                out_dict["weight_mag1c"] = weight_mag1c

            if self.load_auxiliary:
                aux = self.normaliser.normalize_aux(aux)
                out_dict["aux"] = aux

            return out_dict
        except Exception as e:
            print("Dataset error with record", idx, ":", row)
            print(e)
            print("DEBUG")
            print("input_product_names >>", self.input_product_names)
            print("output_product_names >>", self.output_product_names)
            assert False

    def process_from_raw(self, x_raw, y_raw=None):
        # Process from raw, as if we loaded this in the get item ...
        """
        As if we did:
        x _raw = load_emit_data_as_tensors(self.input_product_names, window, event_folder, feature_folder,
                                      extend_window_by_extra=extend_window_by_extra,
                                      available_bands=self.available_bands)
        y _raw = load_band_names_as_tensors(self.output_product_names, window, event_folder, feature_folder,
                                   extend_window_by_extra=extend_window_by_extra)
        """
        # Data augment
        multiple_x = [x_raw]
        y = y_raw

        if self.augmentation_active:
            augment_products = multiple_x
            y_idx = len(multiple_x)
            if y is not None:
                augment_products.append(y)
            augmented_products = self.augmentor.augment(augment_products)
            multiple_x = augmented_products[0:y_idx]
            if y is not None:
                y = augmented_products[y_idx]

        # Get validity mask (after all augmentations)
        global_mask_nodata = None
        for time_idx in range(len(multiple_x)):
            mask_nodata = get_validity_mask(multiple_x[time_idx][[self.validity_band_idx]])
            if global_mask_nodata is None:
                global_mask_nodata = mask_nodata
            else:
                # merge the two masks
                global_mask_nodata = np.where(mask_nodata == 1, global_mask_nodata, 0)  # 1 valid, 0 invalid
                # ^ where the new one is valid, look at the previous one
        mask_nodata = global_mask_nodata

        # Note: we should apply the validity map to all products in x (namely in EMIT if we use bi-temporal data)
        for time_idx in range(len(multiple_x)):
            multiple_x[time_idx] = multiple_x[time_idx] * mask_nodata
            # x = x * mask_nodata

        # Normalise
        if self.normalisation_active:
            for time_idx in range(len(multiple_x)):
                multiple_x[time_idx] = self.normaliser.normalize_x(multiple_x[time_idx])
            if y is not None:
                y = self.normaliser.normalize_y(y)

        out_dict = {
            "y": y,
            "times": len(multiple_x),
            "valid_mask": mask_nodata,
        }

        x = multiple_x[0]
        x = x.nan_to_num(0)
        out_dict["x"] = x # later as [batch size, channels, W, H]
        return out_dict

    def sort_dataframe(self):
        # Sort the dataframe rows according to the size of the plumes they contain
        # (This can be either done when looking at the labels, or by using the qplume value)
        self.dataframe = self.dataframe.sort_values(by=['qplume'], ascending=False)

    def show_item_from_data(self, batch_data):
        plt, fig = self.visualiser.plot_batch(batch_data, prediction=None)
        plt.show()

    def show_item_from_idx(self, idx):
        batch_data = self.__getitem__(idx)
        self.show_item_from_data(batch_data)

    def summarise(self):
        sample = self[0]
        if self.multitemporal:
            print("- MultitemporalDataset", self.mode, "with", len(self), "samples with keys", sample.keys()," and x shape:", sample["x"].shape, "y shape:", sample["y"].shape, "with", sample["times"], "times")
            print("- Input products:", self.input_product_names, "[*", self.multitemporal_bases,"]")
            print("- Output products:", self.output_product_names)
            print("- Auxiliary products:", self.auxiliary_product_names)

        else:
            print("- Dataset", self.mode, "with", len(self), "samples with keys", sample.keys()," and x shape:", sample["x"].shape, "y shape:", sample["y"].shape)
            print("- Input products:", self.input_product_names)
            print("- Output products:", self.output_product_names)
            print("- Auxiliary products:", self.auxiliary_product_names)

    def init_band_names(self):
        if self.band_names_initialised:
            return True

        if self.format == "AVIRIS":
            # Which bands do we have available (Dynamically from the first folder...)
            self.available_bands = get_available_aviris_bands(self.settings.dataset.root_folder)

            # Then load from settings the names of the tif files
            self.input_product_names = input_products_from_settings_AVIRIS(self.settings.dataset.input_products['specific_products'],
                                                                 self.settings.dataset.input_products['band_ranges'],
                                                                 self.available_bands)

            self.output_product_names = [f"{product}.tif" for product in self.settings.dataset.output_products]
            self.auxiliary_product_names = [f"{product}.tif" for product in self.settings.dataset.auxiliary_products]
            self.band_names_initialised = True
            return True
        elif self.format == "EMIT" or self.format == "EMIT_MINERALS":

            # Which bands do we have available (Dynamically from the first folder...)
            self.available_bands = get_available_emit_bands(self.settings.dataset.root_folder)
            self.input_product_names = input_products_from_settings_EMIT(self.settings.dataset.input_products['specific_products'], # < also one by one files
                                                                 self.settings.dataset.input_products['band_ranges'], # < but this one from the big file
                                                                 self.available_bands)

            self.output_product_names = [f"{product}.tif" for product in self.settings.dataset.output_products]
            self.auxiliary_product_names = [f"{product}.tif" for product in self.settings.dataset.auxiliary_products]
            self.band_names_initialised = True

            if self.format == "EMIT_MINERALS":
                self.input_product_names = [name.replace("B_EMIT_", "C_EMIT_") for name in self.input_product_names]
            return True

