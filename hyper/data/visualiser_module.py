# Visualises data given information from which band they are

import torch
from torch.nn import Module
import numpy as np
from hyper.data.plot_utils import show_3_bands, show_1_band
from hyper.utils import to_device
import pylab as plt
from typing import List, Union, Optional, Dict, Any, Tuple
import numpy as np
from torch import Tensor
from matplotlib.axis import Axis
import matplotlib.patches as mpatches

COLORS_DIFFERENCES = np.array([[255, 255, 255],
                                [255, 0, 0],
                                [0, 0, 255],
                                [0, 255, 0],
                        ]) / 255
INTERPRETATION_DIFFERENCES = ["TN", "FN", "FP", "TP"]

class DataVisualiser(Module):
    def __init__(self, settings, dataset):
        super().__init__()
        self.settings = settings
        self.dataset = dataset
        self.format = dataset.format # AVIRIS, EMIT or EMIT_MINERALS

        if self.format == "AVIRIS": # 640,550,460
            self.default_rgb_bands = ['TOA_AVIRIS_640', 'TOA_AVIRIS_550', 'TOA_AVIRIS_460']
            self.default_magic_band = ["mag1c"]
        elif self.format == "EMIT": # 641,551,462
            self.default_rgb_bands = ['B_EMIT_641', 'B_EMIT_551', 'B_EMIT_462'] # ~ for now just B
            self.default_magic_band = ["B_magic30_tile"]
        elif self.format == "EMIT_MINERALS": # 641,551,462
            self.default_rgb_bands = ['C_EMIT_641', 'C_EMIT_551', 'C_EMIT_462']
            self.default_magic_band = ["B_magic30_tile"]

    def band_indices(self, band_names):
        # Finds if we have selected bands in the input or auxiliary data
        band_indices = []
        input_product_names_permissive = [a.replace("nm.tif","").replace(".tif","") for a in self.dataset.input_product_names]
        auxiliary_product_names_permissive = [a.replace("nm.tif","").replace(".tif","") for a in self.dataset.auxiliary_product_names]
        for b in band_names:
            if b in input_product_names_permissive:
                idx = input_product_names_permissive.index(b)
                band_indices.append([idx,"input"])
            elif b in auxiliary_product_names_permissive:
                idx = auxiliary_product_names_permissive.index(b)
                band_indices.append([idx,"aux"])
        return band_indices

    def rgb_from_data(self, x, aux):
        band_indices = self.band_indices(band_names = self.default_rgb_bands)
        bands = []
        for idx, source_str in band_indices:
            source = x if source_str == "input" else aux
            bands.append( np.asanyarray(source[idx]) )

        bands = np.asarray(bands)
        return bands


    def plot_batch(self, batch_data, prediction, figsize_ax=(3, 2)):
        # Put data to CPU
        batch_data = to_device(batch_data, torch.device("cpu"))
        prediction = to_device(prediction, torch.device("cpu"))

        # Adjustment for multi-temporal case ... just show the first time for now ...
        if self.dataset.multitemporal:
            if self.dataset.multitemporal_idx_as_y is None:
                # x shape when multitemporal torch.Size([Batch, Times, Bands, W, H])
                # print("Visualisation of multitemporal samples", batch_data["x"].shape, "-> will use the first temporal sample")
                batch_data["x"] = batch_data["x"][:,0]

        # Plot batch of data
        # - from samples 0 to selected number
        #print("Batch has keys", batch_data.keys(), "and", len(batch_data["x"]),"samples.")
        plot_samples_per_batch = self.settings.training.visualiser.samples_per_batch
        plot_samples_per_batch = min(plot_samples_per_batch, len(batch_data["x"]))

        if prediction is None:
            prediction = torch.zeros_like(batch_data["y"])

        plot_functions = self.settings.training.visualiser.bands
        num_plots = len(plot_functions)

        # Adjust if we are plotting all bands
        if "all_inputs" in plot_functions:
            x = batch_data["x"][0]
            num_plots = num_plots - 1 + len(x)

        ax_idx = 0
        fig, ax = plt.subplots(plot_samples_per_batch, num_plots,
                               figsize=(figsize_ax[0] * num_plots, figsize_ax[1] * plot_samples_per_batch),
                               constrained_layout=True, squeeze=False) # tight_layout=True,
        ax = ax.flatten()

        for sample_i in range(plot_samples_per_batch):
            x = batch_data["x"][sample_i]
            y_rec = batch_data["y"][sample_i]
            y = batch_data["y"][sample_i].long()
            valid_mask = batch_data["valid_mask"][sample_i].long()
            p = prediction[sample_i]
            p_thr = (p > 0.5).long()
            differences = 2 * p_thr.long() + (y == 1).long()

            if hasattr(p, "detach"):
                p = p.detach().numpy()
            if hasattr(p, "differences"):
                differences = differences.detach().numpy()

            aux = None
            if "aux" in batch_data.keys():
                aux = batch_data["aux"][sample_i]

            for plot_function in plot_functions:
                if plot_function == 'first3': # just shows the first 3 bands ...
                    self.plot_first3(x, ax[ax_idx])
                    ax[ax_idx].set_title(f"sample #{sample_i}")

                elif plot_function == 'first1': # just shows the first 3 bands ...
                    self.plot_first1(x, ax[ax_idx])
                    ax[ax_idx].set_title(f"sample #{sample_i}")

                elif plot_function == 'rgb': # specifically searches for the RGB bands
                    self.plot_rgb(x, aux, ax[ax_idx])
                    ax[ax_idx].set_title(f"rgb #{sample_i}")

                elif plot_function == 'y_rgb': # searches for the RGB bands in y
                    self.plot_rgb(y_rec, aux, ax[ax_idx])
                    ax[ax_idx].set_title(f"target rgb #{sample_i}")

                elif plot_function == 'pred_rgb': # searches for the RGB bands in pred
                    self.plot_rgb(p, aux, ax[ax_idx])
                    ax[ax_idx].set_title(f"predicted rgb #{sample_i}")

                elif plot_function == 'labelbinary':
                    self.plot_labelbinary(y, ax[ax_idx])

                    add_str = ""
                    if "qplume_fulltile" in batch_data.keys():
                        add_str = " ("+str(int(batch_data["qplume_fulltile"][sample_i]))+" qp.)"
                    ax[ax_idx].set_title("label"+add_str)

                elif plot_function == 'mag1c':
                    self.plot_mag1c(x, aux, ax[ax_idx])
                    ax[ax_idx].set_title("mag1c")

                elif plot_function == 'prediction':
                    self.plot_prediction(p, ax[ax_idx])
                    ax[ax_idx].set_title("prediction")

                elif plot_function == 'prediction01':
                    self.plot_prediction(np.clip(p,0.,1.), ax[ax_idx])
                    ax[ax_idx].set_title("prediction, clip 0-1")

                elif plot_function == 'valid_mask':
                    #self.plot_labelbinary(valid_mask, ax[ax_idx])
                    show_1_band(valid_mask[0], ax[ax_idx], vmin=0, vmax=1)
                    ax[ax_idx].set_title("valid_mask")

                elif plot_function == 'differences':
                    self.plot_differences(differences, ax[ax_idx])
                    ax[ax_idx].set_title("differences")

                elif plot_function == 'all_inputs':
                    for x_band_i in range(len(x)):
                        show_1_band(x[x_band_i], ax[ax_idx], add_colorbar=True)
                        ax[ax_idx].set_title("x_"+str(x_band_i))
                        ax_idx += 1
                    ax_idx -= 1

                # 1D functions:
                elif plot_function == 'x_spectral':
                    # first let's just assume the single hyper pixel in the center
                    self.plot_spectral_central(ax[ax_idx], x, y)
                    ax[ax_idx].set_title("spectrum of central px")

                # minerals functions
                elif plot_function == 'minerals':
                    # easier to plot just 3 bands:
                    self.plot_minerals_first3(ax[ax_idx], y)
                    ax[ax_idx].set_title("minerals gt")

                elif plot_function == 'minerals_pred':
                    # easier to plot just 3 bands:
                    self.plot_minerals_first3(ax[ax_idx], p_thr) # p or p_thr
                    ax[ax_idx].set_title("minerals pred")

                else:
                    assert False, "Unknown plotting function "+plot_function+" !"

                ax_idx += 1

        # smaller gaps:
        for a in ax:
            a.set_yticklabels([])
            # a.axis('off')
        plt.subplots_adjust(wspace=0.030, hspace=0)

        return plt, fig

    def plot_first1(self, x, ax):
        return show_1_band(x[0], ax, add_colorbar=True)

    def plot_first3(self, x, ax):
        return show_3_bands(x[0:3], ax)

    def plot_labelbinary(self, y, ax):
        return show_1_band(y[0], ax, vmin=0, vmax=1)

    def plot_mag1c(self, x, aux, ax):
        for magic_name in self.default_magic_band:
            # maybe this would be better done just once ...
            try:
                magic_band_i, source_str = self.band_indices(band_names=[magic_name])[0]
                break
            except:
                print("failed finding magic band name =",magic_name)
                print(self.dataset.input_product_names)
                print(self.dataset.auxiliary_product_names)
                pass

        source = x if source_str == "input" else aux
        magic = source[magic_band_i]
        return show_1_band(magic, ax, add_colorbar=True, vmin=0, vmax=2)

    def plot_rgb(self, x, aux, ax):
        if self.format == "AVIRIS":
            rgb = self.rgb_from_data(x,aux)
        elif self.format == "EMIT" or self.format == "EMIT_MINERALS":
            rgb = self.rgb_from_data(x,aux)
            rgb = rgb / np.nanmax(rgb)

        return show_3_bands(rgb, ax)

    def plot_prediction(self, pred, ax):
        return show_1_band(pred[0], ax, add_colorbar=True) #, vmin=0, vmax=1)

    def plot_differences(self, differences, ax, with_legend=True):
        differences = differences[0]

        colors_cmap = COLORS_DIFFERENCES
        if with_legend:
            interpretation = INTERPRETATION_DIFFERENCES
        else:
            interpretation = None
        plot_mask_categorical(differences, values=[0, 1, 2, 3], colors_cmap=colors_cmap,
                                         interpretation=interpretation,
                                         ax=ax)
        if with_legend:
            # move legend ...
            legend = ax.get_legend()
            labels = (x.get_text() for x in legend.get_texts())
            ax.legend(legend.legendHandles, labels, loc='center left', bbox_to_anchor=(1, 0.5))


    def plot_spectral_central(self, ax, x, y):
        CH, W, H = x.shape
        center_w = int(W / 2)
        center_h = int(H / 2)
        center_X = x[:, center_w:center_w+1, center_h:center_h+1].flatten()
        center_Y = y[:, center_w:center_w+1, center_h:center_h+1].flatten()

        if self.dataset.format == "AVIRIS":
            available_bands = self.dataset.available_bands
        elif self.dataset.format == "EMIT" or self.format == "EMIT_MINERALS":
            _, available_bands = self.dataset.available_bands

        xs = -center_X
        ys = available_bands

        ### Zoom in onto a region
        zoom = [2000,2500]
        new_xs = []
        new_ys = []
        for x, y in zip(xs, ys):
            if y > zoom[0] and y < zoom[1]:
                new_xs.append(x)
                new_ys.append(y)
        xs,ys = new_xs, new_ys

        ### Hide gaps:
        new_xs = [xs[0]]
        new_ys = [ys[0]]
        for i in range(1,len(xs)):
            prev_x = xs[i-1]
            x = xs[i]
            prev_y = ys[i-1]
            y = ys[i]
            # if we have a large hole put nan as x
            gap = abs(prev_y - y)
            if gap > 50:
                new_xs.append(x+1)
                new_ys.append(np.nan)

            new_xs.append(x)
            new_ys.append(y)

        ax.scatter(new_ys, new_xs, s=1)
        ax.plot(new_ys, new_xs)

    def plot_minerals_first3(self, ax, minerals):
        minerals3 = minerals[0:3]
        return show_3_bands(minerals3, ax)

def mask_to_rgb(mask:Union[Tensor,np.array], values:Union[List[int],np.array], colors_cmap:np.array) -> np.array:
    """
    Given a 2D mask it assigns to each value of the mask the corresponding color

    Args:
        mask:  array of shape (H, W) with the mask
        values: 1D list with values of the mask to assign to each color in colors_map.
        colors_cmap: 2D array of shape (len(values), 3) or (len(values), 4) with colors to assign to each value in
            values. Colors assumed to be floats in [0,1]

    Returns:
        (3, H, W) or (4, H, W) tensor with rgb(a) values of the mask.
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    mask = np.asanyarray(mask)
    assert len(values) == len(
        colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"

    assert len(mask.shape) == 2, f"Expected only 2D array found {mask.shape}"
    mask_return = np.zeros((colors_cmap.shape[1],) + mask.shape[:2], dtype=np.uint8)
    colores = np.array(np.round(colors_cmap * 255), dtype=np.uint8)
    for i, c in enumerate(colores):
        for _j in range(len(c)):
            mask_return[_j][mask == values[i]] = c[_j]
    return np.transpose(mask_return, (1,2,0))

def mask_to_rgb(mask: Union[Tensor, np.array], values: Union[List[int], np.array],
                colors_cmap: np.array) -> np.array:
    """
    Given a 2D mask it assigns to each value of the mask the corresponding color

    Args:
        mask:  array of shape (H, W) with the mask
        values: 1D list with values of the mask to assign to each color in colors_map.
        colors_cmap: 2D array of shape (len(values), 3) or (len(values), 4) with colors to assign to each value in
            values. Colors assumed to be floats in [0,1]

    Returns:
        (3, H, W) or (4, H, W) tensor with rgb(a) values of the mask.
    """
    if hasattr(mask, "cpu"):
        mask = mask.cpu()
    mask = np.asanyarray(mask)
    assert len(values) == len(
        colors_cmap), f"Values and colors should have same length {len(values)} {len(colors_cmap)}"

    assert len(mask.shape) == 2, f"Expected only 2D array found {mask.shape}"
    mask_return = np.zeros((colors_cmap.shape[1],) + mask.shape[:2], dtype=np.uint8)
    colores = np.array(np.round(colors_cmap * 255), dtype=np.uint8)
    for i, c in enumerate(colores):
        for _j in range(len(c)):
            mask_return[_j][mask == values[i]] = c[_j]
    return np.transpose(mask_return, (1, 2, 0))

def plot_mask_categorical(mask:Union[Tensor, np.ndarray], values:Union[List[int], np.array], colors_cmap:np.array,
                          interpretation:Optional[Union[List[str], np.array]]=None, ax:Optional[Axis]=None,
                          loc_legend:str='upper right') -> Axis:
    rgb_mask = mask_to_rgb(mask, values, colors_cmap)
    if ax is None:
        ax = plt.gca()

    ax.imshow(rgb_mask, interpolation="nearest")
    if interpretation is not None:
        patches = []
        for c, interp in zip(colors_cmap, interpretation):
            patches.append(mpatches.Patch(color=c, label=interp))

        ax.legend(handles=patches, loc=loc_legend)
    return ax


