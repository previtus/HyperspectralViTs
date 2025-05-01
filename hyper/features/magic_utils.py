import numpy as np
import torch
import random
import pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from hyper.features import mag1cmod as mag1c
from timeit import default_timer as timer

def magic_per_tile(data, target, num_iter=30):
    nodata = -9999.0 # was true in the raw data ...
    nodata = 0.0
    covariance_lerp_alpha=1e-4

    data = data.to(torch.float64) # w,h,b
    valid_mask = np.where(np.array(data[:, :, 0]) == nodata, False, True) # w,h ~ np
    data_valid = data[valid_mask,:] # w*h, b ~ stays torch
    data_valid_torch = torch.Tensor(data_valid).unsqueeze(0)

    target_torch = torch.Tensor(target)

    mf_flat, albedo_flat = mag1c.acrwl1mf(data_valid_torch, target_torch, num_iter=num_iter, alpha=covariance_lerp_alpha,) # 1, w*h, 1

    magic = np.zeros(data.shape[:2],dtype=np.float64) + nodata # back into w,h
    magic[valid_mask] = np.array(mf_flat)[0,:,0]

    albedo = np.zeros(data.shape[:2], dtype=np.float64)
    albedo[valid_mask] = np.array(albedo_flat)[0, :, 0]

    return magic, albedo


def robust_matched_filter(data, target):
    nodata = -9999.0 # was true in the raw data ...
    nodata = 0.0

    data = data.to(torch.float64) # w,h,b
    valid_mask = np.where(np.array(data[:, :, 0]) == nodata, False, True) # w,h
    data_valid = data[valid_mask,:] # w*h, b
    data_valid_torch = torch.Tensor(data_valid).unsqueeze(0)

    target_torch = torch.Tensor(target)

    rbf_flat, albedo_rbf_flat = mag1c.rmf(data_valid_torch, target_torch, albedo_override=True)

    rbf = np.zeros(data.shape[:2],dtype=np.float64) + nodata # back into w,h
    rbf[valid_mask] = np.array(rbf_flat)[0,:,0]

    # albedo_rbf = np.zeros(data.shape[:2], dtype=data.dtype)
    # albedo_rbf[valid_mask] = np.array(albedo_rbf_flat)[0, :, 0]

    return rbf, None #albedo_rbf


def magic_per_column(data, target, columns_width = 1, num_iter = 30):
    # columns_width=1 ~ Found 496 groups. Reading by groups of size 50
    # columns_width=5 ~ Found 100 groups. Reading by groups of size 50

    # nodata super important here!
    nodata = -9999.0 # was true in the raw data ...
    nodata = 0.0
    covariance_lerp_alpha = 1e-4

    data = np.asarray(data).astype(np.float64)  # w,h,b
    target_torch = torch.Tensor(target)


    ####
    noalbeldo = False
    nonnegativeoff = False
    no_sparsity = False
    covariance_update_scaling = 1.
    covariance_lerp_alpha = 0.
    function_acfwl1mf = lambda x: mag1c.acrwl1mf(x, target_torch, num_iter=num_iter, albedo_override=noalbeldo,
                                                    zero_override=nonnegativeoff, sparse_override=no_sparsity,
                                                    covariance_update_scaling=covariance_update_scaling,
                                                    alpha=covariance_lerp_alpha,
                                                    mask=None)

    # Make up glt_data - on x axis numbers go from high to low (integers) left to right
    #                    on y axis from high on top to low at bottom
    # glt_data = np.zeros((512,512,2))
    #groups: (H, W) tensor of type int
    groups = np.zeros((data.shape[0], data.shape[1]), dtype=int)
    max_num = 1000 # len(groups) + 10
    for h_i in range(0, groups.shape[0], columns_width):
        for w_i in range(0, groups.shape[1], columns_width):
            target_size = groups[h_i : h_i+columns_width, w_i:w_i+columns_width].shape # for last col/row will be smaller
            groups[h_i : h_i+columns_width, w_i:w_i+columns_width] = np.ones(target_size) * (max_num - w_i)

    groups = np.where(np.array(data[:, :, 0]) == nodata, 0, groups)  # w,h # mask them too


    groups = torch.tensor(groups)
    magic, albedo = mag1c.func_by_groups(function_acfwl1mf, data, groups, NODATA_local=nodata)
    magic = np.where(np.array(data[:, :, 0]) == nodata, 0, magic)  # w,h

    return magic, albedo


def colorbar_next_to(im, ax, size='5%',pad=0.05):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    plt.gcf().colorbar(im, cax=cax, orientation='vertical')


def compute_and_plot_matched_filters(bands_data, target, num_iter = 30, opt_magic_reference = None, opt_rgb = None, opt_y = None,
                                     plots = ["tile", "column_1", "column_5", "robust_matched_filter", "magic_ref", "rgb", "y"],
                                     verbose=True, show=True, save=False, save_name=""):
    times = {}
    products = []
    names = []

    if verbose: print("We have bands data", bands_data.shape)
    bands_data = bands_data.permute((1, 2, 0))

    if "tile" in plots:
        start = timer()
        magic_from_tile, _ = magic_per_tile(bands_data, target, num_iter=num_iter)
        magic_from_tile = np.clip(magic_from_tile / 1750., 0, 2)
        end = timer()
        time = (end - start)
        times["time_from_tile"] = time
        if verbose: print("magic_per_tile time: " + str(time) + "s (" + str(time / 60.0) + "min)")
        products.append(magic_from_tile)
        names.append("from tile")

    for p in plots:
        if "column_" in p:
            columns_width = int(p.split("_")[1])
            start = timer()
            # slow, optionally turn off:
            magic_from_columns, _ = magic_per_column(bands_data, target, columns_width=columns_width, num_iter=num_iter)
            magic_from_columns = np.clip(magic_from_columns / 1750., 0, 2)
            # magic_from_columns = np.zeros_like((bands_data[:,:,0]))
            end = timer()
            time = (end - start)
            times["time_from_column_"+str(columns_width)] = time
            if verbose: print("magic_per_column columns_width="+str(columns_width)+" time: " + str(time) + "s (" + str(time / 60.0) + "min)")

            products.append(magic_from_columns)
            names.append("from column "+str(columns_width))

    if "robust_matched_filter" in plots:
        start = timer()
        matched_filter, _ = robust_matched_filter(bands_data, target)
        matched_filter = np.clip(matched_filter / 1750., 0, 2)
        end = timer()
        time = (end - start)
        times["time_from_rmf"] = time
        if verbose: print("robust_matched_filter time: " + str(time) + "s (" + str(time / 60.0) + "min)")
        products.append(matched_filter)
        names.append("matched filter")

    fig, axes = plt.subplots(1, len(plots), figsize=(15, 5))
    ax_i = 0
    if "rgb" in plots:
        rgb = opt_rgb.numpy().astype(np.float32)
        rgb = np.transpose(rgb, (1, 2, 0))
        axes[ax_i].imshow(np.clip(rgb / 60., 0, 2))
        axes[ax_i].set_title("rgb")
        ax_i+=1

    if "magic_ref" in plots:
        im = axes[ax_i].imshow(opt_magic_reference)
        colorbar_next_to(im, axes[ax_i])
        axes[ax_i].set_title("magic_reference")
        ax_i+=1

    for product, name in zip(products, names):
        im = axes[ax_i].imshow(product)
        colorbar_next_to(im, axes[ax_i])
        axes[ax_i].set_title(name)
        ax_i+=1

    if "y" in plots:
        axes[ax_i].imshow(opt_y)
        axes[ax_i].set_title("GT")
        ax_i+=1

    #name + "_" + str(plot_idx).zfill(3) + ".jpg"
    if save:
        plt.savefig(save_name, dpi=300)
    if show:
        plt.show()

    return times