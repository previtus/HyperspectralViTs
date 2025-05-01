import pylab as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

def show_1_band(band, ax, add_colorbar=False, vmin=None, vmax=None):
    im = ax.imshow(band, vmin=vmin, vmax=vmax)

    if add_colorbar:
        colorbar_next_to(im, ax)
    return im

def show_3_bands(rgb, ax):
    out = np.transpose(rgb, (1, 2, 0))
    # move to range between 0 and 1 to prevent warning messages
    out = np.clip(out, 0., 1.)
    return ax.imshow(out)
    #return ax.imshow((out * 255).astype(np.uint8))

def colorbar_next_to_vert(im, ax, size='5%',pad=0.05, invisible=False):
    # Vertical
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size=size, pad=pad)
    cb = plt.gcf().colorbar(im, cax=cax, orientation='vertical')
    if invisible: cb.remove()

def colorbar_next_to(im, ax, size='5%',pad=0.05, invisible=False):
    # Horizontal
    divider = make_axes_locatable(ax)
    cax = divider.new_vertical(size=size, pad=pad, pack_start=True)
    plt.gcf().add_axes(cax)
    cb = plt.gcf().colorbar(im, cax=cax, orientation='horizontal')
    if invisible: cb.remove()


def plot_rgb(rgb):
    figure = plt.figure(figsize=(8, 4))
    ax = plt.gca()
    show_3_bands(rgb, ax)
    plt.show()



