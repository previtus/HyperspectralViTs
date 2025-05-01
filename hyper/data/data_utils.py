# Has low level functions for reading data

import os
import rasterio
import numpy as np
import glob
import torch
from osgeo import gdal
from hyper.parameters.nodata import NODATA_padded_value, NODATA_AVIRIS
from georeader.geotensor import GeoTensor
from georeader.save_cog import save_cog

def file_exists(path):
    return os.path.exists(path)

def dgal_no_window(ds, bands_to_load):
    data = []

    if bands_to_load is None:
        raster_count = ds.RasterCount
        # print("raster_count", raster_count)
        bands_to_load = list(range(raster_count))
        # print("bands_to_load", bands_to_load)

    for band_i in bands_to_load:
        # print("band i", band_i, "in no-window (from bands_to_load=", bands_to_load,")")
        data_i = ds.GetRasterBand(band_i+1).ReadAsArray()
        data.append(data_i)
    return np.asarray(data)

def dgal_with_window(ds, bands_to_load, xOffset, yOffset, xSize, ySize):
    data = []

    if bands_to_load is None:
        raster_count = ds.RasterCount
        # print("raster_count", raster_count)
        bands_to_load = list(range(raster_count))
        # print("bands_to_load", bands_to_load)

    for band_i in bands_to_load:
        # print("band i", band_i, "in windowed (from bands_to_load=", bands_to_load,")")
        data_i = ds.GetRasterBand(band_i+1).ReadAsArray(xOffset, yOffset, xSize, ySize)
        data.append(data_i)
    return np.asarray(data)

def simple_load_(path, bands_to_load=[0], window = None, extend_window_by_extra=0, mask_nodata=False):
    # instead of just 1 band, we load multiple ...

    ds = gdal.Open(path)
    maxX, maxY = ds.RasterXSize, ds.RasterYSize

    if window is None:
        return dgal_no_window(ds, bands_to_load)

    xOffset = int(window.col_off)
    yOffset = int(window.row_off)
    xSize = int(window.width)
    ySize = int(window.height)

    should_be_x = xSize + 2*extend_window_by_extra
    should_be_y = ySize + 2*extend_window_by_extra

    # Check the start against out of bounds ...
    padding = False
    x_pad_L = 0
    y_pad_L = 0
    # if we won't go into negative
    if xOffset - extend_window_by_extra > 0:
        # then we can extend it
        xOffset = xOffset - extend_window_by_extra
        # but also extend the loaded region then
        xSize = xSize + extend_window_by_extra
    else:
        # we will need to pad the result by extend_window_by_extra instead
        x_pad_L = extend_window_by_extra
        padding = True
    # same for y
    if yOffset - extend_window_by_extra > 0:
        yOffset = yOffset - extend_window_by_extra
        ySize = ySize + extend_window_by_extra
    else:
        y_pad_L = extend_window_by_extra
        padding = True

    # Now check the end
    # if we don't go outside of the maxX, we should extend it
    if xOffset + xSize + extend_window_by_extra <= maxX:
        # then we should extend it
        xSize = xSize + extend_window_by_extra
    else:
        # else we will pad that amount with zeros
        padding = True

    if yOffset + ySize + extend_window_by_extra <= maxY:
        ySize = ySize + extend_window_by_extra
    else:
        padding = True

    data = dgal_with_window(ds, bands_to_load=bands_to_load, xOffset=xOffset, yOffset=yOffset, xSize=xSize, ySize=ySize)

    if padding:
        padded = np.ones((data.shape[0], should_be_y, should_be_x)) * NODATA_padded_value
        padded[:, y_pad_L:y_pad_L+data.shape[1] , x_pad_L:x_pad_L+data.shape[2]] = data
        data = padded

    return data

def simple_load(path, bands_to_load=[0], window = None, extend_window_by_extra=0, mask_nodata=False):
    try:
        return simple_load_(path, bands_to_load=bands_to_load, window=window, extend_window_by_extra=extend_window_by_extra, mask_nodata=mask_nodata)
    except Exception as e:
        print("EXCEPTION", e)
        print("FAILED LOADIND DATA WITH PATH", path)
        print("bands_to_load=", bands_to_load)
        print("window=", window)
        print("extend_window_by_extra=", extend_window_by_extra)
        print("mask_nodata=", mask_nodata)
        assert False # then still fail, but with this ^ report ...

def center_crop(data, w, h):
    assert len(data.shape) == 3 # assuming data in CH,W,H
    start_w = int((data.shape[1] - w) / 2)
    start_h = int((data.shape[2] - h) / 2)
    return data[:, start_w:start_w+w, start_h:start_h+h]

def center_crop_lists(lists, w, h):
    for i, l in enumerate(lists):
        # print(l.shape, "into w,h", w, h)
        lists[i] = center_crop(l, w, h)
        # print("now", lists[i].shape)
    return lists

# def simple_load_RasterioAlternative(path, window):
#     # NOTE:
#     # For some reason this caused memory growth when it was loaded in train dataloader many times over
#     with rasterio.open(path) as src:
#         data = src.read(window=window)
#     return data

def get_validity_mask(data):
    # note: use any input product, except for mag1c
    mask = np.where(data == NODATA_AVIRIS, 0, 1) # 1 valid, 0 invalid
    if NODATA_AVIRIS != NODATA_padded_value:
        mask = np.where(data == NODATA_padded_value, 0, mask) # apply the other check too
    return mask

def load_band_names_as_tensors(band_names, window, event_folder, alternative_folder = None, extend_window_by_extra=0, mask_nodata=False, CH_over_keep_all = False):
    # Create tensor first and then load into it
    CH = len(band_names)
    if CH == 0: return None

    paths = []

    for band_name in band_names:
        path = os.path.join(event_folder, band_name)
        if not file_exists(path):
            # use alt path

            if alternative_folder is None:
                print("didn't provide alt path, can't find", event_folder, band_name)
            path = os.path.join(alternative_folder, band_name)
        paths.append(path)

    if CH_over_keep_all:
        data = simple_load(paths[0], bands_to_load=None, window=window, extend_window_by_extra=extend_window_by_extra)
    else:
        data = simple_load(paths[0], window=window, extend_window_by_extra=extend_window_by_extra)

    first = torch.from_numpy(data) # loads as 1,W,H
    data = None # derefmask_nodata
    del data # del

    if CH_over_keep_all:
        CH, W, H = first.shape
        tensors = first
    else:
        _, W,H = first.shape
        tensors = torch.empty((CH, W, H))
        tensors[0] = first[0]

    if CH == 1:
        return tensors

    for idx, name in enumerate(band_names[1:]):
        data = simple_load(paths[idx + 1], window=window, extend_window_by_extra=extend_window_by_extra) # mask only from the first band
        tensors[idx + 1] = torch.from_numpy(data[0])
        data = None  # deref
        del data  # del

    return tensors

def load_emit_data_as_tensors(band_names, window, event_folder, alternative_folder = None, extend_window_by_extra=0, mask_nodata=False, available_bands=None):
    # Is capable to load part of the data from specific tifs, and the rest from a large single file (using it's description for bands identification)
    specific_products = []
    remaining_products = []
    for b in band_names:
        if ".tif" in b:
            specific_products.append(b)
        else:
            remaining_products.append(b)

    if len(specific_products) > 0:
        specific_tensors = load_band_names_as_tensors(specific_products, window, event_folder, alternative_folder=alternative_folder,
                                   extend_window_by_extra=extend_window_by_extra,mask_nodata=mask_nodata)

    bands_to_load = [emit_wv_to_band_index(emit_band_name_to_wv(b), available_bands) for b in remaining_products]

    if len(bands_to_load) > 0:
        base_of_the_file = remaining_products[0].split("_EMIT_")[0]
        path = os.path.join(event_folder, base_of_the_file)

        data = simple_load(path, bands_to_load=bands_to_load, window=window, extend_window_by_extra=extend_window_by_extra, mask_nodata=mask_nodata)
        remaining_tensors = torch.from_numpy(data)

    if len(specific_products) == 0:
        return remaining_tensors
    if len(bands_to_load) == 0:
        return specific_tensors
    else:
        return torch.cat([specific_tensors, remaining_tensors], dim=0)


def load_band_names_as_numpy(band_names, window, event_folder, alternative_folder = None):
    # Create numpy first and then load into it
    CH = len(band_names)

    paths = []

    for band_name in band_names:
        path = os.path.join(event_folder, band_name)
        if not file_exists(path):
            if alternative_folder is None:
                print("file doesn't exist (and none alt folder given):", path)
            # use alt path
            path = os.path.join(alternative_folder, band_name)
            if file_exists(path):
                print("file doesn't exist (in alt folder):", path)
        paths.append(path)

    first = simple_load(paths[0], window=window)

    _, W,H = first.shape
    arrays = np.zeros((CH, W, H))
    arrays[0] = first[0]

    first = None  # deref
    del first  # del

    if CH == 1:
        return arrays

    for idx, name in enumerate(band_names[1:]):
        data = simple_load(paths[idx + 1], window=window)
        arrays[idx + 1] = data[0]
        data = None  # deref
        del data  # del

    return arrays


def get_available_aviris_bands(root_folder):
    # gets called when creating the dataset, will check once how many bands we actually have in the folder
    # this could have been hardcoded, but a single check at the creation of the dataset is preferable for modularity ...
    folders = [f for f in glob.glob(os.path.join(root_folder, "*")) if ".csv" not in f]

    band_files = sorted(glob.glob(os.path.join(folders[0], "TOA_AVIRIS_*nm.tif")))
    bands_waves = [int(b.split("nm")[0].split("TOA_AVIRIS_")[1]) for b in band_files]

    return sorted(bands_waves)

def fix_emit_descriptions_if_need_be(descriptions):
    if descriptions[0] == "Band 1": # then needs fixing ...
        if len(descriptions) == 86:
            # this is not pretty, as it assumes we don't change the bands in the raw data ...
            return ['462.59888 (462.59888)', '551.8667 (551.8667)', '641.2759 (641.2759)',
                                '1573.3193 (1573.3193)', '1580.7621 (1580.7621)', '1588.205 (1588.205)',
                                '1595.6467 (1595.6467)', '1603.0886 (1603.0886)', '1610.5295 (1610.5295)',
                                '1617.9705 (1617.9705)', '1625.4104 (1625.4104)', '1632.8513 (1632.8513)',
                                '1640.2903 (1640.2903)', '1647.7303 (1647.7303)', '1655.1694 (1655.1694)',
                                '1662.6074 (1662.6074)', '1670.0455 (1670.0455)', '1677.4836 (1677.4836)',
                                '1684.9209 (1684.9209)', '1692.358 (1692.358)', '1699.7952 (1699.7952)',
                                '2004.355 (2004.355)', '2011.7745 (2011.7745)', '2019.1931 (2019.1931)',
                                '2026.6118 (2026.6118)', '2034.0304 (2034.0304)', '2041.4471 (2041.4471)',
                                '2048.865 (2048.865)', '2056.2808 (2056.2808)', '2063.6965 (2063.6965)',
                                '2071.1123 (2071.1123)', '2078.5273 (2078.5273)', '2085.9421 (2085.9421)',
                                '2093.3562 (2093.3562)', '2100.769 (2100.769)', '2108.1821 (2108.1821)',
                                '2115.5942 (2115.5942)', '2123.0063 (2123.0063)', '2130.4175 (2130.4175)',
                                '2137.8289 (2137.8289)', '2145.239 (2145.239)', '2152.6482 (2152.6482)',
                                '2160.0576 (2160.0576)', '2167.467 (2167.467)', '2174.8755 (2174.8755)',
                                '2182.283 (2182.283)', '2189.6904 (2189.6904)', '2197.097 (2197.097)',
                                '2204.5034 (2204.5034)', '2211.9092 (2211.9092)', '2219.3147 (2219.3147)',
                                '2226.7195 (2226.7195)', '2234.1233 (2234.1233)', '2241.5269 (2241.5269)',
                                '2248.9297 (2248.9297)', '2256.3328 (2256.3328)', '2263.7346 (2263.7346)',
                                '2271.1365 (2271.1365)', '2278.5376 (2278.5376)', '2285.9387 (2285.9387)',
                                '2293.3386 (2293.3386)', '2300.7378 (2300.7378)', '2308.136 (2308.136)',
                                '2315.5342 (2315.5342)', '2322.9326 (2322.9326)', '2330.3298 (2330.3298)',
                                '2337.7263 (2337.7263)', '2345.1216 (2345.1216)', '2352.517 (2352.517)',
                                '2359.9126 (2359.9126)', '2367.3071 (2367.3071)', '2374.7007 (2374.7007)',
                                '2382.0935 (2382.0935)', '2389.486 (2389.486)', '2396.878 (2396.878)',
                                '2404.2695 (2404.2695)', '2411.6604 (2411.6604)', '2419.0513 (2419.0513)',
                                '2426.4402 (2426.4402)', '2433.8303 (2433.8303)', '2441.2183 (2441.2183)',
                                '2448.6064 (2448.6064)', '2455.9944 (2455.9944)', '2463.3816 (2463.3816)',
                                '2470.7678 (2470.7678)', '2478.153 (2478.153)']
        elif len(descriptions) == 285:
            ## all bands are:
            return ['381.00558 (381.00558)', '388.4092 (388.4092)', '395.81583 (395.81583)', '403.2254 (403.2254)', '410.638 (410.638)', '418.0536 (418.0536)', '425.47214 (425.47214)', '432.8927 (432.8927)', '440.31726 (440.31726)', '447.7428 (447.7428)', '455.17035 (455.17035)', '462.59888 (462.59888)', '470.0304 (470.0304)', '477.46292 (477.46292)', '484.89743 (484.89743)', '492.33292 (492.33292)', '499.77142 (499.77142)', '507.2099 (507.2099)', '514.6504 (514.6504)', '522.0909 (522.0909)', '529.5333 (529.5333)', '536.9768 (536.9768)', '544.42126 (544.42126)', '551.8667 (551.8667)', '559.3142 (559.3142)', '566.7616 (566.7616)', '574.20905 (574.20905)', '581.6585 (581.6585)', '589.108 (589.108)', '596.55835 (596.55835)', '604.0098 (604.0098)', '611.4622 (611.4622)', '618.9146 (618.9146)', '626.36804 (626.36804)', '633.8215 (633.8215)', '641.2759 (641.2759)', '648.7303 (648.7303)', '656.1857 (656.1857)', '663.6411 (663.6411)', '671.09753 (671.09753)', '678.5539 (678.5539)', '686.0103 (686.0103)', '693.4677 (693.4677)', '700.9251 (700.9251)', '708.38354 (708.38354)', '715.84094 (715.84094)', '723.2993 (723.2993)', '730.7587 (730.7587)', '738.2171 (738.2171)', '745.6765 (745.6765)', '753.1359 (753.1359)', '760.5963 (760.5963)', '768.0557 (768.0557)', '775.5161 (775.5161)', '782.97754 (782.97754)', '790.4379 (790.4379)', '797.89935 (797.89935)', '805.36176 (805.36176)', '812.8232 (812.8232)', '820.2846 (820.2846)', '827.746 (827.746)', '835.2074 (835.2074)', '842.66986 (842.66986)', '850.1313 (850.1313)', '857.5937 (857.5937)', '865.0551 (865.0551)', '872.5176 (872.5176)', '879.98004 (879.98004)', '887.44147 (887.44147)', '894.90393 (894.90393)', '902.3664 (902.3664)', '909.82886 (909.82886)', '917.2913 (917.2913)', '924.7538 (924.7538)', '932.21625 (932.21625)', '939.6788 (939.6788)', '947.14026 (947.14026)', '954.6027 (954.6027)', '962.0643 (962.0643)', '969.5268 (969.5268)', '976.9883 (976.9883)', '984.4498 (984.4498)', '991.9114 (991.9114)', '999.37286 (999.37286)', '1006.8344 (1006.8344)', '1014.295 (1014.295)', '1021.7566 (1021.7566)', '1029.2172 (1029.2172)', '1036.6777 (1036.6777)', '1044.1383 (1044.1383)', '1051.5989 (1051.5989)', '1059.0596 (1059.0596)', '1066.5201 (1066.5201)', '1073.9797 (1073.9797)', '1081.4404 (1081.4404)', '1088.9 (1088.9)', '1096.3597 (1096.3597)', '1103.8184 (1103.8184)', '1111.2781 (1111.2781)', '1118.7368 (1118.7368)', '1126.1964 (1126.1964)', '1133.6552 (1133.6552)', '1141.1129 (1141.1129)', '1148.5717 (1148.5717)', '1156.0304 (1156.0304)', '1163.4882 (1163.4882)', '1170.9459 (1170.9459)', '1178.4037 (1178.4037)', '1185.8616 (1185.8616)', '1193.3184 (1193.3184)', '1200.7761 (1200.7761)', '1208.233 (1208.233)', '1215.6898 (1215.6898)', '1223.1467 (1223.1467)', '1230.6036 (1230.6036)', '1238.0596 (1238.0596)', '1245.5154 (1245.5154)', '1252.9724 (1252.9724)', '1260.4283 (1260.4283)', '1267.8833 (1267.8833)', '1275.3392 (1275.3392)', '1282.7942 (1282.7942)', '1290.2502 (1290.2502)', '1297.7052 (1297.7052)', '1305.1603 (1305.1603)', '1312.6144 (1312.6144)', '1320.0685 (1320.0685)', '1327.5225 (1327.5225)', '1334.9756 (1334.9756)', '1342.4287 (1342.4287)', '1349.8818 (1349.8818)', '1357.3351 (1357.3351)', '1364.7872 (1364.7872)', '1372.2384 (1372.2384)', '1379.6907 (1379.6907)', '1387.1418 (1387.1418)', '1394.5931 (1394.5931)', '1402.0433 (1402.0433)', '1409.4937 (1409.4937)', '1416.944 (1416.944)', '1424.3933 (1424.3933)', '1431.8427 (1431.8427)', '1439.292 (1439.292)', '1446.7404 (1446.7404)', '1454.1888 (1454.1888)', '1461.6372 (1461.6372)', '1469.0847 (1469.0847)', '1476.5321 (1476.5321)', '1483.9796 (1483.9796)', '1491.4261 (1491.4261)', '1498.8727 (1498.8727)', '1506.3192 (1506.3192)', '1513.7649 (1513.7649)', '1521.2104 (1521.2104)', '1528.655 (1528.655)', '1536.1007 (1536.1007)', '1543.5454 (1543.5454)', '1550.9891 (1550.9891)', '1558.4329 (1558.4329)', '1565.8766 (1565.8766)', '1573.3193 (1573.3193)', '1580.7621 (1580.7621)', '1588.205 (1588.205)', '1595.6467 (1595.6467)', '1603.0886 (1603.0886)', '1610.5295 (1610.5295)', '1617.9705 (1617.9705)', '1625.4104 (1625.4104)', '1632.8513 (1632.8513)', '1640.2903 (1640.2903)', '1647.7303 (1647.7303)', '1655.1694 (1655.1694)', '1662.6074 (1662.6074)', '1670.0455 (1670.0455)', '1677.4836 (1677.4836)', '1684.9209 (1684.9209)', '1692.358 (1692.358)', '1699.7952 (1699.7952)', '1707.2314 (1707.2314)', '1714.6667 (1714.6667)', '1722.103 (1722.103)', '1729.5383 (1729.5383)', '1736.9727 (1736.9727)', '1744.4071 (1744.4071)', '1751.8414 (1751.8414)', '1759.2749 (1759.2749)', '1766.7084 (1766.7084)', '1774.1418 (1774.1418)', '1781.5743 (1781.5743)', '1789.007 (1789.007)', '1796.4385 (1796.4385)', '1803.8701 (1803.8701)', '1811.3008 (1811.3008)', '1818.7314 (1818.7314)', '1826.1611 (1826.1611)', '1833.591 (1833.591)', '1841.0206 (1841.0206)', '1848.4495 (1848.4495)', '1855.8773 (1855.8773)', '1863.3052 (1863.3052)', '1870.733 (1870.733)', '1878.16 (1878.16)', '1885.5869 (1885.5869)', '1893.013 (1893.013)', '1900.439 (1900.439)', '1907.864 (1907.864)', '1915.2892 (1915.2892)', '1922.7133 (1922.7133)', '1930.1375 (1930.1375)', '1937.5607 (1937.5607)', '1944.9839 (1944.9839)', '1952.4071 (1952.4071)', '1959.8295 (1959.8295)', '1967.2518 (1967.2518)', '1974.6732 (1974.6732)', '1982.0946 (1982.0946)', '1989.515 (1989.515)', '1996.9355 (1996.9355)', '2004.355 (2004.355)', '2011.7745 (2011.7745)', '2019.1931 (2019.1931)', '2026.6118 (2026.6118)', '2034.0304 (2034.0304)', '2041.4471 (2041.4471)', '2048.865 (2048.865)', '2056.2808 (2056.2808)', '2063.6965 (2063.6965)', '2071.1123 (2071.1123)', '2078.5273 (2078.5273)', '2085.9421 (2085.9421)', '2093.3562 (2093.3562)', '2100.769 (2100.769)', '2108.1821 (2108.1821)', '2115.5942 (2115.5942)', '2123.0063 (2123.0063)', '2130.4175 (2130.4175)', '2137.8289 (2137.8289)', '2145.239 (2145.239)', '2152.6482 (2152.6482)', '2160.0576 (2160.0576)', '2167.467 (2167.467)', '2174.8755 (2174.8755)', '2182.283 (2182.283)', '2189.6904 (2189.6904)', '2197.097 (2197.097)', '2204.5034 (2204.5034)', '2211.9092 (2211.9092)', '2219.3147 (2219.3147)', '2226.7195 (2226.7195)', '2234.1233 (2234.1233)', '2241.5269 (2241.5269)', '2248.9297 (2248.9297)', '2256.3328 (2256.3328)', '2263.7346 (2263.7346)', '2271.1365 (2271.1365)', '2278.5376 (2278.5376)', '2285.9387 (2285.9387)', '2293.3386 (2293.3386)', '2300.7378 (2300.7378)', '2308.136 (2308.136)', '2315.5342 (2315.5342)', '2322.9326 (2322.9326)', '2330.3298 (2330.3298)', '2337.7263 (2337.7263)', '2345.1216 (2345.1216)', '2352.517 (2352.517)', '2359.9126 (2359.9126)', '2367.3071 (2367.3071)', '2374.7007 (2374.7007)', '2382.0935 (2382.0935)', '2389.486 (2389.486)', '2396.878 (2396.878)', '2404.2695 (2404.2695)', '2411.6604 (2411.6604)', '2419.0513 (2419.0513)', '2426.4402 (2426.4402)', '2433.8303 (2433.8303)', '2441.2183 (2441.2183)', '2448.6064 (2448.6064)', '2455.9944 (2455.9944)', '2463.3816 (2463.3816)', '2470.7678 (2470.7678)', '2478.153 (2478.153)', '2485.5386 (2485.5386)', '2492.9238 (2492.9238)']

    return descriptions

def get_available_emit_bands(root_folder):
    # print("debug, get_available_emit_bands called!")
    # gets called when creating the dataset, will check once how many bands we actually have in the folder
    # this could have been hardcoded, but a single check at the creation of the dataset is preferable for modularity ...
    folders = [f for f in glob.glob(os.path.join(root_folder, "*")) if ".csv" not in f]

    try_files = ["A", "B", "C"] # A,B as a pair, B as just a single tile with an event and C as a clean single tile (will be used with minerals)
    available_files = []
    for file in try_files:
        if file_exists(os.path.join(folders[0],file)):
            available_files.append(file)

    if len(available_files) == 0:
        # Special case, we don't have either A or B files in our dataset...

        band_files = sorted(glob.glob(os.path.join(folders[0], "B_EMIT_*nm.tif"))) # for example "B_EMIT_462nm.tif"
        bands_waves = [int(b.split("nm")[0].split("B_EMIT_")[1]) for b in band_files]
        ### print("will return ", bands_waves) # [462, 551, 641]
        return [None, sorted(bands_waves)]
    else:
        # let's assume that all these files will have the same bands...
        with rasterio.open(os.path.join(folders[0],available_files[0])) as src:
            descriptions = fix_emit_descriptions_if_need_be(src.descriptions)

        bands_waves = [emit_descrption_to_wv(d) for d in descriptions]

        # Returns list of available files for EMIT - like A, B
        return [available_files, sorted(bands_waves)]


def wv_to_aviris_band_name(wv):
    return f"TOA_AVIRIS_{wv}nm.tif"

def wv_to_emit_band_name_specific(wv):
    return f"B_EMIT_{wv}nm.tif" # < more complicated with: A_EMIT_... and B_EMIT_...
def wv_to_emit_band_name_bandranges(wv):
    return f"B_EMIT_{wv}"

def emit_descrption_to_wv(emit_desc):
    return int(emit_desc.split(" ")[0].split(".")[0]) # '462.59888 (462.59888)' => 462 ... keep only the start

def emit_band_name_to_wv(emit_band_name):
    if "EMIT_" in emit_band_name:
        # can be B_EMIT_462nm.tif but also A_clean_EMIT_462nm.tif or something ...
        return int(emit_band_name.split("EMIT_")[1].replace("nm.tif", ""))
    else:
        return emit_band_name.replace(".tif", "")

def emit_adjust_base(product_names, base, first_base):
    # product_names: ['B_EMIT_641nm.tif', 'B_EMIT_551nm.tif', 'B_EMIT_462nm.tif', 'B_magic30_tile.tif', 'B_EMIT_2100', 'B_EMIT_2108', 'B_EMIT_2115']
    # base: A_clean
    # -> we will want to prepend all of them with that new base
    adjusted_product_names = []
    for product_name in product_names:
        adjusted_product_names.append(base+product_name.replace(first_base,"",1))
    return adjusted_product_names

def emit_wv_to_band_index(wv, emit_available_bands):
    _, bands = emit_available_bands
    if wv in bands:
        return bands.index(wv)
    else:
        assert False

def aviris_band_name_to_wv(aviris_band_name):
    if "TOA_AVIRIS_" in aviris_band_name:
        return int(aviris_band_name.replace("nm.tif", "").replace("TOA_AVIRIS_", ""))
    else:
        return aviris_band_name.replace(".tif", "")

def input_products_from_settings_AVIRIS(specific_products, band_ranges, available_aviris_bands):
    input_products = []
    # start with 'specific_products', then load 'band_ranges'
    # print(self.settings.dataset.input_products)
    for specific_input_product in specific_products:
        # integers are AVIRIS bands:
        if isinstance(specific_input_product, str) and specific_input_product.isnumeric():
            specific_input_product = int(specific_input_product)
        if isinstance(specific_input_product, int):
            input_products.append(wv_to_aviris_band_name(specific_input_product))
        else:
            input_products.append(f"{specific_input_product}.tif")

    if len(band_ranges) > 0:
        # print("ranges",band_ranges,"meanwhile we have self.available_aviris_bands=", self.available_aviris_bands)
        for r in band_ranges:
            r_start, r_end = r[0], r[1]
            bands_selected = [wv for wv in available_aviris_bands if (wv >= r_start and wv <= r_end)]
            # print("selected:", bands_selected)
            for wv in bands_selected:
                input_products.append(wv_to_aviris_band_name(wv))

    assert len(set(input_products)) == len(input_products), "Input products (ranges and specific products) selected duplicates!"

    return input_products

def input_products_from_settings_EMIT(specific_products, band_ranges, available_emit_bands):
    # specific products convert to their .tif names
    # band ranges into the band indices

    input_products = []
    # start with 'specific_products', then load 'band_ranges'
    # print(self.settings.dataset.input_products)
    for specific_input_product in specific_products:
        # integers are EMIT bands:
        if isinstance(specific_input_product, str) and specific_input_product.isnumeric():
            specific_input_product = int(specific_input_product)
        if isinstance(specific_input_product, int):
            input_products.append(wv_to_emit_band_name_specific(specific_input_product))
        else:
            input_products.append(f"{specific_input_product}.tif")

    available_files, available_bands = available_emit_bands

    if len(band_ranges) > 0:
        #print("ranges",band_ranges,"meanwhile we have self.available_aviris_bands=", available_emit_bands)
        for r in band_ranges:
            r_start, r_end = r[0], r[1]
            bands_selected = [wv for idx, wv in enumerate(available_bands) if (wv >= r_start and wv <= r_end)]
            # indices_of_selected = [idx for idx, wv in enumerate(available_bands) if (wv >= r_start and wv <= r_end)]
            # print("selected:", bands_selected)
            # print("indices_of_selected (in total",len(indices_of_selected),"):", indices_of_selected)

            for wv in bands_selected:
                input_products.append(wv_to_emit_band_name_bandranges(wv))

    # ^ in contrast to AVIRIS, these won't get translated as .tif - instead they will be kept as the numbers, but we still need to mark then with EMIT_

    assert len(set(input_products)) == len(input_products), "Input products (ranges and specific products) selected duplicates!"
    # print("debug input_products_from_settings_EMIT output", "input_products:", input_products)
    return input_products

def file_exists_and_not_empty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0

def mkdir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_data_to_path(data, target_path, geo_reference):
    REF = rasterio.open(geo_reference)
    f = target_path.split("/")[-1].replace(".tif", "")
    save_cog(GeoTensor(data, transform=REF.transform, crs=REF.crs, fill_value_default=None),
             path_tiff_save=target_path, descriptions=[f],
             profile={"BLOCKSIZE": 128, "nodata": REF.nodata})

