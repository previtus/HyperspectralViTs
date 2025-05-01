from georeader.slices import create_windows

def create_windows_regular_tiling(input_dimensions, tile_sizes, tile_overlaps):
    # Ref implementation:
    windows = create_windows(input_dimensions, window_size=tile_sizes, overlap=tile_overlaps,
                             include_incomplete=False)
    return windows
