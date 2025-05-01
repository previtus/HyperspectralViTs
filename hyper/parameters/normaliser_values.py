# Values for the normaliser
# used only if we don't derive normalisation values from the data automatically
# probably stored as dictionaries: { band identifier : values needed for normalisation }

BAND_NORMALIZATION = {
    # AVIRIS rgb
    550: {'offset': 0, 'factor': 60, 'clip': (0, 2)},
    640: {'offset': 0, 'factor': 60, 'clip': (0, 2)},
    460: {'offset': 0, 'factor': 60, 'clip': (0, 2)},
    # ~ data divided by 60 then fills the whole 0-2 range

    # EMIT rgb
    551: {'offset': 0, 'factor': 20, 'clip': (0, 2)},
    641: {'offset': 0, 'factor': 20, 'clip': (0, 2)},
    462: {'offset': 0, 'factor': 20, 'clip': (0, 2)},

    2004: {'offset': 0, 'factor': 1, 'clip': (0, 2)},
    2109: {'offset': 0, 'factor': 5, 'clip': (0, 2)},
    2310: {'offset': 0, 'factor': 4, 'clip': (0, 2)},
    2350: {'offset': 0, 'factor': 3, 'clip': (0, 2)},
    2360: {'offset': 0, 'factor': 3, 'clip': (0, 2)},
    'mag1c': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},

    # EMIT pre-cooked files ...
    'A_magic30_tile.tif': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},
    'B_magic30_tile.tif': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},

    # default setting, keep the data, but clip it between -10 and +10
    'default': {'offset': 0, 'factor': 1, 'clip': (-10, 10)},

    # default for matched filter outputs
    'mag1c_default': {'offset': 0, 'factor': 1750, 'clip': (0, 2)},
}
#torch.clamp((x-offset) / self.factor, self.clip_min_input ,self.clip_max_input)


