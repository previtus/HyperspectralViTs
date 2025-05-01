# Examples of exploring data:

# AVIRIS:
# download the STARCOP_allbands from https://huggingface.co/collections/previtus/starcop-67f13cf30def71591f281a41
# or a minidataset preview on https://huggingface.co/datasets/previtus/starcop_allbands_mini
python -m scripts.data_explore \
          dataset.format="AVIRIS" \
          dataset.input_products.specific_products=[] \
          dataset.auxiliary_products=["mag1c"] \
          dataset.input_products.band_ranges=[[400,2500]] \
          dataset.normalisation.mode_input="hardcoded" \
          dataset.root_folder="/DATASETS/STARCOP_allbands" \
          dataset.train_csv="train.csv" \
          dataset.test_csv="test.csv" \
          training.visualiser.bands=['rgb','mag1c','labelbinary','valid_mask']


# EMIT OxHyperSyntheticCH4:
# download the OxHyperSyntheticCH4 dataset from https://huggingface.co/datasets/previtus/OxHyperSyntheticCH4
# or a minidataset preview on https://huggingface.co/datasets/previtus/OxHyperSyntheticCH4_MINI
python -m scripts.data_explore \
          dataset.format="EMIT" \
          dataset.input_products.specific_products=[] \
          dataset.auxiliary_products=["B_magic30_tile"] \
          dataset.input_products.band_ranges=[[400,2500]] \
          model.multiply_loss_by_mag1c=False \
          dataset.feature_extractor=[] \
          dataset.normalisation.mode_input="hardcoded" \
          dataset.root_folder="/DATASETS/OxHyperSyntheticCH4" \
          dataset.train_csv="train_filtered_v2.csv" \
          dataset.val_csv="val_filtered_v2.csv" \
          dataset.test_csv="test_filtered_v2.csv" \
          training.visualiser.bands=['rgb','mag1c','labelbinary','valid_mask']

# EMIT OxHyperRealCH4:
# download the OxHyperRealCH4 dataset from https://huggingface.co/datasets/previtus/OxHyperRealCH4
# or a minidataset preview on https://huggingface.co/datasets/previtus/OxHyperRealCH4_MINI
# - same as above, but change:
#          dataset.root_folder="/DATASETS/OxHyperRealCH4" \
#          dataset.train_csv="real_train_v3.csv" \
#          dataset.val_csv="real_val_v3.csv" \
#          dataset.test_csv="real_test_v3.csv" \


# EMIT OxHyperMinerals:
# download the OxHyperMinerals dataset from https://huggingface.co/datasets/previtus/OxHyperMinerals_Train
#                                       and https://huggingface.co/datasets/previtus/OxHyperMinerals_TestVal
# or a minidataset preview on https://huggingface.co/datasets/previtus/OxHyperMinerals_MINI
python -m scripts.data_explore \
          dataset.format="EMIT_MINERALS" \
          dataset.input_products.specific_products=[] \
          dataset.auxiliary_products=[] \
          dataset.input_products.band_ranges=[[0,2500]] \
          dataset.output_products=["minerals3ghk"] \
          model.multiply_loss_by_mag1c=False \
          dataset.feature_extractor=[] \
          dataset.normalisation.mode_input="hardcoded" \
          dataset.root_folder="/DATASETS/OxHyperMinerals" \
          dataset.train_csv="train_minerals.csv" \
          dataset.val_csv="val_minerals.csv" \
          dataset.test_csv="test_minerals.csv" \
          training.visualiser.bands=['rgb','minerals','valid_mask']
