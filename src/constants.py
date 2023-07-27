import os
import logging, sys

# Logging
logging.basicConfig(stream=sys.stderr, level=logging.INFO)

# Path
dataset_path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dataset", "raw"))

# Images
dataset_width = 3028
dataset_height = 4080

# Key Lenght
key_length = 128*2

# Nm
compensation_margin = 0
# Nb = L + Nm
num_blocks = key_length + compensation_margin

# Threshold
hamming_threshold = 1