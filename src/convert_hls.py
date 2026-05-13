import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import hls4ml
from tensorflow.keras.models import load_model

# Load trained model
model = load_model('model/best_model.keras')

# hls4ml configuration
config = hls4ml.utils.config_from_keras_model(model, granularity='name')

# Set precision — ap_fixed<16,6> is standard starting point for FPGA
for layer in config['LayerName']:
    config['LayerName'][layer]['Precision']['result'] = 'ap_fixed<16,6>'
    config['LayerName'][layer]['Precision']['weight'] = 'ap_fixed<16,6>'
    config['LayerName'][layer]['Precision']['bias']   = 'ap_fixed<16,6>'

print("HLS Config:")
import pprint
pprint.pprint(config)

# Convert
hls_model = hls4ml.converters.convert_from_keras_model(
    model,
    hls_config=config,
    output_dir='hls_output/malaria_hls',
    part='xc7z020clg400-1'        # PYNQ-Z2 part number
)

hls_model.write()
print("HLS project written successfully.")
