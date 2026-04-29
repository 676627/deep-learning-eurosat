"""
scripts/plot_models.py — Plots model architecture diagrams using keras.utils.plot_model.

Saves one PNG per model version to results/figures/.

Requirements:
    pip install pydot graphviz
    # On Ubuntu/Debian (inside Docker):
    apt-get install -y graphviz

Usage:
    python scripts/plot_models.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
from src.model import build_model

os.makedirs("results/figures", exist_ok=True)

# Define which models to plot and with how many input channels
MODELS = [
    ("baseline",           3,  "RGB_baseline"),
    ("batchnorm",          3,  "RGB_batchnorm"),
    ("batchnorm_3conv",    3,  "RGB_batchnorm_3conv"),
    ("batchnorm_3conv_gap",3,  "RGB_batchnorm_3conv_gap"),
    ("batchnorm_3conv",    13, "MS_batchnorm_3conv"),
]

for version, in_channels, name in MODELS:
    print(f"Plotting {name}...")
    model = build_model(in_channels=in_channels, model_version=version)

    out_path = f"results/figures/model_{name}.png"

    keras.utils.plot_model(
        model,
        to_file=out_path,
        show_shapes=True,        # shows input/output shapes on each layer
        show_dtype=False,
        show_layer_names=True,
        show_layer_activations=True,
        expand_nested=False,
        dpi=150,
    )
    print(f"  Saved to {out_path}")

print("\nDone. Figures saved to results/figures/")