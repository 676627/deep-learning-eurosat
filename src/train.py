import json
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import keras
import wandb
from wandb.integration.keras import WandbMetricsLogger

from src.dataset import load_rgb_dataset, load_ms_dataset
from src.model import build_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--mode",          type=str,    default="rgb", choices=["rgb", "ms"])
parser.add_argument("--band_indices",  type=int,    default=None)
parser.add_argument("--epochs",        type=int,    default=50)
parser.add_argument("--batch_size",    type=int,    default=64)
parser.add_argument("--lr",            type=float,  default=1e-4)
parser.add_argument("--max_per_class", type=int,    default=None)
parser.add_argument("--model_version", type=str,    default=None, choices=["baseline", "batchnorm", "batchnorm_3conv", "batchnorm_3conv_gap"])
parser.add_argument("--dropout",       type=float,  default=0.4)
parser.add_argument("--data_dir",      type=str,    default="data/EuroSAT_RGB")
args = parser.parse_args()

MODE          = args.mode
BAND_INDICES  = args.band_indices  # None = all 13 bands. Example: [3, 2, 1] for RGB-equivalent
EPOCHS        = args.epochs
BATCH_SIZE    = args.batch_size
LEARNING_RATE = args.lr
MAX_PER_CLASS = args.max_per_class # None = use all data

RGB_DATA_DIR  = "data/EuroSAT_RGB"
MS_DATA_DIR   = "data/EuroSAT_MS"

# Used for run naming in W&B and checkpoint files:
dataset_label = "RGB" if MODE == "rgb" else "MS"
subset_label  = f"{args.max_per_class}perclass" if args.max_per_class else "full"
run_name      = f"{dataset_label}_{args.model_version}_{subset_label}"

# Load data
if MODE == "rgb":
    train_ds, val_ds = load_rgb_dataset(RGB_DATA_DIR, batch_size=BATCH_SIZE, max_per_class=MAX_PER_CLASS)
    n_channels = 3
else:
    band_indices = BAND_INDICES if BAND_INDICES is not None else list(range(13))
    train_ds, val_ds, stats = load_ms_dataset(
        MS_DATA_DIR, band_indices=band_indices, batch_size=BATCH_SIZE, max_per_class=MAX_PER_CLASS
    )
    n_channels = len(band_indices)
    stats_path = os.path.join("checkpoints", f"{run_name}_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats, f)
    print(f"Saved normalization stats to {stats_path}")

# Build model
model = build_model(in_channels=n_channels, model_version=args.model_version, dropout=args.dropout)
model.summary()

# Set up W&B logging
wandb.init(
    project="eurosat-cnn",
    name=run_name,
    config={
        "mode": MODE,
        "band_indices": BAND_INDICES,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
    }
)

# Train
model.compile(
    optimizer=keras.optimizers.Adam(LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

os.makedirs("checkpoints", exist_ok=True)

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    callbacks=[
        WandbMetricsLogger(),
        keras.callbacks.ModelCheckpoint(
            f"checkpoints/{run_name}_best.keras",
            monitor="val_accuracy",
            save_best_only=True,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=8
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=15, restore_best_weights=True
        ),
    ],
)

wandb.finish()