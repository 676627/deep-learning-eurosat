import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import keras
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

from src.dataset import load_rgb_dataset, load_ms_dataset, CLASSES

# Configuration — point this at the checkpoint you want to evaluate
CHECKPOINT    = "checkpoints/RGB_baseline_best.keras"
MODE          = "rgb"
RGB_DATA_DIR  = "data/EuroSAT_RGB"
MS_DATA_DIR   = "data/EuroSAT_MS"
BAND_INDICES  = None    # must match what was used during training

# Load the same validation data
if MODE == "rgb":
    _, val_ds = load_rgb_dataset(RGB_DATA_DIR)
else:
    band_indices = BAND_INDICES if BAND_INDICES is not None else list(range(13))
    _, val_ds, _ = load_ms_dataset(MS_DATA_DIR, band_indices=band_indices)

# Load the saved model and run predictions
model = keras.models.load_model(CHECKPOINT)

all_labels, all_preds = [], []
for images, labels in val_ds:
    probs = model.predict(images, verbose=0)
    all_preds.extend(np.argmax(probs, axis=1))
    all_labels.extend(labels.numpy())

all_labels = np.array(all_labels)
all_preds  = np.array(all_preds)

accuracy = (all_labels == all_preds).mean()
print(f"Validation accuracy: {accuracy:.4f}  ({accuracy*100:.2f}%)")

# Per-class metrics
print(classification_report(all_labels, all_preds, target_names=CLASSES))

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm_norm, annot=True, fmt=".2f",
    xticklabels=CLASSES, yticklabels=CLASSES,
    cmap="Blues", vmin=0, vmax=1,
)
plt.xlabel("Predicted class")
plt.ylabel("True class")
plt.title("Confusion matrix (normalised)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()

os.makedirs("results", exist_ok=True)
run_name = os.path.basename(CHECKPOINT).replace("_best.keras", "")
plt.savefig(f"results/{run_name}_confusion_matrix.png", dpi=150, bbox_inches="tight")
plt.show()