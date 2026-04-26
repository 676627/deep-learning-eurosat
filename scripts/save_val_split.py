import sys
sys.path.insert(0, '.')
import numpy as np
from src.dataset import CLASSES
import os

# Reproduce the exact same split used during training
data_dir = "data/EuroSAT_RGB"
seed = 42

# Collect all file paths in the same order as the dataset code
all_paths = []
for class_name in CLASSES:
    class_dir = os.path.join(data_dir, class_name)
    files = sorted(f for f in os.listdir(class_dir) if f.lower().endswith(".jpg"))
    for fname in files:
        all_paths.append(os.path.join(class_dir, fname))

all_paths = np.array(all_paths)

# Reproduce the exact shuffle
rng = np.random.default_rng(seed)
idx   = rng.permutation(len(all_paths))
n_val = int(len(all_paths) * 0.2)

val_paths = all_paths[idx[:n_val]]

# Save to a text file
np.savetxt("val_images.txt", val_paths, fmt="%s")