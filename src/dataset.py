import os
import numpy as np
import keras
import tensorflow as tf
import rasterio
from tqdm import tqdm

# The 10 land cover classes in EuroSAT
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway",
    "Industrial", "Pasture", "PermanentCrop", "Residential",
    "River", "SeaLake",
]

# Sentinel-2 band names, in the order they appear in the .tif files
BAND_NAMES = [
    "B01 (Coastal aerosol)", "B02 (Blue)", "B03 (Green)", "B04 (Red)",
    "B05 (Red Edge 1)", "B06 (Red Edge 2)", "B07 (Red Edge 3)",
    "B08 (NIR)", "B08A (Narrow NIR)", "B09 (Water vapour)",
    "B10 (SWIR cirrus)", "B11 (SWIR 1)", "B12 (SWIR 2)",
]


def load_rgb_dataset(data_dir, image_size=(64, 64), batch_size=64, seed=42, max_per_class=None):
    if max_per_class is None:
        # Original fast path — image_dataset_from_directory handles everything
        train_ds, val_ds = keras.utils.image_dataset_from_directory(
            data_dir,
            validation_split=0.2,
            subset="both",
            seed=seed,
            image_size=image_size,
            batch_size=batch_size,
            label_mode="int",
        )
        rescale = keras.layers.Rescaling(1.0 / 255)
        train_ds = train_ds.map(lambda x, y: (rescale(x), y))
        val_ds   = val_ds.map(lambda x, y: (rescale(x), y))
        return train_ds, val_ds

    # Subset path — collect file paths manually, limit per class, then build dataset
    all_paths, all_labels = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        files = sorted(f for f in os.listdir(class_dir)
                       if f.lower().endswith(".jpg"))[:max_per_class]
        for fname in files:
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(label_idx)

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels, dtype=np.int32)

    # Shuffle and split 80/20
    rng = np.random.default_rng(seed)
    idx   = rng.permutation(len(all_paths))
    n_val = int(len(all_paths) * 0.2)
    train_idx = idx[n_val:]
    val_idx   = idx[:n_val]

    def load_image(fpath, label):
        img = tf.io.read_file(fpath)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, image_size)
        img = tf.cast(img, tf.float32) / 255.0
        return img, label

    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (all_paths[train_idx], all_labels[train_idx])
        )
        .shuffle(len(train_idx), seed=seed)
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )
    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (all_paths[val_idx], all_labels[val_idx])
        )
        .map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds


def load_ms_dataset(data_dir, band_indices=None, batch_size=64, seed=42, max_per_class=None):
    """
    Loads the EuroSAT multispectral dataset (.tif files, 13 bands).

    Images are read lazily from disk one batch at a time during training,
    so the full dataset is never held in RAM.

    Normalisation stats are estimated from a random sample of 2000 images
    rather than the full dataset — accurate enough and avoids a full pass.

    band_indices: list of band indices to use (0-12). Default: all 13.
                  Example: [3, 2, 1] loads only the RGB-equivalent bands.

    max_per_class: if not None, limits the number of images loaded from each class.

    Returns train_ds, val_ds, and stats (mean/std used for normalisation).
    """
    if band_indices is None:
        band_indices = list(range(13))
    band_indices = list(band_indices)   # ensure plain Python list

    # Collect all file paths and labels
    all_paths, all_labels = [], []
    for label_idx, class_name in enumerate(CLASSES):
        class_dir = os.path.join(data_dir, class_name)
        files = sorted(f for f in os.listdir(class_dir) if f.endswith(".tif"))
        if max_per_class is not None:
            rng_class = np.random.default_rng(seed)
            files = rng_class.choice(files, size=max_per_class, replace=False).tolist() 
        for fname in files:
            all_paths.append(os.path.join(class_dir, fname))
            all_labels.append(label_idx)

    all_paths  = np.array(all_paths)
    all_labels = np.array(all_labels, dtype=np.int32)

    # Shuffle and split 80/20 using index arrays (no files read yet)
    rng = np.random.default_rng(seed)
    idx   = rng.permutation(len(all_paths))
    n_val = int(len(all_paths) * 0.2)
    train_idx = idx[n_val:]
    val_idx   = idx[:n_val]

    print(f"Found {len(all_paths)} images | "
          f"{len(train_idx)} train, {n_val} val | "
          f"{len(band_indices)} bands")

    # Estimate normalisation stats from a small sample (2000 train images)
    print("Estimating normalisation stats from 2000 training images...")
    sample_idx = rng.choice(train_idx, size=min(2000, len(train_idx)), replace=False)
    sample_imgs = []
    for fpath in tqdm(all_paths[sample_idx], desc="Sampling"):
        with rasterio.open(fpath) as src:
            img = src.read(list(np.array(band_indices) + 1))  # rasterio is 1-indexed
        img = img.astype(np.float32) / 10000.0
        img = np.transpose(img, (1, 2, 0))                    # (64, 64, n_bands)
        sample_imgs.append(img)
    sample_imgs = np.stack(sample_imgs)                        # (2000, 64, 64, n_bands)
    mean = sample_imgs.mean(axis=(0, 1, 2)).tolist()
    std  = (sample_imgs.std(axis=(0, 1, 2)) + 1e-6).tolist()
    stats = {"mean": mean, "std": std}
    del sample_imgs   # free the sample from RAM immediately

    # Define a function that reads one .tif file and returns a normalised
    # image tensor. tf.data calls this lazily for each file during training.
    mean_t = tf.constant(mean, dtype=tf.float32)    # shape: (n_bands,)
    std_t  = tf.constant(std,  dtype=tf.float32)

    def read_tif(fpath_bytes, label):
        """Reads one .tif file. Called inside tf.py_function."""
        fpath = fpath_bytes.numpy().decode("utf-8")
        with rasterio.open(fpath) as src:
            img = src.read(list(np.array(band_indices) + 1))   # 1-indexed
        img = img.astype(np.float32) / 10000.0
        img = np.transpose(img, (1, 2, 0))                     # (64, 64, n_bands)
        return img, label

    def load_and_normalise(fpath, label):
        """
        tf.data wrapper around read_tif.
        tf.py_function lets us call plain Python/numpy inside a tf.data pipeline.
        """
        img, label = tf.py_function(
            func=read_tif,
            inp=[fpath, label],
            Tout=[tf.float32, tf.int32],
        )
        img = (img - mean_t) / std_t
        # tf.py_function loses shape info, so we restore it explicitly
        img.set_shape([64, 64, len(band_indices)])
        label.set_shape([])
        return img, label

    # Build tf.data pipelines — files are read on demand during training
    train_ds = (
        tf.data.Dataset.from_tensor_slices(
            (all_paths[train_idx], all_labels[train_idx])
        )
        .shuffle(len(train_idx), seed=seed)
        .map(load_and_normalise, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    val_ds = (
        tf.data.Dataset.from_tensor_slices(
            (all_paths[val_idx], all_labels[val_idx])
        )
        .map(load_and_normalise, num_parallel_calls=tf.data.AUTOTUNE)
        .batch(batch_size)
        .prefetch(tf.data.AUTOTUNE)
    )

    return train_ds, val_ds, stats