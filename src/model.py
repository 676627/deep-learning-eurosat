import keras


def build_model(in_channels, model_version="batchnorm_3conv", num_classes=10, dropout=0.4):
    if model_version == "baseline":
        # Baseline architecture for RGB (3 channels) and multispectral (13 channels):
        return keras.Sequential([
            keras.Input(shape=(64, 64, in_channels)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])

    elif model_version == "batchnorm":
        # Add BatchNorm after each 
        return keras.Sequential([
            keras.Input(shape=(64, 64, in_channels)),
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.1),
            keras.layers.Conv2D(64, kernel_size=(3, 3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(64, kernel_size=(3, 3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])

    elif model_version == "batchnorm_3conv":
        # Add an extra Conv layer and BatchNorm after each Conv
        return keras.Sequential([
            keras.Input(shape=(64, 64, in_channels)),
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.1),
            keras.layers.Conv2D(64, kernel_size=(3, 3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(128, kernel_size=(3, 3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Conv2D(256, kernel_size=(3, 3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(pool_size=(2, 2)),
            keras.layers.Flatten(),
            keras.layers.Dropout(dropout),
            keras.layers.Dense(num_classes, activation="softmax"),
        ])

    elif model_version == "batchnorm_3conv_gap":
        # Same as above but with GlobalAveragePooling instead of Flatten and bump Dropout to 0.5
        return keras.Sequential([
            keras.Input(shape=(64, 64, in_channels)),
            keras.layers.RandomFlip("horizontal_and_vertical"),
            keras.layers.RandomRotation(0.1),
            keras.layers.Conv2D(64, (3,3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, (3,3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(128, (3,3), use_bias=False),
            keras.layers.BatchNormalization(),
            keras.layers.Activation("relu"),
            keras.layers.MaxPooling2D(),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.5),
            keras.layers.Dense(10, activation="softmax"),
        ])
    else:
        raise ValueError(
            f"Unknown model_version '{model_version}'. "
        )

