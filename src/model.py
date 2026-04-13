import keras


def build_model(in_channels, num_classes=10):
    """
    Builds a simple CNN for EuroSAT land cover classification.

    Architecture:
        4x (Conv2D → BatchNorm → ReLU → MaxPool)
        → GlobalAveragePooling2D
        → Dropout
        → Dense (10 classes, softmax)

    The only argument that changes between experiments is in_channels:
        3  for RGB
        13 for full multispectral
        n  for a band ablation experiment using n bands
    """
    inputs = keras.Input(shape=(64, 64, in_channels))

    x = inputs
    for filters in [32, 64, 128, 256]:
        x = keras.layers.Conv2D(filters, kernel_size=3, padding="same", use_bias=False)(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.Activation("relu")(x)
        x = keras.layers.MaxPooling2D(pool_size=2)(x)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.5)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs)
    return model