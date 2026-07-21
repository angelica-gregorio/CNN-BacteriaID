"""Custom CNN architecture for 33-class bacterial species classification.

Sized deliberately small/shallow: ~2,700 images across 33 classes is a high
overfitting risk, so the architecture favors regularization (BatchNorm,
Dropout, L2, GlobalAveragePooling instead of Flatten) over depth.
"""

from tensorflow import keras
from tensorflow.keras import layers, regularizers


def build_cnn(
    input_shape: tuple[int, int, int] = (128, 128, 3),
    num_classes: int = 33,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
) -> keras.Model:
    """Build an uncompiled Keras CNN. Compilation (optimizer/loss/metrics) is
    left to train.py so the same architecture can be reused across experiments
    (e.g. raw vs segmented) with different training configs.
    """
    reg = regularizers.l2(l2_reg)
    inputs = keras.Input(shape=input_shape)
    x = layers.Rescaling(1.0 / 255)(inputs)

    # Block 1
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(32, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 2
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(64, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.25)(x)

    # Block 3
    x = layers.Conv2D(128, 3, padding="same", kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D(2)(x)
    x = layers.Dropout(0.3)(x)

    # Global average pooling instead of Flatten: far fewer parameters feed
    # into the dense head, meaningfully reducing overfitting risk.
    x = layers.GlobalAveragePooling2D()(x)

    x = layers.Dense(256, kernel_regularizer=reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(dropout_rate)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs, outputs, name="bacteria_cnn")


if __name__ == "__main__":
    model = build_cnn()
    model.summary()
