import keract
import numpy as np
from keras import layers
from tensorflow import keras


def get_mnist_data(is_model_conv: bool):
    # Model / data parameters
    num_classes = 10
    input_shape = (28, 28, 1) if is_model_conv else (28, 28)

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Scale images to the [0, 1] range
    x_train = x_train.astype("float32") / 255
    x_test = x_test.astype("float32") / 255

    if is_model_conv:
        # Make sure images have shape (28, 28, 1)
        x_train = np.expand_dims(x_train, -1)
        x_test = np.expand_dims(x_test, -1)
    print("x_train shape:", x_train.shape)
    print(x_train.shape[0], "train samples")
    print(x_test.shape[0], "test samples")

    if is_model_conv:
        # convert class vectors to binary class matrices
        y_train = keras.utils.to_categorical(y_train, num_classes)
        y_test = keras.utils.to_categorical(y_test, num_classes)

    return num_classes, input_shape, x_train, x_test, y_train, y_test


def dense_model(num_classes: int, input_shape: tuple[int, ...]):
    model = keras.models.Sequential([
        keras.layers.Flatten(input_shape=input_shape),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate=0.25),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(rate=0.25),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(rate=0.25),
        keras.layers.Dense(num_classes)
    ])
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    return model


def conv_model(num_classes: int, input_shape: tuple[int, ...]):
    model = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            layers.Conv2D(32, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model


def train_model(model, x_train, y_train, batch_size, epochs):
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.1)
    return model


def main():
    # num_classes, input_shape, x_train, x_test, y_train, y_test = get_mnist_data(is_model_conv=True)
    # model = conv_model(num_classes, input_shape)

    num_classes, input_shape, x_train, x_test, y_train, y_test = get_mnist_data(is_model_conv=False)
    model = dense_model(num_classes, input_shape)

    model.summary(line_length=120)

    train_model(model, x_train, y_train, batch_size=128, epochs=10)

    image = x_test[0][None, ...]  # Need to add the batch axis
    activations = keract.get_activations(model, image)

    for name, output in activations.items():
        print()
        print(f'Layer: {name}, shape: {output.shape}')
        print(output)

    return


if __name__ == '__main__':
    main()
