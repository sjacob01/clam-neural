import pathlib

import keract
import numpy
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras import Model


def get_mnist_data():
    # Model / data parameters
    input_shape = (28, 28)

    # the data, split between train and test sets
    (mnist_train, mnist_labels), _ = keras.datasets.mnist.load_data()
    # Scale images to the [0, 1] range
    mnist_train = mnist_train.astype('float32') / 255

    (fashion_train, fashion_labels), _ = keras.datasets.fashion_mnist.load_data()
    # Scale images to the [0, 1] range
    fashion_train = fashion_train.astype('float32') / 255

    return input_shape, mnist_train, mnist_labels, fashion_train, fashion_labels


def dense_model(
        num_classes: int,
        input_shape: tuple[int, int],
        neurons: list[int],
) -> Model:

    dense_layers: list[layers.Layer] = list()
    [dense_layers.extend(
        [layers.Dense(n, activation='relu', name=f'dense_{n}'), layers.Dropout(rate=0.25)]
    ) for n in neurons]
    model = keras.models.Sequential([
        layers.Flatten(input_shape=input_shape),
        *dense_layers,
        layers.Dense(num_classes),
    ])
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


def save_activations(name: str, activations: list[numpy.ndarray], labels: numpy.ndarray):

    data_root = pathlib.Path(__file__).parent.joinpath('data_cross')
    numpy.save(
        str(data_root.joinpath(f'{name}_labels.npy')),
        numpy.asarray(labels, dtype=numpy.uint8),
    )

    for i, output in enumerate(activations[1:], start=1):
        numpy.save(
            str(data_root.joinpath(f'{name}_activations_{i}.npy')),
            numpy.asarray(output, dtype=numpy.float32),
        )

    return


def train_model(
        train_name: str,
        test_name: str,
        input_shape,
        x_train,
        y_train,
        x_test,
        y_test,
):
    model = dense_model(10, input_shape, [588, 392, 196, 64, 32])
    model.build(input_shape=(None, *input_shape))
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.summary(line_length=120)

    model_path = pathlib.Path(__file__).parent.joinpath('saved_models').joinpath(train_name)
    if model_path.exists():
        model.load_weights(str(model_path))
    else:
        callbacks = [
            keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-4, patience=8, mode='min')
        ]
        model.fit(
            x_train,
            y_train,
            batch_size=128,
            epochs=1024,
            validation_split=0.2,
            callbacks=callbacks,
        )

        model.save(str(model_path))

    activations = [
        output
        for i, (_, output) in enumerate(keract.get_activations(model, x_train).items())
        if i % 2 == 0
    ]
    save_activations(train_name, activations, y_train)

    activations = [
        output
        for i, (_, output) in enumerate(keract.get_activations(model, x_test).items())
        if i % 2 == 0
    ]
    save_activations(test_name, activations, y_test)


def main():
    # num_classes, input_shape, x_train, x_test, y_train, y_test = get_mnist_data(is_model_conv=True)
    # model = conv_model(num_classes, input_shape)

    input_shape, mnist_train, mnist_labels, fashion_train, fashion_labels = get_mnist_data()

    train_model(
        'fashion',
        'mnist',
        input_shape,
        fashion_train, fashion_labels,
        mnist_train, mnist_labels,
    )

    return


if __name__ == '__main__':
    main()
