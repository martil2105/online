import warnings
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, losses, metrics
from tensorflow.keras.optimizers import Adam


def general_simple_nn(n, l, m, num_classes, model_name="simple_nn", activationfunction = "relu"):
    """
    To construct a simple neural network.

    Parameters
    ----------
    n : scalar
        the input size
    l : scalar
        the number of hidden layers
    m : scalar or 1D array
        the width vector of hidden layers, if it is a scalar, then the hidden layers of simple neural network have the same nodes.
    num_classes : scalar
        the nodes of output layers, i.e., the number of classes
    model_name : str, optional
        the model name, by default "simple_nn"

    Returns
    -------
    model
        the simple neural network
    """
    input_layer = layers.Input(shape=(n,), name="Input")
    if isinstance(m, int):
        m_vec = np.repeat(m, l)
    elif len(m) == l:
        m_vec = m
    else:
        warnings.warn(
            "The length of width vector must be equal to the number of hidden layers.",
            DeprecationWarning,
        )

    x = layers.Dense(m_vec[0], activation=activationfunction, kernel_regularizer="l2")(input_layer)
    if l >= 2:
        for k in range(l - 1):
            x = layers.Dense(m_vec[k + 1], activation=activationfunction, kernel_regularizer="l2")(
                x
            )

    output_layer = layers.Dense(num_classes)(x)
    model = models.Model(input_layer, output_layer, name=model_name)
    return model


# mymodel = simple_nn(n=100, l=1, m=10, num_classes=2)
# mymodel = simple_nn(n=100, l=3, m=10, num_classes=2)
# mymodel = simple_nn(n=100, l=3, m=[20, 20, 5], num_classes=2)

# build the model, train and save it to disk


def get_optimizer(learning_rate):
    """
    To get the optimizer given the learning rate.

    Parameters
    ----------
    learning_rate : float
        the learning rate for inverse time decay schedule.

    Returns
    -------
    optimizer
        the Adam
    """

    return tf.keras.optimizers.Adam(learning_rate=learning_rate)


def get_callbacks(name, log_dir, epochdots):
    """
    Get callbacks. This function returns the result of epochs during training, if it satisfies some conditions then the training can stop early. At meanwhile, this function also save the results of training in TensorBoard and csv files.

    Parameters
    ----------
    name : str
        the model name
    log_dir : str
        the path of log files
    epochdots : object
        the EpochDots object from tensorflow_docs

    Returns
    -------
    list
        the list of callbacks
    """
    name1 = name + "/log.csv"
    return [
        epochdots,
        tf.keras.callbacks.EarlyStopping(
            monitor="val_sparse_categorical_crossentropy", patience=800, min_delta=1e-3
        ),
        tf.keras.callbacks.TensorBoard(Path(log_dir, name)),
        tf.keras.callbacks.CSVLogger(Path(log_dir, name1)),
    ]


def compile_and_fit(
    model,
    x_train,
    y_train,
    batch_size,
    lr,
    name,
    log_dir,
    epochdots,
    optimizer=None,
    validation_split=0.2,
    max_epochs=10000,
):
    """
    To compile and fit the model

    Parameters
    ----------
    model : Models object
        the simple neural network
    x_train : tf.Tensor
        the tensor of training data
    y_train : tf.Tensor
        the tensor of training data, label
    batch_size : int
        the batch size
    lr : float
        the learning rate
    name : str
        the model name
    log_dir : str
        the path of log files
    epochdots : object
        the EpochDots object from tensorflow_docs
    optimizer : optimizer object or str, optional
        the optimizer, by default None
    max_epochs : int, optional
        the maximum number of epochs, by default 10000

    Returns
    -------
    model.fit object
        a fitted model object
    """
    if optimizer is None:
        optimizer = get_optimizer(lr)
    model.compile(
        optimizer=optimizer,
        loss=losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=[
            metrics.SparseCategoricalCrossentropy(
                from_logits=True, name="sparse_categorical_crossentropy"
            ),
            "accuracy",
        ],
    )
    history = model.fit(
        x_train,
        y_train,
        epochs=max_epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=get_callbacks(name, log_dir, epochdots),
        verbose=0,
    )
    return history

